import gc
import json
import multiprocessing as mp
import re
from pathlib import Path
from string import punctuation

import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from pymystem3 import Mystem
from tqdm import tqdm

from phramer.config import (
    ARTICLES_TAG,
    MAX_TASKS_PER_CHILD,
    PHRAMER_STOP_MESSAGE,
    RIA_DATASET_TAG,
    SUMMARIES_TAG,
)
from phramer.data import Dataset
from phramer.utils.file import chunkify, process_chunk

nltk.download("stopwords")


class RIANewsDataset(Dataset):
    def _parse_html(self, html):
        return BeautifulSoup(html, "lxml").text.replace("\xa0", " ")

    def _filter(self, text):
        text = text.lower()
        text = re.sub(r"https?:\/\/.*[\r\n]*", "", text, flags=re.MULTILINE)
        text = re.sub(r'[_"\-;%()|.,+&=*%]', " ", text)
        text = re.sub(r"\.", " . ", text)
        text = re.sub(r"\!", " !", text)
        text = re.sub(r"\?", " ?", text)
        text = re.sub(r"\,", " ,", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"#", " # ", text)
        text = re.sub(r" . . . ", " ", text)
        text = re.sub(r" .  .  . ", " ", text)
        text = re.sub(r" ! ! ", " ! ", text)
        text = text.replace("\n", "")
        return text

    def _lemmatize(self, text):
        mystem = Mystem()
        russian_stopwords = stopwords.words("russian")
        tokens = mystem.lemmatize(text.lower())
        tokens = [
            token
            for token in tokens
            if token not in russian_stopwords
            and token != " "
            and token.strip() not in punctuation
        ]
        text = " ".join(tokens)
        text = re.sub(" +", " ", text)
        return text

    def _process_article(self, html, should_lemmatize=True):
        text = self._parse_html(html)
        text = self._filter(text)
        if should_lemmatize:
            text = self._lemmatize(text)
        return text

    def parse_record(
        self, record, should_lemmatize, articles_queue, summaries_queue
    ):
        json_data = json.loads(record)

        article = self._process_article(
            json_data["text"], should_lemmatize=should_lemmatize
        )
        articles_queue.put(article)

        title = json_data["title"]
        summaries_queue.put(title)

        return article, title

    def _listener(self, queue, filename):
        with open(filename, "w") as f:
            while True:
                message = queue.get()
                if message == PHRAMER_STOP_MESSAGE:
                    break
                print(str(message), file=f)
                f.flush()

    def _parse_raw(
        self,
        source_filename,
        target_dir,
        num_workers=mp.cpu_count() - 1,
        chunk_size=100,
        skiplines=0,
        **kwargs,
    ):
        with open(source_filename, "rb") as source:
            source_num_lines = sum(1 for _ in source)

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_articles_path = target_dir / "{}.{}".format(
            RIA_DATASET_TAG, ARTICLES_TAG
        )
        target_summaries_path = target_dir / "{}.{}".format(
            RIA_DATASET_TAG, SUMMARIES_TAG
        )

        should_lemmatize = getattr(kwargs, "lemmatize", True)

        manager = mp.Manager()
        pool = mp.Pool(num_workers, maxtasksperchild=MAX_TASKS_PER_CHILD)

        # article writer
        articles_queue = manager.Queue()
        pool.apply_async(
            self._listener, [articles_queue, target_articles_path]
        )
        print("Started the queue for articles...")

        # summary writer
        summaries_queue = manager.Queue()
        pool.apply_async(
            self._listener, [summaries_queue, target_summaries_path]
        )
        print("Started the queue for titles...")

        jobs = chunkify(
            source_filename,
            size=int(1024 * 1024 * chunk_size),
            skiplines=skiplines,
        )
        print(
            "Running {} workers on {} chunks...".format(num_workers, len(jobs))
        )

        jobs = [
            list(job)
            + [
                self.parse_record,
                should_lemmatize,
                articles_queue,
                summaries_queue,
            ]
            for job in jobs
        ]

        with tqdm(
            total=(len(jobs) - 1) // num_workers + 1, desc="Jobs"
        ) as pbar:
            for worker_idx in range(0, len(jobs), num_workers):
                for i, _ in tqdm(
                    enumerate(
                        pool.imap_unordered(
                            process_chunk,
                            jobs[worker_idx : worker_idx + num_workers],
                        )
                    )
                ):
                    pbar.update()
                gc.collect()
        articles_queue.put(PHRAMER_STOP_MESSAGE)
        summaries_queue.put(PHRAMER_STOP_MESSAGE)
        pool.close()
        pool.terminate()
        print("Finished processing the RIA dataset.")
