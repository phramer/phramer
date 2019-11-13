import gc
import json
import multiprocessing as mp
import re
from functools import partial
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
from phramer.utils.file import count_lines

nltk.download("stopwords")


class RIANewsDataset:
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
        text = text.replace("\n", " ")
        return text

    def parse_record(self, record, queue, should_lemmatize=True):
        json_data = json.loads(record)

        article = self._process_article(
            json_data["text"], should_lemmatize=should_lemmatize
        )
        title = json_data["title"]

        queue.put((article, title))

        return article, title

    def _listener(self, queue, articles_fn, summaries_fn):
        with open(articles_fn, "w") as articles, open(
            summaries_fn, "w"
        ) as summaries:
            while True:
                message = queue.get()
                if message == PHRAMER_STOP_MESSAGE:
                    break
                article, summary = message
                articles.write(str(article) + "\n")
                summaries.write(str(summary) + "\n")
                articles.flush()
                summaries.flush()

    def _parse_raw(
        self,
        source_filename,
        target_dir,
        num_workers=mp.cpu_count() - 1,
        skiplines=0,
        **kwargs,
    ):
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_articles_path = target_dir / "{}.{}".format(
            RIA_DATASET_TAG, ARTICLES_TAG
        )
        target_summaries_path = target_dir / "{}.{}".format(
            RIA_DATASET_TAG, SUMMARIES_TAG
        )
        if target_articles_path.exists() or target_summaries_path.exists():
            raise RuntimeError(
                "Aborted the attempt to overwrite "
                + "existing files {} and {}".format(
                    target_articles_path, target_summaries_path
                )
            )

        should_lemmatize = getattr(kwargs, "lemmatize", True)

        manager = mp.Manager()
        pool = mp.Pool(num_workers, maxtasksperchild=MAX_TASKS_PER_CHILD)

        # init writer
        queue = manager.Queue()
        pool.apply_async(
            self._listener,
            [queue, target_articles_path, target_summaries_path],
        )
        num_lines = count_lines(source_filename)
        with open(source_filename, "rb") as source:
            with tqdm(total=num_lines, desc="Lines") as pbar:
                for _ in pool.imap_unordered(
                    partial(
                        self.parse_record,
                        queue=queue,
                        should_lemmatize=should_lemmatize,
                    ),
                    source,
                ):
                    pbar.update()
        queue.put(PHRAMER_STOP_MESSAGE)
        pool.close()
        pool.terminate()
        print("Finished processing the RIA dataset.")
