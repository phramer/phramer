import json
import multiprocessing as mp
import re
from functools import partial
from pathlib import Path
from string import punctuation

import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

from phramer.config import (
    ARTICLES_TAG,
    CNNDM_END_TOKENS,
    CNNDM_SENTENCE_END,
    CNNDM_SENTENCE_START,
    CNNDM_TAG,
    MAX_TASKS_PER_CHILD,
    PHRAMER_STOP_MESSAGE,
    RIA_DATASET_TAG,
    SUMMARIES_TAG,
)
from phramer.utils.file import count_lines, list_files

nltk.download("stopwords")


class CNNDailyMail:
    """
    Handler for CNN DailyMail dataset
    """
    def _fix_missing_period(self, line):
        if line == "":
            return line
        if line[-1] in CNNDM_END_TOKENS:
            return line
        return line + " ."

    def _filter(self, line):
        english_stopwords = stopwords.words("english") 
        line = self._fix_missing_period(line)

        return line

    def _lemmatize(self, line):
        pass
    def _process_line(self, line):
        return self._lemmatize(self._filter(line.lower())) 

    def _parse_lines(self, lines):
        article_lines = []
        highlight_lines = []
        next_is_highlight = False
        for line in map(self._process_line, lines):
            if line == "":
                continue
            elif line.startswith("@highlight"):
                next_is_highlight = True
            elif next_is_highlight:
                highlight_lines.append(line)
            else:
                article_lines.append(line)
        article = " ".join(article_lines)
        summary = " ".join(
            [
                "{} {} {}".format(
                    CNNDM_SENTENCE_START, sent, CNNDM_SENTENCE_END
                )
                for sent in highlight_lines
            ]
        )
        return article, summary

    def _read_text_file(self, filename):
        lines = []
        with open(filename, "r") as f:
            for line in f:
                lines.append(line.strip())
        return lines

    def _reading_listener(self, reading_queue, processing_queue):
        while True:
            message = reading_queue.get()
            if message == PHRAMER_STOP_MESSAGE:
                processing_queue.put(PHRAMER_STOP_MESSAGE)
            processing_queue.put(self._read_text_file(message))

    def _writing_listener(
        self, processing_queue, articles_fn, summaries_fn, num_total_stories
    ):
        with open(articles_fn, "w") as articles, open(
            summaries_fn, "w"
        ) as summaries:
            with tqdm(total=num_total_stories, desc="Stories") as pbar:
                while True:
                    message = processing_queue.get()
                    if message == PHRAMER_STOP_MESSAGE:
                        break
                    article, summary = self._parse_lines(message)
                    articles.write(str(article) + "\n")
                    summaries.write(str(summary) + "\n")
                    articles.flush()
                    summaries.flush()
                    pbar.update()

    def preprocess(
        self,
        cnn_dir,
        dm_dir,
        target_dir,
        num_workers=mp.cpu_count() - 1,
        dataset_tag=CNNDM_TAG,
    ):
        files = list_files(cnn_dir) + list_files(dm_dir)
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_articles_path = target_dir / "{}.{}".format(
            dataset_tag, ARTICLES_TAG
        )
        target_summaries_path = target_dir / "{}.{}".format(
            dataset_tag, SUMMARIES_TAG
        )
        if target_articles_path.exists() or target_summaries_path.exists():
            raise RuntimeError(
                "Aborted the attempt to overwrite "
                + "existing files {} and {}".format(
                    target_articles_path, target_summaries_path
                )
            )

        manager = mp.Manager()
        pool = mp.Pool(num_workers, maxtasksperchild=MAX_TASKS_PER_CHILD)

        reading_queue = manager.Queue()
        processing_queue = manager.Queue()

        pool.apply_async(
            self._reading_listener, [reading_queue, processing_queue]
        )

        pool.apply_async(
            self._writing_listener,
            [
                processing_queue,
                target_articles_path,
                target_summaries_path,
                len(files),
            ],
        )
        for filename in files:
            reading_queue.put(filename)

        reading_queue.put(PHRAMER_STOP_MESSAGE)

        pool.close()
        pool.terminate()


class GigawordDataset:
    """
    Handler for Gigaword dataset
    """
    def preprocess(self):
        pass


class RIANewsDataset:
    """
    Handler for RIA News dataset
    """
    def __init__(self):
        from bs4 import BeautifulSoup
        from pymystem3 import Mystem

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

    def _parse_record(self, record, queue, should_lemmatize=True):
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

    def preprocess(
        self,
        source_filename,
        target_dir,
        num_workers=mp.cpu_count() - 1,
        should_lemmatize=True,
        dataset_tag=RIA_DATASET_TAG,
    ):
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_articles_path = target_dir / "{}.{}".format(
            dataset_tag, ARTICLES_TAG
        )
        target_summaries_path = target_dir / "{}.{}".format(
            dataset_tag, SUMMARIES_TAG
        )
        if target_articles_path.exists() or target_summaries_path.exists():
            raise RuntimeError(
                "Aborted the attempt to overwrite "
                + "existing files {} and {}".format(
                    target_articles_path, target_summaries_path
                )
            )

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
                        self._parse_record,
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
