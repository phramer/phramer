import json
import logging
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

logging.basicConfig(level=logging.INFO)
nltk.download("stopwords")


class CNNDailyMail:
    """
    Handler for CNN DailyMail dataset
    """
    def __init__(self):
        from nltk.stem import WordNetLemmatizer

        self.lemmatizer = WordNetLemmatizer()
        nltk.download("wordnet")

    def _fix_missing_period(self, line):
        if line == "":
            return line
        if line[-1] in CNNDM_END_TOKENS:
            return line
        return line + " ."

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
        text = self._fix_missing_period(text)
        return text

    def _lemmatize(self, line):
        english_stopwords = stopwords.words("english")
        tokens = self.lemmatizer.lemmatize(line).split(" ")
        tokens = [
            token
            for token in tokens
            if token not in english_stopwords
            and token != " "
            and token.strip() not in punctuation
        ]
        text = " ".join(tokens)
        text = re.sub(" +", " ", text)
        return text

    def _parse_lines(self, lines):
        article_lines = []
        highlight_lines = []
        next_is_highlight = False
        for line in map(self._filter, lines):
            if line == "":
                continue
            elif line.startswith("@highlight"):
                next_is_highlight = True
            elif next_is_highlight:
                highlight_lines.append(line)
            else:
                article_lines.append(line)

        article = self._lemmatize(" ".join(article_lines))
        summary = " ".join(
            [
                "{} {} {}".format(
                    CNNDM_SENTENCE_START, sent, CNNDM_SENTENCE_END
                )
                for sent in highlight_lines
            ]
        )
        article = article.replace("\n", "")
        summary = summary.replace("\n", "")
        article = re.sub(" +", " ", article)
        summary = re.sub(" +", " ", summary)
        return article, summary

    def _read_text_file(self, filename):
        lines = []
        with open(filename, "r") as f:
            for line in f:
                lines.append(line.strip())
        return lines

    def _process_story(self, filename, writing_queue):
        lines = self._read_text_file(filename)
        data = self._parse_lines(lines)
        writing_queue.put(data)
        return data

    def _writing_listener(self, writing_queue, articles_fn, summaries_fn):
        with open(articles_fn, "w") as articles, open(
            summaries_fn, "w"
        ) as summaries:
            while True:
                message = writing_queue.get()
                if message == PHRAMER_STOP_MESSAGE:
                    break
                article, summary = message
                articles.write(str(article) + "\n")
                summaries.write(str(summary) + "\n")
                articles.flush()
                summaries.flush()

    def preprocess(
        self,
        cnn_dir,
        dm_dir,
        target_dir,
        num_workers=mp.cpu_count() - 1,
        dataset_tag=CNNDM_TAG,
    ):
    """
    Cleans the CNN/DM stories up and separates the highlights into a separate file. 

    Args:
        cnn_dir: the directory with CNN stories
        dm_dir: the directory with Daily Mail stories
        target_dir: the directory where to save preprocessed data
        num_workers: the number of parallel workers in the pool (default: all cpus but one)
        dataset_tag: the file tag for the dataset (affects the target path) 
                     (default: cnndm)
    """
        logging.info(
            "Preparing to process the CNN dataset. This might take a while..."
        )
        files = list_files(cnn_dir) + list_files(dm_dir)
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_articles_path = target_dir / "{}.{}".format(
            dataset_tag, ARTICLES_TAG
        )
        target_summaries_path = target_dir / "{}.{}".format(
            dataset_tag, SUMMARIES_TAG
        )

        logging.info(
            "Processing {} CNN and DailyMail stories from {} and {} ".format(
                len(files), cnn_dir, dm_dir
            )
            + "to {} and {}...".format(
                target_articles_path, target_summaries_path
            )
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
        logging.info("Started the writing queue...")
        writing_queue = manager.Queue()

        pool.apply_async(
            self._writing_listener,
            [writing_queue, target_articles_path, target_summaries_path],
        )

        with tqdm(total=len(files), desc="Processed texts") as pbar:
            for _ in pool.imap_unordered(
                partial(self._process_story, writing_queue=writing_queue),
                files,
            ):
                pbar.update()

        logging.info("Finished processing the files...")
        writing_queue.put(PHRAMER_STOP_MESSAGE)

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

    def _parse_html(self, html):
        from bs4 import BeautifulSoup

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
        from pymystem3 import Mystem

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
    """
    Parses and cleans up the RIA news dataset into separate files for stories and titles.

    Args:
        source_filename: the file with line-by-line json representation of the data
        target_dir: the target directory for the files
        num_workers: the number of workers in the pool (default: all CPUs but one)
        should_lemmatize: whether to lemmatize the words (default: True)
        dataset_tag: the file tag for the dataset (affects the target path) 
                     (default: ria)
    """
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
        logging.info("Finished processing the RIA dataset.")
