import numpy as np
from tqdm import tqdm
from phramer.config import (
    BY_LENGTH,
    TRAIN_VAL_TEST_SPLIT,
    TRAIN_TAG,
    TEST_TAG,
    VAL_TAG,
)
from phramer.utils.file import count_lines
from pathlib import Path

SAMPLING_METHODS = {BY_LENGTH}


def split(
    texts_path,
    summaries_path,
    target_dir=Path("."),
    ratio=TRAIN_VAL_TEST_SPLIT,
):
    """
    Splits texts and summaries on train/test 
    
    Args:
        texts_path: path to the file with texts
        summaries_path: path to the file with summaries
        target_dir: dir to save splitted texts and summaries
                    (default: Path("."))
        ratio: ratio of train/val/test split
               (default: look at phramer/phramer/config.py)
    """
    articles_num_lines = count_lines(texts_path)
    summaries_num_lines = count_lines(summaries_path)
    assert articles_num_lines == summaries_num_lines, (
        "The number of articles and summaries must be the same, "
        + "got {} and {}".format(articles_num_lines, summaries_num_lines)
    )

    target_dir = Path(target_dir)
    paths = [
        [
            target_dir / "{}.{}".format(tag, Path(data_path).name)
            for data_path in [texts_path, summaries_path]
        ]
        for tag in [TEST_TAG, VAL_TAG, TRAIN_TAG]
    ]

    train, val, _ = np.asarray(ratio) / np.sum(ratio) * articles_num_lines
    thresholds = [articles_num_lines, train + val, train]

    with open(texts_path, "rb") as articles, open(
        summaries_path, "rb"
    ) as summaries:
        with tqdm(total=articles_num_lines, desc="Lines") as pbar:
            threshold = thresholds.pop()
            target_articles_path, target_summaries_path = paths.pop()
            for line_idx, (text, summary) in enumerate(
                zip(articles, summaries)
            ):
                if line_idx >= threshold:
                    target_articles_path, target_summaries_path = paths.pop()
                    threshold = thresholds.pop()
                with open(target_articles_path, "wb") as target_texts, open(
                    target_summaries_path, "wb"
                ) as target_summaries:
                    target_texts.write(text)
                    target_summaries.write(summary)
                pbar.update()


def sample_from_files(
    articles_path,
    summaries_path,
    target_dir="sampled",
    method=BY_LENGTH,
    **kwargs
):
    """
        Samples small
        
        Args:
            articles_path: path to the file with articles
            summaries_path: path to the file with summaries
            target_dir: dir to save sampled articles and summaries
            method: we have only methof BY_LENGTH now, cuz we are lazy lmao
                    (see phramer/phramer/config.py:SAMPLING_METHODS)
    """
    if not target_dir:
        raise ValueError("Please provide a target directory.")

    if Path(target_dir).exists():
        raise ValueError(
            "The directory '{}' already exists, aborting.".format(target_dir)
        )
    target_dir = Path(target_dir)
    target_articles_path = target_dir / Path(articles_path).name
    target_summaries_path = target_dir / Path(summaries_path).name

    if method == BY_LENGTH:
        target_dir.mkdir(parents=True, exist_ok=True)
        max_length = kwargs[BY_LENGTH]

        articles_num_lines = count_lines(articles_path)
        summaries_num_lines = count_lines(summaries_path)
        assert articles_num_lines == summaries_num_lines, (
            "The number of articles and summaries must be the same, "
            + "got {} and {}".format(articles_num_lines, summaries_num_lines)
        )

        total_num_lines = articles_num_lines

        with open(articles_path, "rb") as articles, open(
            summaries_path, "rb"
        ) as summaries, open(
            target_articles_path, "wb"
        ) as target_articles, open(
            target_summaries_path, "wb"
        ) as target_summaries:
            print(
                "Processing articles from '{}' ".format(articles_path)
                + "and summaries from '{}' ".format(summaries_path)
                + "into '{}' and '{}'".format(
                    target_articles_path, target_summaries_path
                )
            )
            cnt = 0
            for article, summary in tqdm(
                zip(articles, summaries), total=articles_num_lines
            ):
                if len(article) < max_length:
                    cnt += 1
                    target_articles.write(article)
                    target_summaries.write(summary)
            print(
                "Sampled {0:.2f}% of the dataset.".format(
                    cnt / total_num_lines * 100
                )
            )
    else:
        raise NotImplementedError(
            "Only {} are available.".format(SAMPLING_METHODS)
            + "This method is not implemented, aborting."
        )
    return
