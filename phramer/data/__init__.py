from tqdm import tqdm
from phramer.config import BY_LENGTH
from pathlib import Path

SAMPLING_METHODS = {BY_LENGTH}


class Dataset:
    @staticmethod
    def sample_from_files(
        articles_path,
        summaries_path,
        target_dir="sampled",
        method=BY_LENGTH,
        **kwargs
    ):
        if not target_dir:
            raise ValueError("Please provide a target directory.")

        if Path(target_dir).exists():
            raise ValueError(
                "The directory '{}' already exists, aborting.".format(
                    target_dir
                )
            )
        target_dir = Path(target_dir)
        target_articles_path = target_dir / Path(articles_path).name
        target_summaries_path = target_dir / Path(summaries_path).name

        if method == BY_LENGTH:
            target_dir.mkdir(parents=True, exist_ok=True)
            max_length = kwargs[BY_LENGTH]
            with open(articles_path, "rb") as articles, open(
                summaries_path, "rb"
            ) as summaries:
                articles_num_lines = sum(1 for _ in articles)
                summaries_num_lines = sum(1 for _ in summaries)
            assert articles_num_lines == summaries_num_lines, (
                "The number of articles and summaries must be the same, "
                + "got {} and {}".format(
                    articles_num_lines, summaries_num_lines
                )
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
