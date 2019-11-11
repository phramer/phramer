import json
import pandas as pd
from bs4 import BeautifulSoup
import argparse
import tqdm
import os, errno


def split_csv():
    parser = argparse.ArgumentParser(description="args to transform file")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="path to file to transform",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default="data/ria_splitted/",
        help="dir where to save articles and summaries",
    )
    args = parser.parse_args()

    # create target dir if don't exist
    try:
        os.makedirs(args.save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    print("Reading data...")
    data = pd.read_csv(args.data_path)
    articles = data["text"].values
    titles = data["title"].values

    f = open(args.save_dir + "/ria.articles", "w")

    print("Processing articles...")
    for article in tqdm.tqdm(articles):
        article = str(article).replace("\n", "")
        f.write(article)
        f.write("\n")
    f.close()

    print("Processing summaries...")
    f = open(args.save_dir + "/ria.summaries", "w")
    for title in tqdm.tqdm(titles):
        title = str(title).replace("\n", "")
        f.write(title)
        f.write("\n")
    f.close()


if __name__ == "__main__":
    split_csv()
