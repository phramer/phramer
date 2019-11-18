import argparse
from phramer.data.dataset import RIANewsDataset

ria = RIANewsDataset()


def process_article(data_path, save_path):
    with open(data_path, 'r+') as f:
        article = f.read()

    article = ria._process_article(article)
    article = article.replace('\n', ' ')

    f = open(save_path, 'w+')
    f.write(article)
    f.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help='path to data to process')
    parser.add_argument("--save_path", type=str, required=True, help='path where to save after processing')
    args = parser.parse_args()
    process_article(args.data_path, args.save_path)

if __name__ == "__main__":
    main()
