from phramer.data.dataset import RIANewsDataset

ria = RIANewsDataset()


def process_article(article):
    print(ria._process_article(article))

def main():
    article = input()
    process_article(article)


if __name__ == "__main__":
    main()
