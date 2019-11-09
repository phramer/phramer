import argparse
import tqdm
import os, errno


def select_short_files(data_dir, save_dir, max_len):
    with open(data_dir + '/ria.preprocessed.articles', 'rb') as f:
        articles = f.readlines()

    print("Selecting files...")
    short_idxs = [i for i in tqdm.tqdm(range(len(articles))) if len(articles[i]) <= max_len]

    with open(data_dir + '/ria.summaries', 'rb') as f:
        summaries = f.readlines()

    print(short_idxs[-10:])
    print("Articles num:", len(articles))
    print("Summaries num:", len(summaries))
    print("Short docs num:", len(short_idxs))

    f_articles = open(save_dir + '/ria.articles', 'wb')
    f_summaries = open(save_dir + '/ria.summaries', 'wb')
 
    print("Writing data...")
    for i in tqdm.tqdm(range(len(short_idxs))):         
        f_articles.write(articles[short_idxs[i]])
        f_summaries.write(summaries[short_idxs[i]])
    
    f_articles.close()
    f_summaries.close() 

def main():
    parser = argparse.ArgumentParser(description="args to select small files")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="path to dir with origin files",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="path to dir where to save short files",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        required=False,
        default=1500,
        help="max length of file",
    )

    args = parser.parse_args()

    try:
        os.makedirs(args.save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    select_short_files(args.data_dir, args.save_dir, args.max_len)

if __name__ == "__main__":
    main()

