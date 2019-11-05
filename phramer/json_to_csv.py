import json
import pandas as pd
from bs4 import BeautifulSoup
import argparse

def json_to_csv():
    parser = argparse.ArgumentParser(description='args to transform file')
    parser.add_argument("--file_path", type=str, required=True, help="path to file to transform")
    parser.add_argument("--save_name", type=str, required=False, default="ria.csv", help="new file name after transform")
    args = parser.parse_args()
    
    with open(args.file_path, 'r') as data:
        json_data = data.readlines()
    
    soups = [BeautifulSoup(json.loads(json_data[i])['text'], 'lxml') for i in range(len(json_data))]
    texts = [soup.text.replace(u'\xa0', u' ').replace('\n', '') for soup in soups]
    titles = [json.loads(json_data[i])['title'] for i in range(len(json_data))]

    data = pd.DataFrame({"text": texts, "title": titles})
    data.to_csv(args.save_name, index=False)


if __name__ == "__main__":
    json_to_csv()
