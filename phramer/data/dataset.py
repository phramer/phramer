import multiprocessing as mp
import pandas as pd
from bs4 import BeautifulSoup
import json
from phramer.data import Dataset


class RIANewsDataset(Dataset):
    def __init__(self):
        pass

    def _parse_raw(self, file_path, num_workers=mp.cpu_count()):
        manager = mp.Manager()
        q = manager.Queue()
        pool = mp.Pool(num_workers)

        with open(file_path, "r") as raw_json:
            total_num_lines = sum(1 for _ in raw_json)

        with open(file_path, "r") as raw_json:
            for line in raw_json:
                json_data = json.loads(line)
                article = (
                    BeautifulSoup(json_data["text"], "lxml")
                    .replace(u"\xa0", u" ")
                    .replace("\n", "")
                )
                title = json_data["title"]
