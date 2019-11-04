import sys
import argparse
import pandas as pd



def main():
    parser = argparse.ArgumentParser(description='hi')
    parser.add_argument("--file_path", type=str, required=True, help="file name to transform")
    args = parser.parse_args()
    print(args)


    data = pd.read_table(args.file_path, header=None)
    columns = {0: 'url', 1: 'title', 2: 'text', 3: 'pub_time'}
    data.rename(columns=columns, inplace=True)

    for column_name in columns.values():
        data[column_name] = data[column_name].astype('str')
        data[column_name] = data[column_name].apply(lambda x: x[len(column_name) + 1:])
    
    data.to_csv(args.file_path + '.csv', index=False)

if __name__ == "__main__":
    main()
