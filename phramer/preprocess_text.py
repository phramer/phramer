import argparse
import tqdm
import re
import nltk
import os
from multiprocessing import Pool
nltk.download("stopwords")
#--------#

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation


#Create lemmatizer and stopwords list
mystem = Mystem()
russian_stopwords = stopwords.words("russian")


ORIGIN_FOLDER_PATH = ''
PREPROCESSED_FOLDER_PATH = ''

#Delete extra symbols
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'[_"\-;%()|.,+&=*%]', ' ', text)
    text = re.sub(r'\.', ' . ', text)
    text = re.sub(r'\!', ' !', text)
    text = re.sub(r'\?', ' ?', text)
    text = re.sub(r'\,', ' ,', text)
    text = re.sub(r':', ' : ', text)
    text = re.sub(r'#', ' # ', text)
    #tweet = re.sub(r'@', ' @ ', tweet)
    text = re.sub(r' . . . ', ' ', text)
    text = re.sub(r' .  .  . ', ' ', text)
    text = re.sub(r' ! ! ', ' ! ', text)
    text = text.replace('\n', '')
    return text

#Preprocess function
def preprocess_text(text):
    text = clean_text(text)
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]

    text = " ".join(tokens)
    text = re.sub(' +', ' ', text)

    return text


### Save preprocessed text
def process_file(file_name):
    with open(ORIGIN_FOLDER_PATH + file_name, 'r') as f:
        text = f.read()
    text = preprocess_text(text)
    
    f = open(PREPROCESSED_FOLDER_PATH + file_name, 'w')
    f.write(text)
    f.close()


def preprocess_corpus(data_path, new_path, process_num=50):
    ### Create folder for origin files
    global ORIGIN_FOLDER_PATH
    print(ORIGIN_FOLDER_PATH)
    origin_folder_path = data_path[:data_path.rfind('/') + 1] + 'origin/'
    ORIGIN_FOLDER_PATH = origin_folder_path
    '''
    try:
        os.mkdir(origin_folder_path)
    except OSError:
        print ("Creation of the directory %s failed" % origin_folder_path)
    else:
        print ("Successfully created the directory %s " % origin_folder_path)
    
    '''
    ### Create folder for processed files
    global PREPROCESSED_FOLDER_PATH
    preprocessed_folder_path = data_path[:data_path.rfind('/') + 1] + 'preprocessed/'
    PREPROCESSED_FOLDER_PATH = preprocessed_folder_path
    '''
    try:
        os.mkdir(preprocessed_folder_path)
    except OSError:
        print ("Creation of the directory %s failed" % preprocessed_folder_path)
    else:
        print ("Successfully created the directory %s " % preprocessed_folder_path)
        
    
    ### Read articles data
    print("Opening file...")
    with open(data_path, 'rb') as f:
        articles = f.readlines()
    print("Processing articles...")
    
    
    
    ### Write articles to separate files
    print("Writing articles to separate files...")
    for i in tqdm.tqdm(range(len(articles))):
        f = open(origin_folder_path + 'article_' + str(i), 'wb+')
        f.write(articles[i])
        f.close()
        
    ### Preprocessing articles
    print("Preprocessing articles...")
    file_names = os.listdir(origin_folder_path)
    p = Pool(process_num)
    list(tqdm.tqdm(p.imap(process_file, file_names), total=len(file_names)))
    '''
    
    ### Uniting files
    print("Uniting files")
    processed_files = os.listdir(preprocessed_folder_path)
    processed_files = sorted(processed_files, key=lambda file:int(file[file.rfind('_') + 1:]))
    f = open(new_path, 'w')
    for file_name in tqdm.tqdm(processed_files):
        with open(preprocessed_folder_path + file_name, 'r') as article_file:
            article = article_file.read()
        f.write(article)
        f.write('\n')
    f.close()

def main():
    parser = argparse.ArgumentParser(description='preprocessing parser')
    parser.add_argument("--data_path", type=str, required=True, help="path to data to preprocess")
    parser.add_argument("--new_path", type=str, required=True, help="new file name after preprocessing")
    parser.add_argument("--process_num", type=int, default=50, help="number of processes to use for preprocessing")
    args = parser.parse_args()

    preprocess_corpus(args.data_path, args.new_path, process_num=args.process_num)
    print(args)

if __name__ == "__main__":
    main()
