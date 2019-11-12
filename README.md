# Phramer

> This repository is under active development.

**Phramer** is an open-source library for extractive and abstractive text summarization. 

## Installation

1. Clone the project:
    ```bash
    git clone git@github.com:phramer/phramer.git
    cd phramer
    ```

2. Please make sure that you run in a Docker container or a virtual environment and install the dependencies:
    ```bash
    python -m venv --no-site-packages .env
    . .env/bin/activate
    pip install -r requirements.txt
    ```

    If you would like to contribute to the project, please install dev dependencies as well: 

    ```bash
    . .env/bin/activate
    pip install -r dev-requirements.txt
    ```

## Howto
Here is instruction how to create you own DVC project by yourself.

0. Make sure you destroy our DVC workspace in repository:
    ```bash
    dvc destroy
    ```

1. Create the DVC project:
    ```bash
    dvc init
    # git commit -m "Initialize DVC project"
    ```

2. Set up remote storage (similar to the way you use Git server to store and share your code):
    * Local Storage
        ```bash
        dvc remote add -d localremote /tmp/dvc-storage
        # flag -d makes it default storage
        ```
    * Google Cloud Storage
        1. Create account on [Google Cloud](https://cloud.google.com/) and go to the *Console*.
        2. Create new bucket (default setup will be okay).
        3. Go to *APIs & Services > Credentials* and make new service account key.
        4. Choose json key type, create and download it to your machine.
        5. Run 
            ```bash
            export GOOGLE_APPLICATION_CREDENTIALS="[PATH]”
            ```
            where `[PATH]` is path to json file (e.g. `/home/user/Downloads/[FILE_NAME].json`)
        6. Make sure you have optional DVC deoendencies for Google Cloud Storage:
            ```bash
            pip install dvc[gs]
            ```
        7. Add Google Cloud bucket as your remote storage to the DVC:
            ```bash
            dvc remote add -d gs gs://yourbucket
            # your bucket url you can find in *Bucket details > Overview*
            # by clicking on your bucket
            ```

3. Part of our project use data from this [repo](https://github.com/RossiyaSegodnya/ria_news_dataset). To keep the data always fresh we will import it to our project using *import-url* command:
    ```bash
    dvc import-url https://github.com/RossiyaSegodnya/ria_news_dataset/raw/master/ria.json.gz
    ```

4. To add other data to DVC project run following command:
    ```bash
    dvc add data/data.xml
    # git add data/.gitignore data/data.xml.dvc
    # git commit -m "Add raw data to project"
