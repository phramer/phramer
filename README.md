# Phramer

> This repository is under active development.

**Phramer** is an open-source library for extractive and abstractive text summarization. 

## Installation
<<<<<<< HEAD

1. Clone the project:
    ```bash
    git clone git@github.com:phramer/phramer.git
    cd phramer
    ```

2. Please make sure that you run in a Docker container or a virtual environment and install requirements:
    ```bash
    virtualenv --no-site-packages .env .env
    . .env/bin/activate
    pip install -r requirements.txt
    ```
=======
Firstly, clone our project:
```console
git clone https://github.com/phramer/phramer.git
cd phramer
```

Now, let's install requirements.<br>
We **strongly** recommend to create a virtual environment with a tool such as module [venv](https://docs.python.org/3/library/venv.html):
```console
python3 -m venv .env
. .env/bin/activate
pip install -r requirements.txt
```

Finally, get the data from preconfigured DVC local remote storage:
```console
dvc pull
```
>>>>>>> b19bfcf52be076ea4e64ad07392bc9a335313c3a

