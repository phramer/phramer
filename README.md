# Phramer

## Installation
Firstly, clone our project:
```console
$ git clone https://github.com/phramer/phramer.git
$ cd phramer
```

Now, let's install requirements.<br>
We **strongly** recommend to create a virtual environment with a tool such as module [venv](https://docs.python.org/3/library/venv.html):
```console
$ python3 -m venv .env
$ . .env/bin/activate
$ pip install -r requirements.txt
```

Finally, get the data from preconfigured DVC local remote storage:
```console
$ dvc pull
```

