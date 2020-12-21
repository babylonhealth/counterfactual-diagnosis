import json
import pickle
from pathlib import Path


def load_from_json(filename):
    with open(filename, "r") as infile:
        return json.load(infile)


def load_from_pickle(filename):
    with open(filename, "rb") as infile:
        return pickle.load(infile)


def write_to_pickle(object, filename):
    with open(filename, "wb") as outfile:
        return pickle.dump(object, outfile)


def ensure_dir(path):
    if type(path) == str:
        path = Path(path)

    if not path.exists():
        path.mkdir(parents=True)
