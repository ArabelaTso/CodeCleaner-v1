import os, re, shutil
import gzip
import json
import astor
import keyword
from ast import parse, FunctionDef
from typing import Iterable, Dict
from inspect import signature
from tqdm import tqdm
from code_analysis import remove_comments


def read_problems(evalset_file) -> Dict[str, Dict]:
    return {"{}_{}".format(task["hexsha"], task['fn_id']): task for task in stream_jsonl(evalset_file)}


def load_jsonl(filename: str, key_str='question_id') -> list:
    list_of_data = [item for item in stream_jsonl(filename)]
    return list_of_data


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)
 
 
def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))
   

def transfer_tsv_to_csv(file):
    with open(file, 'r') as fin:
        with open(file.replace('.tsv', '.csv'), 'w') as fout:
            for line in fin:
                fout.write(line.replace('\t', ','))


def check_non_empty_words(string):
    words = string.split()
    for word in words:
        if word.strip():  # Check if the word is non-empty after stripping whitespace
            return True
    return False
