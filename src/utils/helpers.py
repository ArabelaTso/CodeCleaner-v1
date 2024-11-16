import multiprocessing
from .similarity import *

from typing import Dict 
import tqdm
from pathlib import Path
from datasets import load_from_disk
from datetime import datetime
from human_eval.data import read_problems

the_stack_data = None


def load_the_stack_fns(
    dataset_path: Path, year: int | None = None, jobs: int | None = None
) -> list:
    """
    Load functions from the stack dataset.
    The dataset should be already preprocessed to split functions.
    The whole data is cached in memory.
    @param dataset_path: The path to the dataset
    @param year: The year to filter the dataset
    @param jobs: The number of jobs to use for loading the dataset
    """
    global the_stack_data
    if the_stack_data is None:
        the_stack_data = load_from_disk(str(dataset_path))
    if year:

        def filter(row):
            begin = datetime.fromisoformat(f"{year}-01-01T00:00:00Z")
            end = datetime.fromisoformat(f"{year + 1}-01-01T00:00:00Z")
            return begin <= row["last_update_at"] < end

        data = the_stack_data.filter(filter, num_proc=jobs)
    else:
        data = the_stack_data
    return data


def load_humaneval_fns(dataset_path: Path) -> list:
    data = read_problems(str(dataset_path))
    return [row for row in data.values()]


class SimilarityCalculator:
    def __init__(self) -> None:
        self.sed = StringEditDistance()
        self.ted = ASTEditDistance()
        self.codebert = CodeBert()
        self.graphcodebert = GraphCodeBert()
        self.bleu = BLEU()
        self.code_t5 = CodeT5()
        self.jaccard = JaccardSimilarity()

    def calculate(self, one: str, other: str) -> Dict[str, float]:
        this_tree = ast.parse(one)
        other_tree = ast.parse(other)
        return {
            "sed": self.sed.calculate(one, other),
            "ted": self.ted.calculate(this_tree, other_tree),
            "codebert": self.codebert.calculate(one, other),
            "graphcodebert": self.graphcodebert.calculate(one, other),
            "bleu": self.bleu.calculate(one, other),
            "codet5": self.code_t5.calculate(one, other),
            "jaccard": self.jaccard.calculate(this_tree, other_tree),
        }
