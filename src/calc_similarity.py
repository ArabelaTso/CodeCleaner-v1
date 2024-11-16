import json
from collections import OrderedDict
from enum import Enum
from multiprocessing import Process, Queue
from pathlib import Path
from utils.log import config_logger
from structlog import BoundLogger
from utils.helpers import *
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from human_eval.data import read_problems
import torch

from utils.preprocess import clean_python_code

the_stack_datasets = {
    year: (
        f"the-stack-{year}",
        Path(f"../data2/the-stack-dedup-python-preprocessed-{year}-sample384"),
    )
    for year in range(2018, 2023)
}

human_eval_dataset = (
    "human-eval",
    Path("../data/HumanEval.jsonl.gz"),
)
coder_eval_dataset = (
    "coder-eval",
    Path("../data2/question.jsonl"),
)
repos_datasets = {
    year: (
        f"repos-{year}",
        Path(f"../data2/repos-preprocessed-{year}-sample384"),
    )
    for year in range(2022, 2025)
}
repos_2023_oct_dec_dataset = (
    "the-stack-2023-oct-dec", # should be repos-2023-oct-dec, there is a type here.
    Path("../data2/repos-preprocessed-2023-oct-dec-sample384"),
)
refactored_datasets = [
    ("if", Path("../data2/the-stack-dedup-python-preprocessed-2021-sample384-refactor-if.jsonl")),
    ("loop", Path("../data2/the-stack-dedup-python-preprocessed-2021-sample384-refactor-loop.jsonl")),
    ("var-renamer", Path("../data2/the-stack-dedup-python-preprocessed-2021-sample384-refactor-var-renamer.jsonl")),
    ("fn-args", Path("../data2/the-stack-dedup-python-preprocessed-2021-sample384-refactor-fn-args.jsonl")),
    ("fn-decorator", Path("../data2/the-stack-dedup-python-preprocessed-2021-sample384-refactor-fn-decorator.jsonl")),
]


# dataset pairs: list of pairs of dataset (name, path)
def phase1(args, logger: BoundLogger, dataset_pairs: list[((str, Path), (str, Path))]):
    logger.info("Starting phase 1", args=args)
    datasets = {}
    for (name1, path1), (name2, path2) in dataset_pairs:

        def load(name, path: Path):
            if path.is_dir():
                # load as hugging face dataset
                d = load_from_disk(str(path))
                datasets[name] = d
            elif path.suffix == ".jsonl":
                # load as jsonl
                with open(str(path), "r") as f:
                    data = [json.loads(line) for line in f.readlines()]
                    datasets[name] = data
            elif path.name == "HumanEval.jsonl.gz":
                # load HumanEval dataset
                data = read_problems(str(path))

                def transformer(d):
                    return d

                data = list(map(transformer, data.values()))
                datasets[name] = data
            else:
                raise ValueError(f"Unknown file type: {path}")

        load(name1, path1)
        load(name2, path2)

    logger.info("Loaded datasets", datasets=datasets.keys())

    torch.set_num_threads(int(args.cpus / args.workers))
    torch.set_num_interop_threads(int(args.cpus / args.workers))
    for first, second in dataset_pairs:
        pre = first[0]
        post = second[0]
        out_dir = Path(args.output) / f"{pre}:{post}"
        if out_dir.exists():
            logger.info(
                "Skipping cross compare, output already exists",
                dataset1=pre,
                dataset2=post,
                output=out_dir,
            )
            continue
        dataset_pre = datasets[pre]
        dataset_post = datasets[post]
        total = len(dataset_pre) * len(dataset_post)
        logger.info("Calculating similarity", dataset1=pre, dataset2=post, total=total)

        # workers
        task_queue = Queue(total * 2)
        result_queue = Queue(total * 2)
        ps = []
        for _ in range(args.workers):
            p = Process(target=phase1_workder, args=(task_queue, result_queue))
            p.start()
            ps.append(p)

        # distribute tasks to workers
        logger.info("Distributing tasks to workers", workers=args.workers)
        pbar = tqdm(total=total, desc=f"Progress")
        n = 1
        for i, item_pre in enumerate(dataset_pre):
            fn_pre = get_fn_str(pre, item_pre)
            for j, item_post in enumerate(dataset_post):
                fn_post = get_fn_str(post, item_post)
                task_queue.put((i, j, fn_pre, fn_post))
                pbar.update(1)
                n += 1
        for _ in range(args.workers):
            task_queue.put(None)
        pbar.close()

        # collect results from workers
        logger.info("Collecting results from workers", workers=args.workers)
        pbar = tqdm(total=total, desc=f"Progress")
        pairs = []
        while len(pairs) < total:
            result = result_queue.get()
            i, j, similarity = result
            pair = {
                "pre": dataset_pre[i],
                "post": dataset_post[j],
                "similarity": similarity,
            }
            pairs.append(pair)
            pbar.update(1)
        pbar.close()

        # terminate workers
        for p in ps:
            p.join()

        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Finished cross compare, saving to disk",
            dataset1=pre,
            dataset2=post,
            num_pairs=len(pairs),
            output=out_dir,
        )
        ds = Dataset.from_generator(lambda: pairs)
        ds.save_to_disk(str(out_dir))


def phase1_workder(task_queue: Queue, result_queue: Queue):
    similarity_calculator = SimilarityCalculator()
    task = task_queue.get()
    while task is not None:
        i, j, fn1, fn2 = task
        similarity = similarity_calculator.calculate(fn1, fn2)
        result_queue.put((i, j, similarity))
        task = task_queue.get()


def get_fn_str(_typ, item):
    return (
        item["content"]
        if "content" in item
        else item["prompt"] + item["canonical_solution"]
    )


def phase2(args, logger: BoundLogger):
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect function pairs")
    parser.add_argument(
        "--cpus", "-c", type=int, default=1, help="Number of cpus to use"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="../data2", help="Output directory"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=1, help="Number of workers to use"
    )
    parser.add_argument(
        "--split", "-s", type=int, default=0,
    )

    args = parser.parse_args()
    logger = config_logger()
    dataset_pairs = []
    # dataset_pairs.append((
    #     human_eval_dataset,
    #     human_eval_dataset,
    # ))
    # for year_pre in range(2018, 2023):
    #     pair = (
    #         the_stack_datasets[year_pre],
    #         human_eval_dataset,
    #     )
    #     dataset_pairs.append(pair)
    # for year_pre in range(2018, 2023):
    #     for year_post in range(year_pre, 2023):
    #         pair = (
    #             the_stack_datasets[year_pre],
    #             the_stack_datasets[year_post],
    #         )
    #         dataset_pairs.append(pair)
    # for year_pre in range(2022, 2025):
    #     for year_post in range(year_pre, 2025):
    #         pair = (
    #             repos_datasets[year_pre],
    #             repos_datasets[year_post],
    #         )
    #         dataset_pairs.append(pair)
    # for year_pre in range(2018, 2023):
    #     for year_post in range(2022, 2025):
    #         pair = (
    #             the_stack_datasets[year_pre],
    #             repos_datasets[year_post],
    #         )
    #         dataset_pairs.append(pair)
    # for year_pre in range(2022, 2025):
    #     pair = (
    #         repos_datasets[year_pre],
    #         human_eval_dataset,
    #     )
    #     dataset_pairs.append(pair)
    # dataset_pairs.append(
    #     (coder_eval_dataset, coder_eval_dataset)
    # )
    # dataset_pairs.append(
    #     (coder_eval_dataset, human_eval_dataset)
    # )
    # for year in range(2018, 2023):
    #     pair = (
    #         the_stack_datasets[year],
    #         coder_eval_dataset,
    #     )
    #     dataset_pairs.append(pair)
    # for year in range(2022, 2025):
    #     pair = (
    #         repos_datasets[year],
    #         coder_eval_dataset,
    #     )
    #     dataset_pairs.append(pair)
    for year in range(2018, 2023):
        pair = (
            the_stack_datasets[year],
            repos_2023_oct_dec_dataset,
        )
        dataset_pairs.append(pair)
    for year in range(2022, 2025):
        pair = (
            repos_2023_oct_dec_dataset,
            repos_datasets[year],
        )
        dataset_pairs.append(pair)
    if args.split == 0:
        dataset_pairs = dataset_pairs[0:2]
    elif args.split == 1:
        dataset_pairs = dataset_pairs[2:4]
    elif args.split == 2:
        dataset_pairs = dataset_pairs[4:6]
    elif args.split == 3:
        dataset_pairs = dataset_pairs[6:8]
    else:
        raise ValueError(f"Unknown split: {args.split}")
    phase1(args, logger, dataset_pairs)
