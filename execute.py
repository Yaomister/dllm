import re
import os
import glob
import json
import defaultdict
from datasets import load_dataset
from calculations import calculate_pass_k
from argparse import ArgumentParser



def main():
    pass

def build_humaneval():
    pass

def build_mbpp():
    pass

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--glob", required=True)
    parser.add_argument("--timeout", required=False, default=10.0)

    args = parser.parse_args()

    if args.dataset == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split='test'),
        build = build_humaneval
    elif args.dataset == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "full", split="test")
        build = build_mbpp
    else:
        raise RuntimeError()

    problems = {row['task_id']: row for row in dataset }

    files = sorted(glob.glob(args.glob))

    if not files:
        raise RuntimeError()

    print('loaded in files.')

    by_method = defaultdict(lambda: defaultdict(list))

    for path in files:
        m = re.match(r"samples_[^_]+_(.+?)_\d+_batch\d+\.jsonl$", os.path.basename(path))
        method = m.group(1) if m else os.path.basename(path)

        open

    main()