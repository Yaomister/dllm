import re
import os
import glob
import json
import tempfile
import contextlib
import defaultdict
import multiprocessing
from datasets import load_dataset
from calculations import calculate_pass_k
from argparse import ArgumentParser



def build_humaneval(problem, code):
     return problem["prompt"] + code + "\n" + problem["test"] + "\n" + f"check({problem['entry_point']})"
        
def build_mbpp(problem, code):
    parts = [code]
    if problem.get("test_setup_code"):
        parts.append(problem["test_setup_code"])
    parts.extend(problem['test_list'])
    return "\n".join(parts)

@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname

@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)

def check_correctness(program, results):
    with create_tempdir():
        try:
            exec_globals = {}
            exec(program, exec_globals)
        except TimeoutError:
            results.append("timed out")
        

def execute(program, task_id, timeout):

    manager = multiprocessing.Manager()
    results = manager.list()

    p = multiprocessing.Process(target=check_correctness, args=(program, timeout, results))
    p.start()
    p.join(timeout= timeout + 1)

    if p.is_alive():
        p.kill()

    if not results:
        results.append("timed out")

    return dict(
        task_id = task_id,
        passed = results[0] == 'passed',
        results = results[0]
    )

def calculate(results):
    K = [1, 2, 4, 8, 16, 32]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--glob", required=True)
    parser.add_argument("--timeout", type=float, default=10.0) 

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

        with open(path, "r") as file:
            for line in file:
                rec = json.loads(line)
                by_method[method][rec['task_id']].append(rec["completion"])

    res = {}
    for method in by_method.keys():
        method_dict = by_method[method]
        for task_id in method_dict:
            snippits = []
            for completion in method_dict[task_id]:
                code = build(problems[task_id], completion)
                snippits.append(code)
            res[method] = execute(snippits)
