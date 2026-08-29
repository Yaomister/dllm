import re
import os
import glob
import json
import pandas
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from collections import defaultdict
from calculations import  calculate_pass_k, calculate_entropy_k
from matplotlib.ticker import ScalarFormatter, NullFormatter


K = [1, 2, 4, 8, 16]
N = 16


def stitch_executed_results():
    records = {}
    for path in glob.glob("results/execution_results_*.json"):
        m = re.match(r"execution_results_(\w+)\.json$", os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            data = json.load(f)

        per_dataset = {}
        for method, runs in data.items():
            # runs, passes
            per_method = {}
            counts = defaultdict(lambda: [0, 0])
            for run in runs:
                counts[run['task_id']][0] += 1
                counts[run["task_id"]][1] += run["passed"]

            for k in K:
                per_method[f"pass@{k}"] = np.mean([calculate_pass_k(n, k, c) for n, c in counts.values()]).item()
            per_dataset[method.replace("_", " ")] = per_method
        records[m.group(1)] = per_dataset

    return pandas.DataFrame.from_dict(
        {(d, m): scores for d, methods in records.items() for m, scores in methods.items()},
        orient="index",
    ).rename_axis(["dataset", "method"])
        

def stitch_results():
    records = []
    for path in glob.glob("results/results_*_*_batch*.json"):
        name  = os.path.basename(path)
        dataset, seed, batch = re.match(r"results_(\w+)_(\d+)_batch(\d+)\.json", name).groups()
        data = json.load(open(path))
        for method, d in data["method"].items():
            for index, c, answers, in zip(d['problem_indices'], d['c_counts'], d['answers']):
                records.append({
                    "dataset": dataset,
                    "seed": seed,
                    "problem": index,
                    "method": method,
                    "answers": answers,
                    "c": c
                })

    return pandas.DataFrame(records)


markers_ = ["o", "s", "^", "D", "v", "P", "*"]
shown_methods = ["Cos", "TLC 1",  "TLC 0.55", "Linear",  "Inverse"]


def graph_entropy_k_data(per_seed, ):
    avg = per_seed.groupby(['dataset', 'method']).mean()
    datasets = avg.index.get_level_values("dataset").unique()

    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.6), layout="constrained")
    for i, dataset in enumerate(datasets):
        markers = cycle(markers_)
        for method in avg.loc[dataset].index:
            if method in shown_methods:
                y = [avg.loc[(dataset, method), f"entropy@{k}"] for k in K]
                ax[i].plot(K, y, marker=next(markers), label=method)
            ax[i].set_xlabel("k")
            ax[i].set_xscale("log", base=2)
            ax[i].set_xticks(K)
            ax[i].xaxis.set_major_formatter(ScalarFormatter())
            ax[i].xaxis.set_minor_formatter(NullFormatter())    
            ax[i].set_title(dataset.upper())

            ax[i].grid(True, alpha=0.3)
            ax[i].set_axisbelow(True)

    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=6, frameon=False, fontsize=12)
    ax[0].set_ylabel("entropy@k")
    fig.savefig("entropy_at_k.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def graph_pass_k_data(per_seed, per_execution):
    avg = per_seed.groupby(['dataset', 'method']).mean()
    datasets = list(avg.index.get_level_values("dataset").unique()) + list(per_execution.index.get_level_values("dataset").unique())

    fig, ax = plt.subplots(1, 4, figsize=(14, 4.0), layout="constrained")
    for i, dataset in enumerate(datasets):
        markers = cycle(markers_)
        src = avg if dataset in avg.index.get_level_values("dataset") else per_execution
        for method in src.loc[dataset].index:
            if method in shown_methods:
                if (dataset, method) in avg.index:
                    y = [avg.loc[(dataset, method), f"pass@{k}"] for k in K]
                else: 
                    y = [per_execution.loc[(dataset, method), f"pass@{k}"] for k in K]
                ax[i].plot(K, y, marker=next(markers), label=method)
            ax[i].set_xlabel("k")
            ax[i].set_xscale("log", base=2)
            ax[i].set_xticks(K)
            ax[i].xaxis.set_major_formatter(ScalarFormatter())
            ax[i].xaxis.set_minor_formatter(NullFormatter())
            ax[i].set_title(dataset.upper())
            ax[i].grid(True, alpha=0.3)
            ax[i].set_axisbelow(True)

    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=6, frameon=False, fontsize=14)
    ax[0].set_ylabel("pass@k")
    fig.savefig("pass_at_k.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



    

if __name__ == "__main__":
    stitch_executed_results()

    df = stitch_results()

    for k in K:
        df[f"pass@{k}"] = df['c'].apply(lambda c : calculate_pass_k(N, k, c))
        df[f"entropy@{k}"] = df['answers'].apply(lambda a : calculate_entropy_k(a, k))

    cols = [f'pass@{k}' for k in K] + [f'entropy@{k}' for k in K]
    per_seed = df.groupby(["dataset", "seed", "method"])[cols].mean()   
    

    per_execution = stitch_executed_results()

    print(per_execution)
    print(per_seed.groupby(['dataset', 'method']).mean())
    print(per_seed.groupby(['dataset', 'method']).std())

    graph_pass_k_data(per_seed, per_execution)
    graph_entropy_k_data(per_seed)


