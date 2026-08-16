import re
import os
import glob
import json
import pandas
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from calcualtions import calculate_entropy, calculate_pass_k


K = [1, 2, 4, 8, 16]
N = 16

def stitch_data():
    records = []
    for path in glob.glob("results/results_*_*_batch*.json"):
        name  = os.path.basename(path)
        dataset, seed, batch = re.match(r"results_(\w+)_(\d+)_batch(\d+)\.json", name).groups()
        print(path)
        data = json.load(open(path))
        for method, d in data["method"].items():
            for index, c in zip(d['problem_indices'], d['c_counts']):
                records.append({
                    "dataset": dataset,
                    "seed": seed,
                    "problem": index,
                    "method": method,
                    "c": c
                })

    return pandas.DataFrame(records)



def graph_data(per_seed):
    avg = per_seed.groupby(['dataset', 'method']).mean()
    datasets = avg.index.get_level_values("dataset").unique()

    fig, ax = plt.subplots(1, len(datasets), figsize=(7, 4), layout="constrained")
    shown_methods = ["Cos", "TLC 1", "Linear", "Low Confidence"]
    for i, dataset in enumerate(datasets):
        markers = cycle(["o", "s", "^", "D", "v", "P", "*"])
        for method in avg.loc[dataset].index:
            if method in shown_methods:
                y = [avg.loc[(dataset, method), f"pass@{k}"] for k in K]
                ax[i].plot(K, y, marker=next(markers), label=method)
            ax[i].set_xlabel("k")
            ax[i].set_title(dataset)
            ax[i].set_box_aspect(1)
            ax[i].grid(True, alpha=0.3)
            ax[i].set_axisbelow(True)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.0, 0.95))

    ax[0].set_ylabel("pass@k")
    fig.tight_layout()
    fig.savefig("pass_at_k.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    

if __name__ == "__main__":

    df = stitch_data()
    
    for k in K:
        df[f"pass@{k}"] = df['c'].apply(lambda c : calculate_pass_k(N, k, c))

    per_seed = df.groupby(["dataset", "seed", "method"])[[f'pass@{k}' for k in K]].mean()

    summary = per_seed.groupby(["dataset", "method"]).agg(["mean", "std"])
    print(summary.round(3))

    graph_data(per_seed)


