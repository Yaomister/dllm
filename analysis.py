import re
import os
import glob
import json
import pandas
import numpy as np
import matplotlib.pyplot as plt


K = [1, 2, 4, 8]
N = 8

def calculate_pass_k(n, k, c):
    # n = how many samples you generate
    # c = how many of the samples came out correct
    # k = how many attemps we're asking about

    # not enough failues to fill a subset, so guarenteed hit
    if n - c < k:
        return 1

    return 1 - np.prod(1 - k / np.arange(n - c + 1, n + 1))

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



def graph_data(df):
    for k in K:
        fig, ax = plt.subplots()
        ax.plot(np.arange(k), df[f'pass@{k}'])
        fig.savefig()
    

if __name__ == "__main__":

    df = stitch_data()
    
    for k in K:
        df[f"pass@{k}"] = df['c'].apply(lambda c : calculate_pass_k(N, k, c))

    per_seed = df.groupby(["dataset", "seed", "method"])[[f'pass@{k}' for k in K]].mean()

    summary = per_seed.groupby(["dataset", "method"]).agg(["mean", "std"])
    print(summary.round(3))

    graph_data(df)


