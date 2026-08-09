import re
import os
import glob
import json
import pandas


def calculate_entropy():
    pass

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



def graph_data():
    pass


if __name__ == "__main__":
    df = stitch_data()
    print(df)


