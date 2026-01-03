import os
import pickle
import pandas as pd
import seaborn as sns
from itertools import groupby
from scipy.stats import mannwhitneyu
from matplotlib import pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "link_eval_dir", type=str, help="Directory containing link evaluation data"
)
args = parser.parse_args()
link_eval_dir = args.link_eval_dir

with open(os.path.join(link_eval_dir, "link_eval_data.pkl"), "rb") as fi:
    data = pickle.load(fi)

distances = {
    k: [t[1]["distance"] for t in g]
    for k, g in groupby(data["distances"].items(), key=lambda p: p[0][0][1])
}
constrained_random_distances = {
    k: [t[1]["distance"] for t in g]
    for k, g in groupby(
        data["constrained_random_distances"].items(), key=lambda p: p[0][0][1]
    )
}
random_distances = {
    k: [t[1]["distance"] for t in g]
    for k, g in groupby(data["random_distances"].items(), key=lambda p: p[0][0][1])
}

for edge in distances:
    u = mannwhitneyu(distances[edge], random_distances[edge])
    n1 = len(distances[edge])
    n2 = len(random_distances[edge])
    print(
        f"{edge:<10} - real vs. random      - {u} - Sample sizes: real {n1}, random      {n2}"
    )

print()

for edge in distances:
    u = mannwhitneyu(distances[edge], constrained_random_distances[edge])
    n1 = len(distances[edge])
    n2 = len(constrained_random_distances[edge])
    print(
        f"{edge:<10} - real vs. constrained - {u} - Sample sizes: real {n1}, constrained {n2}"
    )


dfdata = [('true', t, d) for t, l in distances.items() for d in l]
dfdata.extend([('constrained', t, d)
              for t, l in constrained_random_distances.items() for d in l])
dfdata.extend([('random', t, d)
              for t, l in random_distances.items() for d in l])

df = pd.DataFrame(dfdata, columns=["source", "edgeType", "distance"])

