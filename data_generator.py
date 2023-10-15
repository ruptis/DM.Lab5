from sys import argv

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from drawer import draw_data


def generate():
    first_class = []
    second_class = []
    if argv[1] == 'default':
        first_class = [(-1, 0), (1, 1)]
        second_class = [(2, 0), (1, -2)]
    elif argv[1] == 'blobs':
        first_class = make_blobs(n_samples=100, n_features=2, centers=[(1, 1)], cluster_std=0.5)[0].tolist()
        second_class = make_blobs(n_samples=100, n_features=2, centers=[(-1, -1)], cluster_std=0.5)[0].tolist()
    elif argv[1] == 'vertical':
        first_class = [(0, 0), (0, 1)]
        second_class = [(1, 0), (1, 1)]
    elif argv[1] == 'diagonal':
        first_class = [(0, 0.5), (0.5, 0)]
        second_class = [(1, 0.5), (0.5, 1)]
    elif argv[1] == 'spiral':
        first_class = [(0.1 * i * np.cos(i), 0.1 * i * np.sin(i)) for i in np.arange(0, 2 * np.pi, 0.1)]
        second_class = [(0.1 * i * np.cos(i) + 1, 0.1 * i * np.sin(i) + 1) for i in np.arange(0, 2 * np.pi, 0.1)]
    elif argv[1] == 'linear':
        first_class = [(0, 0), (1, 1)]
        second_class = [(0, 1), (1, 0)]
    data = pd.DataFrame(first_class + second_class, columns=['x', 'y'])
    data['classification'] = [True] * len(first_class) + [False] * len(second_class)
    data.to_csv("data.csv", index=False)
    data = [(tuple(row[:2]), row[2]) for row in data.values]
    draw_data(data)


if __name__ == '__main__':
    generate()
