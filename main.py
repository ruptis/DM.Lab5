import numpy as np
import pandas as pd

import drawer
from potential_classifier import PotentialClassifier


def get_limits(data):
    x_min = min([vector[0] for vector, classification in data])
    x_max = max([vector[0] for vector, classification in data])
    y_min = min([vector[1] for vector, classification in data])
    y_max = max([vector[1] for vector, classification in data])
    return x_min, x_max, y_min, y_max


def main():
    data = pd.read_csv("data.csv")
    data = [(tuple(row[:2]), row[2]) for row in data.values]
    classifier = PotentialClassifier()
    classifier.train(data)
    limits = get_limits(data)
    test_data = np.random.uniform(limits[0], limits[1], (250, 2))
    classifications = [(tuple(vector), classifier.classify(vector)) for vector in test_data]
    drawer.draw_data(classifications, classifier.get_separation_function())


if __name__ == '__main__':
    main()
