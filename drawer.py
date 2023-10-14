import matplotlib.pyplot as plt
import numpy as np


def draw_data(data, function=None):
    plt.scatter([vector[0] for vector, classification in data if classification],
                [vector[1] for vector, classification in data if classification], c='r')
    plt.scatter([vector[0] for vector, classification in data if not classification],
                [vector[1] for vector, classification in data if not classification], c='b')

    if function is not None:
        x_min = min([vector[0] for vector, classification in data])
        x_max = max([vector[0] for vector, classification in data])
        x = np.arange(x_min, x_max, 0.001)
        y = np.arange(x_min, x_max, 0.001)
        x, y = np.meshgrid(x, y)
        z = function((x, y))
        plt.contour(x, y, z, [0], colors='k')

    plt.show()
