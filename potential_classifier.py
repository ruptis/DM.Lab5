import numpy as np


class PotentialClassifier:
    def __init__(self):
        self.separation_function = SeparationFunction()

    def train(self, data):
        successful_classifications = 0
        while successful_classifications < len(data):
            for vector, classification in data:
                separation_function_value = self.separation_function(vector)
                correction = self._get_correction(separation_function_value, classification)
                if correction != 0:
                    self._update_separation_function(correction, vector)
                    successful_classifications = 0
                    break
                else:
                    successful_classifications += 1

    def classify(self, vector):
        return self.separation_function(vector) > 0

    def get_separation_function(self):
        return self.separation_function

    def _update_separation_function(self, correction, vector):
        local_potential_function = self._get_local_potential_function(vector)
        self.separation_function = self.separation_function + local_potential_function * correction

    @staticmethod
    def _get_local_potential_function(vector):
        local_potential_function = SeparationFunction()
        local_potential_function.coefficients[0] = 1
        local_potential_function.coefficients[1] = 4 * vector[0]
        local_potential_function.coefficients[2] = 4 * vector[1]
        local_potential_function.coefficients[3] = 16 * vector[0] * vector[1]

        return local_potential_function

    @staticmethod
    def _get_correction(separation_function_value, classification):
        if classification and separation_function_value <= 0:
            return 1
        elif not classification and separation_function_value > 0:
            return -1
        else:
            return 0


class SeparationFunction:
    def __init__(self):
        self.coefficients = np.zeros(4)

    def __call__(self, x):
        return self.coefficients[0] + self.coefficients[1] * x[0] + self.coefficients[2] * x[1] + self.coefficients[3] * \
            x[0] * x[1]

    def __add__(self, other):
        result = SeparationFunction()
        result.coefficients = self.coefficients + other.coefficients
        return result

    def __mul__(self, other):
        result = SeparationFunction()
        result.coefficients = self.coefficients * other
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self.__add__(other.__mul__(-1))

    def __truediv__(self, other):
        result = SeparationFunction()
        result.coefficients = self.coefficients / other
        return result

    def __neg__(self):
        return self.__mul__(-1)

    def __eq__(self, other):
        return np.array_equal(self.coefficients, other.coefficients)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.coefficients)
