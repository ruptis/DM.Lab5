from functools import partial


class PotentialClassifier:
    def __init__(self):
        self.separation_function = lambda x: 0

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
        previous_separation_function = self.separation_function
        local_potential_function = self._get_local_potential_function(vector)

        def separation_function(x):
            return previous_separation_function(x) + correction * local_potential_function(x)

        self.separation_function = partial(separation_function)

    @staticmethod
    def _get_local_potential_function(vector):
        def local_potential_function(x):
            return 1 + 4 * x[0] * vector[0] + 4 * x[1] * vector[1] + 16 * x[0] * x[1] * vector[0] * vector[1]

        return local_potential_function

    @staticmethod
    def _get_correction(separation_function_value, classification):
        if classification and separation_function_value <= 0:
            return 1
        elif not classification and separation_function_value > 0:
            return -1
        else:
            return 0
