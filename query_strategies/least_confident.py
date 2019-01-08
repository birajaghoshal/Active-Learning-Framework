import torch
import numpy as np
from strategy import Strategy


class LeastConfident(Strategy):
    """
    This class is for least confident query method that selects data with the highest uncertainty calculated using the
    least confident method that looks at the highest softmax prediction and selects the values with the lowest values.
    This method is one of the simplest active learning method and is useful for use as a baseline in comparisons with
    other methods.
    """

    def __init__(self, x, y, labeled_indices, model, data_handler, arguments, iterations=1):
        super(LeastConfident, self).__init__(x, y, labeled_indices, model, data_handler, arguments)
        self.number_iterations = iterations

    def query(self, n):
        """
        Method for querying the data to be labeled. This method selects data with the least confident predictions.
        :param n: Amount of data to query.
        :return: Array of indices to data selected to be labeled.
        """

        unlabeled_indices = np.arange(self.pool_size)[~self.labeled_indices]
        probabilities = []

        for i in range(self.number_iterations):
            prob = self.predict(self.x[unlabeled_indices], self.y[unlabeled_indices])[0].numpy()
            probabilities.append(prob)

        if self.number_iterations > 0:
            probabilities = np.var(np.asarray(probabilities), axis=0)

        uncertainties = probabilities.max(1)
        uncertainties = torch.from_numpy(uncertainties)
        query_indices = unlabeled_indices[uncertainties.sort()[1][:n]]
        return query_indices
