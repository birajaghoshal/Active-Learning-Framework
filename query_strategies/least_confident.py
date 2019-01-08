import numpy as np
from strategy import Strategy


class LeastConfident(Strategy):
    """
    This class is for least confident query method that selects data with the highest uncertainty calculated using the
    least confident method that looks at the highest softmax prediction and selects the values with the lowest values.
    This method is one of the simplest active learning method and is useful for use as a baseline in comparisons with
    other methods.
    """

    def query(self, n):
        """
        Method for querying the data to be labeled. This method selects data with the least confident predictions.
        :param n: Amount of data to query.
        :return: Array of indices to data selected to be labeled.
        """

        unlabeled_indices = np.arange(self.pool_size)[~self.labeled_indices]
        probabilities, _ = self.predict(self.x[unlabeled_indices], self.y[unlabeled_indices])
        uncertainties = probabilities.max(1)[0]
        return unlabeled_indices[uncertainties.sort()[1][:n]]
