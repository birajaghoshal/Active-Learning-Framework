import torch
import numpy as np
from strategy import Strategy


class EntropySampling(Strategy):
    """
    This class is for the entropy based query method that selects data with the highest entropy across the softmax
    predictions. It should be noted that with a binary classification problem entropy sampling is equal to least
    confident sampling. This method is a simple method for active learning that could be useful for use as a baseline in
    comparision with other methods.
    """

    def query(self, n):
        """
        Method for querying the data to be labeled. This method selects data with the highest entropy value.
        :param n: Amount of data to query.
        :return: Array of indices to data selected to be labeled.
        """

        unlabeled_indices = np.array(self.pool_size)[~self.labeled_indices]
        probabilities, _ = self.predict(self.x[unlabeled_indices], self.y[unlabeled_indices])
        log_probabilities = torch.log(probabilities)
        uncertainties = (probabilities * log_probabilities).sum(1)
        return unlabeled_indices[uncertainties.sort()[1][:n]]
