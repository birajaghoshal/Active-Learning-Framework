import numpy as np
from strategy import Strategy
from sklearn.cluster import KMeans


class KMeansSoftmaxSampling(Strategy):
    """
    This class is for K-Means sampling using the softmax predictions sampling for active learning. This method uses the
    trained model to obtain predictions for each piece of data. These features are then clustered using the K-Means
    clustering method and the centers of the clusters are then selected to be annotated.
    """

    def query(self, n):
        """
        Method for querying the data to be labeled. This method selects data that are centers of clusters specified by
        the K-Means clusting method.
        :param n: Amount of data to query.
        :return: Array of indices to data selected to be labeled.
        """

        unlabeled_indices = np.arange(self.pool_size)[~self.labeled_indices]
        predictions, _ = self.predict(self.x[unlabeled_indices], self.y[unlabeled_indices])
        predictions = predictions.numpy()
        cluster_learner = KMeans(n_clusters=n, n_jobs=-1)
        cluster_learner.fit(predictions)

        cluster_indices = cluster_learner.predict(predictions)
        centers = cluster_learner.cluster_centers_[cluster_indices]
        distances = (predictions - centers) ** 2
        distances = distances.sum(axis=1)
        query_indices = np.array([np.arange(predictions.shape[0])[cluster_indices == i]
                                  [distances[cluster_indices == i].argmin()] for i in range(n)])

        return unlabeled_indices[query_indices]
