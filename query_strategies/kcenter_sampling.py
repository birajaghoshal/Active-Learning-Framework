import numpy as np
from strategy import Strategy
from scipy.spatial import distance_matrix


class KCentreGreedySampling(Strategy):
    """
    This class is for K-Centre sampling using the embedded features for active learning. This method uses the model
    trained model to extract features from the data. These features are then covered using the K-Centre cover method
    calculated using the greedy method. The centres of the covers are selected to be annotated.
    """

    def query(self, n):
        unlabeled_indices = np.arange(self.pool_size)[~self.labeled_indices]
        labeled_indices = np.arange(self.pool_size)[self.labeled_indices]

        embeddings = self.get_embeddings(self.x, self.y).numpy()

        labeled = embeddings[labeled_indices]
        unlabeled = embeddings[unlabeled_indices]

        greddy_indices = []

        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j + 100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        farthest = np.argmax(min_dist)
        greddy_indices.append(farthest)
        for i in range(n-1):
            dist = distance_matrix(unlabeled[greddy_indices[-1], :].reshape((1, unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greddy_indices.append(farthest)

        return np.array(greddy_indices, dtype=int)
