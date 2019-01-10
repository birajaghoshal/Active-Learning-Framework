from strategy import Strategy

import gc
import numpy as np
import gurobipy as gurobi
from scipy.spatial import distance_matrix
from tensorflow.contrib.keras import backend as K

# TODO: Fix the embeddings to be a numpy ndarray, fix labelled indices list to be the correct format.

class CoreSetSampling(Strategy):
    def query(self, n):
        unlabeled_indices = np.arange(self.pool_size)[~self.labeled_indices]
        embeddings = self.get_embeddings(self.x, self.y)

        print("Calculating Greedy K-Center Solution...")
        new_indices, max_delta = self.greedy_k_center(embeddings[self.labeled_indices],
                                                      embeddings[unlabeled_indices], n)
        new_indices = unlabeled_indices[new_indices]
        outlier_count = int(len(self.x) / 10000)
        submipnodes = 20000

        eps = 0.01
        upper_bound = max_delta
        lower_bound = max_delta / 2.0
        print("Building MIP Model...")
        model, graph = self.mip_model(embeddings, self.labeled_indices, len(self.labeled_indices) + n, upper_bound,
                                      outlier_count, greddy_indices=new_indices)
        model.Params.SubMIPNodes = submipnodes
        points, outliers = model.__data
        model.optimize()
        indices = [i for i in graph if points[i].x == 1]
        current_delta = upper_bound
        while upper_bound - lower_bound > eps:
            print("Upper bound is {ub}, lower bound is {lb}".format(ub=upper_bound, lb=lower_bound))
            if model.getAttr(gurobi.GRB.Attr.Status) in [gurobi.GRB.INFEASIBLE, gurobi.GRB.TIME_LIMIT]:
                print("Optimixation Failed - Infeasible!")

                lower_bound = max(current_delta, self.get_graph_min(embeddings, current_delta))
                current_delta = (upper_bound + lower_bound) / 2.

                del model
                gc.collect()
                model, graph = self.mip_model(embeddings, self.labeled_indices, len(self.labeled_indices) + n,
                                              current_delta, outlier_count, greddy_indices=indices)
                points, outliers = model.__data
                model.Params.SubMIPNodes = submipnodes
            else:
                print("Optimisation Succeeded!")
                upper_bound = min(current_delta, self.get_graph_max(embeddings, current_delta))
                current_delta = (upper_bound + lower_bound) / 2.
                indices = [i for i in graph if points[i].x == 1]

                del model
                gc.collect()
                model, graph = self.mip_model(embeddings, self.labeled_indices, len(self.labeled_indices) + n,
                                              current_delta, outlier_count, greddy_indices=indices)
                points, outliers = model.__data
                model.Params.SubMIPNodes = submipnodes

            if upper_bound - lower_bound > eps:
                model.optimize()

        return np.array(indices)

    def greedy_k_center(self, labeled, unlabeled, n):
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

        return np.array(greddy_indices, dtype=int), np.max(min_dist)

    def mip_model(self, embeddings, labeled_indices, n, delta, outlier_count, greddy_indices):
        model = gurobi.Model("Core Set Selection")

        points = {}
        outliers = {}
        for i in range(embeddings.shape[0]):
            if i in labeled_indices:
                points[i] = model.addVar(ub=1., lb=1., vtype="B", name="points_{}".format(i))
            else:
                points[i] = model.addVar(vtype="B", name="points_{}".format(i))
        for i in range(embeddings.shape[0]):
            outliers[i] = model.addVar(vtype="B", name="outliers_{}".format(i))
            outliers[i].start = 0

        if greddy_indices is not None:
            for i in greddy_indices:
                points[i].start = 1.0

        model.addConstr(sum(outliers[i] for i in outliers) <= outlier_count, "budget")

        model.addConstr(sum(points[i] for i in range(embeddings.shape[0])) == n, "budget")
        neighbors = {}
        graph = {}
        print("Updating Neighborhoods in MIP Model...")
        for i in range(0, embeddings.shape[0], 1000):
            print("At Point " + str(i))
            if i + 1000 > embeddings.shape[0]:
                distances = self.get_distance_matrix(embeddings[i:i + 1000], embeddings)
                amount = embeddings.shape[0] - i
            else:
                distances = self.get_distance_matrix(embeddings[i:i + 1000], embeddings)
                amount = 1000

            distances = np.reshape(distances, (amount, -1))
            for j in range(i, i + amount):
                graph[j] = [(index, distances[j-i, index]) for index in np.reshape(np.where(distances[j-i, :] <= delta), (-1))]
                neighbors[j] = [points[index] for index in np.reshape(np.where(distances[j-i, :] <= delta), (-1))]
                neighbors[j].append(outliers[j])
                model.addConstr(sum(neighbors[j]) >= 1, "converage+outliers")

        model.__data = points, outliers
        model.Params.MIPFocus = 1
        model.params.TIME_LIMIT = 180

        return model, graph

    def get_distance_matrix(self, x, y):
        x_input = K.placeholder((x.shape))
        y_input = K.placeholder(y.shape)
        dot = K.dot(x_input, K.transpose(y_input))
        x_norm = K.reshape(K.sum(K.pow(x_input, 2), axis=1), (-1, 1))
        y_norm = K.reshape(K.sum(K.pow(y_input, 2), axis=1), (1, -1))
        dist_mat = x_norm + y_norm - 2.0*dot
        sqrt_dist_mat = K.sqrt(K.clip(dist_mat, min_value=0, max_value=10000))
        dist_func = K.function([x_input, y_input], [sqrt_dist_mat])

        return dist_func([x, y])[0]

    def get_graph_min(self, embedding, delta):
        print("Getting Graph Minimum...")
        minimum = 10000
        for i in range(0, embedding.shape[0], 1000):
            print("At Point " + str(i))

            if i + 1000 > embedding.shape[0]:
                distances = self.get_distance_matrix(embedding[i:], embedding)
            else:
                distances = self.get_distance_matrix(embedding[i:i + 1000], embedding)

            distances = np.reshape(distances, (-1))
            distances[distances < delta] = 10000
            minimum = min(minimum, np.min(distances))
        return minimum

    def get_graph_max(self, embedding, delta):
        print("Getting Graph Maximum...")
        maximum = 0
        for i in range(0, embedding.shape[0], 1000):
            print("At Point " + str(i))

            if i + 1000 > embedding.shape[0]:
                distances = self.get_distance_matrix(embedding[i:], embedding)
            else:
                distances = self.get_distance_matrix(embedding[i:i + 1000], embedding)

            distances = np.reshape(distances, (-1))
            distances[distances > delta] = 0
            maximum = max(maximum, np.max(distances))

        return maximum
