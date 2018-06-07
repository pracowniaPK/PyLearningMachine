import random
import math
import pandas as pd


def k_mean(data, n_clus, dist, random_seed=1, threshold=0, verbose=0):
    """ Returns array with labes assigining items to clusters

    Args:
        data: iterable of iterables of numbers
        n_clus: expected number of clusters
        dist: function calculating distance of given pair of points (dist(point1, point2))
        random_seed: random seed to be used to generate initial centroids positions

    Return:
        list of labels of cluster to which points are assigned
    """
    random.seed(random_seed)
    data_dim = data.shape[1]
    data_describe = data.describe()

    # initialize centroids
    centroids = pd.DataFrame(columns=data.columns)
    for i in range(n_clus):
        centroids.loc[i] = [0] * data_dim
        for j in range(data_dim):
            centroids.iloc[i, j] = random.uniform(
                data_describe.loc['min'][j], data_describe.loc['max'][j])

    while True:
        old_centroids = centroids.copy()

        # calculate distances to centroids
        # assign to centroids
        assigned_centroids = pd.Series(index=data.index)
        for i in range(data.shape[0]):
            min_dist = math.inf
            assigned_c = -1
            for j in range(n_clus):
                curr_dist = dist(data.iloc[i], centroids.iloc[j])
                if curr_dist < min_dist:
                    assigned_c = j
                    min_dist = curr_dist
            assigned_centroids[i] = assigned_c

        # calculate new centroids position
        for i in range(n_clus):
            assigned_points = assigned_centroids[assigned_centroids == i]
            if len(assigned_points) == 0:
                for j in range(data_dim):
                    centroids.iloc[i, j] = random.uniform(
                        data_describe.loc['min'][j], data_describe.loc['max'][j])
            else:
                new_position = pd.Series(index=data.columns)
                new_position = new_position.fillna(0)
                for j in range(len(assigned_points)):
                    new_position += data.iloc[j]
                centroids.iloc[i] = new_position/len(assigned_points)

        # check if anything changed (halting condition)
        if verbose == 1:
            print(
                f"Current sum distance: {(centroids - old_centroids).applymap(abs).sum().sum()}")
        if (centroids - old_centroids).applymap(abs).sum().sum() <= threshold:
            break

    return assigned_centroids.map(int)
