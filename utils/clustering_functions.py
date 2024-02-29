import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import scipy.linalg as la
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

from signnet_functions import topological_distance_matrix, signed_modularity, frustration_index
from sicomm_functions import communicability_metrics


def compute_clustering(
    G,
    n_factions="optimize",
    ndims="optimize",
    clustering_method="spectral",
    use_matrix="comm_angle",
    validation_metric="silhouette",
):

    def compute_affinity_distance_coordinates(G, use_matrix, nodelist):

        if use_matrix == "adjacency":
            affinity = nx.adjacency_matrix(G, nodelist=nodelist).toarray()
            distance = (
                topological_distance_matrix(np.abs(affinity))
                if (
                    clustering_method == "agglomerative"
                    or validation_metric == "silhouette"
                )
                else np.zeros_like(affinity)
            )
            coordinates = None

        elif use_matrix == "positive_adjacency":
            affinity0 = nx.adjacency_matrix(G, nodelist=nodelist).toarray() + np.ones(
                (G.order(), G.order()), dtype=int
            )
            affinity = np.divide(
                1.0,
                affinity0,
                out=np.zeros_like(affinity0, dtype=np.float64),
                where=affinity0 != 0,
            )
            distance = (
                topological_distance_matrix(affinity)
                if clustering_method == "agglomerative"
                or validation_metric == "silhouette"
                else np.zeros_like(affinity)
            )
            coordinates = None

        elif use_matrix == "comm_distance":
            distance = communicability_metrics(G, nodelist, output="distance")
            affinity = 1 / (distance + 1)
            coordinates = None

        elif use_matrix == "comm_angle":
            theta = communicability_metrics(G, nodelist, output="angle")
            distance = 2 * (1 - np.cos(theta))
            affinity = np.cos(theta)
            coordinates = None

        elif use_matrix == "positive_comm_angle":
            theta = communicability_metrics(G, nodelist, output="angle")
            distance = theta
            affinity = 1 - theta / np.pi
            coordinates = None

        elif use_matrix == "comm_coordinates":
            coordinates = communicability_metrics(
                G, nodelist, output="comm_coordinates"
            ).T  # transpose so that rows are coordinates
            distance = pairwise_distances(coordinates, metric="euclidean")
            affinity = None

        else:

            raise ValueError(
                "The only valid criteria for affinity/distance matrices are 'adjacency', 'positive_adjacency', 'comm_distance', 'comm_angle', or 'positive_comm_angle' or 'comm_coordinates'"
            )

        return affinity, distance, coordinates

    def symmetrize_matrices(affinity, distance):
        if affinity is not None:
            affinity = (affinity + affinity.T) / 2
        if distance is not None:
            distance = (distance + distance.T) / 2
        return affinity, distance

    def process_outliers(labels):
        counter = max(labels)
        labels = [
            (counter := counter + 1) if label == -1 else label for label in labels
        ]
        return labels

    def perform_clustering(affinity, distance, coordinates, n_factions, MDS_dimension):

        if clustering_method == "kmeans":
            clustering = KMeans(n_clusters=n_factions)
            labels = clustering.fit(coordinates).labels_

        elif clustering_method == "DBSCAN":
            # coordinates = StandardScaler().fit_transform(coordinates)
            clustering = DBSCAN(eps=1.5, min_samples=2)
            labels = clustering.fit(coordinates).labels_
            labels = process_outliers(labels)

        elif clustering_method == "spectral":
            clustering = SpectralClustering(
                n_clusters=n_factions,
                assign_labels="discretize",
                affinity="precomputed",
            )
            labels = clustering.fit(affinity).labels_

        elif clustering_method == "agglomerative":
            clustering = AgglomerativeClustering(
                n_clusters=n_factions, affinity="precomputed", linkage="complete"
            )
            labels = clustering.fit(distance).labels_

        elif clustering_method == "MDS_kmeans":
            low_dim_embedding = MDS(
                n_components=MDS_dimension, dissimilarity="precomputed"
            ).fit_transform(distance)
            # low_dim_embedding = StandardScaler().fit_transform(low_dim_embedding)
            clustering = KMeans(n_clusters=n_factions)
            labels = clustering.fit(low_dim_embedding).labels_

        elif clustering_method == "MDS_DBSCAN":
            low_dim_embedding = MDS(
                n_components=MDS_dimension, dissimilarity="precomputed"
            ).fit_transform(distance)
            low_dim_embedding = StandardScaler().fit_transform(low_dim_embedding)
            clustering = DBSCAN(eps=1.5, min_samples=2)
            labels = clustering.fit(low_dim_embedding).labels_
            labels = process_outliers(labels)

        else:
            raise ValueError(
                f"The only valid methods are 'kmeans', 'DBSCAN', 'spectral', 'agglomerative', 'MDS_kmeans', 'MDS_DBSCAN'"
            )
        return labels

    def find_best_clustering(
        affinity, distance, coordinates, validation_metric, n_factions, MDS_dimension
    ):

        MDS_dimension_v = (
            [MDS_dimension]
            if MDS_dimension != "optimize"
            else [2, 3, 5, max(int(N / 10), 7)]
        )
        n_factions_v = (
            [n_factions] if n_factions != "optimize" else range(2, min(10, N))
        )

        max_score = -np.inf
        best_labels = None

        for MDS_dimension in MDS_dimension_v:

            for n_factions in n_factions_v:

                clustering_labels = perform_clustering(
                    affinity, distance, coordinates, n_factions, MDS_dimension
                )
                score = compute_validation_score(
                    distance, clustering_labels, validation_metric
                )
                if score > max_score + 1e-4:
                    max_score = score
                    best_labels = clustering_labels

        return best_labels

    def compute_validation_score(distance, labels, validation_metric):

        if validation_metric == "silhouette":
            if len(set(labels)) != 1 and len(set(labels)) != len(labels):
                return silhouette_score(distance, labels, metric="precomputed")
            else:
                return 0

        elif validation_metric == "frustration":
            return -frustration_index(G, labels)

        else:
            raise ValueError(
                f"The only valid validation metrics are 'silhouette' or 'frustration'"
            )

    ## MAIN ##

    nodelist = G.nodes()
    N = G.order()

    # ensure compatibility of method with 'comm_coordinates'
    if clustering_method in ["kmeans", "DBSCAN"]:
        use_matrix = "comm_coordinates"
    elif use_matrix == "comm_coordinates":
        raise ValueError(
            f"Clustering method {clustering_method} is not compatible with the 'comm_coordinates' matrix"
        )

    # avoid optimizing dimension in non-MDS methods or number of factions in DBSCAN methods
    MDS_dimension = (
        -1 if clustering_method not in ["MDS_kmeans", "MDS_DBSCAN"] else ndims
    )
    n_factions = -1 if clustering_method in ["DBSCAN", "MDS_DBSCAN"] else n_factions

    # compute affinity, distance, coordinates matrices, as required
    affinity, distance, coordinates = compute_affinity_distance_coordinates(
        G, use_matrix, nodelist
    )
    affinity, distance = symmetrize_matrices(affinity, distance)

    # clustering for user-specified number of factions
    if n_factions != "optimize" and MDS_dimension != "optimize":
        labels = perform_clustering(
            affinity, distance, coordinates, n_factions, MDS_dimension
        )

    # clustering for unspecified number of factions
    else:
        labels = find_best_clustering(
            affinity,
            distance,
            coordinates,
            validation_metric,
            n_factions,
            MDS_dimension,
        )

    return labels


def multidimensional_scaling(G, use_matrix, embedding_dimension=2):
    # compute communicability metrics
    nodelist = G.nodes()

    # compute distance matrix according to selected criteria
    if use_matrix == "shortest-path":
        affinity = nx.adjacency_matrix(G, nodelist=nodelist).toarray()
        distance = topological_distance_matrix(np.abs(affinity))

    elif use_matrix == "distance":
        distance = communicability_metrics(G, nodelist, output="distance")

    elif use_matrix == "angle":
        distance = communicability_metrics(G, nodelist, output="angle")

    elif use_matrix == "euclidean_angle":
        theta = communicability_metrics(G, nodelist, output="angle")
        distance = np.sqrt(2 * (1 - np.cos(theta)))

    else:
        raise ValueError(
            "The only valid criteria are 'angle', 'distance', 'euclidean_angle', or 'shortest-path'"
        )

    # correct rounding errors
    assert np.allclose(distance, distance.T, atol=1e-3), "Distance not symmetric"
    distance = (distance + distance.T) / 2

    # embed in a low dimensional space
    embedding = MDS(n_components=embedding_dimension, dissimilarity="precomputed")
    coords = embedding.fit_transform(distance)

    # assign coordinates to each node
    pos = {}
    for k, node in enumerate(G.nodes()):
        pos[node] = tuple(coords[k, :])

    return pos
