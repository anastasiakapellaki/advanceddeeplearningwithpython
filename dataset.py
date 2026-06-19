import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


def create_synthetic_dataset(
    n_samples=5000,
    d=16,
    k=10,
    epsilon=0.15,
    sigma=1.0,
    seed=42
):
    np.random.seed(seed)

    # 1. Δημιουργία labels 0/1
    y = np.random.randint(0, 2, size=n_samples)

    # 2. Μέσοι όροι για τις δύο κλάσεις
    mu0 = np.zeros(d)
    mu1 = np.ones(d) * 3.0

    # 3. Δημιουργία σημείων x_i ~ N(mu_y, sigma^2 I)
    X = np.zeros((n_samples, d))

    for i in range(n_samples):
        if y[i] == 0:
            X[i] = np.random.normal(mu0, sigma, size=d)
        else:
            X[i] = np.random.normal(mu1, sigma, size=d)

    # 4. Βρες k-nearest neighbors
    # βάζουμε k+1 γιατί ο κοντινότερος γείτονας είναι το ίδιο το σημείο
    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    knn.fit(X)

    distances, indices = knn.kneighbors(X)

    # αφαιρούμε το ίδιο το σημείο
    neighbor_indices = indices[:, 1:]

    # 5. Labels γειτόνων
    neighbor_labels = y[neighbor_indices]

    # 6. Μέσο label γειτονιάς
    avg_neighbor_label = neighbor_labels.mean(axis=1)

    # 7. Label flipping
    y_modified = y.copy()

    ambiguous = np.abs(avg_neighbor_label - 0.5) < epsilon
    y_modified[ambiguous] = 1 - y_modified[ambiguous]

    # 8. Features γειτόνων
    neighbor_features = X[neighbor_indices]

    return X, neighbor_features, y, y_modified, neighbor_indices