import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

print("Το script ξεκίνησε")
def createdataset(
    n_samples=5000,
    d=16,
    k=10,
    epsilon=0.15,
    sigma=1.0,
    seed=42
):
    np.random.seed(seed)
    y = np.random.randint(0, 2, size=n_samples)    
    mu0 = np.zeros(d)
    mu1 = np.ones(d) * 3.0
    X = np.zeros((n_samples, d))

    for i in range(n_samples):
        if y[i] == 0:
            X[i] = np.random.normal(mu0, sigma, size=d)
        else:
            X[i] = np.random.normal(mu1, sigma, size=d)

    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    knn.fit(X)

    distances, indices = knn.kneighbors(X)

    
    neighborindices = indices[:, 1:]

    
    neighborlabels = y[neighborindices]

    
    averageneighborlabel = neighborlabels.mean(axis=1)

       
    y_modified = y.copy()

    ambiguous = np.abs(averageneighborlabel - 0.5) < epsilon
    y_modified[ambiguous] = 1 - y_modified[ambiguous]

    
    neighborfeatures = X[neighborindices]

    return X, neighborfeatures, y, y_modified, neighborindices

X, neighbor_features, y_original, y_modified, neighbor_indices = createdataset()

print("X:", X.shape)
print("Neighbors:", neighbor_features.shape)
print("Original labels:", y_original.shape)
print("Modified labels:", y_modified.shape)
class SetDataset(Dataset):
    def __init__(self, X, neighbor_features, labels):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.neighbors = torch.tensor(
            neighbor_features,
            dtype=torch.float32
        )
        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        point = self.X[idx].unsqueeze(0)

        
        neighbors = self.neighbors[idx]

        
        set_input = torch.cat(
            [point, neighbors],
            dim=0
        )

        label = self.labels[idx]

        return set_input, label
   

dataset = SetDataset(
    X,
    neighbor_features,
    y_modified
)
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)