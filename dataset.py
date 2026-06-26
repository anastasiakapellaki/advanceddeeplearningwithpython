import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


def createdataset(
    n_samples=50000,
    d=16,
    k=10,
    epsilon=0.25,
    sigma=1.0,
    seed=42
):
    np.random.seed(seed)
    y = np.random.randint(0, 2, size=n_samples)    
    mu0 = np.zeros(d)
    mu1 = np.ones(d) * 1.0
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
    print("Ambiguous samples:", ambiguous.sum())
    print("Percentage:", 100 * ambiguous.mean(), "%")
    
    neighborfeatures = X[neighborindices]

    return X, neighborfeatures, y, y_modified, neighborindices

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
    
if __name__ == '__main__':  
    X, neighbor_features, y_original, y_modified, neighbor_indices = createdataset()

    print("X:", X.shape)
    print("Neighbors:", neighbor_features.shape)
    print("Original labels:", y_original.shape)
    print("Modified labels:", y_modified.shape)


    dataset = SetDataset(
        X,
        neighbor_features,
        y_modified
    )

    # Train / val / test split, computed ONCE here so every downstream task
        # script trains/evaluates on identical splits. Stratified on y_modified
        # since that's the actual training target (not y_original).
    all_idx = np.arange(len(y_modified))

    #We use train_split in indices in order to achive consistant splitting after loading the dataset.pt in other scripts. y_modified , neighbors etc point to the same order of the 5000 samples.
    train_idx, temp_idx = train_test_split(all_idx, test_size=0.30, random_state=42, stratify=y_modified)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=y_modified[temp_idx])

    print(f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    
    torch.save({
        "X": dataset.X,                      # (n_samples, d) float32 tensor
        "neighbors": dataset.neighbors,      # (n_samples, k, d) float32 tensor
        "y_modified": dataset.labels,        # (n_samples,) long tensor 
        "y_original": torch.tensor(y_original, dtype=torch.long),
        "neighbor_indices": torch.tensor(neighbor_indices, dtype=torch.long),
        "train_idx": torch.tensor(train_idx, dtype=torch.long),
        "val_idx": torch.tensor(val_idx, dtype=torch.long),
        "test_idx": torch.tensor(test_idx, dtype=torch.long),
    }, "dataset.pt")
    
    print("Saved dataset.pt (raw tensors + train/val/test indices)")
 
#How to unpack in future scripts:
'''
data = torch.load("dataset.pt", weights_only=True)
dataset = SetDataset.__new__(SetDataset)  # or just rebuild via constructor
dataset.X = data["X"]
dataset.neighbors = data["neighbors"]
dataset.labels = data["y_modified"]
 
from torch.utils.data import Subset
train_set = Subset(dataset, data["train_idx"])
val_set   = Subset(dataset, data["val_idx"])
test_set  = Subset(dataset, data["test_idx"])
'''