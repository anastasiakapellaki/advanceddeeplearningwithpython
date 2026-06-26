import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import Planetoid


def build_neighbor_sets(edge_index, n_nodes, max_neighbors):
    """
    For each node, collect up to `max_neighbors` citation-neighbor indices
    from edge_index (CiteSeer's edge_index is already symmetrized by the
    Planetoid loader, so this covers both "cites" and "cited by").

    Nodes with fewer than max_neighbors get zero-padded slots, marked False
    in the returned mask. Nodes with more get truncated to the first
    max_neighbors found (only ~0.4% of CiteSeer nodes are affected at
    max_neighbors=20 - see degree distribution discussion).

    Returns:
        neighbor_idx:  (n_nodes, max_neighbors) long  - padded with 0
        neighbor_mask: (n_nodes, max_neighbors) bool  - True = real neighbor
    """
    neighbor_idx = torch.zeros((n_nodes, max_neighbors), dtype=torch.long)
    neighbor_mask = torch.zeros((n_nodes, max_neighbors), dtype=torch.bool)

    src, dst = edge_index[0], edge_index[1]
    for i in range(n_nodes):
        neighbors_i = dst[src == i] #that means find the neighbors by checking the directed edges from node X to node i
        m = min(len(neighbors_i), max_neighbors)
        if m > 0:
            neighbor_idx[i, :m] = neighbors_i[:m]
            neighbor_mask[i, :m] = True

    return neighbor_idx, neighbor_mask


class CiteseerSetDataset(Dataset):
    def __init__(self, x, neighbor_idx, neighbor_mask, labels):
        self.x = x                          # (n, d) float32 - all node features
        self.neighbor_idx = neighbor_idx    # (n, max_neighbors) long
        self.neighbor_mask = neighbor_mask  # (n, max_neighbors) bool
        self.labels = labels                # (n,) long

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        point = self.x[idx].unsqueeze(0)                  # (1, d)
    
        mask_i = self.neighbor_mask[idx]                  # (max_neighbors,)
        neighbor_feats = self.x[self.neighbor_idx[idx]]   # (max_neighbors, d)
        neighbor_feats = neighbor_feats * mask_i.unsqueeze(-1)  # zero out padding

        set_input = torch.cat([point, neighbor_feats], dim=0)   # (1+max_neighbors, d)

        
        full_mask = torch.cat([
            torch.ones(1, dtype=torch.bool),
            mask_i
        ])  # (1+max_neighbors,)

        label = self.labels[idx]
        return set_input, full_mask, label



if __name__ == "__main__":
    MAX_NEIGHBORS = 20  # covers full neighbor set for 99.6% of nodes (13/3327 truncated)

    print("Loading CiteSeer via torch_geometric Planetoid...")
    dataset = Planetoid(root="./citeseer_raw", name="CiteSeer")
    data = dataset[0]

    x = data.x.float()
    y = data.y.long()
    n_nodes = x.shape[0]
    num_classes = dataset.num_classes

    print(f"Nodes: {n_nodes} | Features: {x.shape[1]} | Classes: {num_classes}")
    print(f"Directed edge entries: {data.edge_index.shape[1]}")

    print(f"Building fixed-size (k={MAX_NEIGHBORS}) masked neighbor sets...")

    neighbor_idx, neighbor_mask = build_neighbor_sets(data.edge_index, n_nodes, MAX_NEIGHBORS)

    degrees = neighbor_mask.sum(dim=1)  #how many connection a node has
    print(f"Avg neighbors per node after cap: {degrees.float().mean():.2f}")
    print(f"Isolated nodes (0 neighbors): {(degrees == 0).sum().item()}")
    print(f"Nodes using all {MAX_NEIGHBORS} slots (degree >= cap): {(degrees == MAX_NEIGHBORS).sum().item()}")

    print(f"Standard Planetoid split -> train: {data.train_mask.sum().item()}, "
          f"val: {data.val_mask.sum().item()}, test: {data.test_mask.sum().item()}")

    torch.save({
        "x": x,                                  # (3327, 3703) float32
        "y": y,                                   # (3327,) long
        "neighbor_idx": neighbor_idx,              # (3327, 20) long
        "neighbor_mask": neighbor_mask,            # (3327, 20) bool
        "train_mask": data.train_mask,             # (3327,) bool
        "val_mask": data.val_mask,                 # (3327,) bool
        "test_mask": data.test_mask,               # (3327,) bool
        "num_classes": torch.tensor(num_classes),
        "max_neighbors": torch.tensor(MAX_NEIGHBORS),
    }, "citeseer_dataset.pt")

    print("Saved citeseer_dataset.pt (raw tensors, no pickled objects)")

#How to unpack in future scripts:
'''
import torch
from torch.utils.data import Subset
from citeseerimport CiteseerSetDataset

data = torch.load("citeseer_dataset.pt", weights_only=True)
dataset = CiteseerSetDataset.__new__(CiteseerSetDataset)
dataset.x = data["x"]
dataset.neighbor_idx = data["neighbor_idx"]
dataset.neighbor_mask = data["neighbor_mask"]
dataset.labels = data["y"]

train_set = Subset(dataset, data["train_mask"].nonzero(as_tuple=True)[0])
val_set   = Subset(dataset, data["val_mask"].nonzero(as_tuple=True)[0])
test_set  = Subset(dataset, data["test_mask"].nonzero(as_tuple=True)[0])
'''