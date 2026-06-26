import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import copy


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def run_epoch_synthetic(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device).float()
        if train_mode:
            logits = model(X).squeeze(-1)
            loss = loss_fn(logits, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
        else:
            with torch.no_grad():
                logits = model(X).squeeze(-1)
                loss = loss_fn(logits, Y)
                preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == Y).sum().item()
        total_loss += loss.item()
        total_samples += X.size(0)
    return total_loss / total_samples, total_correct / total_samples


def run_epoch_citeseer(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for X, mask, Y in loader:
        X = X.to(device)
        mask = mask.to(device)
        Y = Y.to(device)
        if train_mode:
            logits = model(X, mask)
            loss = loss_fn(logits, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
        else:
            with torch.no_grad():
                logits = model(X, mask)
                loss = loss_fn(logits, Y)
                preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == Y).sum().item()
        total_loss += loss.item()
        total_samples += X.size(0)
    return total_loss / total_samples, total_correct / total_samples


def run_multi_seed_experiment(
    model_class,
    model_kwargs,
    train_loader,
    val_loader,
    test_loader,
    run_epoch_fn,
    seeds=(42, 43, 44, 45, 46),
    epochs=50,
    lr=1e-3,
    patience=5,
    min_delta=1e-3,
    device="cpu",
    experiment_name="Experiment"
):
    """
    Runs the same experiment across multiple seeds and returns
    mean +/- std of test accuracy.

    Args:
        model_class:      The model class to instantiate
                          (e.g. SetTransformersCiteseer)
        model_kwargs:     Dict of keyword arguments passed to model_class
                          (e.g. {"input_dim": 3703, "d": 32, "num_heads": 4})
        train_loader:     DataLoader for training set — created ONCE outside
                          this function so the train/val/test split is fixed
                          across all seeds; only weight initialization varies.
        val_loader:       DataLoader for validation set
        test_loader:      DataLoader for test set
        run_epoch_fn:     Either run_epoch_synthetic or run_epoch_citeseer
        seeds:            Iterable of integer seeds to run
        epochs:           Maximum number of training epochs per seed
        lr:               Learning rate for Adam optimizer
        patience:         EarlyStopping patience (epochs without improvement)
        min_delta:        EarlyStopping minimum improvement threshold
        device:           torch device string
        experiment_name:  Label printed in logs for readability

    Returns:
        results:  list of per-seed test accuracies
        mean:     float — mean test accuracy across seeds
        std:      float — standard deviation across seeds
    """
    results = []

    for seed in seeds:
        torch.manual_seed(seed)

        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = run_epoch_fn(
                model, train_loader, device, optimizer
            )
            val_loss, val_acc = run_epoch_fn(
                model, val_loader, device
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(
                    f"  [{experiment_name}] Seed {seed} | "
                    f"Early stop at epoch {epoch}"
                )
                break

        # Evaluate the best checkpoint (lowest val loss) on the test set
        model.load_state_dict(best_state)
        test_loss, test_acc = run_epoch_fn(model, test_loader, device)
        results.append(test_acc)
        print(
            f"  [{experiment_name}] Seed {seed} | "
            f"Test accuracy: {test_acc:.4f}"
        )

    mean = np.mean(results)
    std  = np.std(results)
    print(f"\n  [{experiment_name}] Test accuracy: {mean:.4f} +/- {std:.4f}\n")

    return results, mean, std


# Class Definition
class MultiHeadAttention(nn.Module):
    
    # Class Constructor.
    def __init__(self, d_model, num_heads):
        # Call the super class constructor.
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Set internal module parameters.
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Set the linear trandformation layers that will correspond to the
        # weight matrices Qw,Kw and Vw. An additional weight matrix Ow is 
        # required in order to provide the final output of the multi-head 
        # attention layer.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    # This method computes the scaled dot product-based attention scores.
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute the square matrix storing the pairwise scalded ttention 
        # scores.
        # The associated transposition operation on matrix K is applied in the
        # order of dimensions appearing as input arguments.
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            # All zero values will be replaced by -1e9 so that after the 
            # application of the softmax transfer function the corresponding
            # attention probability will be zero.
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        # Row-wise application of the softmax transfer funtion.    
        attn_probs = torch.softmax(attn_scores,-1)
        # Compute the final output of the attention layer which is calculated 
        # by multiplying with matrix V.
        output = torch.matmul(attn_probs, V)
        return output
    
    # This function splits the content of the input sequence to the various 
    # heads. It is assumed that the training input sequences are organized into
    # batches of batch size where each training sequence is composed of seq_len
    # instances so that each instance is of d_model dimensionality.
    # The operation that is being conducted in this function performes the 
    # following transformation to the input batch sequence:
    # [batch_size x seq_len x d_model]         ==> 
    # [batch_size x seq_len x num_heads x d_k] ==>
    # [batch_size x num_heads x seq_len x d_k]
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
    
    # This function combines the contents of the various heads into a batch of
    # training input sequences. In fact, it performs the inverse operation with
    # respect to split_heads method implemented above. 
    def combine_heads(self, x):
        batch_size, _, seq_len, d_k  = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
    
    
    # Forward Pass Function.
    def forward(self, Q, K, V, mask=None):
        # Compute the contents of matrices Q, K and V after being passed through
        # the corresponding linear layers and the split heads function.
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        # Compute the pairwise attention values for each head.
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # Compute the final output of the attention layer.
        output = self.W_o(self.combine_heads(attn_output))
        return output
    

class MAB(nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d, num_heads)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.rff = nn.Sequential(nn.Linear(d,d), nn.ReLU(), nn.Linear(d,d))

    def forward(self, X, Y, mask=None):
        H = self.norm1(X + self.attention(X,Y,Y,mask))
        return self.norm2(H + self.rff(H))
    

class SAB(nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()
        self.mab = MAB(d, num_heads)
    
    def forward(self, X, mask=None):
        return self.mab(X, X, mask)

class PMA(nn.Module):
    def __init__(self, d, num_heads, k=1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, d))  # learned seed vectors
        self.mab = MAB(d, num_heads)
        self.rff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, Z, mask=None):
        S = self.S.expand(Z.size(0), -1, -1) # (1, k, d) -> (batch_size, k, d) "-1: keep the dimension as it is"
        return self.mab(S, self.rff(Z), mask) 


class SetTransformersSynthetic(nn.Module):
    def __init__(self, input_dim, d, num_heads, num_sabs=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d)
        # nn.ModuleList allows num_sabs to be a proper experimental variable
        # rather than toggling commented lines. Each SAB is a separate module
        # with its own weights, applied sequentially in forward().
        self.sabs = nn.ModuleList([SAB(d, num_heads) for _ in range(num_sabs)])
        self.pma = PMA(d, num_heads, k=1)
        self.rho  = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, 1)   
        )

    def forward(self, X):
        X = self.input_proj(X)           # (batch, set_size, d)
        for sab in self.sabs:
            X = sab(X, mask=None)        # (batch, set_size, d)
        X = self.pma(X, mask=None).squeeze(1)  # (batch, d)
        return self.rho(X)               # (batch, 1)
    

class SetTransformersCiteseer(nn.Module):
    def __init__(self, input_dim, d, num_heads, num_sabs=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d)
        # nn.ModuleList allows num_sabs to be a proper experimental variable
        # rather than toggling commented lines. Each SAB is a separate module
        # with its own weights, applied sequentially in forward().
        self.sabs = nn.ModuleList([SAB(d, num_heads) for _ in range(num_sabs)])
        self.pma = PMA(d, num_heads, k=1)
        self.rho  = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, 6)   
        )

    def forward(self, X, mask):
        mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 21) -> (batch, 1, 1, 21)
        X = self.input_proj(X)
        for sab in self.sabs:
            X = sab(X, mask)             # (batch, set_size, d)
        X = self.pma(X, mask).squeeze(1)  # (batch, d)
        return self.rho(X)


# ---------------------------------------------------------------------------
# Synthetic dataset — single training run (unchanged)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataset import SetDataset
    from citeseer import CiteseerSetDataset
    
    data = torch.load("dataset.pt", weights_only=True)
    dataset = SetDataset.__new__(SetDataset)  # or just rebuild via constructor
    dataset.X = data["X"]
    dataset.neighbors = data["neighbors"]
    dataset.labels = data["y_modified"]
    

    train_set_synthetic = Subset(dataset, data["train_idx"])
    val_set_synthetic   = Subset(dataset, data["val_idx"])
    test_set_synthetic  = Subset(dataset, data["test_idx"])

    input_dim = train_set_synthetic[0][0].shape[1]
    HIDDEN_DIM = 32


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MANUAL_SEED = 45
    torch.manual_seed(seed=MANUAL_SEED)

    set_transformers_synthetic = SetTransformersSynthetic(
        input_dim, d=HIDDEN_DIM, num_heads=4
    ).to(device)

    train_loader_synthetic = DataLoader(train_set_synthetic, batch_size=64, shuffle=True)
    val_loader_synthetic = DataLoader(val_set_synthetic, batch_size=64, shuffle=False)
    test_loader_synthetic = DataLoader(test_set_synthetic, batch_size=64, shuffle=False)

    optimizer = torch.optim.Adam(set_transformers_synthetic.parameters(), lr=1e-3)
    epochs = 50
    best_val_loss_synthetic = float("inf")
    best_state_synthetic = None

    early_stopping = EarlyStopping(patience=5, min_delta=1e-3)

    print("Start SetTransformersSynthetic training...")
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch_synthetic(set_transformers_synthetic, train_loader_synthetic, device, optimizer=optimizer)
        val_loss, val_accuracy = run_epoch_synthetic(set_transformers_synthetic, val_loader_synthetic, device, optimizer=None)
        
        print(f"Epoch {epoch:02d} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}\n"
            f"           Train accuracy: {train_accuracy:.6f} | Val accuracy: {val_accuracy:.6f}")
    
        if val_loss < best_val_loss_synthetic:
            best_val_loss_synthetic = val_loss
            best_state_synthetic = copy.deepcopy(set_transformers_synthetic.state_dict())
        
        
        early_stopping(val_loss)
        
        if early_stopping.early_stop:
            print("Early stopping in epoch: ", epoch)
            break

    set_transformers_synthetic.load_state_dict(best_state_synthetic)
    test_loss_synthetic, test_accuracy_synthetic = run_epoch_synthetic(set_transformers_synthetic, test_loader_synthetic, device, optimizer=None)
    print()
    print(f"Best epoch's val loss: {best_val_loss_synthetic:.6f}")
    print(f"Final test loss: {test_loss_synthetic:.6f} | Final test accuracy: {test_accuracy_synthetic:.6f}\n")


    # ---------------------------------------------------------------------------
    # CiteSeer dataset — single training run (unchanged)
    # ---------------------------------------------------------------------------

    data = torch.load("citeseer_dataset.pt", weights_only=True)
    dataset = CiteseerSetDataset.__new__(CiteseerSetDataset)
    dataset.x = data["x"]
    dataset.neighbor_idx = data["neighbor_idx"]
    dataset.neighbor_mask = data["neighbor_mask"]
    dataset.labels = data["y"]

    train_set_citeseer = Subset(dataset, data["train_mask"].nonzero(as_tuple=True)[0])
    val_set_citeseer   = Subset(dataset, data["val_mask"].nonzero(as_tuple=True)[0])
    test_set_citeseer  = Subset(dataset, data["test_mask"].nonzero(as_tuple=True)[0])

    input_dim = train_set_citeseer[0][0].shape[1]
    HIDDEN_DIM = 32

    torch.manual_seed(seed=MANUAL_SEED)
    set_transformers_citeseer = SetTransformersCiteseer(
        input_dim, d=HIDDEN_DIM, num_heads=4
    ).to(device)

    train_loader_citeseer = DataLoader(train_set_citeseer, batch_size=64, shuffle=True)
    val_loader_citeseer = DataLoader(val_set_citeseer, batch_size=64, shuffle=False)
    test_loader_citeseer = DataLoader(test_set_citeseer, batch_size=64, shuffle=False)

    optimizer = torch.optim.Adam(set_transformers_citeseer.parameters(), lr=1e-3)
    epochs = 50
    best_val_loss_citeseer = float("inf")
    best_state_citeseer = None

    early_stopping = EarlyStopping(patience=5, min_delta=1e-3)

    print("Start SetTransformersCiteseer training...")
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch_citeseer(set_transformers_citeseer, train_loader_citeseer, device, optimizer=optimizer)
        val_loss, val_accuracy = run_epoch_citeseer(set_transformers_citeseer, val_loader_citeseer, device, optimizer=None)
        
        print(f"Epoch {epoch:02d} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}\n"
            f"           Train accuracy: {train_accuracy:.6f} | Val accuracy: {val_accuracy:.6f}")
    
        if val_loss < best_val_loss_citeseer:
            best_val_loss_citeseer = val_loss
            best_state_citeseer = copy.deepcopy(set_transformers_citeseer.state_dict())
        
        
        early_stopping(val_loss)
        
        if early_stopping.early_stop:
            print("Early stopping in epoch: ", epoch)
            break

    set_transformers_citeseer.load_state_dict(best_state_citeseer)
    test_loss_citeseer, test_accuracy_citeseer = run_epoch_citeseer(set_transformers_citeseer, test_loader_citeseer, device, optimizer=None)
    print()
    print(f"Best epoch's val loss: {best_val_loss_citeseer:.6f}")
    print(f"Final test loss: {test_loss_citeseer:.6f} | Final test accuracy: {test_accuracy_citeseer:.6f}")


# ---------------------------------------------------------------------------
# Sensitivity Analysis: Impact of Weight Initialization:
#
# During experimentation we observed that changing the random seed
# significantly altered the relative performance of 1-SAB vs 2-SAB
# architectures on CiteSeer, sometimes reversing the ranking entirely.
# The loaders are created once above so the train/val/test split is
# identical across all seeds — only weight initialization varies.
# ---------------------------------------------------------------------------

    SEEDS = [42, 43, 44, 45, 46]

    print("=" * 60)
    print("Sensitivity Analysis: CiteSeer — 1 SAB vs 2 SABs")
    print("=" * 60)

    _, st1_mean, st1_std = run_multi_seed_experiment(
        model_class     = SetTransformersCiteseer,
        model_kwargs    = {"input_dim": input_dim, "d": HIDDEN_DIM,
                        "num_heads": 4, "num_sabs": 1},
        train_loader    = train_loader_citeseer,
        val_loader      = val_loader_citeseer,
        test_loader     = test_loader_citeseer,
        run_epoch_fn    = run_epoch_citeseer,
        seeds           = SEEDS,
        device          = device,
        experiment_name = "SetTransformer 1SAB  CiteSeer"
    )

    _, st2_mean, st2_std = run_multi_seed_experiment(
        model_class     = SetTransformersCiteseer,
        model_kwargs    = {"input_dim": input_dim, "d": HIDDEN_DIM,
                        "num_heads": 4, "num_sabs": 2},
        train_loader    = train_loader_citeseer,
        val_loader      = val_loader_citeseer,
        test_loader     = test_loader_citeseer,
        run_epoch_fn    = run_epoch_citeseer,
        seeds           = SEEDS,
        device          = device,
        experiment_name = "SetTransformer 2SABs CiteSeer"
    )

    print("Summary")
    print("-" * 40)
    print(f"  1 SAB  : {st1_mean:.4f} +/- {st1_std:.4f}")
    print(f"  2 SABs : {st2_mean:.4f} +/- {st2_std:.4f}")