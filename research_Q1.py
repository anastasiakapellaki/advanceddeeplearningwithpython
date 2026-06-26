import numpy as np
import torch 
import torch.nn as nn
from set_transformers import SAB, MAB, PMA
from torch.utils.data import DataLoader, Dataset, random_split
import copy
#Creates 5000 groups each of them has n samples in value_range. 
def generate_max_dataset(n_samples=5000, max_set_size=10, value_range=(0,100)):
    sets = []
    targets = []

    for _ in range(n_samples):
        n = np.random.randint(1, max_set_size+1)
        x = np.random.uniform(*value_range, size=n)
        sets.append(x)
        targets.append(np.max(x))
    
    batch = list(zip(sets, targets))
    return batch

def padding_max_regression(batch, max_size=10):
    sets, targets = zip(*batch)

    padded = torch.zeros(len(sets), max_size, 1)
    masks = torch.zeros(len(sets), max_size, dtype = torch.bool)

    for i,s in enumerate(sets):
        n = len(s)
        padded[i, :n, 0] = torch.tensor(s, dtype=torch.float32)
        masks[i, :n] = True

    targets = torch.tensor(targets, dtype=torch.float32)
    
    return padded, masks, targets

class DeepSetsMax(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, X, mask):
        # X shape: (batch, max_size, 1)
        phi_out = self.phi(X)                        # (batch, max_size, hidden_dim)
        phi_out = phi_out * mask.unsqueeze(-1)       # zero out padding after phi
        pooled  = phi_out.sum(dim=1)                 # (batch, hidden_dim)
        return self.rho(pooled).squeeze(-1)          # (batch,)
    

class SetTransformerMax(nn.Module):
    def __init__(self, d=64, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(1, d)
        self.sab        = SAB(d, num_heads)
        self.pma        = PMA(d, num_heads, k=1)
        self.rho        = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, 1)
        )

    def forward(self, X, mask):
        mask_4d = mask.unsqueeze(1).unsqueeze(2)    # (batch, 1, 1, max_size)
        X = self.input_proj(X)                      # (batch, max_size, d)
        X = self.sab(X, mask_4d)                    # (batch, max_size, d)
        X = self.pma(X, mask_4d).squeeze(1)         # (batch, d)
        return self.rho(X).squeeze(-1)              # (batch,)
    

def run_epoch_max(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    
    loss_fn = nn.L1Loss(reduction="sum")  # MAE loss
    total_loss, total_samples = 0.0, 0
    
    for X, mask, Y in loader:
        X, mask, Y = X.to(device), mask.to(device), Y.to(device)
        
        if train_mode:
            preds = model(X, mask)
            loss  = loss_fn(preds, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = model(X, mask)
                loss  = loss_fn(preds, Y)
        
        total_loss    += loss.item()
        total_samples += X.size(0)
    
    return total_loss / total_samples  # MAE


batch = generate_max_dataset(n_samples=5000, max_set_size=10, value_range=(0,100))

n_total = len(batch)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val



train_data, val_data, test_data = random_split(batch, [n_train, n_val, n_test],  generator=torch.Generator().manual_seed(42))

collate = lambda b: padding_max_regression(b, max_size=10)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True,  collate_fn=collate)
val_loader   = DataLoader(val_data,   batch_size=64, shuffle=False, collate_fn=collate)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False, collate_fn=collate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42
EPOCHS = 50

torch.manual_seed(SEED)
deep_sets_max = DeepSetsMax(hidden_dim=64).to(device)
opt_ds        = torch.optim.Adam(deep_sets_max.parameters(), lr=1e-3)

torch.manual_seed(SEED)
set_transf_max = SetTransformerMax(d=64, num_heads=4).to(device)
opt_st         = torch.optim.Adam(set_transf_max.parameters(), lr=1e-3)

best_mae = float("inf")
best_state = None

print("Training DeepSetsMax...")
for epoch in range(1, EPOCHS + 1):
    train_mae = run_epoch_max(deep_sets_max, train_loader, device, opt_ds)
    val_mae   = run_epoch_max(deep_sets_max, val_loader,   device)
    if val_mae < best_mae:
        best_mae   = val_mae
        best_state = copy.deepcopy(deep_sets_max.state_dict())

    print(f"Epoch {epoch:02d} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

deep_sets_max.load_state_dict(best_state)

test_mae_ds = run_epoch_max(deep_sets_max, test_loader, device)
print(f"\nDeepSets     — Test MAE: {test_mae_ds:.4f}")

best_mae = float("inf")
best_state = None

print("\nTraining SetTransformerMax...")
for epoch in range(1, EPOCHS + 1):
    train_mae = run_epoch_max(set_transf_max, train_loader, device, opt_st)
    val_mae   = run_epoch_max(set_transf_max, val_loader,   device)
    if val_mae < best_mae:
        best_mae   = val_mae
        best_state = copy.deepcopy(set_transf_max.state_dict())
    print(f"Epoch {epoch:02d} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

set_transf_max.load_state_dict(best_state)
test_mae_st = run_epoch_max(set_transf_max, test_loader, device)
print(f"SetTransformer — Test MAE: {test_mae_st:.4f}")

print("\n--- Summary ---")
print(f"DeepSets       Test MAE: {test_mae_ds:.4f}")
print(f"SetTransformer Test MAE: {test_mae_st:.4f}")