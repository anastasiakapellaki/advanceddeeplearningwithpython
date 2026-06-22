import torch
from torch.utils.data import Subset
from dataset import SetDataset
from citeseer import CiteseerSetDataset
import torch.nn as nn
from torch.utils.data import DataLoader
import copy


MANUAL_SEED = 42
torch.manual_seed(seed=MANUAL_SEED)


class DeepSetsModelCiteseer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
       
 
        self.phi = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.rho = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim), nn.ReLU(), nn.Linear(2*hidden_dim, num_classes))
 
    def forward(self, set_input, mask):
        #set_input is (batch, set_size, input_dim)
        phi_out = self.phi(set_input)
        phi_out = phi_out * mask.unsqueeze(-1)
        pooled = phi_out.sum(dim=1)
        logits = self.rho(pooled)
        
        return logits
 
class DeepSetsModelSynthetic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
       
 
        self.phi = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.rho = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim), nn.ReLU(), nn.Linear(2*hidden_dim, 1))
 
    def forward(self, set_input):
        #set_input is (batch, set_size, input_dim)
        phi_out = self.phi(set_input)
        pooled = phi_out.sum(dim=1)
        logits = self.rho(pooled)
        
        return logits



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

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for X, Y in loader:
           
            X = X.to(device)
            Y = Y.to(device).float() #BCELoss needs float


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

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy        


def run_epoch_citeseer(model, loader, device, optimizer=None):
        train_mode = optimizer is not None
        
        if train_mode:
            model.train()
        else:
            model.eval()
 
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
 
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
 
        for X, mask, Y in loader:
 
            X = X.to(device)
            mask = mask.to(device)
            Y = Y.to(device)  # CrossEntropyLoss needs LONG class indices, not float
 
 
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
 
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
 
        return avg_loss, accuracy        

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

set_deep_synthetic = DeepSetsModelSynthetic(input_dim, hidden_dim=HIDDEN_DIM).to(device)

train_loader_synthetic = DataLoader(train_set_synthetic, batch_size=64, shuffle=True)
val_loader_synthetic = DataLoader(val_set_synthetic, batch_size=64, shuffle=False)
test_loader_synthetic = DataLoader(test_set_synthetic, batch_size=64, shuffle=False)

optimizer = torch.optim.Adam(set_deep_synthetic.parameters(), lr=1e-3)
epochs = 50
best_val_loss_synthetic = float("inf")
best_state_synthetic = None

early_stopping = EarlyStopping(patience=5, min_delta=1e-3)

print("Start DeepSetsSynthetic training...")
for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = run_epoch_synthetic(set_deep_synthetic, train_loader_synthetic, device, optimizer=optimizer)
    val_loss, val_accuracy = run_epoch_synthetic(set_deep_synthetic, val_loader_synthetic, device, optimizer=None)
    
    print(f"Epoch {epoch:02d} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}\n"
          f"           Train accuracy: {train_accuracy:.6f} | Val accuracy: {val_accuracy:.6f}")
 
    if val_loss < best_val_loss_synthetic:
        best_val_loss_synthetic = val_loss
        best_state_synthetic = copy.deepcopy(set_deep_synthetic.state_dict())
    
    
    early_stopping(val_loss)
    
    if early_stopping.early_stop:
        print("Early stopping in epoch: ", epoch)
        break

set_deep_synthetic.load_state_dict(best_state_synthetic)
test_loss_synthetic, test_accuracy_synthetic = run_epoch_synthetic(set_deep_synthetic, test_loader_synthetic, device, optimizer=None)
print()
print(f"Best epoch's val loss: {best_val_loss_synthetic:.6f}")
print(f"Final test loss: {test_loss_synthetic:.6f} | Final test accuracy: {test_accuracy_synthetic:.6f}\n")



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

set_deep_citeseer = DeepSetsModelCiteseer(input_dim, hidden_dim=HIDDEN_DIM, num_classes=6).to(device)

train_loader_citeseer = DataLoader(train_set_citeseer, batch_size=64, shuffle=True)
val_loader_citeseer = DataLoader(val_set_citeseer, batch_size=64, shuffle=False)
test_loader_citeseer = DataLoader(test_set_citeseer, batch_size=64, shuffle=False)

optimizer = torch.optim.Adam(set_deep_citeseer.parameters(), lr=1e-3)
epochs = 50
best_val_loss_citeseer = float("inf")
best_state_citeseer = None

early_stopping = EarlyStopping(patience=5, min_delta=1e-3)

print("Start DeepSetsCiteseer training...")
for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = run_epoch_citeseer(set_deep_citeseer, train_loader_citeseer, device, optimizer=optimizer)
    val_loss, val_accuracy = run_epoch_citeseer(set_deep_citeseer, val_loader_citeseer, device, optimizer=None)
    
    print(f"Epoch {epoch:02d} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}\n"
          f"           Train accuracy: {train_accuracy:.6f} | Val accuracy: {val_accuracy:.6f}")
 
    if val_loss < best_val_loss_citeseer:
        best_val_loss_citeseer = val_loss
        best_state_citeseer = copy.deepcopy(set_deep_citeseer.state_dict())
    
    
    early_stopping(val_loss)
    
    if early_stopping.early_stop:
        print("Early stopping in epoch: ", epoch)
        break

set_deep_citeseer.load_state_dict(best_state_citeseer)
test_loss_citeseer, test_accuracy_citeseer = run_epoch_citeseer(set_deep_citeseer, test_loader_citeseer, device, optimizer=None)
print()
print(f"Best epoch's val loss: {best_val_loss_citeseer:.6f}")
print(f"Final test loss: {test_loss_citeseer:.6f} | Final test accuracy: {test_accuracy_citeseer:.6f}")