import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import argparse
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from model import HybridCodeDetector

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train_model(language="python", epochs=100, batch_size=64, learning_rate=1e-5, adversarial=False):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {language.upper()} training on {device}...")
    
    train_x_file = f"{language}_train_adv_X.npy" if adversarial else f"{language}_train_X.npy"
    train_y_file = f"{language}_train_adv_y.npy" if adversarial else f"{language}_train_y.npy"
    
    print(f"Loading training data from: {train_x_file}")
    X_train = np.load(train_x_file)
    y_train = np.load(train_y_file)
    X_val = np.load(f"{language}_validation_X.npy")
    y_val = np.load(f"{language}_validation_y.npy")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    scaler_file = f"{language}_adv_scaler.pkl" if adversarial else f"{language}_scaler.pkl"
    joblib.dump(scaler, scaler_file)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val)), batch_size=batch_size, shuffle=False)
    
    model = HybridCodeDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                preds = (model(batch_X) >= 0.5).int().cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.numpy())
                
        current_f1 = f1_score(val_targets, val_preds, zero_division=0)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Val F1: {current_f1:.4f}")
            
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            model_file = f"{language}_adv_best_model.pt" if adversarial else f"{language}_best_model.pt"
            torch.save(model.state_dict(), model_file)

    print(f"Training complete. Best F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adversarial", action="store_true", help="Load the adversarial augmented training set instead of the clean baseline")
    args = parser.parse_args()
    
    if args.batch_size is None:
        args.batch_size = 64 if args.language == "python" else (32 if args.language == "java" else 16)
            
    train_model(language=args.language, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, adversarial=args.adversarial)
