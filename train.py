import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import argparse
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from model import HybridCodeDetector

def train_model(language="python", epochs=100, batch_size=64, learning_rate=1e-5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {language.upper()} training on {device}...")
    print(f"Hyperparameters -> Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    
    X_train = np.load(f"{language}_train_X.npy")
    y_train = np.load(f"{language}_train_y.npy")
    X_val = np.load(f"{language}_validation_X.npy")
    y_val = np.load(f"{language}_validation_y.npy")
    
    print("Fitting standard scalable normalization...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    joblib.dump(scaler, f"{language}_scaler.pkl")
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val)), batch_size=batch_size, shuffle=False)
    
    model = HybridCodeDetector().to(device)
    criterion = nn.BCELoss()
    # Applied dynamic learning rate here
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    print("Beginning training loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X)
                val_loss += criterion(preds, batch_y).item()
                
        avg_train_loss = total_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{language}_best_model.pt")

    print(f"Training complete. Best model saved as {language}_best_model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Hybrid Code Detector")
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"], help="Target programming language")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for Adam optimizer")
    args = parser.parse_args()
    
    # Auto-assign paper's specified batch size based on language if left blank
    if args.batch_size is None:
        if args.language == "python":
            args.batch_size = 64
        elif args.language == "java":
            args.batch_size = 32
        elif args.language == "cpp":
            args.batch_size = 16
            
    train_model(language=args.language, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
