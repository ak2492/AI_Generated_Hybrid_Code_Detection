import torch
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from model import HybridCodeDetector

def evaluate_model(language="python", batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {language.upper()} evaluation on {device}...")
    
    # Load test data, scaler, and model
    X_test = np.load(f"{language}_test_X.npy")
    y_test = np.load(f"{language}_test_y.npy")
    scaler = joblib.load(f"{language}_scaler.pkl")
    
    X_test_scaled = scaler.transform(X_test)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test)), batch_size=batch_size, shuffle=False)
    
    model = HybridCodeDetector().to(device)
    model.load_state_dict(torch.load(f"{language}_best_model.pt"))
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("Running inference on test split...")
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            probabilities = model(batch_X)
            binary_preds = (probabilities >= 0.5).float()
            
            all_probs.extend(probabilities.cpu().numpy())
            all_preds.extend(binary_preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    # Calculate Metrics
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    
    # Calculate False Positive Rate (FPR)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    fpr = fp / (fp + tn)
    
    print("\n--- FINAL TEST METRICS ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"FPR:       {fpr:.4f}")

if __name__ == "__main__":
    evaluate_model(language="python")
