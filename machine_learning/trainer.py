import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from models.model_01 import LinearBA


# Dataset filepaths
TRAIN_CSV = ""
VAL_CSV   = ""
TEST_CSV  = ""

# Set filepath for output directory
OUT_DIR = ""
os.makedirs(OUT_DIR, exist_ok=True)


# Set seed for reproducibility
SEED = 37
np.random.seed(SEED)
torch.manual_seed(SEED)


# Training hyperparameters
EPOCHS = 200
LR = 1e-3
BATCH_SIZE = 64
LAMBDA = 10.0

# Check cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Load dataset
df_train = pd.read_csv(TRAIN_CSV)
df_val   = pd.read_csv(VAL_CSV)
df_test  = pd.read_csv(TEST_CSV)

# Determine feature columns
reserved_cols = {'age', 'filepath', 'subject_id'}
feature_cols = [c for c in df_train.columns if c not in reserved_cols]

# Extract arrays
X_train_raw = df_train[feature_cols].astype(float).values
y_train = df_train['age'].astype(float).values

X_val_raw = df_val[feature_cols].astype(float).values
y_val = df_val['age'].astype(float).values

X_test_raw = df_test[feature_cols].astype(float).values
y_test = df_test['age'].astype(float).values

# Optional repository transform used previously: sign_log1p
# def sign_log1p_np(X):
#     return np.sign(X) * np.log1p(np.abs(X))
# X_train_trans = sign_log1p_np(X_train_raw)
# X_val_trans   = sign_log1p_np(X_val_raw)
# X_test_trans  = sign_log1p_np(X_test_raw)

# Scale features (fit only on train to prevent leakage)
scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

# Save scaler
scaler_path = os.path.join(OUT_DIR, "feature_scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump({'mean': scaler.mean_, 'scale': scaler.scale_, 'feature_names': feature_cols}, f)

# Create dataloaders
Xtr_t = torch.tensor(X_train, dtype=torch.float32)
ytr_t = torch.tensor(y_train, dtype=torch.float32)
Xva_t = torch.tensor(X_val, dtype=torch.float32)
yva_t = torch.tensor(y_val, dtype=torch.float32)

train_dataset = TensorDataset(Xtr_t, ytr_t)
val_dataset = TensorDataset(Xva_t, yva_t)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    drop_last=True,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=len(val_dataset),
    shuffle=False
)

# Instantiate model and optimizer
model = LinearBA(len(feature_cols))
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)


# Training loop
# (paper loss J = MSE + LAMBDA * |Cov(CA, BA - CA)|)
train_loss_history, val_loss_history = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    n_batches = 0
    
    for Xb, yb in train_loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        BA_pred = model(Xb).squeeze(-1)     # predicted brain age
        err = BA_pred - yb                  # BAI
        mse = torch.mean(err ** 2)
        
        # Cov(CA, BAI) over the batch: mean((CA - mean(CA))*(BAI - mean(BAI)))
        ca_mean = torch.mean(yb)
        bai_mean = torch.mean(err)
        cov = torch.mean((yb - ca_mean) * (err - bai_mean))
        
        loss = mse + LAMBDA * torch.abs(cov)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().cpu().numpy())
        n_batches += 1

    epoch_train_loss = running_loss / max(1, n_batches)
    train_loss_history.append(epoch_train_loss)

    # validation
    model.eval()
    with torch.no_grad():
        for Xv, yv in val_loader:
            Xv = Xv.to(device); yv = yv.to(device)
            BA_v = model(Xv).squeeze(-1)
            err_v = BA_v - yv
            mse_v = torch.mean(err_v ** 2)
            ca_mean_v = torch.mean(yv)
            bai_mean_v = torch.mean(err_v)
            cov_v = torch.mean((yv - ca_mean_v) * (err_v - bai_mean_v))
            val_loss = float((mse_v + LAMBDA * torch.abs(cov_v)).cpu().numpy())
            val_loss_history.append(val_loss)

    print(f"Epoch {epoch}/{EPOCHS}\ntrain_loss: {epoch_train_loss:.6f}\nval_loss: {val_loss:.6f}")


# plot loss
loss_plot = os.path.join(OUT_DIR, "loss.png")
plt.figure(figsize=(6, 4))
plt.plot(train_loss_history, label="train loss")
plt.plot(val_loss_history,   label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE + lambda * |Cov|)")
plt.legend()
plt.tight_layout()
plt.savefig(loss_plot)
plt.close()

# Save trained model weights and metadata
model_path = os.path.join(OUT_DIR, "model.pth")
torch.save({'state_dict': model.state_dict(), 'feature_names': feature_cols, 'lambda': LAMBDA}, model_path)


# Evaluate on all train/val/test sets (for debugging purposes)
model.eval()
with torch.no_grad():
    Xtr_full = torch.tensor(X_train, dtype=torch.float32).to(device)
    Xva_full = torch.tensor(X_val, dtype=torch.float32).to(device)
    Xte_full = torch.tensor(X_test, dtype=torch.float32).to(device)

    BA_tr = model(Xtr_full).cpu().numpy().squeeze()
    BA_va = model(Xva_full).cpu().numpy().squeeze()
    BA_te = model(Xte_full).cpu().numpy().squeeze()

# Create prediction DataFrames
preds_train = pd.DataFrame({'age': y_train, 'BA': BA_tr, 'BAI': BA_tr - y_train})
preds_val   = pd.DataFrame({'age': y_val,   'BA': BA_va, 'BAI': BA_va - y_val})
preds_test  = pd.DataFrame({'age': y_test,  'BA': BA_te, 'BAI': BA_te - y_test})


# Bias Correction
# Compute 5-year sliding bias table for test set
starts = np.arange(15, 76, 5)
rows = []
for s in starts:
    mask = (y_test >= s) & (y_test <= s + 5)
    if np.sum(mask) == 0:
        bias_val = np.nan
    else:
        bias_val = np.mean(y_test[mask] - BA_te[mask])
    rows.append({'CA_min': int(s), 'CA_max': int(s + 5), 'bias': bias_val})
bias_df = pd.DataFrame(rows)

# Apply bias correction to test set
bias_for_samples = np.full_like(y_test, fill_value=np.nan, dtype=float)
for _, row in bias_df.iterrows():
    mask = (y_test >= row['CA_min']) & (y_test <= row['CA_max'])
    bias_for_samples[mask] = row['bias']
bias_for_samples[np.isnan(bias_for_samples)] = 0.0

BA_te_corrected = BA_te + bias_for_samples
BAI_te_corrected = BA_te_corrected - y_test


# Performance metrics
def safe_metrics(df, pred_col='BA', true_col='age'):
    ae = np.mean(np.abs(df[true_col] - df[pred_col]))
    mse = np.mean((df[true_col] - df[pred_col])**2)
    return ae, np.sqrt(mse)

mae_tr, rmse_tr = safe_metrics(preds_train)
mae_va, rmse_va = safe_metrics(preds_val)
mae_te_before, rmse_te_before = safe_metrics(preds_test)
mae_te_after, rmse_te_after = safe_metrics(
    pd.DataFrame({'age': y_test, 'BA_corrected': BA_te_corrected}), 
    pred_col='BA_corrected'
)

metrics = {
    'mae_train': mae_tr, 'rmse_train': rmse_tr,
    'mae_val': mae_va,   'rmse_val': rmse_va,
    'mae_test_before_correction': mae_te_before, 'rmse_test_before_correction': rmse_te_before,
    'mae_test_after_correction': mae_te_after, 'rmse_test_after_correction': rmse_te_after,
    'lambda': LAMBDA, 'epochs': EPOCHS
}
metrics_df = pd.DataFrame([metrics])

metrics_path = os.path.join(OUT_DIR, "metrics_summary.csv")
metrics_df.to_csv(metrics_path, index=False)

print("\n[Performance Summary]")
print(metrics_df.T.to_string(header=False))
