from __future__ import annotations
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from io_utils import set_global_seed

logger = logging.getLogger("eap_ml.models_lstm")


class PanelSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_col="excess_ret_lead", seq_len=12):
        self.samples = []
        for permno, g in df.sort_values(["permno", "date"]).groupby("permno"):
            g = g.reset_index(drop=True)
            X = g[feature_cols].to_numpy(dtype=np.float32)
            y = g[target_col].to_numpy(dtype=np.float32)
            dates = g["date"].to_numpy()
            for t in range(seq_len - 1, len(g)):
                x_seq = X[t-seq_len+1:t+1]
                y_t = y[t]
                self.samples.append((x_seq, y_t, permno, dates[t]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, permno, dt = self.samples[idx]
        return torch.tensor(x), torch.tensor(y), permno, dt


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.head(h).squeeze(-1)


def predictive_r2_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    num = np.sum((y_true - y_pred) ** 2)
    return 1.0 - num / denom if denom > 0 else np.nan


def train_lstm_model(train_df, val_df, feature_cols, seq_len=12, hidden_dim=64, num_layers=1, dropout=0.0, lr=1e-3, batch_size=1024, epochs=20, device=None, random_state=42):
    logger.info(f"Training LSTM: seq_len={seq_len}, hidden={hidden_dim}, layers={num_layers}")
    set_global_seed(random_state=random_state, torch_deterministic=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = PanelSequenceDataset(train_df, feature_cols, seq_len=seq_len)
    val_ds = PanelSequenceDataset(val_df, feature_cols, seq_len=seq_len)
    gen = torch.Generator()
    gen.manual_seed(random_state)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = LSTMRegressor(input_dim=len(feature_cols), hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val_r2 = -np.inf
    for epoch in range(epochs):
        model.train()
        for Xb, yb, _, _ in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for Xb, yb, _, _ in val_loader:
                Xb = Xb.to(device)
                pred = model(Xb).cpu().numpy()
                preds.append(pred)
                trues.append(yb.numpy())
        y_val = np.concatenate(trues) if trues else np.array([])
        p_val = np.concatenate(preds) if preds else np.array([])
        val_r2 = predictive_r2_np(y_val, p_val) if len(y_val) else -np.inf
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"LSTM training completed, best val R2: {best_val_r2:.4f}")
    return {"model": model, "val_r2": best_val_r2}


def tune_lstm_models(train_df, val_df, feature_cols, param_grid, random_state=42):
    logger.info("Tuning LSTM models")
    best = {"model": None, "val_r2": -np.inf, "params": None}
    for seq_len in param_grid["seq_len"]:
        for hidden_dim in param_grid["hidden_dim"]:
            for num_layers in param_grid["num_layers"]:
                for dropout in param_grid["dropout"]:
                    for lr in param_grid["lr"]:
                        res = train_lstm_model(train_df=train_df, val_df=val_df, feature_cols=feature_cols, seq_len=seq_len, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, lr=lr, batch_size=param_grid.get("batch_size", 1024), epochs=param_grid.get("epochs", 20), random_state=random_state)
                        if res["val_r2"] > best["val_r2"]:
                            best = {"model": res["model"], "val_r2": res["val_r2"], "params": {"seq_len": seq_len, "hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout, "lr": lr}}
    logger.info(f"Best LSTM val R2: {best['val_r2']:.4f}, params: {best['params']}")
    return best