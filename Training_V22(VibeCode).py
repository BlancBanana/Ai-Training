import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
import os
import joblib
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import warnings
import gc
import time
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# "V17's 3-Class Target + V21's Multi-Timeframe Features"
# No WF-CV, No Simulation → Just Pure Training!
# ═══════════════════════════════════════════════════════════════════

class Config:
    """V22 Configuration - Optimized for XAUUSD Multi-Timeframe"""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DATA PATH - Change this to your parquet file location
    DATA_PATH = "XAUUSD_Merged_Final.parquet"  # Same directory as script
    
    # MODEL ARCHITECTURE (Reduced to prevent overfitting)
    D_MODEL = 256              # ↓ Reduced from 384
    CNN_FILTERS = 128          # ↓ Reduced from 192
    KERNEL_SIZE = 5        
    NHEAD = 8
    NUM_LAYERS = 3             # ↓ Reduced from 4
    NUM_CLASSES = 3  # Hold(0), Buy(1), Sell(2)
    
    # TRAINING (Anti-overfitting settings)
    BATCH_SIZE = 1024
    GRAD_ACCUM_STEPS = 4
    EPOCHS = 200
    LEARNING_RATE = 5e-5       # ↓ Reduced from 1e-4
    WEIGHT_DECAY = 1e-3        # ↑ Increased 10x for regularization
    GRAD_CLIP = 1.0
    EARLY_STOP_PATIENCE = 20   # ↓ Stop earlier if overfitting
    DROPOUT = 0.4              # ↑ Increased for regularization
    LABEL_SMOOTHING = 0.15     # ↑ More smoothing
    WARMUP_EPOCHS = 10
    MAX_CLASS_WEIGHT = 3.0     # Cap class weights to prevent over-focus
    
    # DATA (Optimized)
    SEQ_LEN = 90               # ↓ Reduced (7.5 hours on M5)
    LOOKAHEAD = 12
    TRAIN_SPLIT = 0.90         # ↓ More validation data (10%)
    EMBARGO_SIZE = 25
    
    # TARGET
    ATR_THRESHOLD = 1.2
    
    MODEL_PATH = "v22.pth"
    SCALER_PATH = "v22_scaler.gz"

# ═══════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

class MultiTimeframeFeatureEngineering:
    @staticmethod
    def process_data(df: pd.DataFrame) -> pd.DataFrame:
        print("🔧 Engineering Multi-Timeframe Features...")
        data = df.copy()
        
        # Set index
        data['datetime'] = pd.to_datetime(data['5m_time'])
        data = data.set_index('datetime').sort_index()
        
        # Extract OHLCV
        m5_open = data['5m_open']
        m5_close = data['5m_close']
        m5_high = data['5m_high']
        m5_low = data['5m_low']
        m5_volume = data['5m_volume']
        h1_close = data['1h_close'].ffill()
        h4_close = data['4h_close'].ffill()
        
        # ═══════════════════════════════════════════════════════════
        # M5 FEATURES
        # ═══════════════════════════════════════════════════════════
        
        data['m5_log_ret'] = np.log(m5_close / m5_close.shift(1))
        data['m5_log_ret_sq'] = data['m5_log_ret'] ** 2
        
        # ATR
        tr = pd.DataFrame({
            'hl': m5_high - m5_low,
            'hc': abs(m5_high - m5_close.shift(1)),
            'lc': abs(m5_low - m5_close.shift(1))
        }).max(axis=1)
        data['m5_atr'] = tr.rolling(14).mean()
        data['m5_atr_pct'] = data['m5_atr'] / m5_close
        
        # Price Action
        data['m5_body'] = m5_close - m5_open
        data['m5_body_pct'] = data['m5_body'] / m5_close
        data['m5_range'] = m5_high - m5_low
        data['m5_range_pct'] = data['m5_range'] / m5_close
        
        # Moving Averages
        for window in [10, 20, 50, 100]:
            ma = m5_close.rolling(window).mean()
            data[f'm5_ma{window}'] = (m5_close - ma) / ma
            data[f'm5_ma{window}_slope'] = ma.diff(5) / ma
        
        # EMA Cross
        ema12 = m5_close.ewm(span=12).mean()
        ema26 = m5_close.ewm(span=26).mean()
        data['m5_ema_cross'] = (ema12 - ema26) / m5_close
        
        # RSI
        delta = m5_close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        data['m5_rsi'] = 100 - (100 / (1 + rs))
        data['m5_rsi_norm'] = (data['m5_rsi'] - 50) / 50
        data['m5_rsi_slope'] = data['m5_rsi'].diff(3)
        
        # MACD
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        data['m5_macd'] = macd / m5_close
        data['m5_macd_signal'] = signal / m5_close
        data['m5_macd_hist'] = (macd - signal) / m5_close
        
        # Bollinger Bands
        bb_ma = m5_close.rolling(20).mean()
        bb_std = m5_close.rolling(20).std()
        data['m5_bb_upper'] = (bb_ma + 2*bb_std - m5_close) / m5_close
        data['m5_bb_lower'] = (m5_close - (bb_ma - 2*bb_std)) / m5_close
        data['m5_bb_width'] = (4 * bb_std) / bb_ma
        
        # Volume
        data['m5_volume_ma'] = (m5_volume - m5_volume.rolling(20).mean()) / (m5_volume.rolling(20).std() + 1e-8)
        data['m5_volume_slope'] = m5_volume.diff(3) / (m5_volume.shift(3) + 1e-8)
        
        # ═══════════════════════════════════════════════════════════
        # MULTI-TIMEFRAME (H1, H4)
        # ═══════════════════════════════════════════════════════════
        
        data['h1_m5_dist'] = (m5_close - h1_close) / h1_close
        h1_ma20 = h1_close.rolling(20).mean()
        data['h1_ma20_dist'] = (h1_close - h1_ma20) / h1_close
        
        h1_ema12 = h1_close.ewm(span=12).mean()
        h1_ema26 = h1_close.ewm(span=26).mean()
        data['h1_trend'] = (h1_ema12 - h1_ema26) / h1_close
        
        h1_delta = h1_close.diff()
        h1_gain = h1_delta.where(h1_delta > 0, 0).rolling(14).mean()
        h1_loss = -h1_delta.where(h1_delta < 0, 0).rolling(14).mean()
        h1_rs = h1_gain / (h1_loss + 1e-10)
        data['h1_rsi'] = 100 - (100 / (1 + h1_rs))
        data['h1_rsi_norm'] = (data['h1_rsi'] - 50) / 50
        
        data['h4_m5_dist'] = (m5_close - h4_close) / h4_close
        h4_ma20 = h4_close.rolling(20).mean()
        data['h4_ma20_dist'] = (h4_close - h4_ma20) / h4_close
        
        h4_ema12 = h4_close.ewm(span=12).mean()
        h4_ema26 = h4_close.ewm(span=26).mean()
        data['h4_trend'] = (h4_ema12 - h4_ema26) / h4_close
        
        data['mtf_alignment'] = (
            np.sign(data['m5_ema_cross']) * 0.4 +
            np.sign(data['h1_trend']) * 0.3 +
            np.sign(data['h4_trend']) * 0.3
        )
        
        # ═══════════════════════════════════════════════════════════
        # ADDITIONAL FEATURES
        # ═══════════════════════════════════════════════════════════
        
        atr_ma50 = data['m5_atr'].rolling(50).mean()
        data['volatility_regime'] = (data['m5_atr'] - atr_ma50) / (atr_ma50 + 1e-10)
        
        high_20 = m5_high.rolling(20).max()
        low_20 = m5_low.rolling(20).min()
        data['price_position'] = (m5_close - low_20) / (high_20 - low_20 + 1e-10)
        
        data['momentum_quality'] = abs(data['m5_ema_cross']) / (data['m5_atr_pct'] + 1e-10)
        
        vol_ma20 = m5_volume.rolling(20).mean()
        vol_std20 = m5_volume.rolling(20).std()
        data['volume_spike'] = (m5_volume - vol_ma20) / (vol_std20 + 1e-10)
        data['volume_trend'] = vol_ma20.pct_change(5)
        
        data['price_acceleration'] = data['m5_log_ret'].diff()
        
        data['upper_wick'] = (m5_high - np.maximum(m5_close, m5_open)) / (data['m5_range'] + 1e-10)
        data['lower_wick'] = (np.minimum(m5_close, m5_open) - m5_low) / (data['m5_range'] + 1e-10)
        
        # ═══════════════════════════════════════════════════════════
        # TIME FEATURES
        # ═══════════════════════════════════════════════════════════
        
        hour = data.index.hour
        day = data.index.dayofweek
        
        data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        data['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        data['day_sin'] = np.sin(2 * np.pi * day / 7)
        data['day_cos'] = np.cos(2 * np.pi * day / 7)
        
        data['asian_session'] = ((hour >= 0) & (hour < 8)).astype(int)
        data['london_session'] = ((hour >= 8) & (hour < 16)).astype(int)
        data['ny_session'] = ((hour >= 13) & (hour < 22)).astype(int)
        data['session_overlap'] = ((hour >= 13) & (hour < 16)).astype(int)
        
        # ═══════════════════════════════════════════════════════════
        # TARGET (3-CLASS WITH ATR)
        # ═══════════════════════════════════════════════════════════
        
        future_close = m5_close.shift(-Config.LOOKAHEAD)
        change = future_close - m5_close
        threshold = data['m5_atr'] * Config.ATR_THRESHOLD
        
        data['target'] = np.select(
            [(change > threshold), (change < -threshold)],
            [1, 2],
            default=0
        )
        
        # Clean up
        data = data.ffill().fillna(0)
        data = data[data['target'].notna()]
        data = data.iloc[200:]  # Warmup
        
        print(f"✅ Feature Engineering Complete")
        print(f"   Shape: {data.shape}")
        print(f"   Date Range: {data.index.min()} to {data.index.max()}")
        print(f"   Target: {dict(data['target'].value_counts().sort_index())}")
        
        return data

# ═══════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len-1]

# ═══════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════

class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ELU()
        self.gate = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.activation(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        gate = torch.sigmoid(self.gate(residual))
        return self.norm(residual + (x * gate))

class HybridImperialV22(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, Config.CNN_FILTERS, kernel_size=Config.KERNEL_SIZE, padding=2),
            nn.BatchNorm1d(Config.CNN_FILTERS),
            nn.ELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Conv1d(Config.CNN_FILTERS, Config.CNN_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm1d(Config.CNN_FILTERS),
            nn.ELU(),
            nn.Dropout(Config.DROPOUT)
        )
        
        self.projection = nn.Linear(Config.CNN_FILTERS, Config.D_MODEL)
        self.grn = GRN(Config.D_MODEL, Config.D_MODEL * 2, Config.DROPOUT)
        self.pos_embedding = nn.Parameter(torch.randn(1, 5000, Config.D_MODEL) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.D_MODEL, nhead=Config.NHEAD,
            dim_feedforward=Config.D_MODEL * 4, dropout=Config.DROPOUT,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.NUM_LAYERS)
        
        self.pooling = nn.Linear(Config.D_MODEL, 1)
        
        self.head = nn.Sequential(
            nn.LayerNorm(Config.D_MODEL),
            nn.Linear(Config.D_MODEL, Config.D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.D_MODEL // 2, Config.NUM_CLASSES)
        )

    def forward(self, x):
        x = torch.clamp(x, -10, 10)
        x = self.input_norm(x)
        
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        
        x = self.projection(x)
        x = self.grn(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        
        attn = F.softmax(self.pooling(x), dim=1)
        x = torch.sum(x * attn, dim=1)
        
        return self.head(x)

# ═══════════════════════════════════════════════════════════════════
# FOCAL LOSS
# ═══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        if self.label_smoothing > 0:
            targets_smooth = (1.0 - self.label_smoothing) * targets_one_hot + (self.label_smoothing / num_classes)
        else:
            targets_smooth = targets_one_hot
        
        pt = (probs * targets_one_hot).sum(1) + 1e-10
        focal = (1 - pt) ** self.gamma
        ce = -torch.sum(targets_smooth * log_probs, dim=1)
        loss = focal * ce
        
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
            
        return loss.mean()

# ═══════════════════════════════════════════════════════════════════
# MAIN - LEAN TRAINING ONLY
# ═══════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print(" LEAN TRAINING MODE")
    print(f"  Device: {Config.DEVICE}")
    print("="*70)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # ═══════════════════════════════════════════════════════════════
    # 1️ LOAD & PROCESS DATA
    # ═══════════════════════════════════════════════════════════════
    
    print("\n Loading data...")
    df = pd.read_parquet(Config.DATA_PATH)
    print(f"   Raw: {df.shape}")
    
    df = MultiTimeframeFeatureEngineering.process_data(df)
    
    # Select features
    exclude = [
        'target', '1m_time', '5m_time', '1h_time', '4h_time',
        '1m_open', '1m_high', '1m_low', '1m_close', '1m_volume', '1m_spread', '1m_real_volume',
        '5m_open', '5m_high', '5m_low', '5m_close', '5m_volume', '5m_spread', '5m_real_volume',
        '1h_open', '1h_high', '1h_low', '1h_close', '1h_volume', '1h_spread', '1h_real_volume',
        '4h_open', '4h_high', '4h_low', '4h_close', '4h_volume', '4h_spread', '4h_real_volume'
    ]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    print(f"\n🧠 Features: {len(feature_cols)}")
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.longlong)
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 Classes: {dict(zip(unique, counts))}")
    
    # ═══════════════════════════════════════════════════════════════
    # TRAIN/VAL SPLIT (with Embargo Gap)
    # ═══════════════════════════════════════════════════════════════
    
    split_idx = int(len(X) * Config.TRAIN_SPLIT)
    val_start_idx = split_idx + Config.EMBARGO_SIZE  # Skip embargo gap
    
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X[:split_idx])
    X_val = scaler.transform(X[val_start_idx:])
    
    y_train = y[:split_idx]
    y_val = y[val_start_idx:]
    
    joblib.dump(scaler, Config.SCALER_PATH)
    print(f"\n✅ Scaler saved: {Config.SCALER_PATH}")
    print(f"   Train: {len(X_train):,} | Embargo: {Config.EMBARGO_SIZE} | Val: {len(X_val):,}")
    
    train_ds = SequenceDataset(X_train, y_train, Config.SEQ_LEN)
    val_ds = SequenceDataset(X_val, y_val, Config.SEQ_LEN)
    
    train_loader = TorchDataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = TorchDataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    # ═══════════════════════════════════════════════════════════════
    # MODEL & TRAINING
    # ═══════════════════════════════════════════════════════════════
    
    model = HybridImperialV22(len(feature_cols)).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # LR Scheduler with Warmup
    def lr_lambda(epoch):
        if epoch < Config.WARMUP_EPOCHS:
            return (epoch + 1) / Config.WARMUP_EPOCHS  # Linear warmup
        else:
            # Cosine decay after warmup
            progress = (epoch - Config.WARMUP_EPOCHS) / (Config.EPOCHS - Config.WARMUP_EPOCHS)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"   LR Scheduler: Warmup {Config.WARMUP_EPOCHS} epochs + Cosine Decay")
    
    # Class weights (capped to prevent over-focus on minority classes)
    train_counts = np.bincount(y_train, minlength=3)
    raw_weight_buy = train_counts[0] / (train_counts[1] + 1)
    raw_weight_sell = train_counts[0] / (train_counts[2] + 1)
    
    # Cap weights to MAX_CLASS_WEIGHT
    capped_weight_buy = min(raw_weight_buy, Config.MAX_CLASS_WEIGHT)
    capped_weight_sell = min(raw_weight_sell, Config.MAX_CLASS_WEIGHT)
    
    weights = torch.tensor([
        1.0,
        capped_weight_buy,
        capped_weight_sell
    ], dtype=torch.float32).to(Config.DEVICE)
    print(f"   Weights: H={weights[0]:.1f}, B={capped_weight_buy:.2f} (raw:{raw_weight_buy:.1f}), S={capped_weight_sell:.2f} (raw:{raw_weight_sell:.1f})")
    
    criterion = FocalLoss(alpha=weights, gamma=2.0, label_smoothing=Config.LABEL_SMOOTHING)
    scaler_amp = GradScaler()
    
    best_action_f1 = 0.0
    patience = 0
    best_epoch = 0
    training_history = []  # Collect training logs
    
    print("\n" + "="*70)
    print(" TRAINING")
    print("="*70 + "\n")
    
    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()  # Timing
        
        # Train
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (x_b, y_b) in enumerate(train_loader):
            x_b, y_b = x_b.to(Config.DEVICE), y_b.to(Config.DEVICE)
            
            with autocast():
                out = model(x_b)
                loss = criterion(out, y_b) / Config.GRAD_ACCUM_STEPS
            
            scaler_amp.scale(loss).backward()
            
            if (batch_idx + 1) % Config.GRAD_ACCUM_STEPS == 0:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * Config.GRAD_ACCUM_STEPS
        
        # Validate
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(Config.DEVICE), y_b.to(Config.DEVICE)
                with autocast():
                    out = model(x_b)
                    val_loss += criterion(out, y_b).item()
                preds = torch.argmax(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_b.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        acc = (all_preds == all_targets).mean() * 100
        precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, labels=[0,1,2], zero_division=0)
        action_f1 = (f1[1] + f1[2]) / 2 * 100
        
        # Timing & Memory
        epoch_time = time.time() - epoch_start
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
            torch.cuda.reset_peak_memory_stats()
        else:
            gpu_mem = 0
        
        is_best = action_f1 > best_action_f1
        mark = " if is_best else ""
        current_lr = scheduler.get_last_lr()[0]
        
        # Main metrics line
        print(f"Ep {epoch+1:3d} | LR: {current_lr:.1e} | Loss: {train_loss/len(train_loader):.4f}/{val_loss/len(val_loader):.4f} | "
              f"Acc: {acc:.1f}% | ActF1: {action_f1:.1f}% | Time: {epoch_time:.1f}s | GPU: {gpu_mem:.1f}GB {mark}")
        
        # Detailed metrics every 5 epochs or when best
        if (epoch + 1) % 5 == 0 or is_best:
            print(f"    F1  : Hold={f1[0]:.3f} | Buy={f1[1]:.3f}  | Sell={f1[2]:.3f}")
            print(f"    Prec: Hold={precision[0]:.3f} | Buy={precision[1]:.3f}  | Sell={precision[2]:.3f}")
            print(f"    Rec : Hold={recall[0]:.3f} | Buy={recall[1]:.3f}  | Sell={recall[2]:.3f}")
        
        # Log training history
        training_history.append({
            'epoch': epoch + 1,
            'lr': current_lr,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'accuracy': acc,
            'action_f1': action_f1,
            'f1_hold': f1[0],
            'f1_buy': f1[1],
            'f1_sell': f1[2],
            'prec_hold': precision[0],
            'prec_buy': precision[1],
            'prec_sell': precision[2],
            'rec_hold': recall[0],
            'rec_buy': recall[1],
            'rec_sell': recall[2],
            'epoch_time': epoch_time,
            'gpu_mem_gb': gpu_mem
        })
        
        if is_best:
            best_action_f1 = action_f1
            best_epoch = epoch + 1  # Track best epoch
            patience = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'action_f1': best_action_f1,
                'features': feature_cols
            }, Config.MODEL_PATH)
            print(f"   Best model saved!")
        else:
            patience += 1
        
        # Save checkpoint every epoch
        os.makedirs('checkpoints', exist_ok=True)
        ckpt_path = f"checkpoints/epoch_{epoch+1:03d}_f1_{action_f1:.1f}.pth"
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'action_f1': action_f1,
            'val_loss': val_loss / len(val_loader),
            'features': feature_cols
        }, ckpt_path)
        
        # Step scheduler
        scheduler.step()
        
        if patience >= Config.EARLY_STOP_PATIENCE:
            print(f"\n Early stop at epoch {epoch+1}")
            break
        
        # Confusion matrix every 20 epochs
        if (epoch + 1) % 20 == 0:
            cm = confusion_matrix(all_targets, all_preds, labels=[0,1,2])
            print(f"   CM: Hold[{cm[0,0]:5d},{cm[0,1]:4d},{cm[0,2]:4d}] Buy[{cm[1,0]:5d},{cm[1,1]:4d},{cm[1,2]:4d}] Sell[{cm[2,0]:5d},{cm[2,1]:4d},{cm[2,2]:4d}]")
    
    # Save training log
    log_df = pd.DataFrame(training_history)
    log_path = 'training_log.csv'
    log_df.to_csv(log_path, index=False)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    epochs_range = log_df['epoch'].values
    
    # Plot 1: Action F1
    axes[0].plot(epochs_range, log_df['action_f1'], 'b-', linewidth=2, label='Action F1')
    axes[0].axhline(y=best_action_f1, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_action_f1:.2f}%')
    axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Action F1 (%)')
    axes[0].set_title('Action F1 Score')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[1].plot(epochs_range, log_df['accuracy'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Validation Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Loss
    axes[2].plot(epochs_range, log_df['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[2].plot(epochs_range, log_df['val_loss'], 'r-', linewidth=2, label='Val Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training & Validation Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = 'training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves: {plot_path}")
    
    print(f"\n" + "="*70)
    print(f" TRAINING COMPLETE")
    print(f"="*70)
    print(f"   Best Action F1: {best_action_f1:.2f}%")
    print(f"   Best Epoch: {best_epoch}")
    print(f"   Model saved: {Config.MODEL_PATH}")
    print(f"   Training log: {log_path}")
    print(f"   Training curves: {plot_path}")
    print(f"="*70)

if __name__ == "__main__":
    main()

