"""
BFEEMD-LSTM-TWSVR-SOA Pipeline (prototype)

Notes:
- This is a single-file prototype implementing the pipeline described in your report.
- Requires: numpy, pandas, scikit-learn, tensorflow (or keras), pyemd (PyEMD), joblib
- TWSVR: placeholder uses sklearn.svm.SVR. Replace `train_twsvr` with a TWSVR implementation if available.
- SOA: simple Seeker-like optimizer implemented here (population + directed search). It's lightweight and intended as a template.
- Dataset: expects GEFCom2014 CSV(s) downloaded locally. Update `DATA_PATH` accordingly.

Run: python BFEEMD_LSTM_TWSVR_SOA_pipeline.py

"""

import os
import math
import json
import random
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from PyEMD import EEMD
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from joblib import dump, load

# -------------------------------
# Config
# -------------------------------
DATA_PATH = './gefcom2014_load.csv'  # change to your downloaded dataset path
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# LSTM / window config (can be tuned via SOA)
DEFAULT_CONFIG = {
    'window_len': 168,    # lookback in hours
    'horizon': 24,        # forecasting horizon
    'lstm_units': 128,
    'lstm_layers': 1,
    'dropout': 0.2,
    'batch_size': 64,
    'epochs': 50,
    'lr': 1e-3,
}

# -------------------------------
# Utility & metrics
# -------------------------------

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(y_true == 0, 1e-6, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

# Winkler score for central (1-alpha) interval
def winkler_score(y, lower, upper, alpha=0.05):
    y = np.array(y)
    lower = np.array(lower)
    upper = np.array(upper)
    score = np.zeros_like(y, dtype=float)
    width = upper - lower
    inside = (y >= lower) & (y <= upper)
    score[inside] = width[inside]
    below = y < lower
    above = y > upper
    score[below] = width[below] + 2.0/alpha * (lower[below] - y[below])
    score[above] = width[above] + 2.0/alpha * (y[above] - upper[above])
    return np.mean(score)

# Prediction Interval Coverage Probability
def picp(y, lower, upper):
    y = np.array(y)
    return np.mean((y >= np.array(lower)) & (y <= np.array(upper)))

# Absolute coverage error (ACE) = |PICP - nominal| (nominal e.g., 0.95)
def ace(y, lower, upper, nominal=0.95):
    return abs(picp(y, lower, upper) - nominal)

# -------------------------------
# Data loading & preprocessing
# -------------------------------

def load_gefcom(csv_path=DATA_PATH):
    # Adjust this loader depending on your CSV structure
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')
    # expect columns: ['load','temp'] or similar
    # rename if needed
    if 'load' not in df.columns:
        # attempt to infer
        for col in df.columns:
            if 'load' in col.lower():
                df = df.rename(columns={col: 'load'})
                break
    return df


def create_time_features(df):
    df = df.copy()
    idx = df.index
    df['hour'] = idx.hour
    df['dow'] = idx.dayofweek
    df['month'] = idx.month
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    return df

# -------------------------------
# FEEMD/EEMD decomposition
# -------------------------------

def eemd_decompose(series, ensembles=100, noise_width=0.2):
    eemd = EEMD()
    eemd.noise_width = noise_width
    eemd.trials = ensembles
    imfs = eemd.eemd(series)
    # imfs.shape -> (n_imfs, n_samples)
    return imfs

# -------------------------------
# Sliding windows & dataset creation
# -------------------------------

def make_windows(df, window_len, horizon, features_cols=['load'], exog_cols=None):
    # features_cols: list of channels to decompose/stack (e.g., IMFs will be passed)
    X, X_exog, Y = [], [], []
    n = len(df)
    for i in range(n - window_len - horizon + 1):
        X.append(df.iloc[i:i+window_len][features_cols].values)
        if exog_cols:
            # take exog at prediction time(s) or aggregated window -- here we append last timestamp exog
            X_exog.append(df.iloc[i+window_len][exog_cols].values)
        Y.append(df.iloc[i+window_len:i+window_len+horizon]['load'].values)
    X = np.array(X)
    X_exog = np.array(X_exog) if len(X_exog) > 0 else None
    Y = np.array(Y)
    return X, X_exog, Y

# -------------------------------
# LSTM encoder & feature extraction
# -------------------------------

def build_lstm_encoder(input_shape, config):
    model = Sequential()
    for layer in range(config['lstm_layers']):
        return_seq = (layer < config['lstm_layers'] - 1)
        if layer == 0:
            model.add(LSTM(config['lstm_units'], return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(LSTM(config['lstm_units'], return_sequences=return_seq))
        if config['dropout'] > 0:
            model.add(Dropout(config['dropout']))
    model.add(Dense(64, activation='linear', name='feature_dense'))
    return model


def train_lstm_encoder(X_train, Y_train_dummy, X_val, Y_val_dummy, config, model_path):
    tf.keras.backend.clear_session()
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_encoder(input_shape, config)
    opt = tf.keras.optimizers.Adam(learning_rate=config['lr'])
    model.compile(optimizer=opt, loss='mse')
    callbacks = [EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                 ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)]
    model.fit(X_train, Y_train_dummy, validation_data=(X_val, Y_val_dummy),
              epochs=config['epochs'], batch_size=config['batch_size'], callbacks=callbacks, verbose=2)
    model.load_weights(model_path)
    return model


def extract_features_from_encoder(model, X):
    # the Dense('feature_dense') output will be used; create a submodel
    sub = tf.keras.Model(inputs=model.input, outputs=model.get_layer('feature_dense').output)
    return sub.predict(X)

# -------------------------------
# TWSVR (placeholder with sklearn SVR)
# -------------------------------

def train_twsvr(X, y, C=1.0, epsilon=0.1, gamma='scale', kernel='rbf'):
    svr = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)
    svr.fit(X, y)
    return svr

# -------------------------------
# Bootstrap methods for PI
# -------------------------------

def wild_bootstrap_intervals(point_preds, residuals, B=1000, alpha=0.05):
    # residuals: 1D array from val set; point_preds: (n_samples,)
    n = len(point_preds)
    boot_preds = np.zeros((B, n))
    # use Rademacher multipliers
    for b in range(B):
        signs = np.random.choice([-1, 1], size=len(residuals))
        rstar = residuals * signs
        # sample residuals with replacement to length n
        r_choice = np.random.choice(rstar, size=n, replace=True)
        boot_preds[b] = point_preds + r_choice
    lower = np.percentile(boot_preds, 100*alpha/2.0, axis=0)
    upper = np.percentile(boot_preds, 100*(1-alpha/2.0), axis=0)
    return lower, upper


def block_bootstrap_intervals(point_preds, residuals, B=1000, block_len=24, alpha=0.05):
    n = len(point_preds)
    # create residual blocks
    blocks = []
    for i in range(0, len(residuals)-block_len+1, block_len):
        blocks.append(residuals[i:i+block_len])
    if len(blocks) == 0:
        # fallback to simple bootstrap
        return wild_bootstrap_intervals(point_preds, residuals, B=B, alpha=alpha)
    boot_preds = np.zeros((B, n))
    for b in range(B):
        resampled = []
        while len(resampled) < n:
            blk = random.choice(blocks)
            resampled.extend(blk.tolist())
        resampled = np.array(resampled[:n])
        boot_preds[b] = point_preds + resampled
    lower = np.percentile(boot_preds, 100*alpha/2.0, axis=0)
    upper = np.percentile(boot_preds, 100*(1-alpha/2.0), axis=0)
    return lower, upper

# -------------------------------
# Simple SOA (Seeker-like) optimizer (lightweight)
# -------------------------------
class SimpleSOA:
    def __init__(self, search_space, pop_size=20, iters=30):
        self.search_space = search_space
        self.pop_size = pop_size
        self.iters = iters
        self.population = []

    def sample_one(self):
        cand = {}
        for k,v in self.search_space.items():
            if v['type']=='int':
                cand[k] = random.randint(v['low'], v['high'])
            elif v['type']=='float':
                if v.get('scale','linear')=='log':
                    cand[k] = 10**np.random.uniform(np.log10(v['low']), np.log10(v['high']))
                else:
                    cand[k] = np.random.uniform(v['low'], v['high'])
            elif v['type']=='cat':
                cand[k] = random.choice(v['choices'])
        return cand

    def init_population(self):
        self.population = [self.sample_one() for _ in range(self.pop_size)]

    def run(self, evaluate_fn):
        # evaluate_fn: callable hyperparams -> score (lower better)
        self.init_population()
        scores = [evaluate_fn(p) for p in self.population]
        best_idx = int(np.argmin(scores))
        best = self.population[best_idx]; best_score = scores[best_idx]
        for it in range(self.iters):
            new_pop = []
            for i,p in enumerate(self.population):
                # local search: perturb continuous params slightly
                q = p.copy()
                for k,v in self.search_space.items():
                    if v['type'] in ['int','float']:
                        span = (v['high'] - v['low']) if v['type']=='float' else max(1, v['high']-v['low'])
                        if random.random() < 0.5:
                            if v['type']=='float':
                                q[k] = q[k] * (1 + np.random.normal(0, 0.1))
                                q[k] = max(v['low'], min(v['high'], q[k]))
                            else:
                                q[k] = q[k] + random.randint(-max(1,int(span*0.1)), max(1,int(span*0.1)))
                                q[k] = int(max(v['low'], min(v['high'], q[k])))
                        else:
                            # global jump
                            q[k] = self.sample_one()[k]
                new_pop.append(q)
            new_scores = [evaluate_fn(p) for p in new_pop]
            # replace if improved
            for i in range(self.pop_size):
                if new_scores[i] < scores[i]:
                    self.population[i] = new_pop[i]
                    scores[i] = new_scores[i]
            cur_best_idx = int(np.argmin(scores))
            if scores[cur_best_idx] < best_score:
                best_score = scores[cur_best_idx]
                best = self.population[cur_best_idx]
            print(f"SOA iter {it+1}/{self.iters} best_score={best_score:.4f}")
        return best, best_score

# -------------------------------
# Main pipeline (prototype run)
# -------------------------------

def main_prototype(config=DEFAULT_CONFIG):
    print('Loading data...')
    df = load_gefcom(DATA_PATH)
    df = create_time_features(df)
    # simple train/val/test split
    n = len(df)
    train_idx = int(0.7*n)
    val_idx = int(0.85*n)
    df_train = df.iloc[:train_idx]
    df_val = df.iloc[train_idx:val_idx]
    df_test = df.iloc[val_idx:]

    # decompose the load series (train only to avoid leakage)
    print('Decomposing using EEMD (this may take time)...')
    imfs_train = eemd_decompose(df_train['load'].values, ensembles=50, noise_width=0.1)
    imfs_full = eemd_decompose(df['load'].values, ensembles=50, noise_width=0.1)
    # stack first few IMFs as channels (or all)
    channels = imfs_full.shape[0]
    print(f'Found {channels} IMFs')
    # prepare a DataFrame of stacked IMF channels aligned with df.index
    imf_df = pd.DataFrame(imfs_full.T, index=df.index, columns=[f'imf_{i+1}' for i in range(channels)])
    df_imf = pd.concat([df, imf_df], axis=1)

    # window creation using IMF channels as inputs
    imf_cols = [c for c in df_imf.columns if c.startswith('imf_')]
    exog_cols = ['hour_sin','hour_cos','dow'] if 'dow' in df_imf.columns else ['hour_sin','hour_cos']

    X_train, Xex_train, Y_train = make_windows(df_imf.iloc[:train_idx+val_idx], config['window_len'], config['horizon'], features_cols=imf_cols, exog_cols=exog_cols)
    # We'll train encoder supervised with next-step load mean as dummy target (for representation)
    Y_train_dummy = Y_train.mean(axis=1)

    # split again into train/val for encoder
    split = int(0.85 * len(X_train))
    X_enc_train, X_enc_val = X_train[:split], X_train[split:]
    Y_enc_train, Y_enc_val = Y_train_dummy[:split], Y_train_dummy[split:]

    model_path = os.path.join(RESULTS_DIR, 'lstm_encoder.weights.h5')
    print('Training LSTM encoder...')
    encoder = train_lstm_encoder(X_enc_train, Y_enc_train, X_enc_val, Y_enc_val, config, model_path)

    print('Extracting features...')
    feat_all = extract_features_from_encoder(encoder, X_train)
    # fit scaler on features
    feat_scaler = StandardScaler().fit(feat_all)
    feat_all_s = feat_scaler.transform(feat_all)

    # target: use horizon=24 aggregated or specific horizon index 0 (1-hour ahead)
    y_point = Y_train[:,0]

    # train TWSVR (here SVR) on features
    print('Training SVR (as TWSVR placeholder)...')
    svr = train_twsvr(feat_all_s, y_point, C=1.0, epsilon=0.1, gamma='scale')

    # prepare test features
    # create windows for test set (use df_imf full)
    X_all, Xex_all, Y_all = make_windows(df_imf, config['window_len'], config['horizon'], features_cols=imf_cols, exog_cols=exog_cols)
    # determine slicing to get test windows
    test_start = len(df_imf) - len(df_test) - config['window_len'] - config['horizon'] + 1
    if test_start < 0:
        test_start = len(X_all) - len(df_test)
    X_test = X_all[test_start:]
    Y_test = Y_all[test_start:]
    feats_test = extract_features_from_encoder(encoder, X_test)
    feats_test_s = feat_scaler.transform(feats_test)

    y_pred = svr.predict(feats_test_s)
    y_true = Y_test[:,0]

    print('Computing residuals from validation for bootstrap')
    # compute residuals using last portion of training windows as val; simple approach using encoder val
    feats_val = feat_all_s[split:]
    y_val = y_point[split:]
    y_val_pred = svr.predict(feats_val)
    residuals = y_val - y_val_pred

    # bootstrap intervals
    print('Generating bootstrap intervals (wild + block)')
    lower_w, upper_w = wild_bootstrap_intervals(y_pred, residuals, B=500, alpha=0.05)
    lower_b, upper_b = block_bootstrap_intervals(y_pred, residuals, B=500, block_len=24, alpha=0.05)

    # evaluate
    print('Evaluating...')
    metrics = {}
    metrics['RMSE'] = rmse(y_true, y_pred)
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['MAPE'] = mape(y_true, y_pred)
    metrics['R2'] = r2_score(y_true, y_pred)
    metrics['Winkler_wild'] = winkler_score(y_true, lower_w, upper_w, alpha=0.05)
    metrics['PICP_wild'] = picp(y_true, lower_w, upper_w)
    metrics['ACE_wild'] = ace(y_true, lower_w, upper_w, nominal=0.95)
    metrics['Winkler_block'] = winkler_score(y_true, lower_b, upper_b, alpha=0.05)
    metrics['PICP_block'] = picp(y_true, lower_b, upper_b)
    metrics['ACE_block'] = ace(y_true, lower_b, upper_b, nominal=0.95)

    print(json.dumps(metrics, indent=2))
    dump(svr, os.path.join(RESULTS_DIR, 'svr_model.joblib'))
    feat_scaler_path = os.path.join(RESULTS_DIR, 'feat_scaler.joblib')
    dump(feat_scaler, feat_scaler_path)
    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'lower_w': lower_w, 'upper_w': upper_w}).to_csv(os.path.join(RESULTS_DIR, 'preds_wild.csv'), index=False)
    print('Done. Results saved in', RESULTS_DIR)


if __name__ == '__main__':
    main_prototype()
