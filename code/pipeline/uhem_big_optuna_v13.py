import sqlite3
import pandas as pd
import numpy as np
import gudhi
import gudhi.subsampling
import optuna
import os
import random
import warnings
import gc
import sys
import json
from datetime import datetime
from scipy.interpolate import interp1d
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, LeaveOneGroupOut, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (fbeta_score, accuracy_score, recall_score,
                             precision_score, f1_score, precision_recall_curve, make_scorer)
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import pdist

# =============================================================================
# 0. MOD SEÇİMİ
# =============================================================================
# "server"  → tam arama uzayı, 200 trial, çok çekirdekli
# "laptop"  → dar arama uzayı, 10 trial, tek çekirdek, birkaç dakikada biter
RUN_MODE = "server"   # ← buradan değiştir: "laptop" veya "server"

# =============================================================================
# 1. CONFIGURATION (V15)
# =============================================================================
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARN)

USER_DIR       = os.getcwd()
DATA_DB_PATH   = os.path.join(USER_DIR, "All_Datasets_Container.db")
RESULT_DB_PATH = os.path.join(USER_DIR, "Results_V15.db")
REPORT_FILE    = os.path.join(USER_DIR, "Results_V15_Report.txt")
OPTUNA_DB_PATH = os.path.join(USER_DIR, "study_v15.db")

if RUN_MODE == "laptop":
    CONFIG = {
        "TARGET_FS"      : 50,
        "LIMIT_POINTS"   : 80,
        "N_TRIALS_GLOBAL": 10,
        "OPTUNA_JOBS"    : 1,
        "MODEL_JOBS"     : 2,

        "SEARCH_SPACE": {
            "win_sec"        : {"low": 2.0, "high": 3.0, "step": 1.0},
            "dim"            : {"low": 2,   "high": 3},
            "delay"          : {"low": 1,   "high": 2},
            "stride_factor"  : [2],
            "complex_type"   : ["Rips"],
            "metrics"        : ["euclidean"],
            "eps_percentile" : [40, 60],
            "use_delay"      : [True],
            "sampling_method": ["maxmin"]
        },

        "MODELS": {
            "LogReg": {
                "estimator": LogisticRegression(class_weight='balanced',
                                                solver='liblinear', max_iter=500),
                "params": {'logreg__C': [0.1, 1], 'logreg__penalty': ['l2']},
                # Daraltılmış grid — Personal model için
                "params_personal": {'logreg__C': [0.1, 1], 'logreg__penalty': ['l2']}
            }
        }
    }
    DATASETS_RUN = ["MobiFall"]
    print("🖥️  LAPTOP MODU — hızlı deneme (dar arama uzayı, tek dataset, LogReg only)")

else:  # "server"
    CONFIG = {
        "TARGET_FS"      : 50,
        "LIMIT_POINTS"   : 150,
        "N_TRIALS_GLOBAL": 200,
        "OPTUNA_JOBS"    : 1,
        "MODEL_JOBS"     : 15,

        "SEARCH_SPACE": {
            "win_sec"        : {"low": 1.0, "high": 5.0, "step": 0.5},
            "dim"            : {"low": 2,   "high": 5},
            "delay"          : {"low": 1,   "high": 5},
            "stride_factor"  : [1, 2, 4],
            "complex_type"   : ["Alpha", "Rips", "SparseRips"],
            "metrics"        : ["euclidean", "cosine", "manhattan", "chebyshev"],
            "eps_percentile" : [20, 40, 60, 80],
            "use_delay"      : [True],
            "sampling_method": ["maxmin"]
        },

        "MODELS": {
            "LogReg": {
                "estimator": LogisticRegression(class_weight='balanced',
                                                solver='liblinear', max_iter=2000),
                # Global GridSearchCV için tam grid
                "params": {
                    'logreg__C'      : [0.01, 0.1, 1, 10, 100],
                    'logreg__penalty': ['l1', 'l2']
                },
                # Personal model için daraltılmış grid — az veriyle CV güvenilirliği
                "params_personal": {
                    'logreg__C'      : [0.1, 1, 10],
                    'logreg__penalty': ['l1', 'l2']
                }
            },
            "SVM": {
                "estimator": SVC(class_weight='balanced', probability=True, cache_size=2000),
                "params": {
                    'svm__C'     : [0.1, 1, 10, 100],
                    'svm__kernel': ['rbf', 'linear'],
                    'svm__gamma' : ['scale']
                },
                "params_personal": {
                    'svm__C'     : [0.1, 1, 10],
                    'svm__kernel': ['linear'],
                    'svm__gamma' : ['scale']
                }
            }
        }
    }
    DATASETS_RUN = ["MobiFall", "SisFall", "FAD_40Hz"]
    print("🖥️  SERVER MODU — tam arama (200 trial, tüm datasetler, LogReg+SVM)")

f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

DATASETS_CONFIG = {
    "MobiFall": {
        "sensor_table": "MobiFall_sensor_data", "trials_table": "MobiFall_trials",
        "subject_col": "subject_id", "label_col": "is_fall", "default_fs": 87,
        "sensor_groups": {
            "Acc": ["acc_x", "acc_y", "acc_z"],
            "Gyr": ["gyr_x", "gyr_y", "gyr_z"],
            "Ori": ["ori_azimuth", "ori_pitch", "ori_roll"]
        }
    },
    "SisFall": {
        "sensor_table": "SisFall_sensor_data", "trials_table": "SisFall_trials",
        "subject_col": "subject_id", "label_col": "is_fall", "default_fs": 200,
        "sensor_groups": {
            "Acc_ADXL": ["acc_adxl345_x", "acc_adxl345_y", "acc_adxl345_z"],
            "Gyr_ITG" : ["gyr_itg3200_x", "gyr_itg3200_y", "gyr_itg3200_z"]
        }
    },
    "FAD_40Hz": {
        "sensor_table": "FAD_40Hz_sensor_data", "trials_table": "FAD_40Hz_trials",
        "subject_col": "subject_id", "label_col": "is_fall", "default_fs": 40,
        "sensor_groups": {
            "Acc": ["acc_x", "acc_y", "acc_z"],
            "Gyr": ["gyr_x", "gyr_y", "gyr_z"]
        }
    }
}

# =============================================================================
# 2. LOGGER & DB
# =============================================================================
class ExperimentLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write("V15 EXPERIMENT REPORT\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {RUN_MODE.upper()}\n")
            f.write("="*120 + "\n\n")

    def log(self, message):
        print(message); sys.stdout.flush()
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    def log_section(self, title):
        self.log("\n" + "="*80)
        self.log(f"  {title}")
        self.log("="*80)

    def log_dict(self, title, data_dict):
        self.log(f"\n--- {title} ---")
        self.log(json.dumps(data_dict, indent=4, default=str))


def _db_write(df, table_name, if_exists='replace'):
    conn = sqlite3.connect(RESULT_DB_PATH)
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.close()

def save_features_to_db(X, y, subjects, ds_name, tda_label):
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df['label'] = y; df['subject'] = subjects
    table = f"features_{ds_name}_{tda_label}"
    _db_write(df, table)
    return table

def save_metrics_to_db(metrics_list, table_name):
    if not metrics_list: return
    _db_write(pd.DataFrame(metrics_list), table_name)

def save_optuna_results_to_db(study, ds_name, tda_label):
    rows = []
    for tr in study.trials:
        rows.append({'trial': tr.number, 'f2': tr.value,
                     'state': str(tr.state), **tr.params})
    if rows:
        _db_write(pd.DataFrame(rows), f"optuna_trials_{ds_name}_{tda_label}")
    _db_write(pd.DataFrame([{'dataset': ds_name, 'tda_label': tda_label,
                              'best_f2': study.best_value, **study.best_params}]),
              f"optuna_best_{ds_name}_{tda_label}")


logger = ExperimentLogger(REPORT_FILE)

# =============================================================================
# 3. DATA PROCESSING
# =============================================================================
def safe_resample(X, orig_fs, target_fs):
    n = len(X); target = int((n / orig_fs) * target_fs)
    if target < 5: return None
    old_t = np.linspace(0, 1, n); new_t = np.linspace(0, 1, target)
    return interp1d(old_t, X, axis=0, kind='linear',
                    fill_value="extrapolate")(new_t)

def fetch_data_full(conn, ds_config, ds_name):
    try:
        s_table   = ds_config["sensor_table"]; t_table = ds_config["trials_table"]
        label_col = ds_config["label_col"];    subj_col = ds_config["subject_col"]
        all_cols  = []
        for grp_cols in ds_config["sensor_groups"].values():
            all_cols.extend(grp_cols)

        logger.log(f"   ⏳ Loading: {ds_name}...")
        query = (f"SELECT s.trial_id, t.{subj_col} as subj_id, s.{label_col}, "
                 f"t.sampling_rate, "
                 f"{', '.join(['s.'+c for c in all_cols])} "
                 f"FROM {s_table} s LEFT JOIN {t_table} t "
                 f"ON s.trial_id = t.trial_id "
                 f"ORDER BY s.trial_id, s.timestamp")
        df = pd.read_sql_query(query, conn)
        if df.empty: return None

        smv_list = []; acc_smv = None
        for grp_name, grp_cols in ds_config["sensor_groups"].items():
            smv = np.sqrt(np.sum(df[grp_cols].values**2, axis=1))
            smv_list.append(smv)
            if "Acc" in grp_name or "acc" in grp_cols[0]: acc_smv = smv

        X_multi  = np.column_stack(smv_list)
        y        = df[label_col].values
        trials   = df['trial_id'].values
        subjects = df['subj_id'].values
        fs_vals  = df['sampling_rate'].fillna(ds_config['default_fs']).values
        if acc_smv is None: acc_smv = smv_list[0]
        del df, smv_list; gc.collect()

        trial_list = []; subj_stats = {}
        max_win_sec = CONFIG["SEARCH_SPACE"]["win_sec"]["high"]

        for tid in np.unique(trials):
            mask  = (trials == tid)
            X_t   = X_multi[mask]; acc_t = acc_smv[mask]; y_t = y[mask]
            subj_t = subjects[mask][0] if any(pd.notna(subjects[mask])) else tid
            fs_t   = np.nanmedian(fs_vals[mask])
            if len(X_t) < 10: continue
            has_fall = 1 if np.sum(y_t) > 0 else 0
            win_len  = int(max_win_sec * fs_t)

            if has_fall:
                peak_idx = np.argmax(acc_t)
                start = max(0, peak_idx - win_len // 2)
                end   = min(len(X_t), peak_idx + win_len // 2)
                X_cut = X_t[start:end]
                if len(X_cut) < win_len // 4: continue
                subj_stats.setdefault(subj_t, {'fall': 0, 'adl': 0})
                subj_stats[subj_t]['fall'] += 1
                trial_list.append({'id': tid, 'subject': subj_t,
                                   'X': X_cut, 'fs': fs_t, 'has_fall': 1})
            else:
                step = win_len
                for i in range(0, len(X_t) - win_len, step):
                    X_cut = X_t[i: i+win_len]
                    subj_stats.setdefault(subj_t, {'fall': 0, 'adl': 0})
                    subj_stats[subj_t]['adl'] += 1
                    trial_list.append({'id': tid, 'subject': subj_t,
                                       'X': X_cut, 'fs': fs_t, 'has_fall': 0})

        valid = [s for s, st in subj_stats.items()
                 if st['fall'] > 1 and st['adl'] > 1]
        return [t for t in trial_list if t['subject'] in valid]
    except Exception as e:
        logger.log(f"   ❌ fetch_data_full hatası: {e}"); return None

# =============================================================================
# 4. TDA ENGINE
# =============================================================================
class Ultimate_Hybrid_Engine:
    def __init__(self, params): self.params = params

    def compute(self, X):
        win_sec    = self.params['win_sec']
        target_len = int(win_sec * CONFIG['TARGET_FS'])
        if len(X) > target_len:
            start = (len(X) // 2) - (target_len // 2)
            X = X[start: start + target_len]

        if self.params.get('use_delay'):
            stride = self.params.get('stride_factor', 1)
            embeddings = [self._embed(X[:, i], self.params['delay'],
                                      self.params['dim'], stride)
                          for i in range(X.shape[1])]
            if not embeddings: return np.zeros(24)
            min_len = min(len(e) for e in embeddings)
            full_pc = np.hstack([e[:min_len] for e in embeddings])
        else:
            full_pc = X

        limit     = CONFIG["LIMIT_POINTS"]
        landmarks = full_pc
        if len(full_pc) > limit:
            try:
                sm = self.params.get('sampling_method', 'maxmin')
                if sm == 'maxmin':
                    landmarks = np.array(
                        gudhi.subsampling.choose_n_farthest_points(
                            points=full_pc, nb_points=limit))
                else:
                    idx = np.random.choice(len(full_pc), limit, replace=False)
                    landmarks = full_pc[idx]
            except:
                idx = np.linspace(0, len(full_pc)-1, limit, dtype=int)
                landmarks = full_pc[idx]

        try:    tda_feats = self._compute_tda(landmarks)
        except: tda_feats = np.zeros(20)

        acc        = X[:, 0]
        stat_feats = np.array([np.max(acc), np.std(acc),
                               np.mean(acc), np.max(acc) - np.min(acc)])
        return np.hstack([tda_feats, stat_feats])

    def _embed(self, signal, delay, dim, stride):
        n = len(signal); req = (dim - 1) * delay + 1
        if n < req: return np.zeros((1, dim))
        max_pts  = 200
        possible = (n - req) // stride + 1
        if possible > max_pts: stride = max(stride, (n - req) // max_pts)
        return np.array([signal[i: i+req: delay]
                         for i in range(0, n - req + 1, stride)])

    def _compute_tda(self, landmarks):
        ct = self.params['complex_type']
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks)
        st = None

        if ct == 'Alpha' and landmarks.shape[1] <= 3:
            try:   st = gudhi.AlphaComplex(points=landmarks).create_simplex_tree()
            except: ct = 'Rips'

        if ct == 'SparseRips':
            try:   st = gudhi.SparseRipsComplex(
                            points=landmarks).create_simplex_tree(max_dimension=2)
            except: ct = 'Rips'

        if ct == 'Rips' or st is None:
            metric = self.params.get('metric', 'euclidean')
            try:    d = pdist(landmarks, metric=metric)
            except: d = pdist(landmarks, metric='euclidean')
            eps = (np.percentile(d, self.params.get('eps_percentile', 70))
                   if len(d) > 0 else 0.5)
            st = gudhi.RipsComplex(
                    points=landmarks, max_edge_length=eps
                 ).create_simplex_tree(max_dimension=2)

        st.persistence()
        return self._extract_features(st, len(landmarks))

    def _extract_features(self, st, n_pts):
        feats = []
        for dim in [0, 1]:
            ivs = st.persistence_intervals_in_dimension(dim)
            if len(ivs) == 0: feats.extend([0]*10); continue
            lt = ivs[:, 1] - ivs[:, 0]
            lt = lt[np.isfinite(lt)]
            if len(lt) == 0: feats.extend([0]*10); continue
            tot   = np.sum(lt); probs = lt / tot
            feats.extend([
                len(lt) / n_pts,
                -np.sum(probs * np.log(probs + 1e-10)),
                np.max(lt),
                np.sum(lt**2) / tot,
                np.mean(lt),
                np.std(lt),
                np.median(lt),
                np.percentile(lt, 25),
                np.percentile(lt, 75),
                np.sum(lt**3) / (tot + 1e-10)
            ])
        return np.array(feats)


def get_xy_matrix(trial_list, params):
    eng = Ultimate_Hybrid_Engine(params)
    X, y, g = [], [], []
    for tr in trial_list:
        xr = safe_resample(tr['X'], tr['fs'], CONFIG["TARGET_FS"])
        if xr is None: continue
        xs = StandardScaler().fit_transform(xr)
        ft = eng.compute(xs)
        if not np.isnan(ft).any() and not np.isinf(ft).any():
            X.append(ft); y.append(tr['has_fall']); g.append(tr['subject'])
    return np.array(X), np.array(y), np.array(g)


def create_global_subset(trial_list):
    falls = [t for t in trial_list if t['has_fall'] == 1]
    adls  = [t for t in trial_list if t['has_fall'] == 0]
    sub   = falls + random.sample(adls, min(len(adls), len(falls) * 3))
    random.shuffle(sub)
    return sub


def balanced_undersample(X, y, seed=None):
    if seed is not None:
        np.random.seed(seed)
    idx0, idx1 = np.where(y==0)[0], np.where(y==1)[0]
    if len(idx1) == 0: return X, y
    n0 = min(len(idx0), len(idx1) * 3)
    if n0 < len(idx0):
        c0 = np.random.choice(idx0, n0, replace=False)
        return (np.concatenate([X[c0], X[idx1]]),
                np.concatenate([y[c0], y[idx1]]))
    return X, y


def find_best_threshold(model, X_val, y_val):
    try:
        probs = model.predict_proba(X_val)[:, 1]
        p, r, t = precision_recall_curve(y_val, probs)
        f2 = np.divide(5*p*r, 4*p+r, out=np.zeros_like(p), where=(4*p+r)!=0)
        return float(t[np.argmax(f2)]) if len(t) > 0 else 0.5
    except:
        return 0.5

# =============================================================================
# 5. OPTUNA OBJECTIVE (tek — model bağımsız TDA arama)
# =============================================================================
def _sample_tda_params(trial):
    sp = CONFIG["SEARCH_SPACE"]
    params = {
        'win_sec'        : trial.suggest_float("win_sec",
                               sp["win_sec"]["low"], sp["win_sec"]["high"],
                               step=sp["win_sec"]["step"]),
        'complex_type'   : trial.suggest_categorical("complex_type",   sp["complex_type"]),
        'use_delay'      : trial.suggest_categorical("use_delay",      sp["use_delay"]),
        'sampling_method': trial.suggest_categorical("sampling_method", sp["sampling_method"])
    }
    if params['use_delay']:
        params['dim']           = trial.suggest_int("dim",   sp["dim"]["low"],   sp["dim"]["high"])
        params['delay']         = trial.suggest_int("delay", sp["delay"]["low"], sp["delay"]["high"])
        params['stride_factor'] = trial.suggest_categorical("stride_factor", sp["stride_factor"])
    if params['complex_type'] in ["Rips", "SparseRips"]:
        params['metric']        = trial.suggest_categorical("metric",         sp["metrics"])
        params['eps_percentile']= trial.suggest_categorical("eps_percentile", sp["eps_percentile"])
    return params


def objective(trial, subset):
    """
    Tek Optuna — TDA parametrelerini model bağımsız optimize eder.
    Her trial'da hem LogReg hem SVM ile kısa CV yapılır, ortalaması alınır.
    Bu şekilde genel olarak iyi özellik çıkaran TDA parametresi seçilir.
    """
    params = _sample_tda_params(trial)
    X, y, _ = get_xy_matrix(subset, params)
    if len(X) < 10: return 0.0
    try:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        # LogReg
        pipe_lr = Pipeline([('scaler', StandardScaler()),
                             ('logreg', LogisticRegression(class_weight='balanced',
                                                           solver='liblinear', max_iter=200))])
        for tr_ix, te_ix in cv.split(X, y):
            Xtr, ytr = balanced_undersample(X[tr_ix], y[tr_ix])
            pipe_lr.fit(Xtr, ytr)
            scores.append(fbeta_score(y[te_ix], pipe_lr.predict(X[te_ix]),
                                      beta=2, zero_division=0))

        # SVM (sadece server modunda)
        if RUN_MODE == "server":
            pipe_sv = Pipeline([('scaler', StandardScaler()),
                                 ('svm', SVC(class_weight='balanced', kernel='linear',
                                             C=1.0, probability=False))])
            for tr_ix, te_ix in cv.split(X, y):
                Xtr, ytr = balanced_undersample(X[tr_ix], y[tr_ix])
                pipe_sv.fit(Xtr, ytr)
                scores.append(fbeta_score(y[te_ix], pipe_sv.predict(X[te_ix]),
                                          beta=2, zero_division=0))

        return float(np.mean(scores))
    except:
        return 0.0

# =============================================================================
# 6. YARDIMCI İSTATİSTİK
# =============================================================================
def mean_ci(values, confidence=0.95):
    """mean ± %95 t-test CI döndürür."""
    arr = np.array([v for v in values
                    if v is not None and not np.isnan(float(v))])
    if len(arr) < 2:
        return (float(np.mean(arr)) if len(arr) == 1 else 0.0), 0.0
    m  = float(np.mean(arr))
    se = float(scipy_stats.sem(arr))
    h  = se * scipy_stats.t.ppf((1 + confidence) / 2., len(arr) - 1)
    return m, float(h)


def fmt_ci(m, h):
    return f"{m:.4f}±{h:.4f}"


def fmt_mean(m):
    return f"{m:.4f}"


def calc_metrics(y_true, y_pred):
    return {
        'acc': float(accuracy_score(y_true, y_pred)),
        'rec': float(recall_score(y_true, y_pred, zero_division=0)),
        'pre': float(precision_score(y_true, y_pred, zero_division=0)),
        'f1' : float(f1_score(y_true, y_pred, zero_division=0)),
        'f2' : float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
    }

# =============================================================================
# 7. THRESHOLD SWEEP — her subject için optimal threshold analizi
# =============================================================================
def threshold_sweep(model, X_sub, y_sub):
    """
    0.05–0.95 arası 19 threshold değerinde tahmin yapar.
    Her threshold için metrikleri döndürür.
    Raporda: optimal threshold vs varsayılan (0.5) F2 karşılaştırması yapılır.
    """
    try:
        probs = model.predict_proba(X_sub)[:, 1]
        rows  = []
        for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
            preds = (probs >= thr).astype(int)
            m = calc_metrics(y_sub, preds)
            m['threshold'] = float(thr)
            rows.append(m)
        return rows
    except:
        return []


def log_threshold_sweep_summary(logger, sweep_rows, subject, model_name, default_thr):
    """
    Threshold sweep özetini raporlar:
    - Varsayılan threshold F2
    - Optimal threshold ve F2
    - Kazanım (delta F2)
    """
    if not sweep_rows:
        return
    # Varsayılan threshold'a en yakın satır
    default_row = min(sweep_rows, key=lambda r: abs(r['threshold'] - default_thr))
    # En iyi F2 veren threshold
    best_row    = max(sweep_rows, key=lambda r: r['f2'])
    delta_f2    = best_row['f2'] - default_row['f2']
    logger.log(
        f"         THR-SWEEP [{model_name} | Sub {subject}]  "
        f"default_thr={default_thr:.2f} → F2={default_row['f2']:.4f}  |  "
        f"opt_thr={best_row['threshold']:.2f} → F2={best_row['f2']:.4f}  |  "
        f"ΔF2={delta_f2:+.4f}"
    )

# =============================================================================
# 8. ANA VALİDASYON FONKSİYONU
# =============================================================================
def run_validation(trial_list, best_tda_params, ds_name, tda_label):
    """
    Üç model tipi (V15):

    A) Naive
       - Tüm veri havuzu, N_NAIVE_REPEATS=10 × train/test split
       - Global GridSearchCV bir kez → sabit hiperparametre
       - Raporda: tek satır global mean (CI yok)

    B) General (LOGO-CV)
       - Global GridSearchCV bir kez → sabit hiperparametre
       - Her subject için N_GENERAL_REPEATS=15 tekrar
         (sadece balanced_undersample seed'i değişir)
       - Raporda: her subject için mean ± CI

    C) Personal
       - Her subject kendi verisiyle, daraltılmış GridSearchCV
       - N_PERS_REPEATS=15 tekrar (train/test split seed değişir)
       - Raporda: her subject için mean ± CI

    Threshold sweep: General modelin son estimatörü ile her subject'e uygulanır,
    özet (varsayılan vs optimal threshold F2) rapora eklenir.
    """
    label = f"{ds_name}_{tda_label}"
    logger.log(f"\n   🔬 [{label}] VALİDASYON BAŞLIYOR...")

    # ── Özellik çıkarımı ─────────────────────────────────────────────────────
    try:
        X_all, y_all, sub_all = get_xy_matrix(trial_list, best_tda_params)
        unique_subjects = np.unique(sub_all)
        tbl = save_features_to_db(X_all, y_all, sub_all, ds_name, tda_label)
        logger.log(f"      ✅ Özellikler: {tbl}  shape={X_all.shape}")
    except Exception as e:
        logger.log(f"      ❌ Özellik çıkarım hatası: {e}"); return

    N_NAIVE_REPEATS   = 10
    N_GENERAL_REPEATS = 15
    N_PERS_REPEATS    = 15

    logo      = LeaveOneGroupOut()
    all_db_rows = []

    for model_name, model_cfg in CONFIG["MODELS"].items():
        logger.log(f"\n      ▶ MODEL: {model_name}  [{label}]")
        logger.log(f"        Naive: n={N_NAIVE_REPEATS} repeat  |  "
                   f"General: n={N_GENERAL_REPEATS} repeat/subject (LOGO, sabit params)  |  "
                   f"Personal: n={N_PERS_REPEATS} repeat/subject (daraltılmış grid)")

        # ── Tablo başlığı ─────────────────────────────────────────────────────
        sep = "-" * 260
        hdr = (
            f"{'SUB':<10} | "
            f"{'G-ACC (m±CI)':>18} {'G-REC (m±CI)':>18} {'G-PRE (m±CI)':>18} "
            f"{'G-F1 (m±CI)':>18} {'G-F2 (m±CI)':>18} {'G-THR':>6} | "
            f"{'P-ACC (m±CI)':>18} {'P-REC (m±CI)':>18} {'P-PRE (m±CI)':>18} "
            f"{'P-F1 (m±CI)':>18} {'P-F2 (m±CI)':>18} {'P-THR':>6} | "
            f"{'G-PARAMS'}"
        )
        logger.log(sep); logger.log(hdr); logger.log(sep)

        # ────────────────────────────────────────────────────────────────────
        # A. GLOBAL NAİVE — global GridSearchCV bir kez, 10 tekrar, mean only
        # ────────────────────────────────────────────────────────────────────
        naive_scores      = {k: [] for k in ['acc','rec','pre','f1','f2']}
        naive_thrs        = []
        naive_params_str  = ''
        naive_best_model  = None

        try:
            # Global hiperparametre araması — tüm veri üzerinde bir kez
            X_tr_n0, X_te_n0, y_tr_n0, y_te_n0 = train_test_split(
                X_all, y_all, test_size=0.4, stratify=y_all, random_state=0)
            X_tr_nb0, y_tr_nb0 = balanced_undersample(X_tr_n0, y_tr_n0, seed=0)
            pipe_n = Pipeline([('scaler', StandardScaler()),
                               (model_name.lower(), model_cfg['estimator'])])
            srch_n = GridSearchCV(pipe_n, model_cfg['params'], cv=3,
                                  scoring=f2_scorer, n_jobs=CONFIG["MODEL_JOBS"])
            srch_n.fit(X_tr_nb0, y_tr_nb0)
            naive_params_str = (str(srch_n.best_params_)
                                .replace("svm__","").replace("logreg__",""))
            naive_best_model = srch_n.best_estimator_

            logger.log(f"      [Naive] Global params: {naive_params_str}")

            # 10 tekrar — sadece split seed değişir, model sabit
            for seed in range(N_NAIVE_REPEATS):
                X_tr_n, X_te_n, y_tr_n, y_te_n = train_test_split(
                    X_all, y_all, test_size=0.4, stratify=y_all, random_state=seed)
                X_tr_nb, y_tr_nb = balanced_undersample(X_tr_n, y_tr_n, seed=seed)
                naive_best_model.fit(X_tr_nb, y_tr_nb)
                thr_n   = find_best_threshold(naive_best_model, X_tr_nb, y_tr_nb)
                preds_n = (naive_best_model.predict_proba(X_te_n)[:, 1] >= thr_n).astype(int)
                m_n     = calc_metrics(y_te_n, preds_n)
                for k in naive_scores: naive_scores[k].append(m_n[k])
                naive_thrs.append(thr_n)

            naive_means = {k: float(np.mean(v)) for k, v in naive_scores.items()}
            naive_mean_thr = float(np.mean(naive_thrs))

            logger.log(
                f"      [Naive ×{N_NAIVE_REPEATS}] "
                f"ACC={fmt_mean(naive_means['acc'])}  "
                f"REC={fmt_mean(naive_means['rec'])}  "
                f"PRE={fmt_mean(naive_means['pre'])}  "
                f"F1={fmt_mean(naive_means['f1'])}  "
                f"F2={fmt_mean(naive_means['f2'])}  "
                f"THR={naive_mean_thr:.2f}  PARAMS={naive_params_str}"
            )

            _db_write(
                pd.DataFrame({
                    'dataset'     : ds_name,
                    'tda_label'   : tda_label,
                    'model'       : model_name,
                    'repeat'      : list(range(N_NAIVE_REPEATS)),
                    'acc'         : naive_scores['acc'],
                    'rec'         : naive_scores['rec'],
                    'pre'         : naive_scores['pre'],
                    'f1'          : naive_scores['f1'],
                    'f2'          : naive_scores['f2'],
                    'thr'         : naive_thrs
                }),
                f"naive_repeats_{ds_name}_{tda_label}",
                if_exists='append'
            )

        except Exception as e:
            logger.log(f"      ❌ Naive model hatası: {e}")

        # ────────────────────────────────────────────────────────────────────
        # B. GENERAL — global GridSearchCV bir kez, sabit params, 15 tekrar
        # ────────────────────────────────────────────────────────────────────
        g_results     = {}
        general_params_str = ''

        try:
            # Global hiperparametre araması — tüm veri üzerinde bir kez
            X_tr_g0, y_tr_g0 = balanced_undersample(X_all, y_all, seed=0)
            pipe_g = Pipeline([('scaler', StandardScaler()),
                               (model_name.lower(), model_cfg['estimator'])])
            srch_g = GridSearchCV(pipe_g, model_cfg['params'], cv=3,
                                  scoring=f2_scorer, n_jobs=CONFIG["MODEL_JOBS"])
            srch_g.fit(X_tr_g0, y_tr_g0)
            general_params_str = (str(srch_g.best_params_)
                                  .replace("svm__","").replace("logreg__",""))
            # Sabit pipeline — sadece fit edilecek
            best_pipe_g = srch_g.best_estimator_

            logger.log(f"      [General] Global params: {general_params_str}")

            logo_folds = list(logo.split(X_all, y_all, groups=sub_all))
            for tr_idx, te_idx in logo_folds:
                test_sub   = sub_all[te_idx][0]
                rep_scores = {k: [] for k in ['acc','rec','pre','f1','f2']}
                rep_thrs   = []
                last_estimator = None

                for rep in range(N_GENERAL_REPEATS):
                    try:
                        # Sadece undersample seed değişir
                        X_tr_g, y_tr_g = balanced_undersample(
                            X_all[tr_idx], y_all[tr_idx], seed=rep)
                        best_pipe_g.fit(X_tr_g, y_tr_g)
                        thr_g   = find_best_threshold(best_pipe_g, X_tr_g, y_tr_g)
                        preds_g = (best_pipe_g.predict_proba(
                                       X_all[te_idx])[:, 1] >= thr_g).astype(int)
                        m_g = calc_metrics(y_all[te_idx], preds_g)
                        for k in rep_scores: rep_scores[k].append(m_g[k])
                        rep_thrs.append(thr_g)
                        last_estimator = best_pipe_g
                    except:
                        continue

                g_ci = {k: mean_ci(v) for k, v in rep_scores.items()}
                g_results[test_sub] = {
                    'ci'       : g_ci,
                    'thr'      : float(np.nanmean(rep_thrs)) if rep_thrs else np.nan,
                    'params'   : general_params_str,
                    'estimator': last_estimator,
                    # mean değerler kolay erişim için
                    'acc': g_ci['acc'][0], 'rec': g_ci['rec'][0],
                    'pre': g_ci['pre'][0], 'f1' : g_ci['f1'][0],
                    'f2' : g_ci['f2'][0],
                }

        except Exception as e:
            logger.log(f"      ❌ General (LOGO) hatası: {e}")

        # ────────────────────────────────────────────────────────────────────
        # C. PERSONAL + RAPORLAMA
        # ────────────────────────────────────────────────────────────────────
        model_rows = []

        for s in unique_subjects:
            try:
                idx      = (sub_all == s)
                X_s, y_s = X_all[idx], y_all[idx]
                gr       = g_results.get(s, {})

                if np.sum(y_s==1) < 2 or np.sum(y_s==0) < 2:
                    logger.log(f"{str(s):<10} | Yetersiz veri — atlandı")
                    continue

                # ── Personal: daraltılmış grid, 15 tekrar ────────────────────
                pers_scores = {k: [] for k in ['acc','rec','pre','f1','f2']}
                pers_thrs   = []
                pers_params = ""
                for seed in range(N_PERS_REPEATS):
                    try:
                        X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
                            X_s, y_s, test_size=0.4, stratify=y_s,
                            random_state=seed)
                        X_tr_pb, y_tr_pb = balanced_undersample(X_tr_p, y_tr_p, seed=seed)
                        pipe_p = Pipeline([('scaler', StandardScaler()),
                                           (model_name.lower(), model_cfg['estimator'])])
                        cv_p = max(2, min(3, int(min(np.sum(y_tr_pb==0),
                                                     np.sum(y_tr_pb==1)))))
                        # Daraltılmış param grid — az veriyle güvenilir CV
                        srch_p = GridSearchCV(pipe_p, model_cfg['params_personal'],
                                              cv=cv_p, scoring=f2_scorer,
                                              n_jobs=CONFIG["MODEL_JOBS"])
                        srch_p.fit(X_tr_pb, y_tr_pb)
                        best_p  = srch_p.best_estimator_
                        thr_p   = find_best_threshold(best_p, X_tr_pb, y_tr_pb)
                        preds_p = (best_p.predict_proba(X_te_p)[:, 1] >= thr_p).astype(int)
                        m_p     = calc_metrics(y_te_p, preds_p)
                        for k in pers_scores: pers_scores[k].append(m_p[k])
                        pers_thrs.append(thr_p)
                        pers_params = (str(srch_p.best_params_)
                                       .replace("svm__","").replace("logreg__",""))
                    except:
                        continue

                p_ci       = {k: mean_ci(v) for k, v in pers_scores.items()}
                p_mean_thr = float(np.nanmean(pers_thrs)) if pers_thrs else np.nan

                # ── Threshold sweep (General estimatör) ──────────────────────
                g_est      = gr.get('estimator', None)
                sweep_rows = threshold_sweep(g_est, X_s, y_s) if g_est else []
                g_thr      = gr.get('thr', 0.5)
                if sweep_rows:
                    log_threshold_sweep_summary(logger, sweep_rows, s,
                                                model_name, g_thr)

                # ── DB satırı ─────────────────────────────────────────────────
                g_ci_sub = gr.get('ci', {k: (np.nan, np.nan)
                                         for k in ['acc','rec','pre','f1','f2']})
                row = {
                    'subject'   : str(s),
                    'model'     : model_name,
                    'tda_label' : tda_label,
                    'ds_name'   : ds_name,
                    # General — mean ± CI
                    'g_acc_mean': g_ci_sub['acc'][0], 'g_acc_ci': g_ci_sub['acc'][1],
                    'g_rec_mean': g_ci_sub['rec'][0], 'g_rec_ci': g_ci_sub['rec'][1],
                    'g_pre_mean': g_ci_sub['pre'][0], 'g_pre_ci': g_ci_sub['pre'][1],
                    'g_f1_mean' : g_ci_sub['f1'][0],  'g_f1_ci' : g_ci_sub['f1'][1],
                    'g_f2_mean' : g_ci_sub['f2'][0],  'g_f2_ci' : g_ci_sub['f2'][1],
                    'g_thr'     : g_thr,
                    'g_params'  : gr.get('params', ''),
                    # Personal — mean ± CI
                    'p_acc_mean': p_ci['acc'][0], 'p_acc_ci': p_ci['acc'][1],
                    'p_rec_mean': p_ci['rec'][0], 'p_rec_ci': p_ci['rec'][1],
                    'p_pre_mean': p_ci['pre'][0], 'p_pre_ci': p_ci['pre'][1],
                    'p_f1_mean' : p_ci['f1'][0],  'p_f1_ci' : p_ci['f1'][1],
                    'p_f2_mean' : p_ci['f2'][0],  'p_f2_ci' : p_ci['f2'][1],
                    'p_thr'     : p_mean_thr,
                    'p_params'  : pers_params,
                    # Threshold sweep JSON
                    'thr_sweep_json': json.dumps(sweep_rows)
                }
                model_rows.append(row)

                # ── Log satırı ───────────────────────────────────────────────
                log_line = (
                    f"{str(s):<10} | "
                    f"{fmt_ci(g_ci_sub['acc'][0], g_ci_sub['acc'][1]):>18} "
                    f"{fmt_ci(g_ci_sub['rec'][0], g_ci_sub['rec'][1]):>18} "
                    f"{fmt_ci(g_ci_sub['pre'][0], g_ci_sub['pre'][1]):>18} "
                    f"{fmt_ci(g_ci_sub['f1'][0],  g_ci_sub['f1'][1]):>18} "
                    f"{fmt_ci(g_ci_sub['f2'][0],  g_ci_sub['f2'][1]):>18} "
                    f"{g_thr if not np.isnan(g_thr) else 0.0:>6.2f} | "
                    f"{fmt_ci(p_ci['acc'][0], p_ci['acc'][1]):>18} "
                    f"{fmt_ci(p_ci['rec'][0], p_ci['rec'][1]):>18} "
                    f"{fmt_ci(p_ci['pre'][0], p_ci['pre'][1]):>18} "
                    f"{fmt_ci(p_ci['f1'][0],  p_ci['f1'][1]):>18} "
                    f"{fmt_ci(p_ci['f2'][0],  p_ci['f2'][1]):>18} "
                    f"{p_mean_thr if not np.isnan(p_mean_thr) else 0.0:>6.2f} | "
                    f"{gr.get('params','')}"
                )
                logger.log(log_line)

            except Exception as e:
                logger.log(f"{str(s):<10} | HATA: {e}"); continue

        all_db_rows.extend(model_rows)

        # ── Özet ─────────────────────────────────────────────────────────────
        if model_rows:
            logger.log(f"\n{'─'*120}")
            logger.log(f"  📊 [{model_name} | {label}] ÖZET")
            logger.log(f"  Naive (global, n={N_NAIVE_REPEATS})  |  "
                       f"General (LOGO, sabit params, n={N_GENERAL_REPEATS}/subject)  |  "
                       f"Personal (daraltılmış grid, n={N_PERS_REPEATS}/subject)")
            logger.log(f"{'─'*120}")

            # Naive — tek satır mean
            logger.log(
                f"  Naive  : "
                f"ACC={fmt_mean(naive_means.get('acc',0))}  "
                f"REC={fmt_mean(naive_means.get('rec',0))}  "
                f"PRE={fmt_mean(naive_means.get('pre',0))}  "
                f"F1={fmt_mean(naive_means.get('f1',0))}  "
                f"F2={fmt_mean(naive_means.get('f2',0))}  "
                f"PARAMS={naive_params_str}"
            )

            # General — cross-subject mean ± CI
            g_parts = []
            for key, lbl in [('g_acc_mean','G-ACC'),('g_rec_mean','G-REC'),
                              ('g_f1_mean','G-F1'),('g_f2_mean','G-F2')]:
                vals = [r[key] for r in model_rows
                        if key in r and r[key] is not None
                        and not np.isnan(float(r[key]))]
                g_parts.append(f"{lbl}: {fmt_ci(*mean_ci(vals))}")
            logger.log("  General: " + "  ".join(g_parts) +
                       f"  PARAMS={general_params_str}")

            # Personal — cross-subject mean ± CI
            p_parts = []
            for key, lbl in [('p_acc_mean','P-ACC'),('p_rec_mean','P-REC'),
                              ('p_f1_mean','P-F1'),('p_f2_mean','P-F2')]:
                vals = [r[key] for r in model_rows
                        if key in r and r[key] is not None
                        and not np.isnan(float(r[key]))]
                p_parts.append(f"{lbl}: {fmt_ci(*mean_ci(vals))}")
            logger.log("  Personal: " + "  ".join(p_parts))

            # Threshold sweep özeti — dataset geneli
            logger.log(f"\n  📈 THRESHOLD SWEEP ÖZETİ [{model_name} | {label}]")
            logger.log(f"  {'SUB':<10} {'DEF_THR':>8} {'DEF_F2':>8} "
                       f"{'OPT_THR':>8} {'OPT_F2':>8} {'ΔF2':>8}")
            logger.log(f"  {'-'*60}")
            delta_f2_list = []
            for r in model_rows:
                sweep = json.loads(r.get('thr_sweep_json', '[]'))
                if not sweep: continue
                def_thr  = r.get('g_thr', 0.5) or 0.5
                def_row  = min(sweep, key=lambda x: abs(x['threshold'] - def_thr))
                best_row = max(sweep, key=lambda x: x['f2'])
                delta    = best_row['f2'] - def_row['f2']
                delta_f2_list.append(delta)
                logger.log(
                    f"  {r['subject']:<10} {def_thr:>8.2f} {def_row['f2']:>8.4f} "
                    f"{best_row['threshold']:>8.2f} {best_row['f2']:>8.4f} {delta:>+8.4f}"
                )
            if delta_f2_list:
                logger.log(f"  {'-'*60}")
                logger.log(
                    f"  {'MEAN':<10} {'':>8} {'':>8} {'':>8} {'':>8} "
                    f"{float(np.mean(delta_f2_list)):>+8.4f}"
                )
            logger.log(f"{'─'*120}\n")

    # ── DB'ye yaz ────────────────────────────────────────────────────────────
    save_metrics_to_db(all_db_rows, f"metrics_{label}")
    logger.log(f"   ✅ [{label}] Tamamlandı.")

# =============================================================================
# 9. ANA AKIŞ
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(DATA_DB_PATH):
        logger.log("⚠️  UYARI: Veri tabanı bulunamadı!")

    conn     = sqlite3.connect(DATA_DB_PATH)
    mode_tag = "LAPTOP MODU (hızlı)" if RUN_MODE == "laptop" else "SERVER MODU (tam)"
    logger.log_section(
        f"V15 — {mode_tag} | TEK OPTUNA | "
        "NAIVE(mean) + GENERAL(CI,sabit-params) + PERSONAL(CI) | THR-SWEEP"
    )
    logger.log_dict("CONFIG", CONFIG)

    for ds_name in DATASETS_RUN:
        cfg = DATASETS_CONFIG[ds_name]
        logger.log_section(f"DATASET: {ds_name}")
        trial_list = fetch_data_full(conn, cfg, ds_name)

        if not trial_list:
            logger.log(f"   ❌ {ds_name} için veri yok — atlandı."); continue

        subset      = create_global_subset(trial_list)
        storage_url = f"sqlite:///{OPTUNA_DB_PATH}"

        # ── TEK OPTUNA — model bağımsız TDA arama ────────────────────────────
        logger.log(f"\n   🔍 [Optuna] {ds_name} — {CONFIG['N_TRIALS_GLOBAL']} trial...")
        study = optuna.create_study(
            storage=storage_url,
            study_name=f"{ds_name}_v15_{RUN_MODE}",
            direction="maximize",
            load_if_exists=True)
        rem = CONFIG["N_TRIALS_GLOBAL"] - len(study.trials)
        if rem > 0:
            try:
                study.optimize(
                    lambda t: objective(t, subset),
                    n_trials=rem, n_jobs=CONFIG["OPTUNA_JOBS"],
                    gc_after_trial=True)
            except Exception as e:
                logger.log(f"   ⚠️  Optuna hatası: {e}")

        logger.log(f"   🏆 Best F2: {study.best_value:.4f}")
        logger.log_dict("   Best TDA Params", study.best_params)
        save_optuna_results_to_db(study, ds_name, f"{RUN_MODE}")

        # ── VALİDASYON ───────────────────────────────────────────────────────
        try:
            run_validation(trial_list, study.best_params,
                           ds_name, tda_label=f"{RUN_MODE}")
        except Exception as e:
            logger.log(f"   ❌ Validasyon hatası: {e}")

    conn.close()
    logger.log_section("✅ V15 TAMAMLANDI")
