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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, LeaveOneGroupOut, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, accuracy_score, recall_score, precision_score, f1_score, precision_recall_curve, make_scorer
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import pdist

# =============================================================================
# 1. CONFIGURATION (V11: FINAL SAFE BEAST MODE WITH F2)
# =============================================================================
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARN)

USER_DIR = os.getcwd()
DATA_DB_PATH = os.path.join(USER_DIR, "All_Datasets_Container.db") 
RESULT_DB_PATH = os.path.join(USER_DIR, "Results_V11_Smart_F2.db")
REPORT_FILE = os.path.join(USER_DIR, "Results_V11_Smart_F2_Report.txt")
OPTUNA_DB_PATH = os.path.join(USER_DIR, "study_v11_smart_f2.db")

CONFIG = {
    # --- AYARLAR ---
    "TARGET_FS": 50,            
    "LIMIT_POINTS": 150,        
    "N_TRIALS_GLOBAL": 200,     
    
    # --- İŞLEMCİ GÜCÜ AYARLARI (KRİTİK DÜZELTME) ---
    # Optuna SQLite kullandığı için paralel yazamaz, bu yüzden 1.
    # Modeller CPU'da eğitildiği için 15 çekirdek serbest.
    "OPTUNA_JOBS": 1,   
    "MODEL_JOBS": 15,    
    
    # --- AKILLI TDA UZAYI ---
    "SEARCH_SPACE": {
        "win_sec": {"low": 1, "high": 5.0, "step": 0.5},
        "dim": {"low": 2, "high": 5},
        "delay": {"low": 1, "high": 5},
        "stride_factor": [1, 2, 4],
        "complex_type": ["Alpha", "Rips", "SparseRips"],
        "metrics": ["euclidean", "cosine", "manhattan", "chebyshev"],
        "eps_percentile": [20, 40, 60, 80], 
        "use_delay": [True],
        "sampling_method": ["maxmin"]
    },
    
    # --- MODELLER ---
    "MODELS": {
        "LogReg": { 
            "estimator": LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=2000),
            "params": {
                'logreg__C': [0.01, 0.1, 1, 10, 100], 
                'logreg__penalty': ['l1', 'l2']
            }
        },
        "SVM": { 
            "estimator": SVC(class_weight='balanced', probability=True, cache_size=2000), 
            "params": {
                'svm__C': [0.1, 1, 10, 100],
                'svm__kernel': ['rbf', 'linear'], 
                'svm__gamma': ['scale']
            }
        }
    }
}

# F2 SCORER (Recall'a 2 kat önem verir)
f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

DATASETS_CONFIG = {
    "MobiFall": {
        "sensor_table": "MobiFall_sensor_data", "trials_table": "MobiFall_trials",
        "subject_col": "subject_id", "label_col": "is_fall", "default_fs": 87,
        "sensor_groups": {"Acc": ["acc_x", "acc_y", "acc_z"], "Gyr": ["gyr_x", "gyr_y", "gyr_z"], "Ori": ["ori_azimuth", "ori_pitch", "ori_roll"]}
    },
    "SisFall": {
        "sensor_table": "SisFall_sensor_data", "trials_table": "SisFall_trials",
        "subject_col": "subject_id", "label_col": "is_fall", "default_fs": 200,
        "sensor_groups": {"Acc_ADXL": ["acc_adxl345_x", "acc_adxl345_y", "acc_adxl345_z"], "Gyr_ITG":  ["gyr_itg3200_x", "gyr_itg3200_y", "gyr_itg3200_z"]}
    },
     "FAD_40Hz": {
        "sensor_table": "FAD_40Hz_sensor_data", "trials_table": "FAD_40Hz_trials",
        "subject_col": "subject_id", "label_col": "is_fall", "default_fs": 40,
        "sensor_groups": {"Acc": ["acc_x", "acc_y", "acc_z"], "Gyr": ["gyr_x", "gyr_y", "gyr_z"]}
    }
}

# =============================================================================
# 2. LOGGER & DB MANAGER
# =============================================================================
class ExperimentLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(f"V11 SMART BEAST EXPERIMENT REPORT (F2 PRIORITY)\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")

    def log(self, message):
        print(message)
        sys.stdout.flush() 
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    def log_section(self, title):
        self.log("\n" + "="*80); self.log(f" {title}"); self.log("="*80)

    def log_dict(self, title, data_dict):
        self.log(f"\n--- {title} ---")
        self.log(json.dumps(data_dict, indent=4, default=str))

def save_features_to_db(X, y, subjects, ds_name):
    conn = sqlite3.connect(RESULT_DB_PATH)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df['label'] = y; df['subject'] = subjects
    table_name = f"features_{ds_name}"
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    return table_name

def save_metrics_to_db(metrics_list, ds_name):
    if not metrics_list: return
    conn = sqlite3.connect(RESULT_DB_PATH)
    pd.DataFrame(metrics_list).to_sql(f"metrics_{ds_name}", conn, if_exists='replace', index=False)
    conn.close()

def save_optuna_results_to_db(study, ds_name):
    conn = sqlite3.connect(RESULT_DB_PATH)
    trials_data = []
    for trial in study.trials:
        trial_dict = {
            'trial_number': trial.number,
            'f2_score': trial.value,
            'state': str(trial.state),
            **trial.params
        }
        trials_data.append(trial_dict)
    
    if trials_data:
        pd.DataFrame(trials_data).to_sql(f"optuna_trials_{ds_name}", conn, if_exists='replace', index=False)
    
    best_params_df = pd.DataFrame([{
        'dataset': ds_name,
        'best_f2_score': study.best_value,
        **study.best_params
    }])
    best_params_df.to_sql(f"optuna_best_params_{ds_name}", conn, if_exists='replace', index=False)
    conn.close()

logger = ExperimentLogger(REPORT_FILE)

# =============================================================================
# 3. DATA PROCESSING
# =============================================================================
def safe_resample(X, orig_fs, target_fs):
    n = len(X); target = int((n / orig_fs) * target_fs)
    if target < 5: return None
    old_t, new_t = np.linspace(0, 1, n), np.linspace(0, 1, target)
    return interp1d(old_t, X, axis=0, kind='linear', fill_value="extrapolate")(new_t)

def fetch_data_full(conn, ds_config, ds_name):
    try:
        s_table = ds_config["sensor_table"]; t_table = ds_config["trials_table"]
        label_col = ds_config["label_col"]; subj_col = ds_config["subject_col"]
        all_cols = []
        for grp_cols in ds_config["sensor_groups"].values(): all_cols.extend(grp_cols)
        
        logger.log(f"   ⏳ Loading data: {ds_name}...")
        query = f"SELECT s.trial_id, t.{subj_col} as subj_id, s.{label_col}, t.sampling_rate, {', '.join(['s.'+c for c in all_cols])} FROM {s_table} s LEFT JOIN {t_table} t ON s.trial_id = t.trial_id ORDER BY s.trial_id, s.timestamp"
        df = pd.read_sql_query(query, conn)
        if df.empty: return None

        smv_list = []; acc_smv = None
        for grp_name, grp_cols in ds_config["sensor_groups"].items():
            sub_data = df[grp_cols].values
            smv = np.sqrt(np.sum(sub_data**2, axis=1))
            smv_list.append(smv)
            if "Acc" in grp_name or "acc" in grp_cols[0]: acc_smv = smv
            
        X_multi = np.column_stack(smv_list)
        y = df[label_col].values; trials = df['trial_id'].values; subjects = df['subj_id'].values
        fs_vals = df['sampling_rate'].fillna(ds_config['default_fs']).values
        if acc_smv is None: acc_smv = smv_list[0]
        del df, smv_list; gc.collect()
        
        trial_list, subj_stats = [], {}
        unique_trials = np.unique(trials)
        max_win_sec = CONFIG["SEARCH_SPACE"]["win_sec"]["high"]
        
        for tid in unique_trials:
            mask = (trials == tid)
            X_t = X_multi[mask]; acc_t = acc_smv[mask]; y_t = y[mask]
            subj_t = subjects[mask][0] if any(pd.notna(subjects[mask])) else tid
            fs_t = np.nanmedian(fs_vals[mask])
            if len(X_t) < 10: continue
            has_fall = 1 if np.sum(y_t) > 0 else 0
            win_len = int(max_win_sec * fs_t)
            
            if has_fall: 
                peak_idx = np.argmax(acc_t)
                start = max(0, peak_idx - win_len // 2)
                end = min(len(X_t), peak_idx + win_len // 2)
                X_cut = X_t[start:end]
                if len(X_cut) < win_len // 4: continue
                if subj_t not in subj_stats: subj_stats[subj_t] = {'fall': 0, 'adl': 0}
                subj_stats[subj_t]['fall'] += 1
                trial_list.append({'id': tid, 'subject': subj_t, 'X': X_cut, 'fs': fs_t, 'has_fall': 1})
            else: 
                step = win_len
                for i in range(0, len(X_t) - win_len, step):
                    X_cut = X_t[i : i+win_len]
                    if subj_t not in subj_stats: subj_stats[subj_t] = {'fall': 0, 'adl': 0}
                    subj_stats[subj_t]['adl'] += 1
                    trial_list.append({'id': tid, 'subject': subj_t, 'X': X_cut, 'fs': fs_t, 'has_fall': 0})

        valid_subjects = [s for s, stats in subj_stats.items() if stats['fall'] > 1 and stats['adl'] > 1]
        return [t for t in trial_list if t['subject'] in valid_subjects]
    except Exception as e: logger.log(f"Error: {e}"); return None

# =============================================================================
# 4. ULTIMATE HYBRID ENGINE
# =============================================================================
class Ultimate_Hybrid_Engine:
    def __init__(self, params): self.params = params
    
    def compute(self, X):
        win_sec = self.params['win_sec']
        target_len = int(win_sec * CONFIG['TARGET_FS'])
        if len(X) > target_len:
            start = (len(X) // 2) - (target_len // 2)
            X = X[start : start+target_len]
        
        # 1. EMBEDDING
        if self.params.get('use_delay'):
            embeddings = []
            stride = self.params.get('stride_factor', 1)
            for i in range(X.shape[1]):
                embeddings.append(self._embed(X[:, i], self.params['delay'], self.params['dim'], stride))
            if not embeddings: return np.zeros(24)
            min_len = min(len(e) for e in embeddings)
            full_point_cloud = np.hstack([e[:min_len] for e in embeddings])
        else: full_point_cloud = X 
        
        # 2. LANDMARK SELECTION
        limit = CONFIG["LIMIT_POINTS"]
        landmarks = full_point_cloud
        
        if len(full_point_cloud) > limit:
            sampling_method = self.params.get('sampling_method', 'maxmin')
            try:
                if sampling_method == 'maxmin':
                    landmarks = np.array(gudhi.subsampling.choose_n_farthest_points(points=full_point_cloud, nb_points=limit))
                else:
                    indices = np.random.choice(len(full_point_cloud), limit, replace=False)
                    landmarks = full_point_cloud[indices]
            except:
                indices = np.linspace(0, len(full_point_cloud)-1, limit, dtype=int)
                landmarks = full_point_cloud[indices]
        
        # 3. TDA EXTRACTION
        try: tda_feats = self._compute_tda_features(landmarks)
        except: tda_feats = np.zeros(20)
        
        acc = X[:, 0]
        stat_feats = np.array([np.max(acc), np.std(acc), np.mean(acc), np.max(acc) - np.min(acc)])
        return np.hstack([tda_feats, stat_feats])
    
    def _embed(self, signal, delay, dim, stride):
        n = len(signal); req = (dim - 1) * delay + 1
        if n < req: return np.zeros((1, dim))
        max_points = 200; possible_points = (n - req) // stride + 1
        if possible_points > max_points: stride = max(stride, (n - req) // max_points)
        embeddings = []
        for i in range(0, n - req + 1, stride): embeddings.append(signal[i:i+req:delay])
        return np.array(embeddings)
    
    def _compute_tda_features(self, landmarks):
        complex_type = self.params['complex_type']
        if not isinstance(landmarks, np.ndarray): landmarks = np.array(landmarks)
        st = None
        
        if complex_type == 'Alpha' and landmarks.shape[1] <= 3:
            try: st = gudhi.AlphaComplex(points=landmarks).create_simplex_tree()
            except: complex_type = 'Rips'
            
        if complex_type == 'SparseRips':
            try:
                st = gudhi.SparseRipsComplex(points=landmarks).create_simplex_tree(max_dimension=2)
            except: complex_type = 'Rips'

        if complex_type == 'Rips' or st is None:
            metric = self.params.get('metric', 'euclidean')
            try: d = pdist(landmarks, metric=metric)
            except: d = pdist(landmarks, metric='euclidean')
            eps = np.percentile(d, self.params.get('eps_percentile', 70)) if len(d) > 0 else 0.5
            st = gudhi.RipsComplex(points=landmarks, max_edge_length=eps).create_simplex_tree(max_dimension=2)
        
        st.persistence()
        return self._extract_features(st, len(landmarks))
    
    def _extract_features(self, st, n_points):
        features = []
        for dim in [0, 1]:
            intervals = st.persistence_intervals_in_dimension(dim)
            if len(intervals) == 0: features.extend([0] * 10); continue
            lifetimes = intervals[:, 1] - intervals[:, 0]
            lifetimes = lifetimes[np.isfinite(lifetimes)]
            if len(lifetimes) == 0: features.extend([0] * 10); continue
            total = np.sum(lifetimes); probs = lifetimes / total
            features.extend([
                len(lifetimes) / n_points, -np.sum(probs * np.log(probs + 1e-10)),
                np.max(lifetimes), np.sum(lifetimes**2) / total, np.mean(lifetimes),
                np.std(lifetimes), np.median(lifetimes), np.percentile(lifetimes, 25),
                np.percentile(lifetimes, 75), np.sum(lifetimes**3) / (total + 1e-10)
            ])
        return np.array(features)

def get_xy_matrix(trial_list, params):
    eng = Ultimate_Hybrid_Engine(params)
    X, y, g = [], [], []
    for tr in trial_list:
        x_r = safe_resample(tr['X'], tr['fs'], CONFIG["TARGET_FS"])
        if x_r is None: continue
        x_sc = StandardScaler().fit_transform(x_r) 
        ft = eng.compute(x_sc)
        if not np.isnan(ft).any() and not np.isinf(ft).any():
            X.append(ft); y.append(tr['has_fall']); g.append(tr['subject'])
    return np.array(X), np.array(y), np.array(g)

def create_global_subset(trial_list):
    falls = [t for t in trial_list if t['has_fall'] == 1]
    adls = [t for t in trial_list if t['has_fall'] == 0]
    subset = falls + random.sample(adls, min(len(adls), len(falls) * 3))
    random.shuffle(subset)
    return subset

def balanced_undersample(X, y):
    idx0, idx1 = np.where(y==0)[0], np.where(y==1)[0]
    if len(idx1) == 0: return X, y
    target_n0 = min(len(idx0), len(idx1) * 3)
    if target_n0 < len(idx0):
        choices0 = np.random.choice(idx0, target_n0, replace=False)
        return np.concatenate([X[choices0], X[idx1]]), np.concatenate([y[choices0], y[idx1]])
    return X, y

def find_best_threshold(model, X_val, y_val):
    try:
        probs = model.predict_proba(X_val)[:, 1]
        p, r, t = precision_recall_curve(y_val, probs)
        # F2-Score based thresholding
        f2 = np.divide(5*p*r, 4*p+r, out=np.zeros_like(p), where=(4*p+r)!=0)
        return t[np.argmax(f2)] if len(t) > 0 else 0.5
    except: return 0.5

# =============================================================================
# 5. OBJECTIVE & VALIDATION
# =============================================================================
def objective_global(trial, subset):
    sp = CONFIG["SEARCH_SPACE"]
    params = {
        'win_sec': trial.suggest_float("win_sec", sp["win_sec"]["low"], sp["win_sec"]["high"], step=sp["win_sec"]["step"]),
        'complex_type': trial.suggest_categorical("complex_type", sp["complex_type"]),
        'use_delay': trial.suggest_categorical("use_delay", sp["use_delay"]),
        'sampling_method': trial.suggest_categorical("sampling_method", sp["sampling_method"])
    }
    if params['use_delay']:
        params['dim'] = trial.suggest_int("dim", sp["dim"]["low"], sp["dim"]["high"])
        params['delay'] = trial.suggest_int("delay", sp["delay"]["low"], sp["delay"]["high"])
        params['stride_factor'] = trial.suggest_categorical("stride_factor", sp["stride_factor"])
    if params['complex_type'] in ["Rips", "SparseRips"]:
        params['metric'] = trial.suggest_categorical("metric", sp["metrics"])
        params['eps_percentile'] = trial.suggest_categorical("eps_percentile", sp["eps_percentile"])
    
    X, y, g = get_xy_matrix(subset, params)
    if len(X) < 10: return 0.0
    
    try:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=200))
        ])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_ix, test_ix in cv.split(X, y):
            X_tr, y_tr = balanced_undersample(X[train_ix], y[train_ix])
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X[test_ix])
            scores.append(fbeta_score(y[test_ix], y_pred, beta=2, zero_division=0))
        return np.mean(scores)
    except: return 0.0

def run_scientific_validation(trial_list, best_tda_params, ds_name):
    logger.log(f"   🔬 [{ds_name}] V11 SMART VALIDATION BAŞLIYOR...")
    
    # 1. Özellik Çıkarımı
    try:
        X_all, y_all, sub_all = get_xy_matrix(trial_list, best_tda_params)
        unique_subjects = np.unique(sub_all)
        table_name = save_features_to_db(X_all, y_all, sub_all, ds_name)
        logger.log(f"      ✅ Özellikler DB'ye yazıldı: {table_name} (Boyut: {X_all.shape})")
    except Exception as e:
        logger.log(f"      ❌ Özellik çıkarımında kritik hata: {e}")
        return 

    logo = LeaveOneGroupOut()
    db_metrics_rows = []
    
    for model_name, model_cfg in CONFIG["MODELS"].items():
        logger.log(f"\n      >>> MODEL: {model_name} ({ds_name})")
        
        header = (
            f"{'SUB':<6} | "
            f"{'G-ACC':<6} {'G-REC':<6} {'G-PRE':<6} {'G-F1':<6} {'G-F2':<6} {'G-THR':<5} | "
            f"{'P-ACC':<6} {'P-REC':<6} {'P-PRE':<6} {'P-F1':<6} {'P-F2':<6} {'P-THR':<5} | "
            f"{'BEST PARAMS'}"
        )
        logger.log("-" * 170)
        logger.log(header)
        logger.log("-" * 170)
        
        g_results = {}

        # --- A. GENERAL MODEL (LOGO) ---
        try:
            for train_idx, test_idx in logo.split(X_all, y_all, groups=sub_all):
                test_sub = sub_all[test_idx][0]
                X_tr, y_tr = balanced_undersample(X_all[train_idx], y_all[train_idx])
                
                pipe = Pipeline([('scaler', StandardScaler()), (model_name.lower(), model_cfg['estimator'])])
                # Güvenli Mod: CPU gücünü burada kullanıyoruz (15)
                search = GridSearchCV(pipe, model_cfg['params'], cv=3, scoring=f2_scorer, n_jobs=CONFIG["MODEL_JOBS"]) 
                search.fit(X_tr, y_tr)
                best = search.best_estimator_
                
                thr = find_best_threshold(best, X_tr, y_tr)
                preds = (best.predict_proba(X_all[test_idx])[:, 1] >= thr).astype(int)
                
                clean_params = str(search.best_params_).replace("svm__", "").replace("logreg__", "")
                
                g_results[test_sub] = {
                    'g_acc': accuracy_score(y_all[test_idx], preds),
                    'g_rec': recall_score(y_all[test_idx], preds, zero_division=0),
                    'g_pre': precision_score(y_all[test_idx], preds, zero_division=0),
                    'g_f1': f1_score(y_all[test_idx], preds, zero_division=0),
                    'g_f2': fbeta_score(y_all[test_idx], preds, beta=2, zero_division=0),
                    'g_thr': thr,
                    'g_params': clean_params
                }
        except Exception as e:
            logger.log(f"      ❌ General Model döngüsünde hata: {e}")

        # --- B. PERSONAL MODEL & RAPORLAMA ---
        for s in unique_subjects:
            try:
                idx = (sub_all == s)
                X_sub, y_sub = X_all[idx], y_all[idx]
                gr = g_results.get(s, {})
                row_data = {'subject': str(s), 'model': model_name, **gr}
                
                if np.sum(y_sub==1) < 2 or np.sum(y_sub==0) < 2:
                    logger.log(f"{str(s):<6} | Yetersiz Veri")
                    continue
                
                X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.4, stratify=y_sub, random_state=42)
                X_tr_p, y_tr_p = balanced_undersample(X_train, y_train)
                
                pipe_p = Pipeline([('scaler', StandardScaler()), (model_name.lower(), model_cfg['estimator'])])
                # Güvenli Mod: CPU gücünü burada kullanıyoruz (15)
                search_p = GridSearchCV(pipe_p, model_cfg['params'], cv=2, scoring=f2_scorer, n_jobs=CONFIG["MODEL_JOBS"])
                search_p.fit(X_tr_p, y_tr_p)
                best_p = search_p.best_estimator_
                
                thr_p = find_best_threshold(best_p, X_tr_p, y_tr_p)
                preds_p = (best_p.predict_proba(X_test)[:, 1] >= thr_p).astype(int)
                
                clean_params_p = str(search_p.best_params_).replace("svm__", "").replace("logreg__", "")
                
                p_metrics = {
                    'p_acc': accuracy_score(y_test, preds_p),
                    'p_rec': recall_score(y_test, preds_p, zero_division=0),
                    'p_pre': precision_score(y_test, preds_p, zero_division=0),
                    'p_f1': f1_score(y_test, preds_p, zero_division=0),
                    'p_f2': fbeta_score(y_test, preds_p, beta=2, zero_division=0),
                    'p_thr': thr_p,
                    'p_params': clean_params_p
                }
                
                row_data.update(p_metrics)
                db_metrics_rows.append(row_data)
                
                log_line = (
                    f"{str(s):<6} | "
                    f"{gr.get('g_acc',0):.4f} {gr.get('g_rec',0):.4f} {gr.get('g_pre',0):.4f} {gr.get('g_f1',0):.4f} {gr.get('g_f2',0):.4f} {gr.get('g_thr',0):.2f} | "
                    f"{p_metrics['p_acc']:.4f} {p_metrics['p_rec']:.4f} {p_metrics['p_pre']:.4f} {p_metrics['p_f1']:.4f} {p_metrics['p_f2']:.4f} {p_metrics['p_thr']:.2f} | "
                    f"{p_metrics['p_params']}"
                )
                logger.log(log_line)

            except Exception as e:
                logger.log(f"{str(s):<6} | HATA: {e}")
                continue 
                
    save_metrics_to_db(db_metrics_rows, ds_name)
    logger.log(f"   ✅ {ds_name} Tamamlandı.")

if __name__ == "__main__":
    if not os.path.exists(DATA_DB_PATH): 
        logger.log("⚠️ UYARI: DB Bulunamadı!")

    conn = sqlite3.connect(DATA_DB_PATH)
    logger.log_section("V11 SMART BEAST - FINAL VERSION START (F2 OPTIMIZED)")
    logger.log_dict("CONFIG", CONFIG)
    
    for ds_name, cfg in DATASETS_CONFIG.items():
        logger.log_section(f"DATASET: {ds_name}")
        trial_list = fetch_data_full(conn, cfg, ds_name)
        
        if not trial_list: 
            logger.log(f"❌ {ds_name} için veri bulunamadı.")
            continue
        
        logger.log(f"   🔍 Optuna Başlıyor (Hedef: {CONFIG['N_TRIALS_GLOBAL']} trials, Optuna Jobs={CONFIG['OPTUNA_JOBS']})...")
        subset = create_global_subset(trial_list)
        
        storage_url = f"sqlite:///{OPTUNA_DB_PATH}"
        study = optuna.create_study(storage=storage_url, study_name=f"{ds_name}_v11", direction="maximize", load_if_exists=True)
        
        remaining_trials = CONFIG["N_TRIALS_GLOBAL"] - len(study.trials)
        if remaining_trials > 0:
            logger.log(f"   🚀 Optimizasyon çalışıyor... (Kalan: {remaining_trials})")
            try:
                # KRİTİK: SQLite kilitlemesini önlemek için Optuna n_jobs=1
                study.optimize(lambda t: objective_global(t, subset), n_trials=remaining_trials, n_jobs=CONFIG["OPTUNA_JOBS"], gc_after_trial=True)
            except Exception as e:
                logger.log(f"⚠️ Optuna hatası: {e}")
        
        logger.log(f"\n   🏆 BEST F2-SCORE: {study.best_value:.4f}")
        logger.log_dict(f"   Best Parameters", study.best_params)
        save_optuna_results_to_db(study, ds_name)
        
        try:
            run_scientific_validation(trial_list, study.best_params, ds_name)
        except Exception as e:
            logger.log(f"❌ Validasyon hatası: {e}")
        
    conn.close()
    logger.log_section("✅ SERVER MISSION COMPLETED")