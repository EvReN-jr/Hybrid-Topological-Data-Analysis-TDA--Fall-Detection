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
from sklearn.metrics import fbeta_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import pdist

# =============================================================================
# 1. KONFİGÜRASYON VE AYARLAR
# =============================================================================
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARN)

# Dosya Yolları
USER_DIR  = os.getcwd()
DATA_DB_PATH = os.path.join(USER_DIR, "All_Datasets_Container.db") 

# ÇIKTI DOSYALARI (YENİ)
RESULT_DB_PATH = os.path.join(USER_DIR, "Results_V23_Storage.db")
REPORT_FILE = os.path.join(USER_DIR, "Results_V23_Detailed_Report.txt")
OPTUNA_DB_PATH = os.path.join(USER_DIR, "study_v23_local.db")

CONFIG = {
    "TARGET_FS": 20,              
    "LIMIT_POINTS": 50,
    "N_TRIALS_GLOBAL": 15,
    
    # TDA Arama Uzayı
    "SEARCH_SPACE": {
        "win_sec": {"low": 2.0, "high": 4.0, "step": 1.0},
        "dim": {"low": 2, "high": 3},                      
        "delay": {"low": 1, "high": 2},                    
        "complex_type": ["Alpha", "Rips"],                 
        "metrics": ["euclidean", "cosine"],                
        "use_delay": [True, False]                         
    },
    
    # SVM Grid
    "SVM_GRID": {
        'svc__C': [0.1, 1, 10, 100],
        'svc__kernel': ['rbf', 'linear', 'sigmoid']
    }
}

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
# 2. LOGGER & DB YÖNETİCİSİ (YENİ)
# =============================================================================
class ExperimentLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        # Dosyayı sıfırla ve başlık at
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(f"V23 DETAILED EXPERIMENT REPORT\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")

    def log(self, message):
        print(message)
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    def log_section(self, title):
        self.log("\n" + "="*80)
        self.log(f" {title}")
        self.log("="*80)

    def log_dict(self, title, data_dict):
        self.log(f"\n--- {title} ---")
        self.log(json.dumps(data_dict, indent=4, default=str))

def save_features_to_db(X, y, subjects, ds_name):
    """ En iyi dönüşümü (X, y) veritabanına kaydeder """
    conn = sqlite3.connect(RESULT_DB_PATH)
    
    # DataFrame oluştur
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df['label'] = y
    df['subject'] = subjects
    
    table_name = f"features_{ds_name}"
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    return table_name

def save_metrics_to_db(metrics_list, ds_name):
    """ Sonuç metriklerini veritabanına kaydeder """
    if not metrics_list: return
    conn = sqlite3.connect(RESULT_DB_PATH)
    df = pd.DataFrame(metrics_list)
    table_name = f"metrics_{ds_name}"
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

# Logger Başlat
logger = ExperimentLogger(REPORT_FILE)

# =============================================================================
# 3. VERİ İŞLEME VE FEATURE ENGINE (AYNI)
# =============================================================================
def safe_resample(X, orig_fs, target_fs):
    n = len(X)
    target = int((n / orig_fs) * target_fs)
    if target < 5: return None
    old_t, new_t = np.linspace(0, 1, n), np.linspace(0, 1, target)
    X_new = interp1d(old_t, X, axis=0, kind='linear', fill_value="extrapolate")(new_t)
    return X_new

def fetch_data_full(conn, ds_config, ds_name):
    try:
        s_table = ds_config["sensor_table"]; t_table = ds_config["trials_table"]
        label_col = ds_config["label_col"]; subj_col = ds_config["subject_col"]
        all_cols = []
        for grp_cols in ds_config["sensor_groups"].values(): all_cols.extend(grp_cols)
        
        logger.log(f"   ⏳ Veri okunuyor: {ds_name}...")
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
                if len(X_cut) < win_len // 2: continue
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
        final_list = [t for t in trial_list if t['subject'] in valid_subjects]
        return final_list
    except Exception as e:
        logger.log(f"Hata: {e}"); return None

class Hybrid_Engine:
    def __init__(self, params): self.params = params
    def compute(self, X):
        win_sec = self.params['win_sec']
        target_len = int(win_sec * CONFIG['TARGET_FS'])
        if len(X) > target_len:
            mid = len(X) // 2
            start = mid - target_len // 2
            X = X[start : start+target_len]
        if self.params.get('use_delay'):
            embeddings = []
            for i in range(X.shape[1]):
                embeddings.append(self._embed(X[:, i], self.params['delay'], self.params['dim']))
            if not embeddings: return np.zeros(8)
            min_len = min(len(e) for e in embeddings)
            pc = np.hstack([e[:min_len] for e in embeddings])
        else: pc = X 
        limit = CONFIG["LIMIT_POINTS"]
        if len(pc) > limit:
            try: pc = np.array(gudhi.subsampling.choose_n_farthest_points(pc, limit, 0))
            except: pc = pc[np.linspace(0, len(pc)-1, limit, dtype=int)]
        try:
            if self.params['complex'] == 'Alpha' and pc.shape[1] <= 3: st = gudhi.AlphaComplex(points=pc).create_simplex_tree()
            else:
                d = pdist(pc, metric=self.params.get('metric', 'euclidean'))
                eps = np.percentile(d, 90) if len(d)>0 else 0.5
                st = gudhi.RipsComplex(points=pc, max_edge_length=eps).create_simplex_tree(max_dimension=2)
            st.persistence()
            tda_feats = self._feats(st, len(pc))
        except: tda_feats = np.zeros(8)
        acc = X[:, 0]
        stat_feats = np.array([np.max(acc), np.std(acc), np.mean(acc), np.max(acc) - np.min(acc)])
        return np.hstack([tda_feats, stat_feats])
    def _embed(self, s, d, dim):
        n = len(s); req = (dim-1)*d + 1
        return np.array([s[i:i+req:d] for i in range(n-req+1)]) if n >= req else np.zeros((1, dim))
    def _feats(self, st, n):
        f = []
        for dim in [0, 1]:
            i = st.persistence_intervals_in_dimension(dim)
            if len(i) == 0: f.extend([0, 0, 0, 0]); continue
            l = i[:, 1] - i[:, 0]; l = l[np.isfinite(l)]; 
            if len(l) == 0: f.extend([0, 0, 0, 0]); continue
            s = np.sum(l); p = l/s
            f.extend([len(l)/n, -np.sum(p*np.log(p+1e-10)), np.max(l), np.sum(l**2)/s])
        return np.array(f)

def get_xy_matrix(trial_list, params):
    eng = Hybrid_Engine(params)
    X, y, g = [], [], []
    for tr in trial_list:
        x_r = safe_resample(tr['X'], tr['fs'], CONFIG["TARGET_FS"])
        if x_r is None: continue
        x_sc = StandardScaler().fit_transform(x_r) 
        ft = eng.compute(x_sc)
        if not np.isnan(ft).any():
            X.append(ft); y.append(tr['has_fall']); g.append(tr['subject'])
    return np.array(X), np.array(y), np.array(g)

def create_global_subset(trial_list):
    falls = [t for t in trial_list if t['has_fall'] == 1]
    adls = [t for t in trial_list if t['has_fall'] == 0]
    subset = falls + random.sample(adls, min(len(adls), len(falls)*3))
    random.shuffle(subset)
    return subset

def balanced_undersample(X, y):
    idx0, idx1 = np.where(y==0)[0], np.where(y==1)[0]
    n0, n1 = len(idx0), len(idx1)
    if n1 == 0: return X, y
    target_n0 = min(n0, n1 * 3)
    if target_n0 < n0:
        choices0 = np.random.choice(idx0, target_n0, replace=False)
        return np.concatenate([X[choices0], X[idx1]]), np.concatenate([y[choices0], y[idx1]])
    return X, y

def find_best_threshold(model, X_val, y_val):
    try:
        probs = model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
        numerator = 5 * precisions * recalls
        denominator = 4 * precisions + recalls
        f2_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        best_idx = np.argmax(f2_scores)
        if best_idx < len(thresholds): return thresholds[best_idx]
        return 0.5
    except: return 0.5

# =============================================================================
# 5. OPTIMIZASYON VE VALİDASYON
# =============================================================================
def objective_global(trial, subset):
    sp = CONFIG["SEARCH_SPACE"]
    params = {
        'win_sec': trial.suggest_float("win_sec", sp["win_sec"]["low"], sp["win_sec"]["high"], step=sp["win_sec"]["step"]),
        'complex': trial.suggest_categorical("complex_type", sp["complex_type"]),
        'use_delay': trial.suggest_categorical("use_delay", sp["use_delay"])
    }
    if params['use_delay']:
        params['dim'] = trial.suggest_int("dim", sp["dim"]["low"], sp["dim"]["high"])
        params['delay'] = trial.suggest_int("delay", sp["delay"]["low"], sp["delay"]["high"])
    if params['complex'] == "Rips":
        params['metric'] = trial.suggest_categorical("metric", sp["metrics"])
    
    X, y, g = get_xy_matrix(subset, params)
    if len(X) < 10: return 0.0
    try:
        pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', class_weight='balanced', C=1.0))])
        cv = StratifiedKFold(n_splits=3)
        scores = []
        for train_ix, test_ix in cv.split(X, y):
            pipeline.fit(X[train_ix], y[train_ix])
            preds = pipeline.predict(X[test_ix])
            scores.append(fbeta_score(y[test_ix], preds, beta=2, zero_division=0))
        return np.mean(scores)
    except: return 0.0

def run_scientific_validation(trial_list, best_tda_params, ds_name):
    logger.log(f"   🔬 [{ds_name}] V23: Veri İşleme ve Validasyon...")
    
    # 1. EN İYİ FEATURE SETİNİ OLUŞTUR VE KAYDET
    X_all, y_all, sub_all = get_xy_matrix(trial_list, best_tda_params)
    unique_subjects = np.unique(sub_all)
    
    table_name = save_features_to_db(X_all, y_all, sub_all, ds_name)
    logger.log(f"      ✅ Özellik Matrisi DB'ye kaydedildi: {table_name}")
    logger.log(f"      ℹ️  Boyut: {X_all.shape}, TDA Params: {best_tda_params}")
    
    report_lines = [
        f"{'SUB':<6} | {'G-ACC':<6} {'G-REC':<6} {'G-PRE':<6} {'G-F1':<6} {'G-THR':<5} | {'P-ACC':<6} {'P-REC':<6} {'P-PRE':<6} {'P-F1':<6} {'P-THR':<5} | {'BEST PARAMS':<30}",
        "-"*160
    ]
    
    logo = LeaveOneGroupOut()
    
    # Veritabanına kaydedilecek metrik listesi
    db_metrics_rows = []
    
    # --- SENARYO 1: GENERAL MODEL ---
    logger.log("      -> General Model (LOSO) Analizi...")
    g_results = {} 
    
    for train_idx, test_idx in logo.split(X_all, y_all, groups=sub_all):
        test_sub = sub_all[test_idx][0]
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        
        X_tr_bal, y_tr_bal = balanced_undersample(X_train, y_train)
        pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(class_weight='balanced', probability=True))])
        search = GridSearchCV(pipeline, CONFIG["SVM_GRID"], cv=3, scoring='f1', n_jobs=1)
        search.fit(X_tr_bal, y_tr_bal)
        
        best_thr = find_best_threshold(search.best_estimator_, X_tr_bal, y_tr_bal)
        probs = search.best_estimator_.predict_proba(X_test)[:, 1]
        preds = (probs >= best_thr).astype(int)
        
        g_results[test_sub] = {
            'g_acc': accuracy_score(y_test, preds),
            'g_rec': recall_score(y_test, preds, zero_division=0),
            'g_pre': precision_score(y_test, preds, zero_division=0),
            'g_f1': f1_score(y_test, preds, zero_division=0),
            'g_thr': best_thr,
            'g_params': str(search.best_params_)
        }

    # --- SENARYO 2: PERSONAL MODEL ---
    logger.log("      -> Personal Model (Fine-Tuning) Analizi...")
    
    for s in unique_subjects:
        idx = (sub_all == s)
        X_sub, y_sub = X_all[idx], y_all[idx]
        
        # Row data for DB
        row_data = {'subject': str(s)}
        
        # General Sonuçları Ekle
        if s in g_results:
            for k, v in g_results[s].items(): row_data[k] = v
        
        if np.sum(y_sub==1) < 2:
            row_data['note'] = "Yetersiz Veri"
            db_metrics_rows.append(row_data)
            report_lines.append(f"{str(s):<6} | Yetersiz Veri")
            continue
            
        X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.4, stratify=y_sub, random_state=42)
        X_tr_bal, y_tr_bal = balanced_undersample(X_train, y_train)
        
        pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(class_weight='balanced', probability=True))])
        
        try:
            cv_inner = StratifiedKFold(n_splits=2)
            search = GridSearchCV(pipeline, CONFIG["SVM_GRID"], cv=cv_inner, scoring='f1', n_jobs=1)
            search.fit(X_tr_bal, y_tr_bal)
            
            best_thr = find_best_threshold(search.best_estimator_, X_tr_bal, y_tr_bal)
            probs = search.best_estimator_.predict_proba(X_test)[:, 1]
            preds = (probs >= best_thr).astype(int)
            
            # Personal Metrics
            p_metrics = {
                'p_acc': accuracy_score(y_test, preds),
                'p_rec': recall_score(y_test, preds, zero_division=0),
                'p_pre': precision_score(y_test, preds, zero_division=0),
                'p_f1': f1_score(y_test, preds, zero_division=0),
                'p_thr': best_thr,
                'p_params': str(search.best_params_).replace("svc__", "")
            }
            row_data.update(p_metrics)
            
            report_lines.append(
                f"{str(s):<6} | "
                f"{row_data.get('g_acc',0):.4f} {row_data.get('g_rec',0):.4f} {row_data.get('g_pre',0):.4f} {row_data.get('g_f1',0):.4f} {row_data.get('g_thr',0):.2f} | "
                f"{row_data.get('p_acc',0):.4f} {row_data.get('p_rec',0):.4f} {row_data.get('p_pre',0):.4f} {row_data.get('p_f1',0):.4f} {row_data.get('p_thr',0):.2f} | "
                f"{p_metrics['p_params']}"
            )
            
        except Exception as e:
            row_data['note'] = f"Error: {str(e)}"
            report_lines.append(f"{str(s):<6} | Error: {str(e)}")
            
        db_metrics_rows.append(row_data)

    # DB'ye Kaydet
    save_metrics_to_db(db_metrics_rows, ds_name)
    logger.log(f"      ✅ Metrikler DB'ye kaydedildi: metrics_{ds_name}")
    
    return "\n".join(report_lines)

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(DATA_DB_PATH): sys.exit("DB Yok")
    conn = sqlite3.connect(DATA_DB_PATH)
    
    logger.log_section("V23 START")
    logger.log_dict("GLOBAL CONFIGURATION", CONFIG)
    
    for ds_name, cfg in DATASETS_CONFIG.items():
        logger.log_section(f"DATASET: {ds_name}")
        trial_list = fetch_data_full(conn, cfg, ds_name)
        if not trial_list: continue
        
        logger.log("   🔍 Global Optuna Optimizasyonu...")
        subset = create_global_subset(trial_list)
        study = optuna.create_study(storage=f"sqlite:///{OPTUNA_DB_PATH}", study_name=f"{ds_name}_v23", direction="maximize", load_if_exists=True)
        
        if len(study.trials) < CONFIG["N_TRIALS_GLOBAL"]:
            study.optimize(lambda t: objective_global(t, subset), n_trials=CONFIG["N_TRIALS_GLOBAL"], n_jobs=1)
            
        logger.log_dict(f"BEST TDA PARAMS ({ds_name})", study.best_params)
        
        rep = run_scientific_validation(trial_list, study.best_params, ds_name)
        logger.log("\n" + rep)
        
    conn.close()
    logger.log_section("COMPLETED")
    print(f"\n✅ BİTTİ.\n📂 Sonuç DB: {RESULT_DB_PATH}\n📄 Detaylı Rapor: {REPORT_FILE}")
