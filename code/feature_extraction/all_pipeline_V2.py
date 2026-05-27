import sqlite3
import pandas as pd
import numpy as np
import os
import gc
import gudhi
import gudhi.representations
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. AYARLAR & SABİTLER
# =============================================================================
BASE_DB_PATH = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataFirstTouch\All_Datasets_Container.db"
OUTPUT_ROOT = r"C:\Users\user\Desktop\KEB\YL_TEZ\TDA_Features_Extraction_V2\V2_TDA_Features"

# =============================================================================
# 2. TDA MOTORU V2.6 (AKILLI DELAY)
# =============================================================================
class TDA_Engine_V2_Pure:
    def __init__(self, target_fs, window_sec):
        self.fs = target_fs
        self.window_sec = window_sec
        self.embedding_dim = 3
        
        # --- AKILLI GECİKME SEÇİMİ ---
        # Kısa pencereler (FD_Ori, FAD_Master) için MİKRO Delay
        if self.window_sec <= 1.5:
            self.delay_seconds = [0.05, 0.10, 0.15, 0.20]
            self.mode = "MICRO (Short Window)"
        else:
            # Standart (SisFall, MobiFall) için NORMAL Delay
            self.delay_seconds = [0.12, 0.24, 0.36, 0.48]
            self.mode = "STANDARD (Long Window)"
            
        # TDA Araçları
        self.betti = gudhi.representations.BettiCurve(resolution=50)
        self.landscape = gudhi.representations.Landscape(resolution=50, num_landscapes=1)

    def _get_delay_embedding(self, signal, delay_pts):
        N = len(signal)
        dim = self.embedding_dim
        required_len = (dim - 1) * delay_pts
        if N <= required_len: return None
        
        num_points = N - required_len
        embedded = np.empty((num_points, dim), dtype=np.float64) 
        for i in range(dim):
            embedded[:, i] = signal[i * delay_pts : i * delay_pts + num_points]
        return embedded

    def _compute_diagram_features(self, persistence_intervals):
        keys = ['W1', 'W_inf', 'Entropy', 'Betti_L1', 'Betti_L2', 'Land_L1', 'Land_L2']
        feats = {k: 0.0 for k in keys}
        if len(persistence_intervals) == 0: return feats
        finite_intervals = persistence_intervals[np.isfinite(persistence_intervals[:, 1])]
        if len(finite_intervals) == 0: return feats

        lifetimes = finite_intervals[:, 1] - finite_intervals[:, 0]
        feats['W1'] = np.sum(lifetimes) / np.sqrt(2)
        feats['W_inf'] = np.max(lifetimes) / np.sqrt(2)
        
        total_life = np.sum(lifetimes)
        if total_life > 1e-10:
            probs = lifetimes / total_life
            feats['Entropy'] = -np.sum(probs * np.log(probs + 1e-10))
            
        diags_format = [finite_intervals]
        try:
            betti_vals = self.betti.fit_transform(diags_format)[0]
            feats['Betti_L1'] = np.linalg.norm(betti_vals, ord=1)
            feats['Betti_L2'] = np.linalg.norm(betti_vals, ord=2)
        except: pass

        try:
            land_vals = self.landscape.fit_transform(diags_format)[0]
            feats['Land_L1'] = np.linalg.norm(land_vals, ord=1)
            feats['Land_L2'] = np.linalg.norm(land_vals, ord=2)
        except: pass
            
        return feats

    def extract_all(self, window_data):
        # Magnitude Hesapla
        if window_data.ndim > 1 and window_data.shape[1] >= 3:
            signal = np.linalg.norm(window_data[:, :3], axis=1)
        else:
            if window_data.size > 0: signal = np.abs(window_data.flatten())
            else: return self._get_default_features()
        
        all_features = {}

        # Delay Döngüsü
        for sec in self.delay_seconds:
            delay_pts = max(1, int(sec * self.fs))
            point_cloud = self._get_delay_embedding(signal, delay_pts)
            
            # Default değerler (0.0)
            for dim in [0, 1]:
                prefix = f"D_{sec:.2f}s_H{dim}"
                for k in ['W1', 'W_inf', 'Entropy', 'Betti_L1', 'Betti_L2', 'Land_L1', 'Land_L2']:
                    all_features[f"{prefix}_{k}"] = 0.0

            if point_cloud is not None and len(point_cloud) > 4:
                try:
                    alpha = gudhi.AlphaComplex(points=point_cloud)
                    st = alpha.create_simplex_tree()
                    st.persistence()
                    for dim in [0, 1]:
                        intervals = st.persistence_intervals_in_dimension(dim)
                        dim_feats = self._compute_diagram_features(intervals)
                        prefix = f"D_{sec:.2f}s_H{dim}"
                        for k, v in dim_feats.items():
                            all_features[f"{prefix}_{k}"] = v
                except: pass

        # Level Sets (Ekstra Topoloji)
        for ls_type in ['Lower', 'Upper']:
            prefix = f"LS_{ls_type}_H0"
            for k in ['W1', 'W_inf', 'Entropy', 'Betti_L1', 'Betti_L2', 'Land_L1', 'Land_L2']:
                all_features[f"{prefix}_{k}"] = 0.0
            try:
                filt_signal = signal if ls_type == 'Lower' else -signal
                cc = gudhi.CubicalComplex(top_dimensional_cells=filt_signal)
                cc.persistence()
                intervals = cc.persistence_intervals_in_dimension(0)
                ls_feats = self._compute_diagram_features(intervals)
                for k, v in ls_feats.items():
                    all_features[f"{prefix}_{k}"] = v
            except: pass
                    
        return all_features
    
    def _get_default_features(self):
        all_features = {}
        keys = ['W1', 'W_inf', 'Entropy', 'Betti_L1', 'Betti_L2', 'Land_L1', 'Land_L2']
        for sec in self.delay_seconds:
            for dim in [0, 1]:
                prefix = f"D_{sec:.2f}s_H{dim}"
                for k in keys: all_features[f"{prefix}_{k}"] = 0.0
        for ls_type in ['Lower', 'Upper']:
            prefix = f"LS_{ls_type}_H0"
            for k in keys: all_features[f"{prefix}_{k}"] = 0.0
        return all_features

# =============================================================================
# 3. YARDIMCI FONKSİYONLAR
# =============================================================================
def setup_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def check_db_columns(conn, table, candidates):
    try:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        exist = {row[1] for row in cursor.fetchall()}
        valid = []
        for c in candidates:
            if c in exist: valid.append(c)
            else:
                if "adx1345" in c and c.replace("adx1345", "adxl345") in exist: valid.append(c.replace("adx1345", "adxl345"))
                elif "adxl345" in c and c.replace("adxl345", "adx1345") in exist: valid.append(c.replace("adxl345", "adx1345"))
        return valid
    except: return []

def universal_subwindowing(signal_data, meta_data, original_fs, target_fs, window_sec, stride_sec):
    if original_fs <= 0: step = 1
    else: step = max(1, int(original_fs / target_fs))
    
    # Sinyali örnekle
    resampled_sig = signal_data[::step]
    # Meta veriyi (Device Ori) aynı oranda örnekle
    resampled_meta = meta_data[::step] if meta_data is not None else None
    
    win_len = int(window_sec * target_fs)
    stride_len = int(stride_sec * target_fs)
    
    if len(resampled_sig) < win_len: return None, None
    
    # Pencerele
    sig_windows = [resampled_sig[i:i + win_len] for i in range(0, len(resampled_sig) - win_len + 1, stride_len)]
    
    meta_windows = None
    if resampled_meta is not None:
        meta_windows = [resampled_meta[i:i + win_len] for i in range(0, len(resampled_meta) - win_len + 1, stride_len)]
        
    return sig_windows, meta_windows

def save_to_disk(buffer, path):
    df = pd.DataFrame(buffer)
    # Meta Sütun Sıralaması (Device_Ori'yi başa yakın tut)
    meta_order = ['Dataset', 'Subject', 'Activity_ID', 'Trial', 'Label', 'Window_Idx', 'Device_Ori']
    
    # Mevcut olanları seç
    final_meta = [c for c in meta_order if c in df.columns]
    feature_cols = [c for c in df.columns if c not in final_meta]
    feature_cols.sort()
    
    df = df[final_meta + feature_cols]
    
    hdr = not os.path.exists(path)
    df.to_csv(path, mode='a', header=hdr, index=False)

# =============================================================================
# 4. PIPELINE KOŞUCUSU
# =============================================================================
def run_tda_pipeline_v2(config):
    name = config['dataset_name']
    print(f"\n{'='*80}")
    print(f"🚀 TDA V2.7 PIPELINE (FD_ORI SPECIAL + META)")
    print(f"📁 Dataset: {name} | Window: {config['window_sec']}s")
    print(f"{'='*80}")

    os.makedirs(os.path.dirname(config['output_csv']), exist_ok=True)
    tda_engine = TDA_Engine_V2_Pure(target_fs=config['target_fs'], window_sec=config['window_sec'])
    print(f"⚙️  Mod: {tda_engine.mode} | Gecikmeler: {tda_engine.delay_seconds}")

    conn = setup_db(config['db_path'])
    
    # 1. Sensör Sütunlarını Çek
    sensor_cols = check_db_columns(conn, config['table_sensor'], config['sensor_columns'])
    
    # 2. Meta Sütunları Çek (Varsa)
    meta_cols = []
    if 'meta_columns' in config:
        meta_cols = check_db_columns(conn, config['table_sensor'], config['meta_columns'])
        if meta_cols: print(f"ℹ️  Bulunan Meta Sütunlar: {meta_cols}")

    if not sensor_cols: conn.close(); return

    try:
        trials = pd.read_sql_query(f"SELECT trial_id, subject_id, activity_id, sampling_rate FROM {config['table_trials']}", conn)
    except:
        trials = pd.read_sql_query(f"SELECT trial_id, subject_id, activity_id FROM {config['table_trials']}", conn)
        trials["sampling_rate"] = config["original_fs"]

    # Resume Check
    if os.path.exists(config['output_csv']):
        try:
            done = pd.read_csv(config['output_csv'], usecols=['Trial'])['Trial'].unique()
            trials = trials[~trials['trial_id'].isin(done)]
            if len(done) > 0: print(f"⏩ {len(done)} trial atlandı.")
        except: pass

    if len(trials) == 0:
        print("✅ Tamamlanmış."); conn.close(); return

    buffer = []
    pbar = tqdm(trials.itertuples(index=False), total=len(trials), desc=name, unit="trial", mininterval=1.0)
    
    # Tüm sütunları birleştirip sorgula
    all_query_cols = sensor_cols + meta_cols

    for row in pbar:
        fs = row.sampling_rate if row.sampling_rate and row.sampling_rate > 0 else config["original_fs"]
        try:
            q = f"SELECT {', '.join(all_query_cols)}, {config['label_column']} FROM {config['table_sensor']} WHERE trial_id = {row.trial_id} ORDER BY timestamp ASC"
            df_sensor = pd.read_sql_query(q, conn)
            
            if df_sensor.empty: continue
            
            # Sadece SENSÖR verisine göre temizlik yap (Meta boş olsa da olur)
            df_sensor.dropna(subset=sensor_cols, inplace=True) 
            if df_sensor.empty: continue

            label = 1 if df_sensor[config["label_column"]].sum() > 0 else 0
            
            # Verileri ayır
            raw_signal = df_sensor[sensor_cols].values
            raw_meta = df_sensor[meta_cols].values if meta_cols else None

            # Pencerele
            sig_wins, meta_wins = universal_subwindowing(
                raw_signal, raw_meta, fs, config["target_fs"], config["window_sec"], config["stride_sec"]
            )
            
            if not sig_wins: continue

            for i, win_data in enumerate(sig_wins):
                feats = tda_engine.extract_all(win_data)
                
                meta_info = {
                    'Dataset': name, 'Subject': row.subject_id, 
                    'Activity_ID': row.activity_id, 'Trial': row.trial_id, 
                    'Label': label, 'Window_Idx': i
                }
                
                # Meta Veriyi Ekle (Varsa)
                if meta_wins is not None and len(meta_cols) > 0:
                    try:
                        # İlk sütunun (device_orientation) ilk değerini al
                        val = meta_wins[i][0, 0] 
                        meta_info['Device_Ori'] = int(val)
                    except:
                        meta_info['Device_Ori'] = 0

                meta_info.update(feats)
                buffer.append(meta_info)

            if len(buffer) >= 1000:
                save_to_disk(buffer, config['output_csv'])
                buffer = []
                gc.collect()
        except: continue

    if buffer: save_to_disk(buffer, config['output_csv'])
    conn.close()
    print(f"✅ {name} Bitti.")

# =============================================================================
# 5. KONFİGÜRASYONLAR (FD_ORI ÖZEL + MOBIFALL STANDART)
# =============================================================================

# 1. FD_ORI: Sadece ACC + Device Orientation + Kısa Pencere
FD_ORI_CONFIG = {
    "dataset_name": "FD_Ori",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(OUTPUT_ROOT, "FD_Ori_V2_Pure.csv"),
    "table_trials": "FD_Ori_trials",
    "table_sensor": "FD_Ori_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z"], # SADECE ACC
    "meta_columns": ["device_orientation"],      # META VAR
    "label_column": "is_fall",
    "original_fs": 100, "target_fs": 50, 
    "window_sec": 1.5, "stride_sec": 0.75 
}

# 2. MOBIFALL: Standart Sensörler + Meta YOK + Uzun Pencere
MOBIFALL_CONFIG = {
    "dataset_name": "MobiFall",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(OUTPUT_ROOT, "MobiFall_V2_Pure.csv"),
    "table_trials": "MobiFall_trials",
    "table_sensor": "MobiFall_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 100, "target_fs": 50, "window_sec": 2.0, "stride_sec": 1.0
}

# 3. DİĞERLERİ
SISFALL_CONFIG = {
    "dataset_name": "SisFall",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(OUTPUT_ROOT, "SisFall_V2_Pure.csv"),
    "table_trials": "SisFall_trials",
    "table_sensor": "SisFall_sensor_data",
    "sensor_columns": ["acc_adx1345_x", "acc_adx1345_y", "acc_adx1345_z", "gyr_itg3200_x", "gyr_itg3200_y", "gyr_itg3200_z"],
    "label_column": "is_fall",
    "original_fs": 200, "target_fs": 50, "window_sec": 2.0, "stride_sec": 1.0
}
FAD_MASTER_CONFIG = {
    "dataset_name": "FAD_Master",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(OUTPUT_ROOT, "FAD_Master_V2_Pure.csv"),
    "table_trials": "FAD_Master_trials",
    "table_sensor": "FAD_Master_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 238, "target_fs": 40, "window_sec": 0.8, "stride_sec": 0.4
}
FAD_40HZ_CONFIG = {
    "dataset_name": "FAD_40Hz",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(OUTPUT_ROOT, "FAD_40Hz_V2_Pure.csv"),
    "table_trials": "FAD_40Hz_trials",
    "table_sensor": "FAD_40Hz_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 40, "target_fs": 40, "window_sec": 2.0, "stride_sec": 1.0
}

if __name__ == "__main__":
    run_tda_pipeline_v2(SISFALL_CONFIG)
    run_tda_pipeline_v2(FAD_MASTER_CONFIG)
    run_tda_pipeline_v2(FD_ORI_CONFIG)
    run_tda_pipeline_v2(MOBIFALL_CONFIG)
    run_tda_pipeline_v2(FAD_40HZ_CONFIG)