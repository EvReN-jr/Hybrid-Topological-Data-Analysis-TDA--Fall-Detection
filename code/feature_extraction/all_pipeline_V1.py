import sqlite3
import pandas as pd
import numpy as np
import os
import gc
import gudhi
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. AYARLAR & KAYIT YOLLARI
# =============================================================================
BASE_DB_PATH = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataFirstTouch\All_Datasets_Container.db"
OUTPUT_ROOT = r"C:\Users\user\Desktop\KEB\YL_TEZ\TDA_Features_Extraction_V1\V1_TDA_Features_MultiScale"
WRITE_BATCH = 1000

# =============================================================================
# 2. V1.8 PURE TDA ENGINE (SMART DELAY + MULTI SCALE)
# =============================================================================
class V1_Pure_TDA_Engine:
    def __init__(self, target_fs, window_sec):
        self.fs = target_fs
        self.window_sec = window_sec
        
        # --- OTOMATİK GECİKME SEÇİMİ (V2/V3 ile Aynı) ---
        if self.window_sec <= 1.5:
            # Kısa Kayıtlar (FD_Ori, FAD_Master) -> Micro Delay
            self.delay_seconds = [0.05, 0.10, 0.15, 0.20]
            self.mode = "MICRO (Short Window)"
        else:
            # Standart Kayıtlar -> Standard Delay
            self.delay_seconds = [0.12, 0.24, 0.36, 0.48]
            self.mode = "STANDARD (Long Window)"
            
        self.homology_dims = [0, 1]

    def _get_embedding(self, signal, delay_pts):
        """3D Gecikmeli Gömme"""
        dim = 3
        N = len(signal)
        required_len = (dim - 1) * delay_pts
        if N <= required_len: return None
        
        num_points = N - required_len
        embedded = np.empty((num_points, dim), dtype=np.float64) 
        for i in range(dim):
            embedded[:, i] = signal[i * delay_pts : i * delay_pts + num_points]
        return embedded

    def extract_features(self, window_data):
        # 1. Sinyal Bileşkesi (Magnitude)
        if window_data.ndim > 1 and window_data.shape[1] >= 3:
            raw_mag = np.linalg.norm(window_data[:, :3], axis=1)
        else:
            if window_data.size > 0: raw_mag = np.abs(window_data.flatten())
            else: return self._get_empty_features()

        all_features = {}

        # 2. Her Gecikme İçin Döngü (Multi-Scale)
        for sec in self.delay_seconds:
            delay_pts = max(1, int(sec * self.fs))
            point_cloud = self._get_embedding(raw_mag, delay_pts)
            
            # Default Değerleri Ata
            for dim in self.homology_dims:
                prefix = f"D_{sec:.2f}s_H{dim}"
                all_features[f'{prefix}_MaxLife'] = 0.0
                all_features[f'{prefix}_WAvgLife'] = 0.0
                all_features[f'{prefix}_Count'] = 0.0
                all_features[f'{prefix}_Entropy'] = 0.0

            if point_cloud is None or len(point_cloud) < 4:
                continue

            try:
                # TDA Hesaplama
                alpha = gudhi.AlphaComplex(points=point_cloud)
                st = alpha.create_simplex_tree()
                st.persistence()
                
                for dim in self.homology_dims:
                    intervals = st.persistence_intervals_in_dimension(dim)
                    if len(intervals) == 0: continue
                    
                    finite_int = intervals[np.isfinite(intervals[:, 1])]
                    if len(finite_int) == 0: continue
                    
                    lifetimes = finite_int[:, 1] - finite_int[:, 0]
                    # Gürültü Filtresi
                    lifetimes = lifetimes[lifetimes > 1e-6]
                    
                    if len(lifetimes) == 0: continue

                    # --- İSTATİSTİKLER ---
                    prefix = f"D_{sec:.2f}s_H{dim}"
                    
                    # 1. MaxLife
                    all_features[f'{prefix}_MaxLife'] = np.max(lifetimes)
                    
                    # 2. Weighted AvgLife
                    sum_life = np.sum(lifetimes)
                    sum_sq_life = np.sum(lifetimes**2)
                    if sum_life > 0:
                        all_features[f'{prefix}_WAvgLife'] = sum_sq_life / sum_life
                    
                    # 3. Count
                    all_features[f'{prefix}_Count'] = len(lifetimes)
                    
                    # 4. Entropy
                    if sum_life > 0:
                        probs = lifetimes / sum_life
                        all_features[f'{prefix}_Entropy'] = -np.sum(probs * np.log(probs + 1e-10))
                        
            except Exception: pass

        return all_features

    def _get_empty_features(self):
        feats = {}
        for sec in self.delay_seconds:
            for dim in self.homology_dims:
                prefix = f"D_{sec:.2f}s_H{dim}"
                feats[f'{prefix}_MaxLife'] = 0.0
                feats[f'{prefix}_WAvgLife'] = 0.0
                feats[f'{prefix}_Count'] = 0.0
                feats[f'{prefix}_Entropy'] = 0.0
        return feats

# =============================================================================
# 3. YARDIMCI FONKSİYONLAR
# =============================================================================
def universal_subwindowing(signal_data, meta_data, original_fs, target_fs, window_sec, stride_sec):
    if original_fs <= 0: step = 1
    else: step = max(1, int(original_fs / target_fs))
    
    resampled_sig = signal_data[::step]
    resampled_meta = meta_data[::step] if meta_data is not None else None
    
    win_len = int(window_sec * target_fs)
    stride_len = int(stride_sec * target_fs)
    
    if len(resampled_sig) < win_len: return None, None
    
    sig_windows = [resampled_sig[i:i + win_len] for i in range(0, len(resampled_sig) - win_len + 1, stride_len)]
    
    meta_windows = None
    if resampled_meta is not None:
        meta_windows = [resampled_meta[i:i + win_len] for i in range(0, len(resampled_meta) - win_len + 1, stride_len)]
        
    return sig_windows, meta_windows

def setup_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def check_cols(conn, table, candidates):
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        exist = {row[1] for row in cur.fetchall()}
        valid = []
        for c in candidates:
            if c in exist: valid.append(c)
            else:
                if "adx1345" in c and c.replace("adx1345", "adxl345") in exist: valid.append(c.replace("adx1345", "adxl345"))
                elif "adxl345" in c and c.replace("adxl345", "adx1345") in exist: valid.append(c.replace("adxl345", "adx1345"))
        return valid
    except: return []

def save_to_disk(buffer, path):
    df = pd.DataFrame(buffer)
    meta_order = ['Dataset', 'Subject', 'Activity_ID', 'Trial', 'Label', 'Window_Idx', 'Device_Ori']
    
    final_meta = [c for c in meta_order if c in df.columns]
    feature_cols = [c for c in df.columns if c not in final_meta]
    feature_cols.sort()
    
    df = df[final_meta + feature_cols]
    
    hdr = not os.path.exists(path)
    df.to_csv(path, mode='a', header=hdr, index=False)

# =============================================================================
# 4. PIPELINE KOŞUCUSU
# =============================================================================
def run_pipeline(config):
    name = config['dataset_name']
    print(f"\n{'='*80}")
    print(f"🚀 TDA V1.8 PIPELINE (SMART DELAY + META): {name}")
    print(f"🛠️  Motor: V1 Pure TDA (Scalar Stats)")
    print(f"⏳ Window: {config['window_sec']}s")
    print(f"{'='*80}")
    
    os.makedirs(os.path.dirname(config['output_csv']), exist_ok=True)
    engine = V1_Pure_TDA_Engine(config['target_fs'], config['window_sec'])
    print(f"⚙️  Mod: {engine.mode} | Gecikmeler: {engine.delay_seconds}")

    conn = setup_db(config['db_path'])
    cols = check_cols(conn, config['table_sensor'], config['sensor_columns'])
    
    # Meta Sütun Kontrolü
    meta_cols = []
    if 'meta_columns' in config:
        meta_cols = check_cols(conn, config['table_sensor'], config['meta_columns'])
        if meta_cols: print(f"ℹ️  Bulunan Meta Sütunlar: {meta_cols}")

    if not cols: print("❌ Kolon hatası!"); conn.close(); return

    try:
        trials = pd.read_sql_query(f"SELECT trial_id, subject_id, activity_id, sampling_rate FROM {config['table_trials']}", conn)
    except:
        trials = pd.read_sql_query(f"SELECT trial_id, subject_id, activity_id FROM {config['table_trials']}", conn)
        trials['sampling_rate'] = config['original_fs']

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
    
    all_query_cols = cols + meta_cols

    for row in pbar:
        fs = row.sampling_rate if row.sampling_rate > 0 else config['original_fs']
        try:
            q = f"SELECT {', '.join(all_query_cols)}, {config['label_column']} FROM {config['table_sensor']} WHERE trial_id={row.trial_id} ORDER BY timestamp ASC"
            df_sensor = pd.read_sql_query(q, conn)
            
            if df_sensor.empty: continue
            df_sensor.dropna(subset=cols, inplace=True)
            if df_sensor.empty: continue
            
            label = 1 if df_sensor[config['label_column']].sum() > 0 else 0
            
            raw_data = df_sensor[cols].values
            raw_meta = df_sensor[meta_cols].values if meta_cols else None
            
            sig_wins, meta_wins = universal_subwindowing(
                raw_data, raw_meta, fs, config['target_fs'], config['window_sec'], config['stride_sec']
            )
            
            if not sig_wins: continue
            
            for i, win in enumerate(sig_wins):
                feats = engine.extract_features(win)
                meta = {'Dataset': name, 'Subject': row.subject_id, 'Activity_ID': row.activity_id, 
                        'Trial': row.trial_id, 'Label': label, 'Window_Idx': i}
                
                # Meta Veri (Device Ori)
                if meta_wins is not None and len(meta_cols) > 0:
                    try: meta['Device_Ori'] = int(meta_wins[i][0, 0])
                    except: meta['Device_Ori'] = 0
                
                meta.update(feats)
                buffer.append(meta)
            
            if len(buffer) >= WRITE_BATCH:
                save_to_disk(buffer, config['output_csv'])
                buffer = []
                gc.collect()
        except: continue

    if buffer: save_to_disk(buffer, config['output_csv'])
    conn.close()
    print(f"✅ {name} Tamamlandı!")

# =============================================================================
# 5. KONFİGÜRASYONLAR (GÜNCELLENMİŞ)
# =============================================================================

# 1. FD_ORI (ÖZEL AYAR)
FD_ORI_CONFIG = {
    "dataset_name": "FD_Ori",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(OUTPUT_ROOT, "FD_Ori_V1_Pure.csv"),
    "table_trials": "FD_Ori_trials",
    "table_sensor": "FD_Ori_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z"], # SADECE ACC
    "meta_columns": ["device_orientation"],      # META VAR
    "label_column": "is_fall",
    "original_fs": 100, "target_fs": 50, 
    "window_sec": 1.5, "stride_sec": 0.75 
}

# 2. MOBIFALL (META İPTAL)
MOBIFALL_CONFIG = {
    "dataset_name": "MobiFall",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(OUTPUT_ROOT, "MobiFall_V1_Pure.csv"),
    "table_trials": "MobiFall_trials",
    "table_sensor": "MobiFall_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    # "meta_columns": ["device_orientation"], <-- İPTAL
    "label_column": "is_fall",
    "original_fs": 100, "target_fs": 50, "window_sec": 2.0, "stride_sec": 1.0
}

# 3. DİĞERLERİ
SISFALL_CONFIG = {
    "dataset_name": "SisFall",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(OUTPUT_ROOT, "SisFall_V1_Pure.csv"),
    "table_trials": "SisFall_trials",
    "table_sensor": "SisFall_sensor_data",
    "sensor_columns": ["acc_adx1345_x", "acc_adx1345_y", "acc_adx1345_z", "gyr_itg3200_x", "gyr_itg3200_y", "gyr_itg3200_z"],
    "label_column": "is_fall",
    "original_fs": 200, "target_fs": 50, "window_sec": 2.0, "stride_sec": 1.0
}
FAD_MASTER_CONFIG = {
    "dataset_name": "FAD_Master",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(OUTPUT_ROOT, "FAD_Master_V1_Pure.csv"),
    "table_trials": "FAD_Master_trials",
    "table_sensor": "FAD_Master_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 238, "target_fs": 40, "window_sec": 0.8, "stride_sec": 0.4
}
FAD_40HZ_CONFIG = {
    "dataset_name": "FAD_40Hz",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(OUTPUT_ROOT, "FAD_40Hz_V1_Pure.csv"),
    "table_trials": "FAD_40Hz_trials",
    "table_sensor": "FAD_40Hz_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 40, "target_fs": 40, "window_sec": 2.0, "stride_sec": 1.0
}

if __name__ == "__main__":
    CONFIGS = [SISFALL_CONFIG, FAD_MASTER_CONFIG, FD_ORI_CONFIG, MOBIFALL_CONFIG, FAD_40HZ_CONFIG]
    for conf in CONFIGS:
        run_pipeline(conf)