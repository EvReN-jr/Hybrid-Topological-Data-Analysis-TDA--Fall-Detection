import sqlite3
import pandas as pd
import numpy as np
import os
import gc
import gudhi
import gudhi.representations
from tqdm import tqdm
import warnings

# Gereksiz uyarıları kapat
warnings.filterwarnings("ignore")

# =============================================================================
# 1. AYARLAR & KAYIT YOLLARI
# =============================================================================
BASE_DB_PATH = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataFirstTouch\All_Datasets_Container.db"
OUTPUT_ROOT = r"C:\Users\user\Desktop\KEB\YL_TEZ\TDA_Features_Extraction_V3\V3_TDA_Features_MultiScale"

# Yazma Sıklığı
WRITE_BATCH = 1000 

# =============================================================================
# 2. TDA MOTORU V3.3 (SMART DELAY - CSV ONLY)
# =============================================================================
class TDA_Engine_V3_Smart:
    def __init__(self, target_fs, window_sec):
        self.fs = target_fs
        self.window_sec = window_sec
        self.embedding_dim = 3
        
        # --- OTOMATİK GECİKME SEÇİMİ ---
        # Kısa kayıtlar için (FAD_Master, FD_Ori) MİKRO ölçek
        if self.window_sec <= 1.5:
            self.delay_seconds = [0.05, 0.10, 0.15, 0.20]
            self.mode = "MICRO (Short Window)"
        else:
            # Standart (SisFall, MobiFall)
            self.delay_seconds = [0.12, 0.24, 0.36, 0.48]
            self.mode = "STANDARD (Long Window)"
        
        # --- PERSISTENCE IMAGE (5x5 Grid) ---
        self.pi_csv = gudhi.representations.PersistenceImage(
            bandwidth=0.2, weight=lambda x: x[1]**2,
            resolution=[5, 5], im_range=[0, 20, 0, 20] 
        )

    def _get_delay_embedding(self, signal, delay_pts):
        """Hızlı 3D Gömme"""
        N = len(signal)
        dim = self.embedding_dim
        required_len = (dim - 1) * delay_pts
        if N <= required_len: return None
        num_points = N - required_len
        embedded = np.empty((num_points, dim), dtype=np.float64) 
        for i in range(dim):
            embedded[:, i] = signal[i * delay_pts : i * delay_pts + num_points]
        return embedded

    def extract_features(self, window_data):
        feats = {}
        
        # 1. Sinyal Bileşkesi (Magnitude)
        if window_data.ndim > 1 and window_data.shape[1] >= 3:
            signal = np.linalg.norm(window_data[:, :3], axis=1)
        else:
            if window_data.size > 0: signal = np.abs(window_data.flatten())
            else: return self._get_empty_features()

        # 2. Her Gecikme Faktörü İçin Döngü
        for sec in self.delay_seconds:
            delay_pts = max(1, int(sec * self.fs))
            point_cloud = self._get_delay_embedding(signal, delay_pts)
            
            # Sütun isimlerini rezerve et
            for dim in [0, 1]:
                prefix = f"PI_D{sec:.2f}s_H{dim}"
                for px in range(25):
                    feats[f"{prefix}_P{px}"] = 0.0

            if point_cloud is None or len(point_cloud) < 5:
                continue

            try:
                # TDA Hesaplama
                alpha = gudhi.AlphaComplex(points=point_cloud)
                st = alpha.create_simplex_tree()
                st.persistence()
                
                for dim in [0, 1]:
                    intervals = st.persistence_intervals_in_dimension(dim)
                    if len(intervals) == 0: continue
                    finite_int = intervals[np.isfinite(intervals[:, 1])]
                    if len(finite_int) == 0: continue
                    
                    lifetimes = finite_int[:, 1] - finite_int[:, 0]
                    finite_int = finite_int[lifetimes > 0.05]
                    
                    if len(finite_int) > 0:
                        pi_vec = self.pi_csv.fit_transform([finite_int])[0]
                        prefix = f"PI_D{sec:.2f}s_H{dim}"
                        for px, val in enumerate(pi_vec):
                            feats[f"{prefix}_P{px}"] = val
            except: pass
                    
        return feats

    def _get_empty_features(self):
        """Hata durumunda boş sözlük"""
        feats = {}
        for sec in self.delay_seconds:
            for dim in [0, 1]:
                prefix = f"PI_D{sec:.2f}s_H{dim}"
                for px in range(25):
                    feats[f"{prefix}_P{px}"] = 0.0
        return feats

# =============================================================================
# 3. YARDIMCI FONKSİYONLAR
# =============================================================================
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

def save_csv(buffer, path):
    df = pd.DataFrame(buffer)
    # Meta Sütun Sıralaması (Device_Ori'yi başa yakın tut)
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
def run_visual_pipeline(config):
    name = config['dataset_name']
    csv_filename = f"{name}_V3_MultiScale_Features.csv"
    csv_path = os.path.join(OUTPUT_ROOT, csv_filename)
    
    print(f"\n{'='*80}")
    print(f"📊 TDA V3.3 PIPELINE (FD_ORI FIX + META): {name}")
    print(f"📁 Kayıt: {csv_path}")
    print(f"⏳ Window: {config['window_sec']}s")
    print(f"{'='*80}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    tda_engine = TDA_Engine_V3_Smart(target_fs=config['target_fs'], window_sec=config['window_sec'])
    print(f"⚙️  Mod: {tda_engine.mode} | Gecikmeler: {tda_engine.delay_seconds}")

    conn = setup_db(config['db_path'])
    
    # 1. Sensör Sütunlarını Çek
    cols = check_cols(conn, config['table_sensor'], config['sensor_columns'])
    
    # 2. Meta Sütunları Çek (Varsa)
    meta_cols = []
    if 'meta_columns' in config:
        meta_cols = check_cols(conn, config['table_sensor'], config['meta_columns'])
        if meta_cols: print(f"ℹ️  Bulunan Meta Sütunlar: {meta_cols}")

    if not cols: conn.close(); return

    try:
        trials = pd.read_sql_query(f"SELECT trial_id, subject_id, activity_id, sampling_rate FROM {config['table_trials']}", conn)
    except:
        trials = pd.read_sql_query(f"SELECT trial_id, subject_id, activity_id FROM {config['table_trials']}", conn)
        trials["sampling_rate"] = config["original_fs"]

    # Resume Check
    if os.path.exists(csv_path):
        try:
            done = pd.read_csv(csv_path, usecols=['Trial'])['Trial'].unique()
            trials = trials[~trials['trial_id'].isin(done)]
            if len(done) > 0: print(f"⏩ {len(done)} trial atlandı.")
        except: pass

    if len(trials) == 0:
        print("✅ Tamamlanmış.")
        conn.close(); return

    buffer = []
    total_wins = 0
    pbar = tqdm(trials.itertuples(index=False), total=len(trials), desc=name, unit="trial", mininterval=1.0)
    
    all_query_cols = cols + meta_cols

    for row in pbar:
        fs = row.sampling_rate if row.sampling_rate and row.sampling_rate > 0 else config["original_fs"]
        try:
            q = f"SELECT {', '.join(all_query_cols)}, {config['label_column']} FROM {config['table_sensor']} WHERE trial_id = {row.trial_id} ORDER BY timestamp ASC"
            df_sensor = pd.read_sql_query(q, conn)
            
            if df_sensor.empty: continue
            
            # Sadece SENSÖR verisine göre temizlik
            df_sensor.dropna(subset=cols, inplace=True)
            if df_sensor.empty: continue

            label = 1 if df_sensor[config["label_column"]].sum() > 0 else 0
            
            raw_data = df_sensor[cols].values
            raw_meta = df_sensor[meta_cols].values if meta_cols else None

            # Pencerele
            sig_wins, meta_wins = universal_subwindowing(
                raw_data, raw_meta, fs, config["target_fs"], config["window_sec"], config["stride_sec"]
            )
            
            if not sig_wins: continue

            for i, window in enumerate(sig_wins):
                feats = tda_engine.extract_features(window)
                if not feats: continue
                
                meta = {'Dataset': name, 'Subject': row.subject_id, 
                        'Activity_ID': row.activity_id, 'Trial': row.trial_id, 
                        'Label': label, 'Window_Idx': i}
                
                # Meta Veriyi Ekle (Varsa)
                if meta_wins is not None and len(meta_cols) > 0:
                    try:
                        val = meta_wins[i][0, 0]
                        meta['Device_Ori'] = int(val)
                    except:
                        meta['Device_Ori'] = 0
                
                meta.update(feats)
                buffer.append(meta)
                total_wins += 1

            if len(buffer) >= WRITE_BATCH: 
                save_csv(buffer, csv_path)
                buffer = []
                gc.collect()
        except: continue

    if buffer: save_csv(buffer, csv_path)
    conn.close()
    print(f"✅ {name} Bitti. Toplam: {total_wins}")

# =============================================================================
# 5. KONFİGÜRASYONLAR (FİNAL)
# =============================================================================

# 1. FD_ORI (GÜNCELLENMİŞ)
# Jiroskop iptal, Meta (Device Ori) aktif, Pencere 1.5 sn
FD_ORI_CONFIG = {
    "dataset_name": "FD_Ori",
    "db_path": BASE_DB_PATH,
    "table_trials": "FD_Ori_trials",
    "table_sensor": "FD_Ori_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z"], # SADECE ACC
    "meta_columns": ["device_orientation"],      # META VAR
    "label_column": "is_fall",
    "original_fs": 100, "target_fs": 50, 
    "window_sec": 1.5, "stride_sec": 0.75 
}

# 2. MOBIFALL (GÜNCELLENMİŞ)
# Meta iptal edildi (İçi boş)
MOBIFALL_CONFIG = {
    "dataset_name": "MobiFall",
    "db_path": BASE_DB_PATH,
    "table_trials": "MobiFall_trials",
    "table_sensor": "MobiFall_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 100, "target_fs": 50, "window_sec": 2.0, "stride_sec": 1.0
}

# 3. DİĞERLERİ (STANDART)
SISFALL_CONFIG = {
    "dataset_name": "SisFall",
    "db_path": BASE_DB_PATH,
    "table_trials": "SisFall_trials",
    "table_sensor": "SisFall_sensor_data",
    "sensor_columns": ["acc_adx1345_x", "acc_adx1345_y", "acc_adx1345_z", "gyr_itg3200_x", "gyr_itg3200_y", "gyr_itg3200_z"],
    "label_column": "is_fall",
    "original_fs": 200, "target_fs": 50, "window_sec": 2.0, "stride_sec": 1.0
}
FAD_MASTER_CONFIG = {
    "dataset_name": "FAD_Master",
    "db_path": BASE_DB_PATH,
    "table_trials": "FAD_Master_trials",
    "table_sensor": "FAD_Master_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 238, "target_fs": 40, "window_sec": 0.8, "stride_sec": 0.4
}
FAD_40HZ_CONFIG = {
    "dataset_name": "FAD_40Hz",
    "db_path": BASE_DB_PATH,
    "table_trials": "FAD_40Hz_trials",
    "table_sensor": "FAD_40Hz_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 40, "target_fs": 40, "window_sec": 2.0, "stride_sec": 1.0
}

if __name__ == "__main__":
    run_visual_pipeline(SISFALL_CONFIG)
    run_visual_pipeline(FAD_MASTER_CONFIG)
    run_visual_pipeline(FD_ORI_CONFIG)
    run_visual_pipeline(MOBIFALL_CONFIG)
    run_visual_pipeline(FAD_40HZ_CONFIG)