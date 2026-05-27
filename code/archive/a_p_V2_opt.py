import sqlite3
import pandas as pd
import numpy as np
import os
import gc
import gudhi
import gudhi.representations
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. OPTİMİZE EDİLMİŞ TDA MOTORU (V2 - 70 ÖZNİTELİK)
# =============================================================================
class TDA_Engine_V2_Optimized:
    def __init__(self, target_fs):
        self.fs = target_fs
        self.delay_factors = [0.5, 1.0, 1.5, 2.0]
        self.embedding_dim = 3
        
        # Gudhi Vektörleştiriciler
        self.betti = gudhi.representations.BettiCurve(resolution=50)
        self.landscape = gudhi.representations.Landscape(resolution=50, num_landscapes=1)

    def _get_delay_embedding(self, signal, delay):
        """Hafıza dostu gecikmeli gömme (Delay Embedding)"""
        N = len(signal)
        dim = self.embedding_dim
        
        if N <= (dim - 1) * delay:
            return None
        
        num_points = N - (dim - 1) * delay
        # np.empty ile hızlı yer ayırma (float64 hassasiyeti korunur)
        embedded = np.empty((num_points, dim), dtype=np.float64) 
        
        for i in range(dim):
            embedded[:, i] = signal[i * delay : i * delay + num_points]
        return embedded

    def _compute_diagram_features(self, persistence_intervals):
        """Diyagramdan istatistiksel öznitelik çıkarma"""
        keys = ['W1', 'W_inf', 'Entropy', 'Betti_L1', 'Betti_L2', 'Land_L1', 'Land_L2']
        feats = {k: 0.0 for k in keys}
        
        if len(persistence_intervals) == 0:
            return feats

        finite_intervals = persistence_intervals[np.isfinite(persistence_intervals[:, 1])]
        
        if len(finite_intervals) == 0:
            return feats

        lifetimes = finite_intervals[:, 1] - finite_intervals[:, 0]
        
        # Temel Metrikler
        feats['W1'] = np.sum(lifetimes) / np.sqrt(2)
        feats['W_inf'] = np.max(lifetimes) / np.sqrt(2)
        
        total_life = np.sum(lifetimes)
        if total_life > 1e-10:
            probs = lifetimes / total_life
            feats['Entropy'] = -np.sum(probs * np.log(probs + 1e-10))
            
        # --- DÜZELTME: diags_format tanımlandı ---
        diags_format = [finite_intervals]
            
        try:
            betti_vals = self.betti.fit_transform(diags_format)[0]
            feats['Betti_L1'] = np.linalg.norm(betti_vals, ord=1)
            feats['Betti_L2'] = np.linalg.norm(betti_vals, ord=2)
        except:
            pass

        try:
            land_vals = self.landscape.fit_transform(diags_format)[0]
            feats['Land_L1'] = np.linalg.norm(land_vals, ord=1)
            feats['Land_L2'] = np.linalg.norm(land_vals, ord=2)
        except:
            pass
            
        return feats

    def extract_all(self, window_data):
        """
        Ana öznitelik çıkarma fonksiyonu.
        window_data: (N, features) matrisi.
        ÖNEMLİ: İlk 3 sütun her zaman İvmeölçer (Acc X,Y,Z) olmalıdır.
        """
        # 1. Bileşke (Magnitude) Hesapla
        # Veri setindeki ilk 3 sütunu (Acc X, Y, Z) alıp normunu hesaplar.
        if window_data.ndim > 1 and window_data.shape[1] >= 3:
            signal = np.linalg.norm(window_data[:, :3], axis=1)
        else:
            signal = window_data.flatten()
        
        # 2. Dejenere Sinyal Kontrolü (Early Exit)
        # Dümdüz çizgileri (sabit duran cihaz) atlar, işlemciyi yormaz.
        if len(np.unique(signal)) < 5:
            return self._get_default_features()
            
        # 3. Normalizasyon (TDA için zorunlu)
        scaler = MinMaxScaler()
        signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        
        all_features = {}
        
        # --- A. TIME DELAY EMBEDDINGS (4 Gecikme x 2 Homoloji) ---
        for factor in self.delay_factors:
            delay = max(1, int(factor * self.fs))
            point_cloud = self._get_delay_embedding(signal, delay)
            
            # Varsayılan değerler
            for dim in [0, 1]:
                prefix = f"D_{factor}fs_H{dim}"
                for k in ['W1', 'W_inf', 'Entropy', 'Betti_L1', 'Betti_L2', 'Land_L1', 'Land_L2']:
                    all_features[f"{prefix}_{k}"] = 0.0

            if point_cloud is not None and len(point_cloud) > 5:
                try:
                    # Gudhi Rips Complex
                    rips = gudhi.RipsComplex(points=point_cloud, max_edge_length=1.75)
                    st = rips.create_simplex_tree(max_dimension=2)
                    st.persistence()
                    
                    for dim in [0, 1]:
                        intervals = st.persistence_intervals_in_dimension(dim)
                        dim_feats = self._compute_diagram_features(intervals)
                        
                        prefix = f"D_{factor}fs_H{dim}"
                        for k, v in dim_feats.items():
                            all_features[f"{prefix}_{k}"] = v
                except:
                    pass

        # --- B. LEVEL SETS (Alt ve Üst Seviye Kümeleri) ---
        for ls_type in ['Lower', 'Upper']:
            prefix = f"LS_{ls_type}_H0"
            for k in ['W1', 'W_inf', 'Entropy', 'Betti_L1', 'Betti_L2', 'Land_L1', 'Land_L2']:
                all_features[f"{prefix}_{k}"] = 0.0

            try:
                # Upper level set için sinyali ters çeviriyoruz
                filt_signal = signal if ls_type == 'Lower' else -signal
                
                # 1D sinyal için Cubical Complex çok hızlıdır
                cc = gudhi.CubicalComplex(top_dimensional_cells=filt_signal)
                cc.persistence()
                intervals = cc.persistence_intervals_in_dimension(0)
                finite_int = intervals[np.isfinite(intervals[:, 1])]
                
                ls_feats = self._compute_diagram_features(finite_int)
                for k, v in ls_feats.items():
                    all_features[f"{prefix}_{k}"] = v
            except:
                pass
                    
        return all_features
    
    def _get_default_features(self):
        """Hızlı sıfır döndürme yardımcısı"""
        all_features = {}
        keys = ['W1', 'W_inf', 'Entropy', 'Betti_L1', 'Betti_L2', 'Land_L1', 'Land_L2']
        
        for factor in self.delay_factors:
            for dim in [0, 1]:
                prefix = f"D_{factor}fs_H{dim}"
                for k in keys: all_features[f"{prefix}_{k}"] = 0.0
                
        for ls_type in ['Lower', 'Upper']:
            prefix = f"LS_{ls_type}_H0"
            for k in keys: all_features[f"{prefix}_{k}"] = 0.0
            
        return all_features

# =============================================================================
# 2. VERİTABANI VE PENCERELEME YARDIMCILARI
# =============================================================================
def create_indexes(conn, table_name):
    """Sorgu hızını artırmak için indeks oluşturur"""
    sql = f"CREATE INDEX IF NOT EXISTS idx_{table_name}_trial_time ON {table_name} (trial_id, timestamp);"
    conn.execute(sql)
    conn.commit()

def check_db_columns(conn, table_name, candidates):
    """Veritabanındaki gerçek sütun isimlerini kontrol eder ve düzeltir"""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cursor.fetchall()}
    valid = []
    
    for col in candidates:
        if col in existing:
            valid.append(col)
        else:
            # SisFall PDF'indeki olası yazım hatası kontrolü (1345 vs l345)
            if col.startswith("acc_adx1345"):
                fixed = col.replace("acc_adx1345", "acc_adxl345")
                if fixed in existing: valid.append(fixed)
            # Tam tersi durum için
            elif col.startswith("acc_adxl345"):
                fixed = col.replace("acc_adxl345", "acc_adx1345")
                if fixed in existing: valid.append(fixed)
                
    return valid

def universal_subwindowing_memory(signal_data, original_fs, target_fs, window_sec, stride_sec, min_ratio=0.8):
    """RAM üzerinde pencereleme yapar (Dosya oluşturmaz)"""
    if original_fs is None or original_fs <= 0: step = 1
    else: step = max(1, int(original_fs / target_fs))
    
    resampled = signal_data[::step]
    
    win_len = int(window_sec * target_fs)
    stride_len = int(stride_sec * target_fs)
    min_len = int(min_ratio * win_len)

    if len(resampled) < min_len:
        return None

    # Pre-allocation: Hız için boş array oluştur
    num_windows = (len(resampled) - win_len) // stride_len + 1
    if resampled.ndim > 1:
        windows = np.empty((num_windows, win_len, resampled.shape[1]), dtype=resampled.dtype)
    else:
        windows = np.empty((num_windows, win_len), dtype=resampled.dtype)
    
    for idx, i in enumerate(range(0, len(resampled) - win_len + 1, stride_len)):
        if idx < num_windows:
            windows[idx] = resampled[i:i + win_len]

    return windows

# =============================================================================
# 3. ANA PIPELINE (İŞLETİCİ)
# =============================================================================
def run_tda_pipeline_v2(config):
    print(f"\n{'='*80}")
    print(f"🚀 TDA V2 PIPELINE (OPTIMIZED)")
    print(f"📁 İşleniyor: {config['dataset_name']}")
    print(f"💾 Hedef Dosya: {config['output_csv']}")
    print(f"{'='*80}")

    os.makedirs(os.path.dirname(config['output_csv']), exist_ok=True)
    
    # TDA Motorunu Başlat
    tda_engine = TDA_Engine_V2_Optimized(target_fs=config['target_fs'])

    # DB Bağlantısı
    conn = sqlite3.connect(config['db_path'])
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA cache_size=-32000;") # 32MB Cache
    
    create_indexes(conn, config['table_sensor'])
    
    # Sütunları PDF şemasına göre doğrula
    sensor_cols = check_db_columns(conn, config['table_sensor'], config['sensor_columns'])
    
    if not sensor_cols:
        print("❌ Sensör sütunu bulunamadı, tabloyu kontrol edin.")
        conn.close(); return

    # Trial Listesini Çek
    try:
        trials_df = pd.read_sql_query(f"SELECT trial_id, subject_id, activity_id, sampling_rate FROM {config['table_trials']}", conn)
    except:
        trials_df = pd.read_sql_query(f"SELECT trial_id, subject_id, activity_id FROM {config['table_trials']}", conn)
        trials_df["sampling_rate"] = config["original_fs"]

    # Kaldığı yerden devam etme (Resume)
    if os.path.exists(config['output_csv']):
        try:
            processed_trials = pd.read_csv(config['output_csv'], usecols=['Trial'])['Trial'].unique()
            trials_df = trials_df[~trials_df['trial_id'].isin(processed_trials)]
            print(f"🔄 Devam ediliyor... {len(processed_trials)} trial atlandı.")
        except: pass

    total_wins = 0
    buffer = []
    WRITE_EVERY = 30 # RAM koruması için sık yazma

    pbar = tqdm(trials_df.itertuples(index=False), total=len(trials_df), desc=f"{config['dataset_name']}", unit="trial")
    
    for row in pbar:
        t_id, s_id, a_id = row.trial_id, row.subject_id, row.activity_id
        fs = row.sampling_rate if row.sampling_rate and row.sampling_rate > 0 else config["original_fs"]

        try:
            query = f"SELECT {', '.join(sensor_cols)}, {config['label_column']} FROM {config['table_sensor']} WHERE trial_id = {t_id} ORDER BY timestamp ASC"
            df_sensor = pd.read_sql_query(query, conn)
        except: continue

        if df_sensor.empty: continue
        
        # Boş veri kontrolü
        active_cols = [c for c in sensor_cols if not df_sensor[c].isna().all()]
        if not active_cols: continue
        df_sensor.dropna(subset=active_cols, inplace=True)
        if df_sensor.empty: continue

        label = 1 if df_sensor[config["label_column"]].sum() > 0 else 0
        raw_data = df_sensor[active_cols].values

        # Subwindowing (RAM'de)
        windows = universal_subwindowing_memory(
            raw_data, fs, config["target_fs"], 
            config["window_sec"], config["stride_sec"]
        )

        if windows is None: continue

        # TDA Hesaplama
        for i, window in enumerate(windows):
            try:
                feats = tda_engine.extract_all(window)
                row_data = {
                    'Dataset': config['dataset_name'],
                    'Subject': s_id, 'Activity_ID': a_id,
                    'Trial': t_id, 'Label': label, 'Window_Idx': i
                }
                row_data.update(feats)
                buffer.append(row_data)
            except: pass

        # Diske Yazma
        if len(buffer) >= WRITE_EVERY:
            df_batch = pd.DataFrame(buffer)
            meta_cols = ['Dataset', 'Subject', 'Activity_ID', 'Trial', 'Label', 'Window_Idx']
            other_cols = [c for c in df_batch.columns if c not in meta_cols]
            df_batch = df_batch[meta_cols + other_cols]

            file_exists = os.path.exists(config['output_csv'])
            df_batch.to_csv(config['output_csv'], mode='a', header=not file_exists, index=False)
            
            total_wins += len(df_batch)
            buffer = []
            gc.collect() # RAM temizliği
            pbar.set_postfix({'Windows': total_wins})

    # Kalan Son Veriler
    if buffer:
        df_batch = pd.DataFrame(buffer)
        meta_cols = ['Dataset', 'Subject', 'Activity_ID', 'Trial', 'Label', 'Window_Idx']
        other_cols = [c for c in df_batch.columns if c not in meta_cols]
        df_batch = df_batch[meta_cols + other_cols]
        file_exists = os.path.exists(config['output_csv'])
        df_batch.to_csv(config['output_csv'], mode='a', header=not file_exists, index=False)
        total_wins += len(df_batch)

    conn.close()
    gc.collect()
    print(f"✅ {config['dataset_name']} Tamamlandı. Toplam Pencere: {total_wins}")

# =============================================================================
# 4. KONFİGÜRASYONLAR (PDF ŞEMASINA GÖRE GÜNCELLENDİ)
# =============================================================================
BASE_DB_PATH = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataFirstTouch\All_Datasets_Container.db"
BASE_OUT_DIR = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataOrganize_V2\TDA_V2_Features"

# 1. SISFALL (PDF'teki 'acc_adx1345' yazım hatası burada handle edildi)
# ÖNEMLİ: İlk 3 kolon İvmeölçer olmalı (Bileşke hesabı için)
SISFALL_CONFIG = {
    "dataset_name": "SisFall",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(BASE_OUT_DIR, "SisFall_V2_Optimized.csv"),
    "table_trials": "SisFall_trials",
    "table_sensor": "SisFall_sensor_data",
    "sensor_columns": [
        "acc_adx1345_x", "acc_adx1345_y", "acc_adx1345_z", # Primary Acc (Önce)
        "gyr_itg3200_x", "gyr_itg3200_y", "gyr_itg3200_z",
        "acc_mma8451q_x", "acc_mma8451q_y", "acc_mma8451q_z" # Secondary Acc (Sonra)
    ],
    "label_column": "is_fall",
    "original_fs": 200, "target_fs": 50, "window_sec": 2.0, "stride_sec": 1.0
}

# 2. FAD_40Hz
FAD_40HZ_CONFIG = {
    "dataset_name": "FAD_40Hz",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(BASE_OUT_DIR, "FAD_40Hz_V2_Optimized.csv"),
    "table_trials": "FAD_40Hz_trials",
    "table_sensor": "FAD_40Hz_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 40, "target_fs": 40, "window_sec": 2.0, "stride_sec": 1.0
}

# 3. FAD_MASTER
FAD_MASTER_CONFIG = {
    "dataset_name": "FAD_Master",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(BASE_OUT_DIR, "FAD_Master_V2_Optimized.csv"),
    "table_trials": "FAD_Master_trials",
    "table_sensor": "FAD_Master_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 238, "target_fs": 40, "window_sec": 0.8, "stride_sec": 0.4
}

# 4. FD_ORI
FD_ORI_CONFIG = {
    "dataset_name": "FD_Ori",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(BASE_OUT_DIR, "FD_Ori_V2_Optimized.csv"),
    "table_trials": "FD_Ori_trials",
    "table_sensor": "FD_Ori_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 100, "target_fs": 50, "window_sec": 2.0, "stride_sec": 1.0
}

# 5. MOBIFALL
MOBIFALL_CONFIG = {
    "dataset_name": "MobiFall",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(BASE_OUT_DIR, "MobiFall_V2_Optimized.csv"),
    "table_trials": "MobiFall_trials",
    "table_sensor": "MobiFall_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 100, "target_fs": 50, "window_sec": 2.0, "stride_sec": 1.0
}

# =============================================================================
# ÇALIŞTIRMA
# =============================================================================
if __name__ == "__main__":
    print("\n🖥️  LOW-END PC OPTIMIZATION ACTIVE")
    print("✅ QUALITY: Resolution=50, float64, max_edge=1.75")
    print("💾 MEMORY: Frequent writes (30 windows) + GC")
    print("⚡ BUG FIX: diags_format defined correctly")
    print("="*80)
    
    # Hepsini sırayla çalıştır
    run_tda_pipeline_v2(SISFALL_CONFIG)
    run_tda_pipeline_v2(FAD_40HZ_CONFIG)
    run_tda_pipeline_v2(FAD_MASTER_CONFIG)
    run_tda_pipeline_v2(FD_ORI_CONFIG)
    run_tda_pipeline_v2(MOBIFALL_CONFIG)
    
    print("\n✅ TÜM VERİ SETLERİ TAMAMLANDI!")