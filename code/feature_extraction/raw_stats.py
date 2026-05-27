import sqlite3
import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. AYARLAR
# =============================================================================
BASE_DB_PATH = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataFirstTouch\All_Datasets_Container.db"
OUTPUT_ROOT = r"C:\Users\user\Desktop\KEB\YL_TEZ\RAW_Stats\Raw_Features_V2"

# Yazma sıklığı
WRITE_BATCH = 2000 

# =============================================================================
# 2. RAW FEATURE MOTORU (GELİŞMİŞ İSTATİSTİK)
# =============================================================================
class Raw_Feature_Engine:
    def __init__(self):
        pass

    def extract(self, window_data):
        """
        Ham veri penceresinden gelişmiş fiziksel istatistikleri çıkarır.
        Girdi: (N, 3) veya (N, 6) numpy array. (AccX, AccY, AccZ, ...)
        """
        feats = {}
        
        # Veri boyutu kontrolü
        if window_data.size == 0: return None
        
        # --- A. BİLEŞKE (MAGNITUDE) BAZLI ÖZELLİKLER ---
        # Amaç: Hareketin toplam şiddetini ölçmek (Yön bağımsız)
        
        if window_data.ndim > 1 and window_data.shape[1] >= 3:
            # Sadece ilk 3 sütunu (Acc) al
            acc_data = window_data[:, :3]
            mag = np.linalg.norm(acc_data, axis=1)
        else:
            # Tek eksense direkt mutlak değer
            acc_data = window_data
            mag = np.abs(window_data.flatten())

        # 1. Temel İstatistikler (Magnitude)
        feats['Raw_Mag_Max'] = np.max(mag)       # En sert darbe
        feats['Raw_Mag_Min'] = np.min(mag)
        feats['Raw_Mag_Mean'] = np.mean(mag)     # Genel aktivite seviyesi
        feats['Raw_Mag_Std'] = np.std(mag)       # Hareketlilik varyasyonu
        
        # 2. Enerji (Magnitude)
        feats['Raw_Mag_Energy'] = np.sum(mag**2) / len(mag)
        
        # 3. Jerk (Sarsıntı - Magnitude Türevi)
        jerk = np.diff(mag)
        if len(jerk) > 0:
            feats['Raw_Jerk_Max'] = np.max(np.abs(jerk))
            feats['Raw_Jerk_Mean'] = np.mean(np.abs(jerk))
        else:
            feats['Raw_Jerk_Max'] = 0.0
            feats['Raw_Jerk_Mean'] = 0.0

        # --- B. EKSEN BAZLI ÖZELLİKLER (YÖN VE DURUŞ) ---
        # Amaç: Eylem tipini ve düşme yönünü (Öne/Arkaya/Yana) belirlemek.
        # Sadece veri en az 3 eksenliyse (X, Y, Z) hesaplanır.
        
        if window_data.ndim > 1 and window_data.shape[1] >= 3:
            # X, Y, Z Eksenlerini Ayır
            ax = acc_data[:, 0]
            ay = acc_data[:, 1]
            az = acc_data[:, 2]
            
            # 4. Eksen Ortalamaları (Posture / Duruş Bilgisi)
            # Örn: Yerde yatarken bir eksen daima 1g (yerçekimi) gösterir, diğerleri 0g.
            feats['Raw_Axis_Mean_X'] = np.mean(ax)
            feats['Raw_Axis_Mean_Y'] = np.mean(ay)
            feats['Raw_Axis_Mean_Z'] = np.mean(az)
            
            # 5. Eksen Standart Sapmaları (Hangi eksende titreme var?)
            feats['Raw_Axis_Std_X'] = np.std(ax)
            feats['Raw_Axis_Std_Y'] = np.std(ay)
            feats['Raw_Axis_Std_Z'] = np.std(az)

            # 6. Korelasyon (Koordinasyon Bilgisi)
            # Örn: Yürürken X ve Y belli bir uyumla hareket eder. Düşerken kaos olur.
            # 0 varyans durumunda (sabit duruş) korelasyon NaN döner, bunu 0 yapmalıyız.
            try:
                # Rowvar=False sütunlar arası korelasyonu verir
                corr_matrix = np.corrcoef(acc_data, rowvar=False)
                
                # NaN kontrolü (Eğer bir eksen tamamen sabitse std=0 olur, bölme hatası verir)
                if np.isnan(corr_matrix).any():
                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

                feats['Raw_Corr_XY'] = corr_matrix[0, 1]
                feats['Raw_Corr_XZ'] = corr_matrix[0, 2]
                feats['Raw_Corr_YZ'] = corr_matrix[1, 2]
            except:
                # Herhangi bir matematiksel hatada 0 ata
                feats['Raw_Corr_XY'] = 0.0
                feats['Raw_Corr_XZ'] = 0.0
                feats['Raw_Corr_YZ'] = 0.0
        
        else:
            # Veri 3 eksenli değilse (örn. sadece magnitude verildiyse) bu özellikleri 0 doldur
            for k in ['Raw_Axis_Mean_X', 'Raw_Axis_Mean_Y', 'Raw_Axis_Mean_Z',
                      'Raw_Axis_Std_X',  'Raw_Axis_Std_Y',  'Raw_Axis_Std_Z',
                      'Raw_Corr_XY',     'Raw_Corr_XZ',     'Raw_Corr_YZ']:
                feats[k] = 0.0

        return feats

# =============================================================================
# 3. YARDIMCI FONKSİYONLAR (Pencereleme & DB)
# =============================================================================
def universal_subwindowing(signal_data, original_fs, target_fs, window_sec, stride_sec):
    """
    TDA kodlarıyla senkronizasyon için standart pencereleme.
    """
    if original_fs is None or original_fs <= 0: step = 1
    else: step = max(1, int(original_fs / target_fs))
    
    resampled = signal_data[::step]
    win_len = int(window_sec * target_fs)
    stride_len = int(stride_sec * target_fs)
    
    if len(resampled) < win_len: return None
    
    return [resampled[i:i + win_len] for i in range(0, len(resampled) - win_len + 1, stride_len)]

def setup_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def check_cols(conn, table, candidates):
    cur = conn.execute(f"PRAGMA table_info({table})")
    exist = {row[1] for row in cur.fetchall()}
    valid = []
    for c in candidates:
        if c in exist: valid.append(c)
        else:
            # SisFall typo fixleri
            if "adx1345" in c and c.replace("adx1345", "adxl345") in exist:
                valid.append(c.replace("adx1345", "adxl345"))
            elif "adxl345" in c and c.replace("adxl345", "adx1345") in exist:
                valid.append(c.replace("adxl345", "adx1345"))
    return valid

def save_to_disk(buffer, path):
    if not buffer: return
    df = pd.DataFrame(buffer)
    # Kolon sırası: Meta -> İstatistikler (Alfabetik veya mantıksal sıralama yapmak iyidir)
    meta = ['Dataset', 'Subject', 'Activity_ID', 'Trial', 'Label', 'Window_Idx']
    other = [c for c in df.columns if c not in meta]
    # İstatistik sütunlarını alfabetik sırala ki her dosyada aynı olsun
    other.sort() 
    df = df[meta + other]
    
    hdr = not os.path.exists(path)
    df.to_csv(path, mode='a', header=hdr, index=False)

# =============================================================================
# 4. PIPELINE KOŞUCUSU
# =============================================================================
def run_raw_stats_pipeline(config):
    name = config['dataset_name']
    print(f"\n{'='*80}")
    print(f"📊 GELİŞMİŞ İSTATİSTİK MOTORU: {name}")
    print(f"⚡ Özellikler: Magnitude + Eksen Ortalamaları + Korelasyon")
    print(f"{'='*80}")
    
    os.makedirs(os.path.dirname(config['output_csv']), exist_ok=True)
    
    conn = setup_db(config['db_path'])
    cols = check_cols(conn, config['table_sensor'], config['sensor_columns'])
    
    if not cols:
        print("❌ Sensör sütunları bulunamadı!"); return

    # Trial Listesi
    try:
        trials = pd.read_sql_query(f"SELECT trial_id, subject_id, activity_id, sampling_rate FROM {config['table_trials']}", conn)
    except:
        trials = pd.read_sql_query(f"SELECT trial_id, subject_id, activity_id FROM {config['table_trials']}", conn)
        trials['sampling_rate'] = config['original_fs']

    # Resume (Kaldığı yerden devam)
    if os.path.exists(config['output_csv']):
        try:
            done = pd.read_csv(config['output_csv'], usecols=['Trial'])['Trial'].unique()
            trials = trials[~trials['trial_id'].isin(done)]
            if len(done) > 0: print(f"⏩ {len(done)} trial zaten işlenmiş.")
        except: pass

    if len(trials) == 0:
        print("✅ Tüm veriler zaten tamamlanmış.")
        conn.close(); return

    engine = Raw_Feature_Engine()
    buffer = []
    total_wins = 0
    
    pbar = tqdm(trials.itertuples(index=False), total=len(trials), desc=name, unit="trial")
    
    for row in pbar:
        try:
            q = f"SELECT {', '.join(cols)}, {config['label_column']} FROM {config['table_sensor']} WHERE trial_id={row.trial_id} ORDER BY timestamp ASC"
            df_sensor = pd.read_sql_query(q, conn)
            
            if df_sensor.empty: continue
            df_sensor.dropna(subset=cols, inplace=True)
            if df_sensor.empty: continue
            
            label = 1 if df_sensor[config['label_column']].sum() > 0 else 0
            raw_data = df_sensor[cols].values
            fs = row.sampling_rate if row.sampling_rate > 0 else config['original_fs']
            
            # Pencereleme
            windows = universal_subwindowing(raw_data, fs, config['target_fs'], config['window_sec'], config['stride_sec'])
            
            if not windows: continue
            
            for i, win in enumerate(windows):
                feats = engine.extract(win)
                if feats:
                    meta = {
                        'Dataset': name, 'Subject': row.subject_id, 
                        'Activity_ID': row.activity_id, 'Trial': row.trial_id, 
                        'Label': label, 'Window_Idx': i
                    }
                    meta.update(feats)
                    buffer.append(meta)
            
            # Diske Yaz
            if len(buffer) >= WRITE_BATCH:
                save_to_disk(buffer, config['output_csv'])
                total_wins += len(buffer)
                buffer = []
                gc.collect()
                
        except Exception: continue

    if buffer:
        save_to_disk(buffer, config['output_csv'])
        total_wins += len(buffer)
        
    conn.close()
    print(f"✅ {name} Bitti! Toplam Pencere: {total_wins}")

# =============================================================================
# 5. KONFİGÜRASYONLAR (SENKRONİZASYON İÇİN SABİT)
# =============================================================================
BASE_DB = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataFirstTouch\All_Datasets_Container.db"

# OUTPUT_ROOT yolunu değiştirmeyi unutmayın!
CONFIGS = [
    {
        "dataset_name": "SisFall",
        "db_path": BASE_DB,
        "output_csv": os.path.join(OUTPUT_ROOT, "SisFall_Raw_Stats.csv"),
        "table_trials": "SisFall_trials",
        "table_sensor": "SisFall_sensor_data",
        "sensor_columns": ["acc_adx1345_x", "acc_adx1345_y", "acc_adx1345_z", "gyr_itg3200_x", "gyr_itg3200_y", "gyr_itg3200_z"],
        "label_column": "is_fall",
        "original_fs": 200, "target_fs": 50, "window_sec": 2.0, "stride_sec": 1.0
    },
    {
        "dataset_name": "FAD_Master",
        "db_path": BASE_DB,
        "output_csv": os.path.join(OUTPUT_ROOT, "FAD_Master_Raw_Stats.csv"),
        "table_trials": "FAD_Master_trials",
        "table_sensor": "FAD_Master_sensor_data",
        "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
        "label_column": "is_fall",
        "original_fs": 238, "target_fs": 40, "window_sec": 0.8, "stride_sec": 0.4
    },
    {
        "dataset_name": "FD_Ori",
        "db_path": BASE_DB,
        "output_csv": os.path.join(OUTPUT_ROOT, "FD_Ori_Raw_Stats.csv"),
        "table_trials": "FD_Ori_trials",
        "table_sensor": "FD_Ori_sensor_data",
        "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
        "label_column": "is_fall",
        "original_fs": 100, "target_fs": 50, "window_sec": 2.0, "stride_sec": 1.0
    },
    {
        "dataset_name": "MobiFall",
        "db_path": BASE_DB,
        "output_csv": os.path.join(OUTPUT_ROOT, "MobiFall_Raw_Stats.csv"),
        "table_trials": "MobiFall_trials",
        "table_sensor": "MobiFall_sensor_data",
        "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
        "label_column": "is_fall",
        "original_fs": 100, "target_fs": 50, "window_sec": 2.0, "stride_sec": 1.0
    },
    {
        "dataset_name": "FAD_40Hz",
        "db_path": BASE_DB,
        "output_csv": os.path.join(OUTPUT_ROOT, "FAD_40Hz_Raw_Stats.csv"),
        "table_trials": "FAD_40Hz_trials",
        "table_sensor": "FAD_40Hz_sensor_data",
        "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
        "label_column": "is_fall",
        "original_fs": 40, "target_fs": 40, "window_sec": 2.0, "stride_sec": 1.0
    }
]

if __name__ == "__main__":
    for conf in CONFIGS:
        run_raw_stats_pipeline(conf)