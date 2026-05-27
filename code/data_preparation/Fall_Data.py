import sqlite3
import pandas as pd
import os
from tqdm import tqdm

# ==========================================
# AYARLAR
# ==========================================
DATA_FOLDER = r"C:\Users\user\Desktop\KEB\YL_TEZ\Data\fall_data"
DB_NAME = "FallDetection_WithOrientation.db"

# 1. Aktivite Eşleşmesi (is_fall: 1 = Fall, 0 = ADL)
ACTIVITY_MAPPING = {
    "downSit":  (1, 0, "Sitting down"),
    "freeFall": (2, 1, "Free fall"),
    "runFall":  (3, 1, "Running fall"),
    "runSit":   (4, 0, "Running then sitting"),
    "walkFall": (5, 1, "Walking fall"),
    "walkSit":  (6, 0, "Walking then sitting")
}

# 2. Cihaz Yönelimi Eşleşmesi (Senin kodundan alındı)
# Bu veri ML modelinde "Feature" olarak kullanılabilir veya çıkarılabilir.
ORIENTATION_MAPPING = {
    "portrait": 1,
    "portraitUpsideDown": 2,
    "faceUp": 3,
    "faceDown": 4,
    "landscapeRight": 5,
    "landscapeLeft": 6,
    "Unknown": 99
}

# ==========================================
# TABLO OLUŞTURMA
# ==========================================
def create_tables(conn):
    cursor = conn.cursor()

    # Subjects
    cursor.execute("""CREATE TABLE IF NOT EXISTS subjects (
        subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_code TEXT UNIQUE
    )""")

    # Activity Info
    cursor.execute("""CREATE TABLE IF NOT EXISTS activity_info (
        activity_id INTEGER PRIMARY KEY,
        description TEXT,
        class_type TEXT
    )""")

    # Trials (Metadata)
    cursor.execute("""CREATE TABLE IF NOT EXISTS trials (
        trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER,
        activity_id INTEGER,
        dataset_version TEXT,
        device_pos TEXT,
        sampling_rate INTEGER,
        trial_no INTEGER,
        FOREIGN KEY(subject_id) REFERENCES subjects(subject_id),
        FOREIGN KEY(activity_id) REFERENCES activity_info(activity_id)
    )""")

    # --- SENSOR DATA (Orientation Eklendi) ---
    cursor.execute("""CREATE TABLE IF NOT EXISTS sensor_data (
        data_id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER,
        trial_id INTEGER,
        activity_id INTEGER,
        is_fall INTEGER,         -- 1: Fall, 0: ADL
        timestamp REAL,          -- Normalized seconds
        device_orientation INTEGER, -- YENİ EKLENDİ (1-6 veya 99)
        acc_x REAL, acc_y REAL, acc_z REAL,
        gyr_x REAL, gyr_y REAL, gyr_z REAL,
        mag_x REAL, mag_y REAL, mag_z REAL,
        bar_pressure REAL,
        FOREIGN KEY(subject_id) REFERENCES subjects(subject_id),
        FOREIGN KEY(trial_id) REFERENCES trials(trial_id),
        FOREIGN KEY(activity_id) REFERENCES activity_info(activity_id)
    )""")
    
    # İndeksler
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_grouping ON sensor_data(subject_id, activity_id, trial_id)")
    
    conn.commit()

# ==========================================
# VERİ İŞLEME
# ==========================================
def process_kaggle_data():
    if not os.path.exists(DATA_FOLDER):
        print(f"HATA: Klasör bulunamadı -> {DATA_FOLDER}")
        return

    conn = sqlite3.connect(DB_NAME)
    create_tables(conn)
    cursor = conn.cursor()

    # Tekil kullanıcı oluştur
    DEFAULT_SUBJECT_CODE = "Kaggle_Anon_User"
    cursor.execute("INSERT OR IGNORE INTO subjects (original_code) VALUES (?)", (DEFAULT_SUBJECT_CODE,))
    cursor.execute("SELECT subject_id FROM subjects WHERE original_code=?", (DEFAULT_SUBJECT_CODE,))
    subject_id = cursor.fetchone()[0]

    # Aktiviteleri kaydet
    for folder_name, (act_id, is_fall, desc) in ACTIVITY_MAPPING.items():
        class_type = "FALL" if is_fall == 1 else "ADL"
        cursor.execute("INSERT OR IGNORE INTO activity_info (activity_id, description, class_type) VALUES (?,?,?)",
                       (act_id, desc, class_type))
    conn.commit()

    # Dosyaları Bul
    all_files = []
    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            if file.endswith(".csv"):
                all_files.append(os.path.join(root, file))

    print(f"Toplam {len(all_files)} CSV dosyası bulundu. İşleniyor...")

    trial_counter = 1
    
    for file_path in tqdm(all_files, desc="Processing CSVs"):
        folder_name = os.path.basename(os.path.dirname(file_path))
        
        if folder_name not in ACTIVITY_MAPPING:
            continue

        act_id, is_fall, _ = ACTIVITY_MAPPING[folder_name]

        try:
            # CSV Oku
            df = pd.read_csv(file_path, delimiter=';')
            df.columns = df.columns.str.strip().str.lower()
            
            # Zorunlu alan kontrolü (Sadece ACC ve Orientation yeterli)
            if 'accelerationx' not in df.columns:
                continue

            # --- TIMESTAMP HESAPLAMA ---
            timestamps = []
            if 'timestamp' in df.columns:
                try:
                    first_ts = df['timestamp'].iloc[0]
                    timestamps = (df['timestamp'] - first_ts) / 1000.00 
                except:
                    timestamps = df.index / 50.0 
            else:
                timestamps = df.index / 50.0 

            # --- TRIAL KAYDI ---
            cursor.execute("""
                INSERT INTO trials (subject_id, activity_id, dataset_version, device_pos, sampling_rate, trial_no)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (subject_id, act_id, "Kaggle_FallData_CSV", "Mobile_Device", 50, trial_counter))
            current_trial_id = cursor.lastrowid

            # --- SENSOR DATA HAZIRLAMA ---
            batch_data = []
            for i in range(len(df)):
                ts = float(timestamps.iloc[i]) if hasattr(timestamps, 'iloc') else float(timestamps[i])
                ax = float(df['accelerationx'].iloc[i])
                ay = float(df['accelerationy'].iloc[i])
                az = float(df['accelerationz'].iloc[i])
                
                # --- ORIENTATION PARSE ETME (Eski kodundan restore edildi) ---
                raw_ori = str(df['deviceorientation'].iloc[i]) if 'deviceorientation' in df.columns else "Unknown"
                ori_val = ORIENTATION_MAPPING.get(raw_ori, 99) # Bulamazsa 99 basar
                
                batch_data.append((
                    subject_id,
                    current_trial_id,
                    act_id,
                    is_fall,
                    ts,
                    ori_val, # <-- device_orientation buraya eklendi
                    ax, ay, az,
                    None, None, None, # Gyro
                    None, None, None, # Mag
                    None              # Bar
                ))

            # Veriyi Bas
            cursor.executemany("""
                INSERT INTO sensor_data 
                (subject_id, trial_id, activity_id, is_fall, timestamp, device_orientation,
                 acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z, bar_pressure)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, batch_data)

            trial_counter += 1

        except Exception as e:
            print(f"Hata oluştu ({file_path}): {e}")

    conn.commit()
    conn.close()
    print(f"\n✅ İşlem tamamlandı! Veritabanı: {DB_NAME}")

if __name__ == "__main__":
    process_kaggle_data()