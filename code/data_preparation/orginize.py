#%% PLD FULL Data
# -*- coding: utf-8 -*-
"""
FallAllD JSON -> SQLite DB
Memory-safe streaming ile işlenir ve trial ilerlemesi gösterilir.
"""

import sqlite3
import os
import ijson
from tqdm import tqdm
import decimal

# ---------------------------
# Ayarlar
# ---------------------------
json_file = r"C:\Users\user\Desktop\KEB\YL_TEZ\Data\archive_pckl\FallAllD.json"
db_file = "FallAllD.db"
DEVICE_MAP = {"Waist": 1, "Wrist": 2, "Neck": 3}

# Sampling rate map (Hz)
SAMPLING_RATE = {
    "Acc": 238,
    "Gyr": 238,
    "Mag": 80,
    "Bar": 10
}

# ---------------------------
# Veritabanı oluştur
# ---------------------------
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Subjects tablosu
cursor.execute("""
CREATE TABLE IF NOT EXISTS PLD_Subjects(
    subject_id INTEGER PRIMARY KEY,
    age INTEGER,
    gender TEXT
)
""")

# Devices tablosu
cursor.execute("""
CREATE TABLE IF NOT EXISTS PLD_Devices(
    device_code INTEGER PRIMARY KEY,
    device_name TEXT
)
""")
for name, code in DEVICE_MAP.items():
    cursor.execute("INSERT OR IGNORE INTO PLD_Devices(device_code, device_name) VALUES (?,?)", (code,name))

# Activities tablosu
cursor.execute("""
CREATE TABLE IF NOT EXISTS PLD_Activities(
    activity_id INTEGER PRIMARY KEY,
    description TEXT,
    numeric_code INTEGER,
    activity_type TEXT
)
""")

# Trials tablosu
cursor.execute("""
CREATE TABLE IF NOT EXISTS PLD_Trials(
    experimental_id INTEGER PRIMARY KEY,
    subject_id INTEGER,
    device_code INTEGER,
    activity_id INTEGER,
    trial_no INTEGER,
    FOREIGN KEY(subject_id) REFERENCES PLD_Subjects(subject_id),
    FOREIGN KEY(device_code) REFERENCES PLD_Devices(device_code),
    FOREIGN KEY(activity_id) REFERENCES PLD_Activities(activity_id)
)
""")

# Sensor Data tablosu
cursor.execute("""
CREATE TABLE IF NOT EXISTS PLD_Data(
    data_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experimental_id INTEGER,
    activity_id INTEGER,
    timestamp REAL,
    acc_x REAL, acc_y REAL, acc_z REAL,
    gyr_x REAL, gyr_y REAL, gyr_z REAL,
    mag_x REAL, mag_y REAL, mag_z REAL,
    bar REAL,
    FOREIGN KEY(experimental_id) REFERENCES PLD_Trials(experimental_id),
    FOREIGN KEY(activity_id) REFERENCES PLD_Activities(activity_id)
)
""")
conn.commit()

# ---------------------------
# Yardımcı fonksiyon: decimal to float
# ---------------------------
def decimal_to_float(value):
    """Decimal değeri float'a çevirir, diğer tipleri olduğu gibi bırakır"""
    if isinstance(value, decimal.Decimal):
        return float(value)
    elif isinstance(value, list):
        return [decimal_to_float(item) for item in value]
    elif isinstance(value, tuple):
        return tuple(decimal_to_float(item) for item in value)
    else:
        return value

# ---------------------------
# JSON streaming ile oku ve DB'ye ekle
# ---------------------------
print("JSON dosyası okunuyor... Bu biraz zaman alabilir.")

experimental_id_counter = 1
activity_numeric_map = {}
numeric_counter = 1

with open(json_file, "r") as f:
    # ijson ile JSON listesindeki her trial için iterator
    for trial in tqdm(ijson.items(f, "item"), desc="Trials işleniyor"):

        subject_id = trial.get("SubjectID")
        device_name = trial.get("Device")
        device_code = DEVICE_MAP.get(device_name, 0)
        activity_id = trial.get("ActivityID")
        trial_no = trial.get("TrialNo", 1)

        # Subjects tablosu
        cursor.execute("INSERT OR IGNORE INTO PLD_Subjects(subject_id) VALUES (?)", (subject_id,))

        # Activities tablosu
        if activity_id not in activity_numeric_map:
            activity_numeric_map[activity_id] = numeric_counter
            numeric_counter += 1
            act_type = "ADL" if activity_id < 100 else "Fall"
            cursor.execute("INSERT OR IGNORE INTO PLD_Activities(activity_id, description, numeric_code, activity_type) VALUES (?,?,?,?)",
                           (activity_id, f"Activity {activity_id}", activity_numeric_map[activity_id], act_type))

        # Trials tablosu
        experimental_id = experimental_id_counter
        cursor.execute("""INSERT INTO PLD_Trials(experimental_id, subject_id, device_code, activity_id, trial_no)
                          VALUES (?,?,?,?,?)""", (experimental_id, subject_id, device_code, activity_id, trial_no))

        # Sensor verisi ekleme - decimal değerleri float'a çevir
        acc_data = decimal_to_float(trial.get("Acc", []))
        gyr_data = decimal_to_float(trial.get("Gyr", []))
        mag_data = decimal_to_float(trial.get("Mag", []))
        bar_data = decimal_to_float(trial.get("Bar", []))
        
        n_samples = max(len(acc_data), len(gyr_data), len(mag_data), len(bar_data))

        for i in range(n_samples):
            t_acc = i / SAMPLING_RATE["Acc"] if i < len(acc_data) else 0
            t_gyr = i / SAMPLING_RATE["Gyr"] if i < len(gyr_data) else 0
            t_mag = i / SAMPLING_RATE["Mag"] if i < len(mag_data) else 0
            t_bar = i / SAMPLING_RATE["Bar"] if i < len(bar_data) else 0

            acc_x, acc_y, acc_z = acc_data[i] if i < len(acc_data) else (0,0,0)
            gyr_x, gyr_y, gyr_z = gyr_data[i] if i < len(gyr_data) else (0,0,0)
            mag_x, mag_y, mag_z = mag_data[i] if i < len(mag_data) else (0,0,0)
            
            # Barometre verisi için özel işleme
            bar_value = 0
            if i < len(bar_data):
                bar_item = bar_data[i]
                # Eğer bar_item bir liste ise, ilk elemanı al
                if isinstance(bar_item, list):
                    bar_value = float(bar_item[0]) if len(bar_item) > 0 else 0
                else:
                    bar_value = float(bar_item)

            cursor.execute("""INSERT INTO PLD_Data(experimental_id, activity_id, timestamp,
                             acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z,
                             mag_x, mag_y, mag_z, bar)
                             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                           (experimental_id, activity_id, t_acc,
                            acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z,
                            mag_x, mag_y, mag_z, bar_value))

        if experimental_id_counter % 10 == 0:
            conn.commit()  # Her 10 trial'da bir commit yap, RAM kullanımını azalt

        experimental_id_counter += 1

conn.commit()
conn.close()
print(f"✅ FallAllD JSON verisi DB'ye başarıyla işlendi ve '{db_file}' oluşturuldu!")


#%% PLD 40 Sample
# -*- coding: utf-8 -*-
"""
FallAllD 40Hz JSON -> SQLite DB
Memory-safe streaming ile işlenir, sadece Trials ve Data tabloları oluşturulur.
"""

import sqlite3
import os
import ijson
from tqdm import tqdm
import decimal

# --------------------------- 
# Ayarlar
# --------------------------- 
json_file = r"C:\Users\kbybt\Desktop\EVREN\MatMüh_İTÜ_ARŞ_GÖR\YL_TEZ\Data\archive_pckl\FallAllD_40SamplesPerSec_ActivityIdsFiltered.json"
db_file = "FallAllD_40Hz.db"
DEVICE_MAP = {"Waist": 1, "Wrist": 2}  # 40Hz için Neck yok

# Sampling rate map (Hz)
SAMPLING_RATE = {
    "Acc": 40,
    "Gyr": 40
}

# --------------------------- 
# Veritabanı oluştur
# --------------------------- 
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Devices tablosu
cursor.execute("""
CREATE TABLE IF NOT EXISTS PLSD_Devices(
    device_code INTEGER PRIMARY KEY,
    device_name TEXT
)
""")
for name, code in DEVICE_MAP.items():
    cursor.execute("INSERT OR IGNORE INTO PLSD_Devices(device_code, device_name) VALUES (?,?)", (code,name))

# Activities tablosu
cursor.execute("""
CREATE TABLE IF NOT EXISTS PLSD_Activities(
    activity_id INTEGER PRIMARY KEY,
    description TEXT,
    numeric_code INTEGER,
    activity_type TEXT
)
""")

# Trials tablosu
cursor.execute("""
CREATE TABLE IF NOT EXISTS PLSD_Trials(
    experimental_id INTEGER PRIMARY KEY,
    subject_id INTEGER,
    device_code INTEGER,
    activity_id INTEGER,
    trial_no INTEGER,
    FOREIGN KEY(device_code) REFERENCES PLSD_Devices(device_code),
    FOREIGN KEY(activity_id) REFERENCES PLSD_Activities(activity_id)
)
""")

# Sensor Data tablosu
cursor.execute("""
CREATE TABLE IF NOT EXISTS PLSD_Data(
    data_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experimental_id INTEGER,
    activity_id INTEGER,
    timestamp REAL,
    acc_x REAL, acc_y REAL, acc_z REAL,
    gyr_x REAL, gyr_y REAL, gyr_z REAL,
    FOREIGN KEY(experimental_id) REFERENCES PLSD_Trials(experimental_id),
    FOREIGN KEY(activity_id) REFERENCES PLSD_Activities(activity_id)
)
""")
conn.commit()

# --------------------------- 
# Yardımcı fonksiyon: decimal to float
# --------------------------- 
def decimal_to_float(value):
    """Decimal değeri float'a çevirir, diğer tipleri olduğu gibi bırakır"""
    if isinstance(value, decimal.Decimal):
        return float(value)
    elif isinstance(value, list):
        return [decimal_to_float(item) for item in value]
    elif isinstance(value, tuple):
        return tuple(decimal_to_float(item) for item in value)
    else:
        return value

# --------------------------- 
# JSON streaming ile oku ve DB'ye ekle
# --------------------------- 
print("40Hz JSON dosyası okunuyor... Bu biraz zaman alabilir.")

experimental_id_counter = 1
activity_numeric_map = {}
numeric_counter = 1

with open(json_file, "r") as f:
    for trial in tqdm(ijson.items(f, "item"), desc="Trials işleniyor"):
        subject_id = trial.get("SubjectID")
        device_name = trial.get("Device")
        device_code = DEVICE_MAP.get(device_name, 0)
        activity_id = trial.get("ActivityID")
        trial_no = trial.get("TrialNo", 1)

        # Activities tablosu
        if activity_id not in activity_numeric_map:
            activity_numeric_map[activity_id] = numeric_counter
            numeric_counter += 1
            act_type = "ADL" if activity_id < 100 else "Fall"
            cursor.execute(
                "INSERT OR IGNORE INTO PLSD_Activities(activity_id, description, numeric_code, activity_type) VALUES (?,?,?,?)",
                (activity_id, f"Activity {activity_id}", activity_numeric_map[activity_id], act_type)
            )

        # Trials tablosu
        experimental_id = experimental_id_counter
        cursor.execute(
            "INSERT INTO PLSD_Trials(experimental_id, subject_id, device_code, activity_id, trial_no) VALUES (?,?,?,?,?)",
            (experimental_id, subject_id, device_code, activity_id, trial_no)
        )

        # Sensor verisi ekleme
        acc_data = decimal_to_float(trial.get("Acc", []))
        gyr_data = decimal_to_float(trial.get("Gyr", []))
        n_samples = max(len(acc_data), len(gyr_data))

        for i in range(n_samples):
            t_acc = i / SAMPLING_RATE["Acc"] if i < len(acc_data) else 0
            t_gyr = i / SAMPLING_RATE["Gyr"] if i < len(gyr_data) else 0

            acc_x, acc_y, acc_z = acc_data[i] if i < len(acc_data) else (0,0,0)
            gyr_x, gyr_y, gyr_z = gyr_data[i] if i < len(gyr_data) else (0,0,0)

            cursor.execute(
                "INSERT INTO PLSD_Data(experimental_id, activity_id, timestamp, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z) VALUES (?,?,?,?,?,?,?,?,?)",
                (experimental_id, activity_id, t_acc, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
            )

        if experimental_id_counter % 10 == 0:
            conn.commit()  # RAM kullanımını azalt

        experimental_id_counter += 1

conn.commit()
conn.close()
print(f"✅ FallAllD 40Hz JSON verisi DB'ye başarıyla işlendi ve '{db_file}' oluşturuldu!")

#%%  Fall Data
import sqlite3
import pandas as pd
import os
from datetime import datetime

# -------------------------------
# Ayarlar
# -------------------------------
db_file = "FAD.db"
data_folder = r"C:\Users\kbybt\Desktop\EVREN\MatMüh_İTÜ_ARŞ_GÖR\YL_TEZ\Data\fall_data"  # FALL_DATA klasörü

# FALL DATA aktivite eşleştirmesi
activity_mapping = {
    "downSit": 1,    # FD - Sitting down
    "freeFall": 2,   # FD - Free fall
    "runFall": 3,    # FD - Running fall
    "runSit": 4,     # FD - Running then sitting
    "walkFall": 5,   # FD - Walking fall
    "walkSit": 6     # FD - Walking then sitting
}

# FD aktivite açıklamaları
fd_activity_descriptions = {
    1: "Sitting down",
    2: "Free fall",
    3: "Running fall",
    4: "Running then sitting",
    5: "Walking fall",
    6: "Walking then sitting"
}

device_orientation_mapping = {
    "portrait": 1,
    "portraitUpsideDown":2,
    "faceUp": 3,
    "faceDown":4,
    "landscapeRight":5,    
    "landscapeLeft":6,
    "Unknown": 99
}

device_orientation_descriptions = {
    1:"portrait",
    2:"portraitUpsideDown",
    3:"faceUp", 
    4:"faceDown",
    5: "landscapeRight",
    6: "landscapeLeft",
    99:"Unknown"
}
# -------------------------------
# Veritabanına bağlan
# -------------------------------
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# FD (Fall Data) için tamamen yeni tablolar oluştur
cursor.execute("""
CREATE TABLE IF NOT EXISTS FAD_Activities (
    activity_id INTEGER PRIMARY KEY,
    description TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS FAD_DeviceOrientation (
    device_orientation_id INTEGER PRIMARY KEY,
    description TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS FAD_Trials (
    experimental_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    activity_id INTEGER,
    trial_no INTEGER,
    FOREIGN KEY(activity_id) REFERENCES FD_activity_descriptions(activity_id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS FAD_Data (
    data_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experimental_id INTEGER,
    activity_id INTEGER,
    timestamp TEXT,
    device_orientation INTEGER,
    acc_x REAL,
    acc_y REAL,
    acc_z REAL,
    FOREIGN KEY(experimental_id) REFERENCES FD_trials(experimental_id),
    FOREIGN KEY(activity_id) REFERENCES FD_activity_descriptions(activity_id)
)
""")

# -------------------------------
# FD_activity_descriptions tablosunu doldur
# -------------------------------
for activity_id, description in fd_activity_descriptions.items():
    cursor.execute("""
        INSERT OR IGNORE INTO FAD_Activities (activity_id, description)
        VALUES (?, ?)
    """, (activity_id, description))

conn.commit()

# -------------------------------
# FD_device_orientation_descriptions tablosunu doldur
# -------------------------------
for orientation_id, description in device_orientation_descriptions.items():
    cursor.execute("""
        INSERT OR IGNORE INTO FAD_DeviceOrientation (device_orientation_id, description)
        VALUES (?, ?)
    """, (orientation_id, description))

conn.commit()

# -------------------------------
# Mevcut en yüksek experimental_id'yi bul (FD_trials için)
# -------------------------------
cursor.execute("SELECT MAX(experimental_id) FROM FAD_trials")
result = cursor.fetchone()
experimental_id_counter = result[0] + 1 if result[0] is not None else 1

print(f"Starting FD experimental_id from: {experimental_id_counter}")

# -------------------------------
# FALL DATA CSV dosyalarını işle
# -------------------------------
processed_files = 0
total_records = 0

# Tüm alt klasörleri ve CSV dosyalarını bul
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith('.csv'):
            csv_path = os.path.join(root, file)
            folder_name = os.path.basename(root)
            
            # Klasör adına göre activity_id belirle
            activity_id = activity_mapping.get(folder_name, 99)  # 99 = Unknown activity
            
            # Dosya adından trial_no'yu çıkar
            try:
                base_name = os.path.splitext(file)[0]
                digits = ''.join(filter(str.isdigit, base_name))
                trial_no = int(digits) if digits else 1
            except:
                trial_no = 1
            
            print(f"Processing: {csv_path}")
            print(f"  Folder: {folder_name}, Activity ID: {activity_id}, Trial: {trial_no}")
            
            # CSV dosyasını oku
            try:
                df = pd.read_csv(csv_path, delimiter=';')
                
                # Sütun isimlerini standartlaştır
                df.columns = df.columns.str.strip().str.lower()
                
                # Gerekli sütunları kontrol et
                required_columns = ['accelerationx', 'accelerationy', 'accelerationz']
                if all(col in df.columns for col in required_columns):
                    # Veriyi temizle ve dönüştür
                    df = df.dropna(subset=required_columns)
                    
                    # Timestamp oluştur
                    if 'timestamp' in df.columns:
                        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                    
                    # NaN timestamp'leri temizle
                    df = df.dropna(subset=['datetime'])
                    
                    if len(df) == 0:
                        print(f"  Warning: No valid data in {csv_path}")
                        continue
                    
                    # FD_trials tablosuna ekle
                    user_id = 0
                    
                    cursor.execute("""
                        INSERT INTO FAD_Trials (experimental_id, user_id, activity_id, trial_no)
                        VALUES (?, ?, ?, ?)
                    """, (experimental_id_counter, user_id, activity_id, trial_no))
                    
                    # FD_sensor_data tablosuna ekle
                    record_count = 0
                    for _, row in df.iterrows():
                        try:
                            device_orientation = device_orientation_mapping.get(
                                str(row['deviceorientation']),  # CSV’deki orijinal değer
                                99                              # mapping’de yoksa 99
                                )
                            
                            if device_orientation == 99:
                                print(row.get('deviceorientation'))
                                
                            cursor.execute("""
                                INSERT INTO FAD_Data (experimental_id, activity_id, timestamp, device_orientation, acc_x, acc_y, acc_z)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                experimental_id_counter,
                                activity_id,
                                row['datetime'].isoformat(),
                                device_orientation,
                                row['accelerationx'],
                                row['accelerationy'],
                                row['accelerationz']
                            ))
                            record_count += 1
                        except Exception as e:
                            print(f"    Error inserting record: {e}")
                            continue
                    
                    total_records += record_count
                    processed_files += 1
                    
                    print(f"  Added {record_count} records to FD database (Experimental ID: {experimental_id_counter})")
                    
                    experimental_id_counter += 1
                    
                else:
                    print(f"  Warning: Missing acceleration columns in {csv_path}")
                    print(f"  Available columns: {list(df.columns)}")
                    
            except Exception as e:
                print(f"  Error processing {csv_path}: {str(e)}")
                import traceback
                traceback.print_exc()

# -------------------------------
# Değişiklikleri kaydet ve bağlantıyı kapat
# -------------------------------
conn.commit()

# Tablo istatistiklerini göster
cursor.execute("SELECT COUNT(*) FROM FAD_Activities")
fd_activity_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM FAD_Trials")
fd_trials_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM FAD_Data")
fd_sensor_count = cursor.fetchone()[0]

print(f"\n{'='*50}")
print("FALL DATA TABLO İSTATİSTİKLERİ:")
print(f"{'='*50}")
print(f"FD_activity_descriptions: {fd_activity_count} kayıt")
print(f"FD_trials: {fd_trials_count} deneme")
print(f"FD_sensor_data: {fd_sensor_count} sensör verisi")
print(f"{'='*50}")


conn.close()

print("\nProcessing completed!")
print(f"Processed files: {processed_files}")
print(f"Total records added: {total_records}")
print(f"Last FD experimental_id: {experimental_id_counter - 1}")
print(f"Veritabanı '{db_file}' güncellendi!")
full_path = os.path.abspath(db_file)
print(f"Veritabanı tam yolu: {full_path}")

#%% MobiFall

import os
import sqlite3

# ---------------------------
# Veritabanı oluşturma
# ---------------------------
db_name = "MobiFallDataset.db"
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# ---------------------------
# Tablolar
# ---------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS MFD_Subjects (
    subject_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    age INTEGER,
    height_cm INTEGER,
    weight_kg INTEGER,
    gender TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS MFD_Activities (
    activity_id INTEGER PRIMARY KEY,
    code TEXT UNIQUE,
    name TEXT,
    type TEXT,
    trials INTEGER,
    duration TEXT,
    description TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS MFD_Data (
    data_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id INTEGER,
    subject_id INTEGER,
    activity_id INTEGER,
    is_fall INTEGER,
    trial_no INTEGER,
    timestamp_ns_acc INTEGER,
    timestamp_ns_gyro INTEGER,
    timestamp_ns_ori INTEGER,
    acc_x REAL, acc_y REAL, acc_z REAL,
    gyro_x REAL, gyro_y REAL, gyro_z REAL,
    ori_azimuth REAL, ori_pitch REAL, ori_roll REAL
)
""")

conn.commit()

# ---------------------------
# Subjects ekleme
# ---------------------------
subjects = [
    (1, "pat1", "pat1", 32, 180, 85, "M"),
    (2, "pat2", "pat2", 26, 169, 64, "M"),
    (3, "pat3", "pat3", 26, 164, 55, "F"),
    (4, "pat4", "pat4", 32, 186, 93, "M"),
    (5, "pat5", "pat5", 36, 160, 50, "F"),
    (6, "pat6", "pat6", 22, 172, 62, "F"),
    (7, "pat7", "pat7", 25, 189, 80, "M"),
    (8, "pat8", "pat8", 22, 183, 93, "M"),
    (9, "pat9", "pat9", 30, 177, 102, "M"),
    (10, "pat10", "pat10", 26, 170, 90, "F"),
    (11, "Pat11", "pat11", 26, 168, 80, "F"),
    (12, "sub12", "sub12", 29, 178, 83, "M"),
    (13, "sub13", "sub13", 24, 177, 62, "M"),
    (14, "sub14", "sub14", 24, 178, 85, "M"),
    (15, "sub15", "sub15", 25, 173, 82, "M"),
    (16, "sub16", "sub16", 27, 172, 56, "F"),
    (17, "sub17", "sub17", 25, 173, 67, "M"),
    (18, "sub18", "sub18", 25, 176, 73, "M"),
    (19, "sub19", "sub19", 25, 161, 63, "F"),
    (20, "sub20", "sub20", 26, 178, 71, "M"),
    (21, "sub21", "sub21", 25, 180, 70, "M"),
    (29, "sub29", "sub29", 27, 186, 103, "M"),
    (30, "sub30", "sub30", 47, 172, 90, "M"),
    (31, "sub31", "sub31", 27, 170, 75, "M")
]

cursor.executemany("INSERT OR IGNORE INTO MFD_Subjects VALUES (?,?,?,?,?,?,?)", subjects)

# ---------------------------
# Activities ekleme
# ---------------------------
activities = [
    # ADL Activities
    (1, "STD", "Standing", "ADL", 1, "5m", "Standing with subtle movements"),
    (2, "WAL", "Walking", "ADL", 1, "5m", "Normal walking"),
    (3, "JOG", "Jogging", "ADL", 3, "30s", "Jogging"),
    (4, "JUM", "Jumping", "ADL", 3, "30s", "Continuous jumping"),
    (5, "STU", "Stairs up", "ADL", 6, "10s", "Stairs up (10 stairs)"),
    (6, "STN", "Stairs down", "ADL", 6, "10s", "Stairs down (10 stairs)"),
    (7, "SCH", "Sit chair", "ADL", 6, "6s", "Sitting on a chair"),
    (8, "CSI", "Car-step in", "ADL", 6, "6s", "Step in a car"),
    (9, "CSO", "Car-step out", "ADL", 6, "6s", "Step out a car"),

    # Fall Activities
    (10, "FOL", "Forward-lying", "FALL", 3, "10s", "Fall forward from standing, use of hands to dampen fall"),
    (11, "FKL", "Front-knees-lying", "FALL", 3, "10s", "Fall forward from standing, first impact on knees"),
    (12, "BSC", "Back-sitting-chair", "FALL", 3, "10s", "Fall backward while trying to sit on a chair"),
    (13, "SDL", "Sideward-lying", "FALL", 3, "10s", "Fall sidewards from standing, bending legs")
]

cursor.executemany("INSERT OR IGNORE INTO MFD_Activities VALUES (?,?,?,?,?,?,?)", activities)
conn.commit()

# ---------------------------
# Dosya işleme fonksiyonları
# ---------------------------
def parse_sensor_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('@') or line.startswith('#') or line == '' or line.startswith('DATA'):
                continue
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    timestamp = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    data.append((timestamp, x, y, z))
                except ValueError:
                    continue
    return data

def find_matching_files(base_dir):
    file_groups = {}
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                parts = file.split('_')
                if len(parts) >= 4:
                    activity_code = parts[0]
                    sensor_type = parts[1]
                    subject_id = int(parts[2])
                    trial_no = int(parts[3].replace('.txt',''))
                    key = f"{activity_code}_{subject_id}_{trial_no}"
                    if key not in file_groups:
                        file_groups[key] = {}
                    file_groups[key][sensor_type] = os.path.join(root, file)
    return file_groups

def process_file_group(key, files_dict, trial_counter, activity_map):
    if 'acc' not in files_dict or 'gyro' not in files_dict or 'ori' not in files_dict:
        print(f"Skipping incomplete group: {key}")
        return trial_counter
    
    parts = key.split('_')
    activity_code = parts[0]
    subject_id = int(parts[1])
    trial_no = int(parts[2])
    
    is_fall = 1 if activity_map[activity_code]['type'] == 'FALL' else 0
    activity_id = activity_map[activity_code]['id']
    
    acc_data = parse_sensor_file(files_dict['acc'])
    gyro_data = parse_sensor_file(files_dict['gyro'])
    ori_data = parse_sensor_file(files_dict['ori'])
    
    min_length = min(len(acc_data), len(gyro_data), len(ori_data))
    
    records = []
    for i in range(min_length):
        ts_acc, acc_x, acc_y, acc_z = acc_data[i]
        ts_gyro, gyro_x, gyro_y, gyro_z = gyro_data[i]
        ts_ori, ori_azimuth, ori_pitch, ori_roll = ori_data[i]
        records.append((
            trial_counter, subject_id, activity_id, is_fall, trial_no,
            ts_acc, ts_gyro, ts_ori,
            acc_x, acc_y, acc_z,
            gyro_x, gyro_y, gyro_z,
            ori_azimuth, ori_pitch, ori_roll
        ))
    
    cursor.executemany("""
    INSERT INTO MFD_Data 
    (trial_id, subject_id, activity_id, is_fall, trial_no, 
     timestamp_ns_acc, timestamp_ns_gyro, timestamp_ns_ori,
     acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, 
     ori_azimuth, ori_pitch, ori_roll)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, records)
    
    print(f"Processed Trial {trial_counter}: Subject {subject_id}, Activity {activity_code}, Records: {min_length}")
    return trial_counter + 1

# ---------------------------
# Ana işlem
# ---------------------------
def main():
    base_dir = r"C:\Users\kbybt\Desktop\EVREN\MatMüh_İTÜ_ARŞ_GÖR\YL_TEZ\Data\MobiFall_Dataset_v2.0"
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return
    
    cursor.execute("SELECT activity_id, code, type FROM MFD_Activities")
    activity_map = {row[1]: {'id': row[0], 'type': row[2]} for row in cursor.fetchall()}
    
    file_groups = find_matching_files(base_dir)
    print(f"Found {len(file_groups)} file groups")
    
    trial_counter = 1
    processed_count = 0
    
    for key, files_dict in file_groups.items():
        trial_counter = process_file_group(key, files_dict, trial_counter, activity_map)
        processed_count += 1
        if processed_count % 5 == 0:
            print(f"Committed {processed_count} groups so far...")
            conn.commit()
    
    conn.commit()
    print(f"Processing complete! Total trials: {trial_counter-1}")
    print(f"Database saved as: {db_name}")

if __name__ == "__main__":
    main()
    conn.close()

#%% SIS FALL
import os
import sqlite3
import re
import time



# Veritabanı dosyası
DB_FILE = "sisfall.db"

# Sensör özellikleri (README'den)
SENSOR_SPECS = {
    'ADXL345': {'range': 16, 'resolution': 13, 'conversion_factor': (2*16)/(2**13)},
    'ITG3200': {'range': 2000, 'resolution': 16, 'conversion_factor': (2*2000)/(2**16)},
    'MMA8451Q': {'range': 8, 'resolution': 14, 'conversion_factor': (2*8)/(2**14)}
}

# Activity type mapping: ADL=0, FALL=1
ACTIVITY_TYPE_MAP = {'ADL':0, 'FALL':1}

# Activity numeric mapping
# ADL 1-19, FALL 20-34
ACTIVITY_NUMERIC = {
    'D01':1,'D02':2,'D03':3,'D04':4,'D05':5,'D06':6,'D07':7,'D08':8,'D09':9,'D10':10,
    'D11':11,'D12':12,'D13':13,'D14':14,'D15':15,'D16':16,'D17':17,'D18':18,'D19':19,
    'F01':20,'F02':21,'F03':22,'F04':23,'F05':24,'F06':25,'F07':26,'F08':27,'F09':28,
    'F10':29,'F11':30,'F12':31,'F13':32,'F14':33,'F15':34
}

# Subjects mapping: numeric id
SUBJECT_NUMERIC = {}  # SA01 -> 1, SE01 -> 24 vb, main script dolduracak

def create_database_tables():
    """Tüm tabloları oluştur"""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    
    # Subjects tablosu
    cur.execute("""
    CREATE TABLE IF NOT EXISTS SFD_Subjects (
        subject_id INTEGER PRIMARY KEY,
        subject_code TEXT,
        age_group TEXT NOT NULL,
        age INTEGER NOT NULL,
        height INTEGER NOT NULL,
        weight REAL NOT NULL,
        gender TEXT NOT NULL
    )
    """)
    
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS SFD_Activities (
        activity_code INTEGER PRIMARY KEY,   -- numeric code
        original_code TEXT UNIQUE,           -- orijinal D/F kodu
        activity_type INTEGER,               -- 0=ADL, 1=FALL
        description TEXT,
        typical_duration INTEGER,
        typical_trials INTEGER
    )
    """)
    
    # Sensor dataset raw
    cur.execute("""
    CREATE TABLE IF NOT EXISTS SFD_DataRaw (
        subject_id INTEGER REFERENCES subjects(subject_id),
        activity_code INTEGER REFERENCES activity_info(activity_code),
        activity_type INTEGER,
        trial_number INTEGER,
        sample_index INTEGER,
        adxl345_x INTEGER,
        adxl345_y INTEGER,
        adxl345_z INTEGER,
        itg3200_x INTEGER,
        itg3200_y INTEGER,
        itg3200_z INTEGER,
        mma8451q_x INTEGER,
        mma8451q_y INTEGER,
        mma8451q_z INTEGER
    )
    """)
    
    # Sensor dataset converted
    cur.execute("""
    CREATE TABLE IF NOT EXISTS SFD_DataConverted (
        subject_id INTEGER REFERENCES subjects(subject_id),
        activity_code INTEGER REFERENCES activity_info(activity_code),
        activity_type INTEGER,
        trial_number INTEGER,
        sample_index INTEGER,
        adxl345_x_g REAL,
        adxl345_y_g REAL,
        adxl345_z_g REAL,
        itg3200_x_dps REAL,
        itg3200_y_dps REAL,
        itg3200_z_dps REAL,
        mma8451q_x_g REAL,
        mma8451q_y_g REAL,
        mma8451q_z_g REAL
    )
    """)
    
    conn.commit()
    conn.close()
    print("Tablolar oluşturuldu.")

def insert_subject_data():
    """Subjects verilerini ekle"""
    subjects_list = [
        ('SA01', 'SA', 26, 165, 53.0, 'F'),
        ('SA02', 'SA', 23, 176, 58.5, 'M'),
        ('SA03', 'SA', 19, 156, 48.0, 'F'),
        ('SA04', 'SA', 23, 170, 72.0, 'M'),
        ('SA05', 'SA', 22, 172, 69.5, 'M'),
        ('SA06', 'SA', 21, 169, 58.0, 'M'),
        ('SA07', 'SA', 21, 156, 63.0, 'F'),
        ('SA08', 'SA', 21, 149, 41.5, 'F'),
        ('SA09', 'SA', 24, 165, 64.0, 'M'),
        ('SA10', 'SA', 21, 177, 67.0, 'M'),
        ('SA11', 'SA', 19, 170, 80.5, 'M'),
        ('SA12', 'SA', 25, 153, 47.0, 'F'),
        ('SA13', 'SA', 22, 157, 55.0, 'F'),
        ('SA14', 'SA', 27, 160, 46.0, 'F'),
        ('SA15', 'SA', 25, 160, 52.0, 'F'),
        ('SA16', 'SA', 20, 169, 61.0, 'F'),
        ('SA17', 'SA', 23, 182, 75.0, 'M'),
        ('SA18', 'SA', 23, 181, 73.0, 'M'),
        ('SA19', 'SA', 30, 170, 76.0, 'M'),
        ('SA20', 'SA', 30, 150, 42.0, 'F'),
        ('SA21', 'SA', 30, 183, 68.0, 'M'),
        ('SA22', 'SA', 19, 158, 50.5, 'F'),
        ('SA23', 'SA', 24, 156, 48.0, 'F'),
        ('SE01', 'SE', 71, 171, 102.0, 'M'),
        ('SE02', 'SE', 75, 150, 57.0, 'F'),
        ('SE03', 'SE', 62, 150, 51.0, 'F'),
        ('SE04', 'SE', 63, 160, 59.0, 'F'),
        ('SE05', 'SE', 63, 165, 72.0, 'M'),
        ('SE06', 'SE', 60, 163, 79.0, 'M'),
        ('SE07', 'SE', 65, 168, 76.0, 'M'),
        ('SE08', 'SE', 68, 163, 72.0, 'F'),
        ('SE09', 'SE', 66, 167, 65.0, 'M'),
        ('SE10', 'SE', 64, 156, 66.0, 'F'),
        ('SE11', 'SE', 66, 169, 63.0, 'F'),
        ('SE12', 'SE', 69, 164, 56.5, 'M'),
        ('SE13', 'SE', 65, 171, 72.5, 'M'),
        ('SE14', 'SE', 67, 163, 58.0, 'M'),
        ('SE15', 'SE', 64, 150, 50.0, 'F')
    ]
    
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    for idx, s in enumerate(subjects_list, start=1):
        SUBJECT_NUMERIC[s[0]] = idx  # numeric mapping
        cur.execute("INSERT OR IGNORE INTO SFD_Subjects(subject_id, subject_code, age_group, age, height, weight, gender) VALUES (?,?,?,?,?,?,?)",
                    (idx, s[0], s[1], s[2], s[3], s[4], s[5]))
    conn.commit()
    conn.close()
    print("Subjects verileri eklendi.")

def insert_activity_info():
    """Tüm activity bilgilerini tek tabloda ekle"""
    
    activities = [
        # ADL
        ('D01','Walking slowly',100,1),('D02','Walking quickly',100,1),
        ('D03','Jogging slowly',100,1),('D04','Jogging quickly',100,1),
        ('D05','Walking upstairs and downstairs slowly',25,5),
        ('D06','Walking upstairs and downstairs quickly',25,5),
        ('D07','Slowly sit in a half height chair, wait a moment, and up slowly',12,5),
        ('D08','Quickly sit in a half height chair, wait a moment, and up quickly',12,5),
        ('D09','Slowly sit in a low height chair, wait a moment, and up slowly',12,5),
        ('D10','Quickly sit in a low height chair, wait a moment, and up quickly',12,5),
        ('D11','Sitting a moment, trying to get up, and collapse into a chair',12,5),
        ('D12','Sitting a moment, lying slowly, wait a moment, and sit again',12,5),
        ('D13','Sitting a moment, lying quickly, wait a moment, and sit again',12,5),
        ('D14','Being on one s back change to lateral position, wait a moment, and change to one s back',12,5),
        ('D15','Standing, slowly bending at knees, and getting up',12,5),
        ('D16','Standing, slowly bending without bending knees, and getting up',12,5),
        ('D17','Standing, get into a car, remain seated and get out of the car',25,5),
        ('D18','Stumble while walking',12,5),
        ('D19','Gently jump without falling (trying to reach a high object)',12,5),
        # FALL
        ('F01','Fall forward while walking caused by a slip',15,5),('F02','Fall backward while walking caused by a slip',15,5),
        ('F03','Lateral fall while walking caused by a slip',15,5),('F04','Fall forward while walking caused by a trip',15,5),
        ('F05','Fall forward while jogging caused by a trip',15,5),('F06','Vertical fall while walking caused by fainting',15,5),
        ('F07','Fall while walking, with use of hands in a table to dampen fall, caused by fainting',15,5),
        ('F08','Fall forward when trying to get up',15,5),('F09','Lateral fall when trying to get up',15,5),
        ('F10','Fall forward when trying to sit down',15,5),('F11','Fall backward when trying to sit down',15,5),
        ('F12','Lateral fall when trying to sit down',15,5),
        ('F13','Fall forward while sitting, caused by fainting or falling asleep',15,5),
        ('F14','Fall backward while sitting, caused by fainting or falling asleep',15,5),
        ('F15','Lateral fall while sitting, caused by fainting or falling asleep',15,5)
    ]
    
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    
    
    
    # Activity numeric map oluştur
    activity_numeric = {a[0]:i+1 for i,a in enumerate(activities)}
    
    # Tabloya ekle
    for a in activities:
        code, desc, dur, trials = a
        act_type = 0 if code.startswith('D') else 1
        act_num = activity_numeric[code]
        cur.execute("""
        INSERT OR IGNORE INTO SFD_Activities
        (activity_code, original_code, activity_type, description, typical_duration, typical_trials)
        VALUES (?,?,?,?,?,?)
        """, (act_num, code, act_type, desc, dur, trials))
    
    conn.commit()
    conn.close()
    print("✅ Tüm activity bilgileri SFD_Activities tablosuna eklendi.")


def parse_filename(filename):
    """Dosya adından info çıkar"""
    pattern = r'^(D|F)(\d{2})_(SA|SE)(\d{2})_R(\d{2})\.txt$'
    match = re.match(pattern, filename)
    if not match: return None
    orig_code = f"{match.group(1)}{match.group(2)}"
    activity_code = ACTIVITY_NUMERIC[orig_code]
    activity_type = ACTIVITY_TYPE_MAP['ADL'] if match.group(1)=='D' else ACTIVITY_TYPE_MAP['FALL']
    subject_code = f"{match.group(3)}{match.group(4)}"
    subject_id = SUBJECT_NUMERIC[subject_code]
    trial_number = int(match.group(5))
    return {'subject_id':subject_id, 'activity_code':activity_code, 'activity_type':activity_type, 'trial_number':trial_number}

def clean_and_parse_line(line):
    values = line.replace(';','').strip().split(',')
    result = []
    for v in values:
        v = v.strip()
        if v:
            try: result.append(int(v))
            except: result.append(0)
    while len(result)<9: result.append(0)
    return result[:9]

def convert_sensor_data(raw):
    adxl_factor = SENSOR_SPECS['ADXL345']['conversion_factor']
    itg_factor = SENSOR_SPECS['ITG3200']['conversion_factor']
    mma_factor = SENSOR_SPECS['MMA8451Q']['conversion_factor']
    return {
        'adxl345_x_g': raw[0]*adxl_factor, 'adxl345_y_g': raw[1]*adxl_factor, 'adxl345_z_g': raw[2]*adxl_factor,
        'itg3200_x_dps': raw[3]*itg_factor, 'itg3200_y_dps': raw[4]*itg_factor, 'itg3200_z_dps': raw[5]*itg_factor,
        'mma8451q_x_g': raw[6]*mma_factor, 'mma8451q_y_g': raw[7]*mma_factor, 'mma8451q_z_g': raw[8]*mma_factor
    }

def process_data_files(dataset_path):
    """Tüm veri dosyalarını oku ve raw+converted tablolara ekle"""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    
    all_files = [os.path.join(root,f) for root,_,files in os.walk(dataset_path) for f in files if f.endswith('.txt')]
    total_files = len(all_files)
    print(f"Toplam {total_files} dosya işlenecek")
    start_time = time.time()
    
    processed_files = 0
    for file_path in all_files:
        filename = os.path.basename(file_path)
        info = parse_filename(filename)
        if not info: continue
        subject_id = info['subject_id']
        activity_code = info['activity_code']
        activity_type = info['activity_type']
        trial_number = info['trial_number']
        
        try:
            with open(file_path,'r',encoding='utf-8',errors='ignore') as f:
                lines = f.readlines()
            
            for idx, line in enumerate(lines):
                raw = clean_and_parse_line(line)
                
                # Ham veri ekle
                cur.execute("""INSERT INTO sensor_dataset_raw
                    (subject_id, activity_code, activity_type, trial_number, sample_index,
                    adxl345_x, adxl345_y, adxl345_z, itg3200_x, itg3200_y, itg3200_z,
                    mma8451q_x, mma8451q_y, mma8451q_z)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (subject_id, activity_code, activity_type, trial_number, idx,
                     raw[0],raw[1],raw[2],raw[3],raw[4],raw[5],raw[6],raw[7],raw[8])
                )
                
                converted = convert_sensor_data(raw)
                
                # Dönüştürülmüş veri ekle
                cur.execute("""INSERT INTO sensor_dataset_converted
                    (subject_id, activity_code, activity_type, trial_number, sample_index,
                     adxl345_x_g, adxl345_y_g, adxl345_z_g,
                     itg3200_x_dps, itg3200_y_dps, itg3200_z_dps,
                     mma8451q_x_g, mma8451q_y_g, mma8451q_z_g)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (subject_id, activity_code, activity_type, trial_number, idx,
                     converted['adxl345_x_g'], converted['adxl345_y_g'], converted['adxl345_z_g'],
                     converted['itg3200_x_dps'], converted['itg3200_y_dps'], converted['itg3200_z_dps'],
                     converted['mma8451q_x_g'], converted['mma8451q_y_g'], converted['mma8451q_z_g'])
                )
            
            conn.commit()
            processed_files += 1
            if processed_files % 10 == 0:
                print(f"{processed_files}/{total_files} dosya işlendi")
                
        except Exception as e:
            conn.rollback()
            print(f"Hata: {filename} -> {e}")
    
    elapsed = time.time()-start_time
    print(f"İşlem tamamlandı: {processed_files}/{total_files} dosya işlendi. Süre: {elapsed:.2f} sn")
    conn.close()

def main():
    dataset_path = r"C:\Users\kbybt\Desktop\EVREN\MatMüh_İTÜ_ARŞ_GÖR\YL_TEZ\Data\SisFall_dataset"
    create_database_tables()
    insert_subject_data()
    insert_activity_info()
    process_data_files(dataset_path)

if __name__=="__main__":
    main()

#%%
import os

# Bekleyen shutdown işlemini iptal et
#os.system("shutdown /s /t 900")
#os.system("shutdown /a")
#print("Shutdown iptal edildi.")

