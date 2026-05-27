import os
import sqlite3
from tqdm import tqdm

# ==========================================
# AYARLAR
# ==========================================
BASE_DIR = r"C:\Users\user\Desktop\KEB\YL_TEZ\Data\MobiFall_Dataset_v2.0"
DB_NAME = "MobiFall_ML_Ready.db"

# ==========================================
# VERİ TANIMLAMALARI (METADATA)
# ==========================================

# Katılımcı Listesi (Readme'den alındı)
SUBJECTS_DATA = [
    (1, "pat1", 32, 180, 85, "M"), (2, "pat2", 26, 169, 64, "M"), (3, "pat3", 26, 164, 55, "F"),
    (4, "pat4", 32, 186, 93, "M"), (5, "pat5", 36, 160, 50, "F"), (6, "pat6", 22, 172, 62, "F"),
    (7, "pat7", 25, 189, 80, "M"), (8, "pat8", 22, 183, 93, "M"), (9, "pat9", 30, 177, 102, "M"),
    (10, "pat10", 26, 170, 90, "F"), (11, "pat11", 26, 168, 80, "F"), (12, "sub12", 29, 178, 83, "M"),
    (13, "sub13", 24, 177, 62, "M"), (14, "sub14", 24, 178, 85, "M"), (15, "sub15", 25, 173, 82, "M"),
    (16, "sub16", 27, 172, 56, "F"), (17, "sub17", 25, 173, 67, "M"), (18, "sub18", 25, 176, 73, "M"),
    (19, "sub19", 25, 161, 63, "F"), (20, "sub20", 26, 178, 71, "M"), (21, "sub21", 25, 180, 70, "M"),
    (29, "sub29", 27, 186, 103, "M"), (30, "sub30", 47, 172, 90, "M"), (31, "sub31", 27, 170, 75, "M")
]

# Aktivite Listesi ve Türleri
# (Code, Name, Class_Type)
ACTIVITIES_DATA = [
    # ADL
    ("STD", "Standing", "ADL"), ("WAL", "Walking", "ADL"), ("JOG", "Jogging", "ADL"),
    ("JUM", "Jumping", "ADL"), ("STU", "Stairs up", "ADL"), ("STN", "Stairs down", "ADL"),
    ("SCH", "Sit chair", "ADL"), ("CSI", "Car-step in", "ADL"), ("CSO", "Car-step out", "ADL"),
    # Falls
    ("FOL", "Forward-lying", "FALL"), ("FKL", "Front-knees-lying", "FALL"),
    ("BSC", "Back-sitting-chair", "FALL"), ("SDL", "Sideward-lying", "FALL")
]

# ==========================================
# TABLO OLUŞTURMA
# ==========================================
def create_tables(conn):
    cur = conn.cursor()

    # Subjects (Detaylı bilgi içerir)
    cur.execute("""CREATE TABLE IF NOT EXISTS subjects (
        subject_id INTEGER PRIMARY KEY,
        original_code TEXT UNIQUE,
        age INTEGER,
        height INTEGER,
        weight INTEGER,
        gender TEXT
    )""")

    # Activity Info
    cur.execute("""CREATE TABLE IF NOT EXISTS activity_info (
        activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT UNIQUE,
        description TEXT,
        class_type TEXT
    )""")

    # Trials (Dataset Version ile)
    cur.execute("""CREATE TABLE IF NOT EXISTS trials (
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

    # --- SENSOR DATA (MobiFall için genişletildi) ---
    # ori_azimuth, pitch, roll eklendi.
    # Önceki setlerdeki 'device_orientation' (integer) kolonu burada NULL kalacak.
    cur.execute("""CREATE TABLE IF NOT EXISTS sensor_data (
        data_id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER,
        trial_id INTEGER,
        activity_id INTEGER,
        is_fall INTEGER,
        timestamp REAL,
        
        -- Sensörler
        acc_x REAL, acc_y REAL, acc_z REAL,
        gyr_x REAL, gyr_y REAL, gyr_z REAL,
        
        -- MobiFall Özel (Oryantasyon Açıları)
        ori_azimuth REAL, ori_pitch REAL, ori_roll REAL,
        
        -- Diğer setlerden gelenler (Burada NULL kalacaklar)
        mag_x REAL, mag_y REAL, mag_z REAL,
        bar_pressure REAL,
        device_orientation INTEGER, 

        FOREIGN KEY(subject_id) REFERENCES subjects(subject_id),
        FOREIGN KEY(trial_id) REFERENCES trials(trial_id),
        FOREIGN KEY(activity_id) REFERENCES activity_info(activity_id)
    )""")
    
    cur.execute("CREATE INDEX IF NOT EXISTS idx_grouping ON sensor_data(subject_id, activity_id, trial_id)")
    conn.commit()

# ==========================================
# DOSYA İŞLEME FONKSİYONLARI
# ==========================================
def parse_sensor_file(file_path):
    """Dosyayı okur ve (timestamp_ns, x, y, z) listesi döner"""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('@', '#', 'DATA')):
                    continue
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        ts = int(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        data.append((ts, x, y, z))
                    except ValueError:
                        continue
    except FileNotFoundError:
        return []
    return data

def find_file_groups(base_dir):
    """Aynı deneye ait Acc, Gyro ve Ori dosyalarını gruplar"""
    groups = {}
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                # Format: Code_Sensor_SubjectID_TrialNo.txt
                # Örn: FOL_acc_1_1.txt
                parts = file.split('_')
                if len(parts) >= 4:
                    code = parts[0]
                    sensor = parts[1] # acc, gyro, ori
                    sub_id = int(parts[2])
                    try:
                        trial_no = int(parts[3].replace('.txt', ''))
                    except ValueError:
                        continue # Hatalı dosya ismi
                    
                    # Benzersiz Grup Anahtarı
                    key = (code, sub_id, trial_no)
                    
                    if key not in groups:
                        groups[key] = {}
                    groups[key][sensor] = os.path.join(root, file)
    return groups

# ==========================================
# ANA İŞLEM
# ==========================================
def process_mobifall():
    if not os.path.exists(BASE_DIR):
        print(f"HATA: Klasör bulunamadı -> {BASE_DIR}")
        return

    conn = sqlite3.connect(DB_NAME)
    create_tables(conn)
    cur = conn.cursor()

    # 1. Subject Verilerini Ekle
    print("Subjects ekleniyor...")
    for s in SUBJECTS_DATA:
        # s: (id, code, age, height, weight, gender)
        cur.execute("INSERT OR IGNORE INTO subjects (subject_id, original_code, age, height, weight, gender) VALUES (?,?,?,?,?,?)",
                    (s[0], s[1], s[2], s[3], s[4], s[5]))
    
    # 2. Activity Verilerini Ekle
    print("Activities ekleniyor...")
    act_map = {} # Code -> (id, class_type)
    for code, name, c_type in ACTIVITIES_DATA:
        cur.execute("INSERT OR IGNORE INTO activity_info (code, description, class_type) VALUES (?,?,?)",
                    (code, name, c_type))
        # ID'yi geri çek
        cur.execute("SELECT activity_id FROM activity_info WHERE code=?", (code,))
        aid = cur.fetchone()[0]
        act_map[code] = (aid, c_type)
    
    conn.commit()

    # 3. Dosyaları Grupla ve İşle
    groups = find_file_groups(BASE_DIR)
    print(f"Toplam {len(groups)} deney grubu bulundu. İşleniyor...")

    for key, files in tqdm(groups.items(), desc="Trials İşleniyor"):
        code, sub_id, trial_no = key
        
        # Eğer acc, gyro veya ori dosyalarından biri eksikse atlayalım (Veri bütünlüğü için)
        if not all(k in files for k in ('acc', 'gyro', 'ori')):
            # İstersen sadece acc ve gyro ile de devam edebilirsin, burada katı kural uyguladık.
            continue
            
        # Aktivite Bilgileri
        if code not in act_map: continue
        act_id, class_type = act_map[code]
        is_fall = 1 if class_type == 'FALL' else 0

        # Trial Kaydı
        # MobiFall Samsung Galaxy S3 ile genelde bel cebinde veya belde taşınarak toplandı.
        cur.execute("""INSERT INTO trials 
            (subject_id, activity_id, dataset_version, device_pos, sampling_rate, trial_no)
            VALUES (?,?,?,?,?,?)""",
            (sub_id, act_id, "MobiFall_v2.0", "Waist/Pocket", 87, trial_no)) # Yaklaşık 87Hz
        
        trial_id = cur.lastrowid

        # Dosyaları Oku
        acc_raw = parse_sensor_file(files['acc'])
        gyr_raw = parse_sensor_file(files['gyro'])
        ori_raw = parse_sensor_file(files['ori'])

        # Senkronizasyon: En kısa olanın uzunluğunu al
        min_len = min(len(acc_raw), len(gyr_raw), len(ori_raw))
        
        if min_len == 0: continue

        # Başlangıç zamanı (İvmeölçerin ilk zamanı referans)
        start_time_ns = acc_raw[0][0]

        batch_data = []
        for i in range(min_len):
            # Timestamp: Nanosaniye -> Saniye (Normalize)
            # t_ns, x, y, z formatında geliyor parse fonksiyonundan
            t_curr_ns = acc_raw[i][0]
            timestamp_sec = (t_curr_ns - start_time_ns) / 1_000_000_000.0

            # Sensör Değerleri
            ax, ay, az = acc_raw[i][1], acc_raw[i][2], acc_raw[i][3]
            gx, gy, gz = gyr_raw[i][1], gyr_raw[i][2], gyr_raw[i][3]
            # Oryantasyon (Azimuth, Pitch, Roll)
            o_az, o_pi, o_ro = ori_raw[i][1], ori_raw[i][2], ori_raw[i][3]

            batch_data.append((
                sub_id, trial_id, act_id, is_fall, timestamp_sec,
                ax, ay, az,
                gx, gy, gz,
                o_az, o_pi, o_ro
            ))
        
        # Veriyi Bas
        cur.executemany("""INSERT INTO sensor_data
            (subject_id, trial_id, activity_id, is_fall, timestamp,
             acc_x, acc_y, acc_z, 
             gyr_x, gyr_y, gyr_z, 
             ori_azimuth, ori_pitch, ori_roll)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", batch_data)

    conn.commit()
    conn.close()
    print(f"\n✅ İşlem tamamlandı! Veritabanı: {DB_NAME}")

if __name__ == "__main__":
    process_mobifall()