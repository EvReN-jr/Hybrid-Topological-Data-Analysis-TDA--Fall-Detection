import os
import sqlite3
import re
from tqdm import tqdm

# ==========================================
# AYARLAR
# ==========================================
BASE_DIR = r"C:\Users\user\Desktop\KEB\YL_TEZ\Data\SisFall_dataset"
DB_NAME = "SisFall_ML_Ready_Detailed.db"
SAMPLING_RATE = 200  # 200 Hz

# ==========================================
# SENSÖR DÖNÜŞÜM FAKTÖRLERİ (Readme'den)
# ==========================================
# Formül: [(2 * Range) / (2 ^ Resolution)]

# 1. ADXL345 (Primary Acc) - Düşme Odaklı
ADXL_FACTOR = (2 * 16) / (2 ** 13) 

# 2. ITG3200 (Gyro) - Yönelim Odaklı
ITG_FACTOR = (2 * 2000) / (2 ** 16)

# 3. MMA8451Q (Secondary Acc) - Hassasiyet/ADL Odaklı
MMA_FACTOR = (2 * 8) / (2 ** 14)

# ==========================================
# METADATA (Konfigürasyon)
# ==========================================
SUBJECTS_DB = {
    'SA01': (26, 165, 53, 'F'), 'SA02': (23, 176, 58.5, 'M'), 'SA03': (19, 156, 48, 'F'),
    'SA04': (23, 170, 72, 'M'), 'SA05': (22, 172, 69.5, 'M'), 'SA06': (21, 169, 58, 'M'),
    'SA07': (21, 156, 63, 'F'), 'SA08': (21, 149, 41.5, 'F'), 'SA09': (24, 165, 64, 'M'),
    'SA10': (21, 177, 67, 'M'), 'SA11': (19, 170, 80.5, 'M'), 'SA12': (25, 153, 47, 'F'),
    'SA13': (22, 157, 55, 'F'), 'SA14': (27, 160, 46, 'F'), 'SA15': (25, 160, 52, 'F'),
    'SA16': (20, 169, 61, 'F'), 'SA17': (23, 182, 75, 'M'), 'SA18': (23, 181, 73, 'M'),
    'SA19': (30, 170, 76, 'M'), 'SA20': (30, 150, 42, 'F'), 'SA21': (30, 183, 68, 'M'),
    'SA22': (19, 158, 50.5, 'F'), 'SA23': (24, 156, 48, 'F'),
    'SE01': (71, 171, 102, 'M'), 'SE02': (75, 150, 57, 'F'), 'SE03': (62, 150, 51, 'F'),
    'SE04': (63, 160, 59, 'F'), 'SE05': (63, 165, 72, 'M'), 'SE06': (60, 163, 79, 'M'),
    'SE07': (65, 168, 76, 'M'), 'SE08': (68, 163, 72, 'F'), 'SE09': (66, 167, 65, 'M'),
    'SE10': (64, 156, 66, 'F'), 'SE11': (66, 169, 63, 'F'), 'SE12': (69, 164, 56.5, 'M'),
    'SE13': (65, 171, 72.5, 'M'), 'SE14': (67, 163, 58, 'M'), 'SE15': (64, 150, 50, 'F')
}

ACTIVITIES_DB = {
    'D01': 'Walking slowly', 'D02': 'Walking quickly', 'D03': 'Jogging slowly', 'D04': 'Jogging quickly',
    'D05': 'Walking upstairs and downstairs slowly', 'D06': 'Walking upstairs and downstairs quickly',
    'D07': 'Slowly sit in a half height chair', 'D08': 'Quickly sit in a half height chair',
    'D09': 'Slowly sit in a low height chair', 'D10': 'Quickly sit in a low height chair',
    'D11': 'Sitting a moment, trying to get up, and collapse', 'D12': 'Sitting a moment, lying slowly',
    'D13': 'Sitting a moment, lying quickly', 'D14': 'Change to lateral position',
    'D15': 'Standing, slowly bending at knees', 'D16': 'Standing, slowly bending without knees',
    'D17': 'Car-step in and out', 'D18': 'Stumble while walking', 'D19': 'Gently jump',
    'F01': 'Fall forward (slip)', 'F02': 'Fall backward (slip)', 'F03': 'Lateral fall (slip)',
    'F04': 'Fall forward (trip)', 'F05': 'Fall forward jog (trip)', 'F06': 'Vertical fall (fainting)',
    'F07': 'Fall while walking (dampen)', 'F08': 'Fall forward getting up', 'F09': 'Lateral fall getting up',
    'F10': 'Fall forward sitting down', 'F11': 'Fall backward sitting down', 'F12': 'Lateral fall sitting down',
    'F13': 'Fall forward sitting', 'F14': 'Fall backward sitting', 'F15': 'Lateral fall sitting'
}

# ==========================================
# VERİTABANI OLUŞTURMA
# ==========================================
def create_tables(conn):
    cur = conn.cursor()

    # 1. Subjects
    cur.execute("""CREATE TABLE IF NOT EXISTS subjects (
        subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_code TEXT UNIQUE,
        dataset_name TEXT,
        age INTEGER,
        height INTEGER,
        weight REAL,
        gender TEXT
    )""")

    # 2. Activity Info
    cur.execute("""CREATE TABLE IF NOT EXISTS activity_info (
        activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT UNIQUE,
        description TEXT,
        class_type TEXT
    )""")

    # 3. Trials
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

    # --- YENİ TABLO: Sensör Açıklamaları (Metadata) ---
    cur.execute("""CREATE TABLE IF NOT EXISTS sensor_descriptions (
        sensor_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sensor_model TEXT UNIQUE,   -- Örn: ADXL345
        sensor_type TEXT,           -- Accelerometer / Gyroscope
        range_val TEXT,             -- +-16g
        resolution TEXT,            -- 13 bits
        unit TEXT,                  -- g veya deg/s
        suitability TEXT            -- Hangi amaçla kullanılmalı?
    )""")

    # --- SENSOR DATA (Model İsimleri ile) ---
    cur.execute("""CREATE TABLE IF NOT EXISTS sensor_data (
        data_id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER,
        trial_id INTEGER,
        activity_id INTEGER,
        is_fall INTEGER,
        timestamp REAL,
        
        -- Ana İvmeölçer (ADXL345)
        acc_adxl345_x REAL, acc_adxl345_y REAL, acc_adxl345_z REAL,
        
        -- Jiroskop (ITG3200)
        gyr_itg3200_x REAL, gyr_itg3200_y REAL, gyr_itg3200_z REAL,
        
        -- İkincil İvmeölçer (MMA8451Q)
        acc_mma8451q_x REAL, acc_mma8451q_y REAL, acc_mma8451q_z REAL,
        
        -- Diğer Veri Setleri İçin Placeholder (NULL)
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
# METADATA EKLEME (Sensör Bilgileri Burada)
# ==========================================
def insert_metadata(conn):
    cur = conn.cursor()
    
    # Sensör Bilgilerini Ekle (Gelecekte hatırlamak için)
    sensors = [
        ("ADXL345", "Accelerometer", "+-16g", "13 bits", "g", 
         "MAIN SENSOR for fall detection. Thanks to its high g-range, it does not experience data loss (clipping) during sharp falls."),
        
        ("ITG3200", "Gyroscope", "+-2000 deg/s", "16 bits", "deg/s", 
         "It measures the rotation and orientation changes during a fall. It is critical for orientation determination."),
        
        ("MMA8451Q", "Accelerometer", "+-8g", "14 bits", "g", 
         "It is more suitable for Daily Activities (ADL). Because its resolution is higher than ADXL, it measures low-intensity movements more accurately.")
    ]
    
    cur.executemany("""INSERT OR IGNORE INTO sensor_descriptions 
        (sensor_model, sensor_type, range_val, resolution, unit, suitability)
        VALUES (?,?,?,?,?,?)""", sensors)
    
    # Subject ve Aktiviteleri Ekle
    for code, info in SUBJECTS_DB.items():
        cur.execute("""INSERT OR IGNORE INTO subjects 
            (original_code, dataset_name, age, height, weight, gender)
            VALUES (?,?,?,?,?,?)""", (code, "SisFall", info[0], info[1], info[2], info[3]))
        
    for code, desc in ACTIVITIES_DB.items():
        class_type = "FALL" if code.startswith("F") else "ADL"
        cur.execute("INSERT OR IGNORE INTO activity_info (code, description, class_type) VALUES (?,?,?)",
                    (code, desc, class_type))
    
    conn.commit()

# ==========================================
# DOSYA İŞLEME VE DÖNÜŞTÜRME
# ==========================================
def parse_filename(filename):
    match = re.match(r"([DF]\d{2})_(S[AE]\d{2})_R(\d{2})\.txt", filename)
    if match: return match.groups()
    return None

def process_sisfall():
    if not os.path.exists(BASE_DIR):
        print(f"HATA: Klasör bulunamadı -> {BASE_DIR}")
        return

    conn = sqlite3.connect(DB_NAME)
    create_tables(conn)
    insert_metadata(conn) # Sensör bilgilerini bas
    cur = conn.cursor()

    # Dosyaları Bul
    all_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith(".txt") and parse_filename(file):
                all_files.append(os.path.join(root, file))

    print(f"Toplam {len(all_files)} dosya işlenecek.")

    # Dosyaları İşle
    for file_path in tqdm(all_files, desc="SisFall Verisi İşleniyor"):
        filename = os.path.basename(file_path)
        parsed = parse_filename(filename)
        if not parsed: continue
        
        act_code, sub_code, trial_str = parsed
        trial_no = int(trial_str)
        
        cur.execute("SELECT subject_id FROM subjects WHERE original_code=?", (sub_code,))
        res_sub = cur.fetchone()
        cur.execute("SELECT activity_id, class_type FROM activity_info WHERE code=?", (act_code,))
        res_act = cur.fetchone()
        
        if not res_sub or not res_act: continue
            
        sub_id = res_sub[0]
        act_id, class_type = res_act
        is_fall = 1 if class_type == "FALL" else 0

        # Trial Kaydı
        cur.execute("""INSERT INTO trials 
            (subject_id, activity_id, dataset_version, device_pos, sampling_rate, trial_no)
            VALUES (?,?,?,?,?,?)""",
            (sub_id, act_id, "SisFall_Original", "Waist", SAMPLING_RATE, trial_no))
        
        trial_id = cur.lastrowid

        batch_data = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                parts = line.replace(';', '').strip().split(',')
                if len(parts) < 9: continue
                
                try:
                    raw = [int(p) for p in parts if p.strip()]
                    if len(raw) < 9: continue

                    # Dönüşümler (Readme formülleri)
                    # ADXL345 (Ana İvmeölçer)
                    ax = raw[0] * ADXL_FACTOR
                    ay = raw[1] * ADXL_FACTOR
                    az = raw[2] * ADXL_FACTOR
                    
                    # ITG3200 (Gyro)
                    gx = raw[3] * ITG_FACTOR
                    gy = raw[4] * ITG_FACTOR
                    gz = raw[5] * ITG_FACTOR
                    
                    # MMA8451Q (Hassas/Yedek İvmeölçer)
                    mma_x = raw[6] * MMA_FACTOR
                    mma_y = raw[7] * MMA_FACTOR
                    mma_z = raw[8] * MMA_FACTOR
                    
                    ts = i / float(SAMPLING_RATE)

                    batch_data.append((
                        sub_id, trial_id, act_id, is_fall, ts,
                        ax, ay, az,       
                        gx, gy, gz,       
                        mma_x, mma_y, mma_z 
                    ))

                except ValueError:
                    continue
            
            # Veriyi Bas (Yeni kolon isimleriyle)
            cur.executemany("""INSERT INTO sensor_data
                (subject_id, trial_id, activity_id, is_fall, timestamp,
                 acc_adxl345_x, acc_adxl345_y, acc_adxl345_z, 
                 gyr_itg3200_x, gyr_itg3200_y, gyr_itg3200_z, 
                 acc_mma8451q_x, acc_mma8451q_y, acc_mma8451q_z)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", batch_data)

        except Exception as e:
            print(f"Hata: {filename} - {e}")

    conn.commit()
    conn.close()
    print(f"\n✅ İşlem tamamlandı! Veritabanı: {DB_NAME}")
    print("✅ Sensör açıklamaları 'sensor_descriptions' tablosuna eklendi.")

if __name__ == "__main__":
    process_sisfall()