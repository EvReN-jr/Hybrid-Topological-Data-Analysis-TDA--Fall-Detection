import sqlite3
import ijson
import os
from tqdm import tqdm
import decimal

# ==========================================
# AYARLAR
# ==========================================
#JSON_FILE_PATH = r"C:\Users\user\Desktop\KEB\YL_TEZ\Data\archive_pckl\FallAllD.json"
#DB_NAME = "FallAllD_Master_ML_Ready.db"

JSON_FILE_PATH = r"C:\Users\user\Desktop\KEB\YL_TEZ\Data\archive_pckl\FallAllD_40SamplesPerSec_ActivityIdsFiltered.json"
DB_NAME = "FallAllD_40Hz_ML_Ready.db"

# Hangi seti işliyorsan burayı ayarla:
# A) Original Set
"""
CURRENT_CONFIG = {
    "VERSION_NAME": "Original_238Hz",
    "SAMPLING_RATE": 238,      
    "HAS_MAG_BAR": True,       
    "EXCLUDE_NECK": False
}
"""
# B) Derived Set (Kullanırken yorumu kaldır)
CURRENT_CONFIG = {
     "VERSION_NAME": "Derived_40Hz",
     "SAMPLING_RATE": 40,       
     "HAS_MAG_BAR": False,      
     "EXCLUDE_NECK": True       
}

# ==========================================
# TABLO OLUŞTURMA (GÜNCELLENMİŞ)
# ==========================================
def create_tables(conn):
    cur = conn.cursor()
    
    # Subjects
    cur.execute("""CREATE TABLE IF NOT EXISTS subjects (
        subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_code INTEGER UNIQUE
    )""")
    
    # Activity Info
    cur.execute("""CREATE TABLE IF NOT EXISTS activity_info (
        activity_id INTEGER PRIMARY KEY,
        description TEXT,
        class_type TEXT
    )""")
    
    # Trials (Metadata)
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
    
    # --- GÜNCELLEME: is_fall KOLONU EKLENDİ ---
    # is_fall: 1 ise Düşme (Fall), 0 ise Günlük Aktivite (ADL)
    cur.execute("""CREATE TABLE IF NOT EXISTS sensor_data (
        data_id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_id INTEGER,     -- ML Gruplama
        trial_id INTEGER,       -- Deney takibi
        activity_id INTEGER,    -- Aktivite Tipi
        is_fall INTEGER,        -- ML Binary Label (1=Fall, 0=ADL) - YENİ EKLENDİ
        timestamp REAL,
        acc_x REAL, acc_y REAL, acc_z REAL,
        gyr_x REAL, gyr_y REAL, gyr_z REAL,
        mag_x REAL, mag_y REAL, mag_z REAL,
        bar_pressure REAL,
        FOREIGN KEY(subject_id) REFERENCES subjects(subject_id),
        FOREIGN KEY(trial_id) REFERENCES trials(trial_id),
        FOREIGN KEY(activity_id) REFERENCES activity_info(activity_id)
    )""")
    
    # ML sorguları için çoklu indeks (Hız için kritik)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ml_ready ON sensor_data(subject_id, activity_id, is_fall, trial_id)")
    
    conn.commit()

# ==========================================
# YARDIMCI FONKSİYONLAR
# ==========================================
def decimal_to_float(val):
    if isinstance(val, decimal.Decimal): return float(val)
    if isinstance(val, list): return [decimal_to_float(x) for x in val]
    return val

def determine_class(act_id):
    # FallAllD genel kuralı: 
    # Genelde ID >= 100 Fall, < 100 ADL (veya tam tersi, veri setine göre kontrol et)
    # Varsayım: ID > 100 ise FALL
    return "FALL" if int(act_id) > 100 else "ADL"

# ==========================================
# ANA İŞLEM
# ==========================================
def process_data():
    if not os.path.exists(JSON_FILE_PATH):
        print(f"HATA: Dosya bulunamadı -> {JSON_FILE_PATH}")
        return

    conn = sqlite3.connect(DB_NAME)
    create_tables(conn)
    cur = conn.cursor()
    
    print(f"İşleniyor: {CURRENT_CONFIG['VERSION_NAME']} | Hz: {CURRENT_CONFIG['SAMPLING_RATE']}")
    
    subj_cache = {}
    
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        # ijson streaming
        for item in tqdm(ijson.items(f, "item"), desc="Veritabanına Yazılıyor"):
            
            # 1. Veri Okuma
            subj_code = item.get("SubjectID")
            device = item.get("Device")
            act_id = item.get("ActivityID")
            trial_no = item.get("TrialNo", 1)
            
            # Filtreleme
            if CURRENT_CONFIG["EXCLUDE_NECK"] and device == "Neck":
                continue
            
            # 2. Subject ID Yönetimi
            if subj_code not in subj_cache:
                cur.execute("INSERT OR IGNORE INTO subjects (original_code) VALUES (?)", (subj_code,))
                cur.execute("SELECT subject_id FROM subjects WHERE original_code=?", (subj_code,))
                subj_cache[subj_code] = cur.fetchone()[0]
            s_id = subj_cache[subj_code]
            
            # 3. Activity ID ve Class Belirleme
            class_str = determine_class(act_id)
            
            # --- YENİ MANTIK: is_fall ---
            # Eğer class_str 'FALL' ise 1, 'ADL' ise 0
            is_fall_val = 1 if class_str == "FALL" else 0
            
            cur.execute("SELECT 1 FROM activity_info WHERE activity_id=?", (act_id,))
            if not cur.fetchone():
                cur.execute("INSERT INTO activity_info (activity_id, description, class_type) VALUES (?,?,?)",
                            (act_id, f"Activity_{act_id}", class_str))
            
            # 4. Trial Kaydı
            cur.execute("""INSERT INTO trials 
                (subject_id, activity_id, dataset_version, device_pos, sampling_rate, trial_no)
                VALUES (?,?,?,?,?,?)""", 
                (s_id, act_id, CURRENT_CONFIG["VERSION_NAME"], device, CURRENT_CONFIG["SAMPLING_RATE"], trial_no))
            
            t_id = cur.lastrowid
            
            # 5. Sensör Verisi Hazırlama
            acc = decimal_to_float(item.get("Acc", []))
            gyr = decimal_to_float(item.get("Gyr", []))
            
            mag = decimal_to_float(item.get("Mag", [])) if CURRENT_CONFIG["HAS_MAG_BAR"] else []
            bar = decimal_to_float(item.get("Bar", [])) if CURRENT_CONFIG["HAS_MAG_BAR"] else []
            
            n_samples = max(len(acc), len(gyr))
            batch_data = []
            
            for i in range(n_samples):
                ts = i / CURRENT_CONFIG["SAMPLING_RATE"]
                
                ax, ay, az = acc[i] if i < len(acc) else (None, None, None)
                gx, gy, gz = gyr[i] if i < len(gyr) else (None, None, None)
                
                mx, my, mz = (None, None, None)
                if i < len(mag): mx, my, mz = mag[i]
                
                bp = None
                if i < len(bar): 
                    val = bar[i]
                    bp = val[0] if isinstance(val, list) else val
                
                # --- VERİ LİSTESİNE is_fall_val EKLENDİ ---
                batch_data.append((
                    s_id,           # subject_id
                    t_id,           # trial_id
                    act_id,         # activity_id
                    is_fall_val,    # is_fall (1 veya 0) <-- BURASI EKLENDİ
                    ts, ax, ay, az, gx, gy, gz, mx, my, mz, bp
                ))
            
            # 6. Veri Basma (Yeni is_fall kolonu ile)
            cur.executemany("""INSERT INTO sensor_data 
                (subject_id, trial_id, activity_id, is_fall, timestamp, 
                 acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, 
                 mag_x, mag_y, mag_z, bar_pressure)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", batch_data)
                
        conn.commit()
        conn.close()
        print(f"✅ İşlem Tamamlandı! Veritabanı: {DB_NAME}")

if __name__ == "__main__":
    process_data()