import sqlite3
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# =============================================================================
# 1. VERİTABANI OPTİMİZASYONU (HAYAT KURTARAN KISIM)
# =============================================================================
def create_indexes(conn, table_name):
    """
    trial_id + timestamp index’i oluşturur.
    Yoksa ekler, varsa geçer.
    """
    sql = f"""
    CREATE INDEX IF NOT EXISTS idx_{table_name}_trial_time
    ON {table_name} (trial_id, timestamp);
    """
    conn.execute(sql)
    conn.commit()


def check_db_columns(conn, table_name, candidates):
    """
    DB şemasına bakar, gerçekten var olan sensör sütunlarını döndürür.
    SisFall typo problemini otomatik çözer.
    """
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cursor.fetchall()}

    valid = []
    for col in candidates:
        if col in existing:
            valid.append(col)
        else:
            # SisFall typo fix
            if col.startswith("acc_adx1345"):
                fixed = col.replace("acc_adx1345", "acc_adxl345")
                if fixed in existing:
                    valid.append(fixed)
    return valid


# =============================================================================
# 2. SUBWINDOW MOTORU (TOLERANSLI)
# =============================================================================
def universal_subwindowing(signal_data, original_fs, target_fs,
                           window_sec, stride_sec, min_ratio=0.8):
    if original_fs is None or original_fs <= 0:
        step = 1
    else:
        step = int(original_fs / target_fs)
        step = max(step, 1)

    resampled = signal_data[::step]

    win_len = int(window_sec * target_fs)
    stride_len = int(stride_sec * target_fs)
    min_len = int(min_ratio * win_len)

    if len(resampled) < min_len:
        return None

    windows = []
    for i in range(0, len(resampled) - win_len + 1, stride_len):
        windows.append(resampled[i:i + win_len])

    return np.array(windows)


# =============================================================================
# 3. ANA PIPELINE
# =============================================================================
def run_dataset_pipeline(config):
    print(f"\n{'='*70}")
    print(f"BAŞLATILIYOR: {config['dataset_name']}")
    print(f"Tablo: {config['table_sensor']}")
    print(f"Hedef: {config['output_dir']}")
    print(f"{'='*70}")

    os.makedirs(config['output_dir'], exist_ok=True)
    conn = sqlite3.connect(config['db_path'])

    # --- SQLITE OPTİMİZASYON AYARLARI ---
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    # --- INDEX OLUŞTUR ---
    create_indexes(conn, config['table_sensor'])

    # --- GERÇEK SENSÖR KOLONLARINI BUL ---
    sensor_cols = check_db_columns(
        conn, config['table_sensor'], config['sensor_columns']
    )

    if not sensor_cols:
        print("❌ Sensör sütunu bulunamadı, çıkılıyor.")
        conn.close()
        return

    print(f"✅ Aktif sensör sayısı: {len(sensor_cols)}")

    # --- TRIAL LİSTESİ ---
    try:
        trials_df = pd.read_sql_query(
            f"SELECT trial_id, subject_id, activity_id, sampling_rate "
            f"FROM {config['table_trials']}", conn
        )
    except Exception:
        trials_df = pd.read_sql_query(
            f"SELECT trial_id, subject_id, activity_id "
            f"FROM {config['table_trials']}", conn
        )
        trials_df["sampling_rate"] = config["original_fs"]

    success = 0
    skipped = 0

    # --- ANA DÖNGÜ (itertuples = hızlı) ---
    for row in tqdm(trials_df.itertuples(index=False),
                    total=len(trials_df),
                    desc=config["dataset_name"],
                    unit="trial"):

        t_id = row.trial_id
        s_id = row.subject_id
        a_id = row.activity_id
        fs = row.sampling_rate if row.sampling_rate and row.sampling_rate > 0 else config["original_fs"]

        query = f"""
        SELECT {", ".join(sensor_cols)}, {config['label_column']}
        FROM {config['table_sensor']}
        WHERE trial_id = {t_id}
        ORDER BY timestamp ASC
        """

        try:
            df = pd.read_sql_query(query, conn)
        except Exception:
            skipped += 1
            continue

        if df.empty:
            skipped += 1
            continue

        # --- TAMAMI NaN OLAN KOLONLARI AT ---
        empty_cols = [c for c in sensor_cols if df[c].isna().all()]
        active_cols = [c for c in sensor_cols if c not in empty_cols]

        if not active_cols:
            skipped += 1
            continue

        df.dropna(axis=0, how="any", subset=active_cols, inplace=True)
        if df.empty:
            skipped += 1
            continue

        label = 1 if df[config["label_column"]].sum() > 0 else 0
        raw = df[active_cols].values

        windows = universal_subwindowing(
            raw, fs,
            config["target_fs"],
            config["window_sec"],
            config["stride_sec"]
        )

        if windows is None:
            skipped += 1
            continue

        fname = (
            f"{config['dataset_name']}_"
            f"Sub_{s_id}_Act_{a_id}_Trial_{t_id}_Fall_{label}.npy"
        )
        np.save(os.path.join(config["output_dir"], fname), windows)
        success += 1

    conn.close()
    print(f"\n✅ {config['dataset_name']} TAMAMLANDI")
    print(f"   Başarılı: {success}")
    print(f"   Atlanan : {skipped}")


# =============================================================================
# 4. KONFİGÜRASYONLAR
# =============================================================================

BASE_DB_PATH = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataOrganize_V2\All_Datasets_Container.db"
BASE_OUT_DIR = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataOrganize_V2\SubWindowing_V2"

SISFALL_CONFIG = {
    "dataset_name": "SisFall",
    "db_path": BASE_DB_PATH,
    "output_dir": os.path.join(BASE_OUT_DIR, "SisFall_SubWindow"),
    "table_trials": "SisFall_trials",
    "table_sensor": "SisFall_sensor_data",
    "sensor_columns": [
        "acc_adx1345_x", "acc_adx1345_y", "acc_adx1345_z",
        "gyr_itg3200_x", "gyr_itg3200_y", "gyr_itg3200_z",
        "acc_mma8451q_x", "acc_mma8451q_y", "acc_mma8451q_z",
        "mag_x", "mag_y", "mag_z", "bar_pressure"
    ],
    "label_column": "is_fall",
    "original_fs": 200,
    "target_fs": 50,
    "window_sec": 2.0,
    "stride_sec": 1.0
}

FAD_40HZ_CONFIG = {
    "dataset_name": "FAD_40Hz",
    "db_path": BASE_DB_PATH,
    "output_dir": os.path.join(BASE_OUT_DIR, "FAD_40Hz_SubWindow"),
    "table_trials": "FAD_40Hz_trials",
    "table_sensor": "FAD_40Hz_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z","mag_x","mag_y","mag_z","bar_pressure"],
    "label_column": "is_fall",
    "original_fs": 40,
    "target_fs": 40,
    "window_sec": 2.0,
    "stride_sec": 1.0
}

FAD_MASTER_CONFIG = { # Fixed empty output
    "dataset_name": "FAD_Master",
    "db_path": BASE_DB_PATH,
    "output_dir": os.path.join(BASE_OUT_DIR, "FAD_Master_SubWindow"),
    "table_trials": "FAD_Master_trials",
    "table_sensor": "FAD_Master_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z","mag_x","mag_y","mag_z","bar_pressure"],
    "label_column": "is_fall",
    "original_fs": 238,
    "target_fs": 40,
    "window_sec": 0.8,  # 0.8 seconds × 40 Hz = 32 samples (fits in 33!)
    "stride_sec": 0.4   # 50% overlap
}

FD_ORI_CONFIG = {
    "dataset_name": "FD_Ori",
    "db_path": BASE_DB_PATH,
    "output_dir": os.path.join(BASE_OUT_DIR, "FD_Ori_SubWindow"),
    "table_trials": "FD_Ori_trials",
    "table_sensor": "FD_Ori_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z","mag_x","mag_y","mag_z","bar_pressure"],
    "label_column": "is_fall",
    "original_fs": 100,
    "target_fs": 50,
    "window_sec": 2.0,
    "stride_sec": 1.0
}

MOBIFALL_CONFIG = {
    "dataset_name": "MobiFall",
    "db_path": BASE_DB_PATH,
    "output_dir": os.path.join(BASE_OUT_DIR, "MobiFall_SubWindow"),
    "table_trials": "MobiFall_trials",
    "table_sensor": "MobiFall_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z",
                       "ori_azimuth","ori_pitch","ori_roll",
                       "mag_x","mag_y","mag_z","bar_pressure"],
    "label_column": "is_fall",
    "original_fs": 100,
    "target_fs": 50,
    "window_sec": 2.0,
    "stride_sec": 1.0
}


# =============================================================================
# 5. ÇALIŞTIRMA
# =============================================================================
if __name__ == "__main__":
    #run_dataset_pipeline(SISFALL_CONFIG)
    #run_dataset_pipeline(FAD_40HZ_CONFIG)
    run_dataset_pipeline(FAD_MASTER_CONFIG)
    #run_dataset_pipeline(FD_ORI_CONFIG)
    #run_dataset_pipeline(MOBIFALL_CONFIG)
