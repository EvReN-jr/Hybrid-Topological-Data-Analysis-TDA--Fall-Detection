import sqlite3
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# =============================================================================
# DEEP DIAGNOSTIC - Find out WHY trials are being skipped
# =============================================================================
def deep_diagnose_fad_master(db_path, num_trials_to_check=5):
    """
    Checks individual trials in detail to see what's wrong
    """
    print("\n" + "="*70)
    print("DEEP DIAGNOSTIC MODE: Checking individual trials")
    print("="*70)
    
    conn = sqlite3.connect(db_path)
    
    # Get first few trials
    trials_df = pd.read_sql_query(
        f"SELECT trial_id, subject_id, activity_id, sampling_rate FROM FAD_Master_trials LIMIT {num_trials_to_check}",
        conn
    )
    
    print(f"\nChecking first {num_trials_to_check} trials in detail...\n")
    
    for idx, row in trials_df.iterrows():
        t_id = row['trial_id']
        s_id = row['subject_id']
        a_id = row['activity_id']
        fs = row['sampling_rate']
        
        print(f"\n{'='*60}")
        print(f"Trial #{idx+1}: trial_id={t_id}, subject={s_id}, activity={a_id}, fs={fs}")
        print(f"{'='*60}")
        
        # Check sensor data for this trial
        query = f"""
        SELECT COUNT(*) as row_count,
               MIN(timestamp) as min_time,
               MAX(timestamp) as max_time
        FROM FAD_Master_sensor_data
        WHERE trial_id = {t_id}
        """
        stats = pd.read_sql_query(query, conn).iloc[0]
        print(f"  📊 Sensor data rows: {stats['row_count']}")
        print(f"  ⏱️  Time range: {stats['min_time']:.3f} to {stats['max_time']:.3f}")
        print(f"  ⏱️  Duration: {stats['max_time'] - stats['min_time']:.3f} seconds")
        
        # Get actual data
        data_query = f"""
        SELECT acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, 
               mag_x, mag_y, mag_z, bar_pressure, is_fall
        FROM FAD_Master_sensor_data
        WHERE trial_id = {t_id}
        ORDER BY timestamp ASC
        """
        df = pd.read_sql_query(data_query, conn)
        
        print(f"\n  🔍 Data quality check:")
        print(f"     Total rows fetched: {len(df)}")
        
        # Check for NaN values
        sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 
                       'mag_x', 'mag_y', 'mag_z', 'bar_pressure']
        
        for col in sensor_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"     ⚠️  {col}: {nan_count} NaN values ({nan_count/len(df)*100:.1f}%)")
        
        # Check which columns are all NaN
        empty_cols = [c for c in sensor_cols if df[c].isna().all()]
        if empty_cols:
            print(f"     ❌ Completely empty columns: {empty_cols}")
        
        active_cols = [c for c in sensor_cols if c not in empty_cols]
        print(f"     ✅ Active columns ({len(active_cols)}): {active_cols}")
        
        # After dropping NaN rows
        df_clean = df.dropna(subset=active_cols, how='any')
        print(f"     📉 Rows after dropping NaN: {len(df_clean)} (lost {len(df) - len(df_clean)} rows)")
        
        if len(df_clean) == 0:
            print(f"     ❌ NO DATA LEFT after cleaning!")
            continue
        
        # Check windowing
        original_fs = 238
        target_fs = 40
        window_sec = 2.0
        
        step = int(original_fs / target_fs)
        resampled_len = len(df_clean[::step])
        win_len = int(window_sec * target_fs)
        
        print(f"\n  🪟 Windowing simulation:")
        print(f"     Original length: {len(df_clean)}")
        print(f"     After resampling (step={step}): {resampled_len}")
        print(f"     Window length needed: {win_len}")
        print(f"     Min length required (80%): {int(0.8 * win_len)}")
        
        if resampled_len >= int(0.8 * win_len):
            num_windows = (resampled_len - win_len) // int(1.0 * target_fs) + 1
            print(f"     ✅ CAN CREATE {num_windows} windows!")
        else:
            print(f"     ❌ TOO SHORT - cannot create windows!")
        
        # Check label
        label = 1 if df['is_fall'].sum() > 0 else 0
        print(f"  🏷️  Label: {'FALL' if label == 1 else 'NON-FALL'}")
    
    conn.close()
    print("\n" + "="*70)


# =============================================================================
# FIXED PIPELINE WITH VERBOSE OUTPUT
# =============================================================================
def create_indexes(conn, table_name):
    sql = f"""
    CREATE INDEX IF NOT EXISTS idx_{table_name}_trial_time
    ON {table_name} (trial_id, timestamp);
    """
    try:
        conn.execute(sql)
        conn.commit()
    except Exception as e:
        pass


def check_db_columns(conn, table_name, candidates):
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cursor.fetchall()}
    
    valid = []
    for col in candidates:
        if col in existing:
            valid.append(col)
        else:
            if col.startswith("acc_adx1345"):
                fixed = col.replace("acc_adx1345", "acc_adxl345")
                if fixed in existing:
                    valid.append(fixed)
    
    return valid


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


def run_dataset_pipeline_verbose(config):
    print(f"\n{'='*70}")
    print(f"BAŞLATILIYOR: {config['dataset_name']}")
    print(f"Tablo: {config['table_sensor']}")
    print(f"Hedef: {config['output_dir']}")
    print(f"{'='*70}")

    os.makedirs(config['output_dir'], exist_ok=True)
    conn = sqlite3.connect(config['db_path'])

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    create_indexes(conn, config['table_sensor'])

    sensor_cols = check_db_columns(
        conn, config['table_sensor'], config['sensor_columns']
    )

    if not sensor_cols:
        print("❌ Sensör sütunu bulunamadı, çıkılıyor.")
        conn.close()
        return

    print(f"✅ Aktif sensör sayısı: {len(sensor_cols)}")

    trials_df = pd.read_sql_query(
        f"SELECT trial_id, subject_id, activity_id, sampling_rate FROM {config['table_trials']}",
        conn
    )

    success = 0
    skipped = 0
    skip_reasons = {
        'empty_query': 0,
        'all_nan': 0,
        'no_data_after_clean': 0,
        'too_short': 0,
        'query_error': 0
    }
    
    # Process first 10 with detailed output, then use progress bar
    first_trials = min(10, len(trials_df))

    for idx, row in enumerate(trials_df.itertuples(index=False)):
        t_id = row.trial_id
        s_id = row.subject_id
        a_id = row.activity_id
        fs = row.sampling_rate if row.sampling_rate and row.sampling_rate > 0 else config["original_fs"]

        # Verbose output for first 10 trials
        if idx < first_trials:
            print(f"\n--- Trial {idx+1}/{len(trials_df)}: trial_id={t_id} ---")

        query = f"""
        SELECT {", ".join(sensor_cols)}, {config['label_column']}
        FROM {config['table_sensor']}
        WHERE trial_id = {t_id}
        ORDER BY timestamp ASC
        """

        try:
            df = pd.read_sql_query(query, conn)
        except Exception as e:
            if idx < first_trials:
                print(f"  ❌ Query error: {e}")
            skip_reasons['query_error'] += 1
            skipped += 1
            continue

        if df.empty:
            if idx < first_trials:
                print(f"  ❌ Empty result")
            skip_reasons['empty_query'] += 1
            skipped += 1
            continue

        empty_cols = [c for c in sensor_cols if df[c].isna().all()]
        active_cols = [c for c in sensor_cols if c not in empty_cols]

        if not active_cols:
            if idx < first_trials:
                print(f"  ❌ All columns are NaN")
            skip_reasons['all_nan'] += 1
            skipped += 1
            continue

        df.dropna(axis=0, how='any', subset=active_cols, inplace=True)
        
        if df.empty:
            if idx < first_trials:
                print(f"  ❌ No data after cleaning NaN")
            skip_reasons['no_data_after_clean'] += 1
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
            if idx < first_trials:
                print(f"  ❌ Too short for windowing (length={len(raw)})")
            skip_reasons['too_short'] += 1
            skipped += 1
            continue

        fname = (
            f"{config['dataset_name']}_"
            f"Sub_{s_id}_Act_{a_id}_Trial_{t_id}_Fall_{label}.npy"
        )
        fpath = os.path.join(config["output_dir"], fname)
        np.save(fpath, windows)
        
        if idx < first_trials:
            print(f"  ✅ SUCCESS: {windows.shape[0]} windows saved to {fname}")
        
        success += 1

    conn.close()
    print(f"\n{'='*70}")
    print(f"✅ {config['dataset_name']} TAMAMLANDI")
    print(f"   Başarılı: {success}")
    print(f"   Atlanan : {skipped}")
    print(f"\n📊 Skip Reasons:")
    for reason, count in skip_reasons.items():
        if count > 0:
            print(f"   - {reason}: {count}")
    print(f"{'='*70}")


# =============================================================================
# CONFIGURATIONS
# =============================================================================
BASE_DB_PATH = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataOrganize_V2\All_Datasets_Container.db"
BASE_OUT_DIR = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataOrganize_V2\SubWindowing_V2"

# PROBLEM: Trials only have 200 samples at 238 Hz = 0.84 seconds
# After resampling to 40 Hz: only 33 samples
# 2.0 second window needs 80 samples → TOO SHORT!

# SOLUTION 1: Smaller window (RECOMMENDED for consistency with other datasets)
FAD_MASTER_CONFIG_SMALL_WINDOW = {
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

# SOLUTION 2: No downsampling (keeps original 238 Hz)
FAD_MASTER_CONFIG_NO_RESAMPLE = {
    "dataset_name": "FAD_Master",
    "db_path": BASE_DB_PATH,
    "output_dir": os.path.join(BASE_OUT_DIR, "FAD_Master_SubWindow_238Hz"),
    "table_trials": "FAD_Master_trials",
    "table_sensor": "FAD_Master_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z","mag_x","mag_y","mag_z","bar_pressure"],
    "label_column": "is_fall",
    "original_fs": 238,
    "target_fs": 238,  # No resampling!
    "window_sec": 0.8,  # 0.8 seconds × 238 Hz = 190 samples (fits in 200!)
    "stride_sec": 0.4
}

# SOLUTION 3: Use ALL available data (no windowing requirements)
FAD_MASTER_CONFIG_FULL_TRIAL = {
    "dataset_name": "FAD_Master",
    "db_path": BASE_DB_PATH,
    "output_dir": os.path.join(BASE_OUT_DIR, "FAD_Master_SubWindow_FullTrial"),
    "table_trials": "FAD_Master_trials",
    "table_sensor": "FAD_Master_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z","mag_x","mag_y","mag_z","bar_pressure"],
    "label_column": "is_fall",
    "original_fs": 238,
    "target_fs": 40,
    "window_sec": 0.5,  # Very small window
    "stride_sec": 0.25
}


# =============================================================================
# ÇALIŞTIRMA
# =============================================================================
if __name__ == "__main__":
    # Choose ONE of these solutions:
    
    # SOLUTION 1: Small window with downsampling (RECOMMENDED)
    # 0.8 sec window at 40 Hz = 32 samples (fits in 33 after resampling)
    print("\n🚀 Running SOLUTION 1: Small window (0.8s) with 40Hz resampling")
    run_dataset_pipeline_verbose(FAD_MASTER_CONFIG_SMALL_WINDOW)
    
    # SOLUTION 2: No downsampling (keeps 238 Hz)
    # Uncomment to use:
    # print("\n🚀 Running SOLUTION 2: No resampling (238 Hz)")
    # run_dataset_pipeline_verbose(FAD_MASTER_CONFIG_NO_RESAMPLE)
    
    # SOLUTION 3: Very small windows
    # Uncomment to use:
    # print("\n🚀 Running SOLUTION 3: Very small windows (0.5s)")
    # run_dataset_pipeline_verbose(FAD_MASTER_CONFIG_FULL_TRIAL)