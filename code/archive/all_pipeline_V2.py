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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# 1. ADVANCED TDA ENGINE (VERSION 2 - 70 FEATURES)
# =============================================================================
class TDA_Engine_V2:
    def __init__(self, target_fs):
        self.fs = target_fs
        # Delay factors from the paper: 0.5, 1.0, 1.5, 2.0 * fs
        self.delay_factors = [0.5, 1.0, 1.5, 2.0]
        self.embedding_dim = 3  # Standard dimension for 3D embedding
        
        # Gudhi Vectorizers (for functional norms)
        # Resolution=50 is a good balance between speed and accuracy
        self.betti = gudhi.representations.BettiCurve(resolution=50)
        self.landscape = gudhi.representations.Landscape(resolution=50, num_landscapes=1)

    def _get_delay_embedding(self, signal, delay):
        """Creates a Time-Delay Embedding point cloud from a univariate signal."""
        N = len(signal)
        dim = self.embedding_dim
        # Check if signal is long enough
        if N <= (dim - 1) * delay:
            return None
            
        embedded = np.zeros((N - (dim - 1) * delay, dim))
        for i in range(dim):
            embedded[:, i] = signal[i * delay : i * delay + len(embedded)]
        return embedded

    def _compute_diagram_features(self, persistence_intervals):
        """
        Extracts 7 statistical features from a single persistence diagram.
        Source: Karan & Kaygun (2021)
        """
        # Initialize with zeros
        keys = ['W1', 'W_inf', 'Entropy', 'Betti_L1', 'Betti_L2', 'Land_L1', 'Land_L2']
        feats = {k: 0.0 for k in keys}
        
        if len(persistence_intervals) == 0:
            return feats

        # Filter for finite intervals only
        finite_intervals = persistence_intervals[np.isfinite(persistence_intervals[:, 1])]
        
        if len(finite_intervals) == 0:
            return feats

        lifetimes = finite_intervals[:, 1] - finite_intervals[:, 0]
        
        # 1. Wasserstein (p=1) Distance to Empty: Sum of lifetimes / sqrt(2)
        feats['W1'] = np.sum(lifetimes) / np.sqrt(2)
        
        # 2. Bottleneck Distance to Empty: Max lifetime / sqrt(2)
        feats['W_inf'] = np.max(lifetimes) / np.sqrt(2)
        
        # 3. Persistent Entropy
        total_life = np.sum(lifetimes)
        if total_life > 0:
            probs = lifetimes / total_life
            # Add epsilon to avoid log(0)
            feats['Entropy'] = -np.sum(probs * np.log(probs + 1e-10))
            
        # --- Functional Norms (Gudhi Representations) ---
        # Gudhi expects a list of diagrams
        diags_format = [finite_intervals]
        
        # 4 & 5. Betti Curve Norms
        try:
            betti_vals = self.betti.fit_transform(diags_format)[0]
            feats['Betti_L1'] = np.linalg.norm(betti_vals, ord=1)
            feats['Betti_L2'] = np.linalg.norm(betti_vals, ord=2)
        except:
            pass

        # 6 & 7. Landscape Norms
        try:
            land_vals = self.landscape.fit_transform(diags_format)[0]
            feats['Land_L1'] = np.linalg.norm(land_vals, ord=1)
            feats['Land_L2'] = np.linalg.norm(land_vals, ord=2)
        except:
            pass
            
        return feats

    def extract_all(self, window_data):
        """
        Main extraction function.
        Input: window_data (N, 3) or (N, 6) etc.
        Output: Dictionary of ~70 features.
        """
        # 1. Calculate Magnitude (Univariate Signal)
        # We use the norm of the first 3 columns (usually Acc X,Y,Z)
        if window_data.ndim > 1 and window_data.shape[1] >= 3:
            signal = np.linalg.norm(window_data[:, :3], axis=1)
        else:
            signal = window_data.flatten()
            
        # 2. Normalization (Crucial for fixed edge length TDA)
        scaler = MinMaxScaler()
        signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        
        all_features = {}
        
        # --- A. TIME DELAY EMBEDDINGS (4 Delays x 2 Dims x 7 Feats = 56 Features) ---
        for factor in self.delay_factors:
            delay = int(factor * self.fs)
            if delay < 1: delay = 1 # Safety check
            
            point_cloud = self._get_delay_embedding(signal, delay)
            
            # Default values in case of failure
            for dim in [0, 1]:
                prefix = f"D_{factor}fs_H{dim}"
                for k in ['W1', 'W_inf', 'Entropy', 'Betti_L1', 'Betti_L2', 'Land_L1', 'Land_L2']:
                    all_features[f"{prefix}_{k}"] = 0.0

            if point_cloud is not None and len(point_cloud) > 5:
                try:
                    # Rips Complex (max_edge 1.75 from Version 1 logic)
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
                    pass # Keep defaults

        # --- B. LEVEL SETS (2 Sets x 1 Dim x 7 Feats = 14 Features) ---
        for ls_type in ['Lower', 'Upper']:
            prefix = f"LS_{ls_type}_H0"
            # Default values
            for k in ['W1', 'W_inf', 'Entropy', 'Betti_L1', 'Betti_L2', 'Land_L1', 'Land_L2']:
                all_features[f"{prefix}_{k}"] = 0.0

            try:
                if ls_type == 'Lower':
                    filt_signal = signal
                else:
                    filt_signal = -1 * signal # Invert for upper level sets
                
                # Cubical Complex for 1D signal (very fast)
                cc = gudhi.CubicalComplex(top_dimensional_cells=filt_signal)
                cc.persistence()
                intervals = cc.persistence_intervals_in_dimension(0)
                
                # Filter out infinite intervals (global min/max)
                finite_int = intervals[np.isfinite(intervals[:, 1])]
                
                ls_feats = self._compute_diagram_features(finite_int)
                for k, v in ls_feats.items():
                    all_features[f"{prefix}_{k}"] = v
            except:
                pass
                    
        return all_features

# =============================================================================
# 2. SUBWINDOWING & DB HELPERS (FROM YOUR CODE)
# =============================================================================
def create_indexes(conn, table_name):
    sql = f"CREATE INDEX IF NOT EXISTS idx_{table_name}_trial_time ON {table_name} (trial_id, timestamp);"
    conn.execute(sql)
    conn.commit()

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
                if fixed in existing: valid.append(fixed)
    return valid

def universal_subwindowing_memory(signal_data, original_fs, target_fs, window_sec, stride_sec, min_ratio=0.8):
    """
    Performs subwindowing in memory without saving files.
    """
    if original_fs is None or original_fs <= 0: step = 1
    else:
        step = int(original_fs / target_fs)
        step = max(step, 1)

    resampled = signal_data[::step]
    win_len = int(window_sec * target_fs)
    stride_len = int(stride_sec * target_fs)
    min_len = int(min_ratio * win_len)

    if len(resampled) < min_len: return None

    windows = []
    for i in range(0, len(resampled) - win_len + 1, stride_len):
        windows.append(resampled[i:i + win_len])

    return np.array(windows)

# =============================================================================
# 3. MAIN INTEGRATED PIPELINE
# =============================================================================
def run_tda_pipeline_v2(config):
    print(f"\n{'='*80}")
    print(f"🚀 TDA VERSION 2 PIPELINE (70 Features)")
    print(f"📁 Processing: {config['dataset_name']}")
    print(f"💾 Output CSV: {config['output_csv']}")
    print(f"{'='*80}")

    # Output directory
    os.makedirs(os.path.dirname(config['output_csv']), exist_ok=True)
    
    # Initialize TDA Engine
    tda_engine = TDA_Engine_V2(target_fs=config['target_fs'])

    # Connect DB
    conn = sqlite3.connect(config['db_path'])
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    create_indexes(conn, config['table_sensor'])
    sensor_cols = check_db_columns(conn, config['table_sensor'], config['sensor_columns'])
    
    if not sensor_cols:
        print("❌ No sensor columns found.")
        conn.close(); return

    # Get Trials
    try:
        trials_df = pd.read_sql_query(f"SELECT trial_id, subject_id, activity_id, sampling_rate FROM {config['table_trials']}", conn)
    except:
        trials_df = pd.read_sql_query(f"SELECT trial_id, subject_id, activity_id FROM {config['table_trials']}", conn)
        trials_df["sampling_rate"] = config["original_fs"]

    # Check for existing progress to resume (Optional)
    if os.path.exists(config['output_csv']):
        try:
            processed_trials = pd.read_csv(config['output_csv'], usecols=['Trial'])['Trial'].unique()
            trials_df = trials_df[~trials_df['trial_id'].isin(processed_trials)]
            print(f"🔄 Resuming... Skipping {len(processed_trials)} already processed trials.")
        except:
            pass # File might be empty or corrupt, start over

    total_wins = 0
    buffer = []
    BATCH_SIZE = 50 # Write to disk every 50 trials (or when buffer gets big)

    pbar = tqdm(trials_df.itertuples(index=False), total=len(trials_df), desc=f"{config['dataset_name']}", unit="trial")
    
    for row in pbar:
        t_id, s_id, a_id = row.trial_id, row.subject_id, row.activity_id
        fs = row.sampling_rate if row.sampling_rate and row.sampling_rate > 0 else config["original_fs"]

        # Read Sensor Data
        try:
            query = f"SELECT {', '.join(sensor_cols)}, {config['label_column']} FROM {config['table_sensor']} WHERE trial_id = {t_id} ORDER BY timestamp ASC"
            df_sensor = pd.read_sql_query(query, conn)
        except: continue

        if df_sensor.empty: continue
        
        # Clean NaNs
        active_cols = [c for c in sensor_cols if not df_sensor[c].isna().all()]
        if not active_cols: continue
        df_sensor.dropna(subset=active_cols, inplace=True)
        if df_sensor.empty: continue

        # Logic
        label = 1 if df_sensor[config["label_column"]].sum() > 0 else 0
        raw_data = df_sensor[active_cols].values

        # Subwindowing (In Memory)
        windows = universal_subwindowing_memory(
            raw_data, fs, config["target_fs"], 
            config["window_sec"], config["stride_sec"]
        )

        if windows is None: continue

        # TDA Feature Extraction Loop for Windows
        for i, window in enumerate(windows):
            try:
                # Extract 70 Features
                feats = tda_engine.extract_all(window)
                
                # Add Metadata
                row_data = {
                    'Dataset': config['dataset_name'],
                    'Subject': s_id,
                    'Activity_ID': a_id,
                    'Trial': t_id,
                    'Label': label,
                    'Window_Idx': i
                }
                row_data.update(feats)
                buffer.append(row_data)
            except Exception as e:
                pass # Skip problematic window

        # Batch Write
        if len(buffer) >= 200: # Write when buffer has 200 windows
            df_batch = pd.DataFrame(buffer)
            # Ensure column order
            meta_cols = ['Dataset', 'Subject', 'Activity_ID', 'Trial', 'Label', 'Window_Idx']
            other_cols = [c for c in df_batch.columns if c not in meta_cols]
            df_batch = df_batch[meta_cols + other_cols]

            file_exists = os.path.exists(config['output_csv'])
            df_batch.to_csv(config['output_csv'], mode='a', header=not file_exists, index=False)
            
            total_wins += len(df_batch)
            buffer = []
            gc.collect()

    # Final Write
    if buffer:
        df_batch = pd.DataFrame(buffer)
        meta_cols = ['Dataset', 'Subject', 'Activity_ID', 'Trial', 'Label', 'Window_Idx']
        other_cols = [c for c in df_batch.columns if c not in meta_cols]
        df_batch = df_batch[meta_cols + other_cols]
        
        file_exists = os.path.exists(config['output_csv'])
        df_batch.to_csv(config['output_csv'], mode='a', header=not file_exists, index=False)
        total_wins += len(df_batch)

    conn.close()
    print(f"✅ Finished {config['dataset_name']}. Total Windows: {total_wins}")


# =============================================================================
# 4. CONFIGURATIONS
# =============================================================================
BASE_DB_PATH = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataFirstTouch\All_Datasets_Container.db"
BASE_OUT_DIR = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataOrganize_V2\TDA_V2_Features" # New Folder for Version 2

SISFALL_CONFIG = {
    "dataset_name": "SisFall",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(BASE_OUT_DIR, "SisFall_V2.csv"),
    "table_trials": "SisFall_trials",
    "table_sensor": "SisFall_sensor_data",
    "sensor_columns": ["acc_adx1345_x", "acc_adx1345_y", "acc_adx1345_z", "gyr_itg3200_x", "gyr_itg3200_y", "gyr_itg3200_z"],
    "label_column": "is_fall",
    "original_fs": 200,
    "target_fs": 50,
    "window_sec": 2.0,
    "stride_sec": 1.0
}

FAD_40HZ_CONFIG = {
    "dataset_name": "FAD_40Hz",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(BASE_OUT_DIR, "FAD_40Hz_V2.csv"),
    "table_trials": "FAD_40Hz_trials",
    "table_sensor": "FAD_40Hz_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 40,
    "target_fs": 40,
    "window_sec": 2.0,
    "stride_sec": 1.0
}

FAD_MASTER_CONFIG = {
    "dataset_name": "FAD_Master",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(BASE_OUT_DIR, "FAD_Master_V2.csv"),
    "table_trials": "FAD_Master_trials",
    "table_sensor": "FAD_Master_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 238,
    "target_fs": 40,
    "window_sec": 0.8,
    "stride_sec": 0.4
}

FD_ORI_CONFIG = {
    "dataset_name": "FD_Ori",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(BASE_OUT_DIR, "FD_Ori_V2.csv"),
    "table_trials": "FD_Ori_trials",
    "table_sensor": "FD_Ori_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 100,
    "target_fs": 50,
    "window_sec": 2.0,
    "stride_sec": 1.0
}

MOBIFALL_CONFIG = {
    "dataset_name": "MobiFall",
    "db_path": BASE_DB_PATH,
    "output_csv": os.path.join(BASE_OUT_DIR, "MobiFall_V2.csv"),
    "table_trials": "MobiFall_trials",
    "table_sensor": "MobiFall_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "label_column": "is_fall",
    "original_fs": 100,
    "target_fs": 50,
    "window_sec": 2.0,
    "stride_sec": 1.0
}

# =============================================================================
# 5. EXECUTION
# =============================================================================
if __name__ == "__main__":
    # You can comment/uncomment these lines to run specific datasets
    run_tda_pipeline_v2(SISFALL_CONFIG)
    run_tda_pipeline_v2(FAD_40HZ_CONFIG)
    run_tda_pipeline_v2(FAD_MASTER_CONFIG)
    run_tda_pipeline_v2(FD_ORI_CONFIG)
    run_tda_pipeline_v2(MOBIFALL_CONFIG)