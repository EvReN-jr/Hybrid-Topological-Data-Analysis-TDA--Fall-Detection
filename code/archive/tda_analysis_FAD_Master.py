import os
import numpy as np
import pandas as pd
import gudhi
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import multiprocessing

# =============================================================================
# AYARLAR
# =============================================================================
INPUT_DIR = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataOrganize_V2\SubWindowing_V2"
OUTPUT_DIR = r"C:\Users\user\Desktop\KEB\YL_TEZ\TDA"

# TDA Parametreleri
HOMOLOGY_DIMS = [0, 1]
N_JOBS = -1  # Tüm çekirdekleri kullan
BATCH_SIZE = 500  # HER 500 DOSYADA BİR KAYDEDER (RAM ŞİŞMEZ)

def extract_tda_features(point_cloud):
    """Tek bir pencere için TDA hesaplar."""
    features = {}
    try:
        # max_edge_length=1.5 optimizasyonu
        rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=1.5)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        persistence = simplex_tree.persistence()
    except Exception:
        return None
    
    for dim in HOMOLOGY_DIMS:
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        if len(intervals) == 0:
            features[f'H{dim}_Entropy'] = 0; features[f'H{dim}_MaxLife'] = 0; features[f'H{dim}_AvgLife'] = 0; features[f'H{dim}_Count'] = 0
            continue
            
        finite_intervals = intervals[np.isfinite(intervals[:, 1])]
        if len(finite_intervals) == 0:
            features[f'H{dim}_Entropy'] = 0; features[f'H{dim}_MaxLife'] = 0; features[f'H{dim}_AvgLife'] = 0; features[f'H{dim}_Count'] = len(intervals)
            continue
            
        lifetimes = finite_intervals[:, 1] - finite_intervals[:, 0]
        total_life = np.sum(lifetimes)
        entropy = -np.sum((lifetimes/total_life) * np.log(lifetimes/total_life)) if total_life > 0 else 0
            
        features[f'H{dim}_Entropy'] = entropy
        features[f'H{dim}_MaxLife'] = np.max(lifetimes)
        features[f'H{dim}_AvgLife'] = np.mean(lifetimes)
        features[f'H{dim}_Count'] = len(intervals)
    return features

def parse_filename(filename):
    """Dosya isminden metaveri çeker."""
    try:
        name = filename.replace('.npy', '')
        parts = name.split('_')
        label = int(parts[-1])
        try: trial_id = int(parts[parts.index('Trial') + 1])
        except: trial_id = -1
        try: subject_id = parts[parts.index('Sub') + 1]
        except: subject_id = "Unknown"
        dataset_name = parts[0]
        if len(parts) > 1 and (parts[1] == "40Hz" or parts[1] == "Master" or parts[1] == "Ori"):
            dataset_name = f"{parts[0]}_{parts[1]}"
        return {"Dataset": dataset_name, "Subject": subject_id, "Trial": trial_id, "Label": label}
    except: return None

def process_single_file(file_info):
    """Paralel işlem birimi."""
    file_path, filename = file_info
    file_results = []
    try: windows_batch = np.load(file_path)
    except: return []
    
    meta = parse_filename(filename)
    if not meta: return []

    for i in range(windows_batch.shape[0]):
        window = windows_batch[i]
        if np.isnan(window).any() or np.isinf(window).any(): continue
        try:
            scaler = MinMaxScaler()
            window_norm = scaler.fit_transform(window)
            feats = extract_tda_features(window_norm)
            if feats:
                row = meta.copy()
                row['Window_Idx'] = i
                row.update(feats)
                file_results.append(row)
        except: continue
    return file_results

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- Sadece FAD_Master ---
    subfolders = ["FAD_Master_SubWindow"] 
    
    for folder_name in subfolders:
        folder_path = os.path.join(INPUT_DIR, folder_name)
        if not os.path.exists(folder_path): continue

        npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        if not npy_files: continue
            
        print(f"🚀 {folder_name} işleniyor... ({len(npy_files)} dosya)")
        
        # Çıktı Dosya Yolu
        clean_name = folder_name.replace("_SubWindow", "") 
        output_csv_path = os.path.join(OUTPUT_DIR, f"{clean_name}_Features.csv")
        
        # Eğer dosya varsa üzerine yazmasın, sıfırdan başlatsın
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)

        # --- BATCH PROCESSING (RAM KORUYUCU) ---
        file_list = [(os.path.join(folder_path, f), f) for f in npy_files]
        total_batches = (len(file_list) + BATCH_SIZE - 1) // BATCH_SIZE
        
        total_processed_windows = 0
        
        for i in range(0, len(file_list), BATCH_SIZE):
            batch_files = file_list[i : i + BATCH_SIZE]
            current_batch_num = (i // BATCH_SIZE) + 1
            
            print(f"   📦 Batch {current_batch_num}/{total_batches} işleniyor... ({len(batch_files)} dosya)")
            
            # Paralel İşlem (Sadece bu batch için)
            results = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(process_single_file)(f) for f in batch_files
            )
            
            # Sonuçları Düzleştir
            flat_results = [item for sublist in results for item in sublist]
            
            if flat_results:
                df_batch = pd.DataFrame(flat_results)
                
                # Diske Yaz (Append Mode)
                # İlk batch ise başlıkları yaz (header=True), sonrakilerde yazma (header=False)
                is_first_batch = (i == 0)
                df_batch.to_csv(output_csv_path, mode='a', header=is_first_batch, index=False)
                
                total_processed_windows += len(df_batch)
                
                # RAM TEMİZLİĞİ
                del df_batch
                del flat_results
                del results
                
        print(f"✅ {clean_name} tamamlandı! Toplam {total_processed_windows} pencere kaydedildi.\n")

    print("🏁 İŞLEM BİTTİ.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()