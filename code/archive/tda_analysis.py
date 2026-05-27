import os
import numpy as np
import pandas as pd
import gudhi
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import gc

# =============================================================================
# AYARLAR - DÜŞÜK RAM SİSTEMLER İÇİN OPTİMİZE EDİLDİ
# =============================================================================
INPUT_DIR = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataOrganize\SubWindowing"
OUTPUT_DIR = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataOrganize\TDA"

# TDA Parametreleri
HOMOLOGY_DIMS = [0, 1]

# Performans Ayarları (Düşük RAM İçin)
WRITE_EVERY = 50  # Her 50 pencerede bir diske yaz
MAX_EDGE_LENGTH = 1.75  # Olası en büyük mesafe

def extract_tda_features(point_cloud):
    """Tek bir pencere için TDA özelliklerini hesaplar"""
    features = {}
    try:
        rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=MAX_EDGE_LENGTH)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        simplex_tree.persistence()
    except:
        return None
    
    for dim in HOMOLOGY_DIMS:
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        
        if len(intervals) == 0:
            features[f'H{dim}_Entropy'] = 0
            features[f'H{dim}_MaxLife'] = 0
            features[f'H{dim}_AvgLife'] = 0
            features[f'H{dim}_Count'] = 0
            continue
            
        finite_intervals = intervals[np.isfinite(intervals[:, 1])]
        
        if len(finite_intervals) == 0:
            features[f'H{dim}_Entropy'] = 0
            features[f'H{dim}_MaxLife'] = 0
            features[f'H{dim}_AvgLife'] = 0
            features[f'H{dim}_Count'] = len(intervals)
            continue
            
        lifetimes = finite_intervals[:, 1] - finite_intervals[:, 0]
        total_life = np.sum(lifetimes)
        
        if total_life > 0:
            probs = lifetimes / total_life
            entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            entropy = 0
            
        features[f'H{dim}_Entropy'] = entropy
        features[f'H{dim}_MaxLife'] = np.max(lifetimes)
        features[f'H{dim}_AvgLife'] = np.mean(lifetimes)
        features[f'H{dim}_Count'] = len(intervals)
        
    return features

def parse_filename(filename):
    """Dosya isminden meta bilgileri çıkar"""
    try:
        name = filename.replace('.npy', '')
        parts = name.split('_')
        info = {}

        if 'Sub' in parts:
            idx = parts.index('Sub')
            info['Subject'] = parts[idx + 1]
            info['Dataset'] = "_".join(parts[:idx])
        else:
            info['Subject'] = 'Unknown'
            info['Dataset'] = parts[0]

        if 'Act' in parts:
            try:
                idx = parts.index('Act')
                info['Activity_ID'] = int(parts[idx + 1])
            except:
                info['Activity_ID'] = -1
        else:
            info['Activity_ID'] = -1

        if 'Trial' in parts:
            try:
                idx = parts.index('Trial')
                info['Trial'] = int(parts[idx + 1])
            except:
                info['Trial'] = -1
        else:
            info['Trial'] = -1

        if 'Fall' in parts:
            try:
                idx = parts.index('Fall')
                info['Label'] = int(parts[idx + 1])
            except:
                info['Label'] = int(parts[-1])
        else:
            try:
                info['Label'] = int(parts[-1])
            except:
                info['Label'] = -1
            
        return info
    except:
        return None

def process_file_sequential(file_path, filename, output_csv_path):
    """
    Tek bir dosyayı sıralı olarak işle
    RAM'i korumak için her WRITE_EVERY pencerede bir diske yaz
    """
    try:
        windows_batch = np.load(file_path)
    except:
        return 0
    
    meta = parse_filename(filename)
    if not meta:
        return 0
    
    num_windows = windows_batch.shape[0]
    buffer = []
    total_saved = 0
    scaler = MinMaxScaler()
    
    # Progress bar kaldırıldı - sadak klasör seviyesinde gösterim
    for i in range(num_windows):
        window = windows_batch[i]
        
        # Bozuk veri kontrolü
        if np.isnan(window).any() or np.isinf(window).any():
            continue
        
        try:
            window_norm = scaler.fit_transform(window)
        except ValueError:
            continue

        # TDA Hesapla
        feats = extract_tda_features(window_norm)
        
        if feats:
            row = meta.copy()
            row['Window_Idx'] = i
            row.update(feats)
            buffer.append(row)
            total_saved += 1
            
            # Buffer dolduğunda diske yaz ve temizle
            if len(buffer) >= WRITE_EVERY:
                df = pd.DataFrame(buffer)
                
                # Sütun düzenleme
                meta_cols = ['Dataset', 'Subject', 'Activity_ID', 'Trial', 'Label', 'Window_Idx']
                other_cols = [c for c in df.columns if c not in meta_cols]
                df = df[meta_cols + other_cols]
                
                # Append mode
                file_exists = os.path.exists(output_csv_path)
                df.to_csv(output_csv_path, mode='a', header=not file_exists, index=False)
                
                # Buffer temizle
                buffer.clear()
                del df
                gc.collect()
    
    # Kalan buffer'ı yaz
    if buffer:
        df = pd.DataFrame(buffer)
        meta_cols = ['Dataset', 'Subject', 'Activity_ID', 'Trial', 'Label', 'Window_Idx']
        other_cols = [c for c in df.columns if c not in meta_cols]
        df = df[meta_cols + other_cols]
        
        file_exists = os.path.exists(output_csv_path)
        df.to_csv(output_csv_path, mode='a', header=not file_exists, index=False)
        del df
        gc.collect()
    
    # Hafızayı temizle
    del windows_batch
    gc.collect()
    
    return total_saved

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*80)
    print("🔬 TDA Feature Extraction (Düşük RAM Optimizasyonlu)")
    print("="*80)
    print(f"📂 Kaynak    : {INPUT_DIR}")
    print(f"💾 Hedef     : {OUTPUT_DIR}")
    print(f"⚙️  Mod      : Sequential (RAM dostu)")
    print(f"📝 Yazma Sıklığı: Her {WRITE_EVERY} pencere")
    print("="*80 + "\n")
    
    subfolders = [f for f in os.listdir(INPUT_DIR) 
                  if os.path.isdir(os.path.join(INPUT_DIR, f))]
    
    print(subfolders)
    subfolders=['MobiFall_SubWindow']
    
    if not subfolders:
        print("❌ Klasör bulunamadı.")
        return

    # Klasörler üzerinde döngü
    for folder_idx, folder_name in enumerate(subfolders, 1):
        folder_path = os.path.join(INPUT_DIR, folder_name)
        npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
        
        if not npy_files:
            print(f"⚠️  [{folder_idx}/{len(subfolders)}] {folder_name} - BOŞ, Atlanıyor\n")
            continue
        
        print(f"📁 [{folder_idx}/{len(subfolders)}] {folder_name}")
        print(f"   Dosya Sayısı: {len(npy_files)}")
        
        # Çıktı dosyası
        clean_name = folder_name.replace("_SubWindow", "")
        output_csv_path = os.path.join(OUTPUT_DIR, f"{clean_name}_Features.csv")
        
        # Eski dosyayı sil
        if os.path.exists(output_csv_path):
            try:
                os.remove(output_csv_path)
            except PermissionError:
                print(f"   ❌ HATA: {output_csv_path} açık! Kapatıp tekrar deneyin.\n")
                continue
        
        # Dosya bazında işlem (tqdm ile)
        total_windows = 0
        
        file_pbar = tqdm(npy_files, 
                        desc=f"   [{folder_idx}/{len(subfolders)}] İşleniyor", 
                        ncols=100,
                        unit="dosya",
                        bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for npy_file in file_pbar:
            file_path = os.path.join(folder_path, npy_file)
            windows_saved = process_file_sequential(file_path, npy_file, output_csv_path)
            total_windows += windows_saved
            
            # Her dosya sonrası garbage collection
            gc.collect()
        
        file_pbar.close()
        
        print(f"   ✅ Tamamlandı: {total_windows:,} pencere kaydedildi")
        print(f"   💾 Dosya: {output_csv_path}\n")

    print("="*80)
    print("🏁 TÜM İŞLEMLER TAMAMLANDI")
    print("="*80)

if __name__ == "__main__":
    main()