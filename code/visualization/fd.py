import sqlite3
import pandas as pd
import numpy as np

# Veritabanı yolunuz
DB_PATH = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataFirstTouch\All_Datasets_Container.db"

def inspect_fd_ori():
    print(f"{'='*60}")
    print("🔍 FD_Ori DIAGNOSTIC TOOL")
    print(f"{'='*60}")
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Tablo Doluluk Kontrolü
    try:
        count = pd.read_sql_query("SELECT COUNT(*) as cnt FROM FD_Ori_sensor_data", conn)
        print(f"📊 Toplam Sensör Verisi Satır Sayısı: {count['cnt'][0]}")
        
        trial_count = pd.read_sql_query("SELECT COUNT(*) as cnt FROM FD_Ori_trials", conn)
        print(f"📄 Toplam Trial Sayısı: {trial_count['cnt'][0]}")
    except Exception as e:
        print(f"❌ HATA: Tablolar okunamadı. {e}")
        conn.close()
        return

    # 2. Örnek Bir Trial İncelemesi (Süre Kontrolü)
    try:
        # Rastgele bir trial seç
        trial_id = pd.read_sql_query("SELECT trial_id FROM FD_Ori_trials LIMIT 1", conn)['trial_id'][0]
        
        # O trial'ın verisini çek
        df = pd.read_sql_query(f"SELECT * FROM FD_Ori_sensor_data WHERE trial_id = {trial_id}", conn)
        
        if df.empty:
            print(f"⚠️ UYARI: Trial ID {trial_id} için sensör verisi BULUNAMADI!")
        else:
            duration_samples = len(df)
            # Eğer frekans 100Hz ise süre: samples / 100
            estimated_duration = duration_samples / 50 # Varsayılan 50Hz tahmini
            print(f"\n⏱️  Örnek Trial (ID: {trial_id}) Analizi:")
            print(f"   - Satır Sayısı: {duration_samples}")
            print(f"   - Tahmini Süre (50Hz ise): {duration_samples/50:.2f} saniye")
            print(f"   - Tahmini Süre (100Hz ise): {duration_samples/100:.2f} saniye")
            
            if duration_samples < 100: # 2 saniye (50Hz) için 100 örnek lazım
                print("\n🚨 KRİTİK TESPİT: Kayıt süresi çok kısa! 2.0 saniye pencere buraya sığmaz.")
                print("✅ ÇÖZÜM: Config dosyasında FD_Ori için 'window_sec' değerini 1.0 veya 0.8 yapın.")
            else:
                print("\n✅ Veri uzunluğu yeterli görünüyor. Sorun sütun isimlerinde olabilir.")
                print(f"   - Sütunlar: {list(df.columns)}")

    except Exception as e:
        print(f"❌ Veri analizi sırasında hata: {e}")

    conn.close()

if __name__ == "__main__":
    inspect_fd_ori()
    
    
import sqlite3
import pandas as pd
import numpy as np

# =============================================================================
# AYARLAR
# =============================================================================
DB_PATH = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataFirstTouch\All_Datasets_Container.db"

# Kullanılan Config (V2/V3 ile aynı)
CONFIG = {
    "table_trials": "FD_Ori_trials",
    "table_sensor": "FD_Ori_sensor_data",
    "sensor_columns": ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
    "target_fs": 50,
    "window_sec": 1.5,
    "stride_sec": 0.75,
    "original_fs": 100
}

def debug_step_by_step():
    print(f"{'='*60}")
    print("🔍 FD_Ori DETAYLI HATA ANALİZİ")
    print(f"{'='*60}\n")
    
    conn = sqlite3.connect(DB_PATH)
    
    # ADIM 1: Sütun İsimlerini Doğrula
    print("--- ADIM 1: Sütun İsimleri Kontrolü ---")
    cursor = conn.execute(f"PRAGMA table_info({CONFIG['table_sensor']})")
    db_cols = [row[1] for row in cursor.fetchall()]
    print(f"🗄️  Veritabanındaki Sütunlar: {db_cols}")
    print(f"⚙️  İstenen Sütunlar: {CONFIG['sensor_columns']}")
    
    found_cols = [c for c in CONFIG['sensor_columns'] if c in db_cols]
    if len(found_cols) != len(CONFIG['sensor_columns']):
        print(f"❌ KRİTİK HATA: İstenen sütunlar veritabanında yok!")
        print(f"   Bulunanlar: {found_cols}")
        print(f"   Eksikler: {set(CONFIG['sensor_columns']) - set(db_cols)}")
        return
    else:
        print("✅ Sütun eşleşmesi BAŞARILI.\n")

    # ADIM 2: İlk Trial'ı Bul
    print("--- ADIM 2: Trial ID Kontrolü ---")
    try:
        trial_row = pd.read_sql_query(f"SELECT trial_id, sampling_rate FROM {CONFIG['table_trials']} LIMIT 1", conn).iloc[0]
        trial_id = trial_row['trial_id']
        sampling_rate = trial_row['sampling_rate']
        print(f"📄 Bulunan İlk Trial ID: {trial_id} (Tip: {type(trial_id)})")
        print(f"   Sampling Rate: {sampling_rate}")
    except Exception as e:
        print(f"❌ Trial tablosu okunamadı: {e}")
        return

    # ADIM 3: Sensör Verisini Çek (SQL Sorgusu)
    print("\n--- ADIM 3: Veri Çekme Testi ---")
    q = f"SELECT {', '.join(CONFIG['sensor_columns'])} FROM {CONFIG['table_sensor']} WHERE trial_id = {trial_id}"
    print(f"💻 SQL Sorgusu: {q}")
    
    df_sensor = pd.read_sql_query(q, conn)
    print(f"📊 Çekilen Satır Sayısı: {len(df_sensor)}")
    
    if df_sensor.empty:
        print("❌ HATA: Bu Trial ID için sensör verisi BOŞ dönüyor!")
        # ID Uyuşmazlığı kontrolü
        check_any = pd.read_sql_query(f"SELECT trial_id FROM {CONFIG['table_sensor']} LIMIT 5", conn)
        print(f"   İpucu: Sensör tablosundaki örnek trial_id'ler: {check_any['trial_id'].tolist()}")
        return
    else:
        print("✅ Veri başarıyla çekildi.")
        print("   İlk 3 satır:")
        print(df_sensor.head(3))

    # ADIM 4: NaN Kontrolü (dropna)
    print("\n--- ADIM 4: Temizlik (dropna) Kontrolü ---")
    before_len = len(df_sensor)
    df_sensor.dropna(subset=CONFIG['sensor_columns'], inplace=True)
    after_len = len(df_sensor)
    print(f"🧹 Temizlik Öncesi: {before_len} -> Sonrası: {after_len}")
    
    if after_len == 0:
        print("❌ HATA: Tüm veriler NaN (boş) olduğu için silindi!")
        return
    else:
        print("✅ Veriler temiz.")

    # ADIM 5: Pencereleme Matematiği
    print("\n--- ADIM 5: Pencereleme (Subwindowing) Testi ---")
    raw_data = df_sensor.values
    fs = sampling_rate if sampling_rate and sampling_rate > 0 else CONFIG['original_fs']
    target_fs = CONFIG['target_fs']
    
    # Matematik
    step = max(1, int(fs / target_fs))
    resampled = raw_data[::step]
    win_len_samples = int(CONFIG['window_sec'] * target_fs)
    
    print(f"🧮 Kullanılan Frekans (FS): {fs}")
    print(f"🧮 Downsample Adımı (Step): {step}")
    print(f"📏 Ham Veri Uzunluğu: {len(raw_data)}")
    print(f"b' Resample Edilmiş Uzunluk: {len(resampled)}")
    print(f"🎯 Hedef Pencere Boyutu (Sample): {win_len_samples}")
    
    if len(resampled) < win_len_samples:
        print(f"❌ KRİTİK HATA: Veri çok kısa! ({len(resampled)} < {win_len_samples})")
        print("   Çözüm: window_sec değerini düşürün.")
    else:
        num_windows = (len(resampled) - win_len_samples) // int(CONFIG['stride_sec'] * target_fs) + 1
        print(f"✅ BAŞARILI! Bu trial'dan {num_windows} adet pencere çıkarılabilir.")

    conn.close()

if __name__ == "__main__":
    debug_step_by_step()