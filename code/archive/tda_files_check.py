import pandas as pd
import os

# Yolları kontrol et
CSV_PATH = r"C:\Users\user\Desktop\KEB\YL_TEZ\TDA\MobiFall_Features.csv"
ORI_FOLDER = r"C:\Users\user\Desktop\KEB\YL_TEZ\DataOrganize_V2\SubWindowing_V2\MobiFall_SubWindow"

if os.path.exists(CSV_PATH):
    # 1. CSV'deki benzersiz dosya (veya Subject/Trial) sayısını bulalım
    # Not: Senin kodunda 'Dataset' ve 'Subject' bilgileri vardı. 
    # Kaç farklı denemenin (trial) başarıyla kaydedildiğine bakalım.
    df = pd.read_csv(CSV_PATH)
    
    # Senin isimlendirme formatına göre kaç tane benzersiz veri grubu var?
    # Dataset, Subject, Trial ve Activity_ID sütunlarını birleştirerek kaç dosya bittiğine bakalım
    processed_files_count = len(df.groupby(['Subject', 'Activity_ID', 'Trial']))
    
    # 2. Orijinal klasördeki toplam .npy dosyası sayısını alalım
    total_files_in_folder = len([f for f in os.listdir(ORI_FOLDER) if f.endswith('.npy')])
    
    print(f"📊 MOBIFALL KONTROL SONUCU:")
    print(f"---------------------------")
    print(f"📁 Klasördeki toplam dosya sayısı: {total_files_in_folder}")
    print(f"✅ CSV'ye aktarılan dosya sayısı: {processed_files_count}")
    
    fark = total_files_in_folder - processed_files_count
    if fark <= 0:
        print(f"\n🚀 SONUÇ: MobiFall TAMAMLANMIŞ! Hiç eksik yok.")
    elif fark > 0:
        print(f"\n⚠️ SONUÇ: MobiFall YARIM KALMIŞ! {fark} adet dosya işlenemeden bilgisayar kapanmış.")
else:
    print("❌ MobiFall CSV dosyası bulunamadı!")