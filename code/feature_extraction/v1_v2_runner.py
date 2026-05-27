import subprocess
import sys
import time
import os

# =============================================================================
# 🌍 EVRENSEL ÇALIŞTIRICI (Farklı Klasörler İçin - TQDM Fix)
# =============================================================================

# Dosya yollarını buraya tanımlayın
SCRIPTS = [
    r"C:\Users\user\Desktop\KEB\YL_TEZ\TDA_Features_Extraction_V1\all_pipeline_V1.py",
    # 1. V2 TDA (Smart Delay)
    r"C:\Users\user\Desktop\KEB\YL_TEZ\TDA_Features_Extraction_V2\all_pipeline_V2.py",
    # 2. V3 TDA (Multi Scale)
    r"C:\Users\user\Desktop\KEB\YL_TEZ\TDA_Features_Extraction_V3\all_pipeline_V3.py"
]

def run_script_from_anywhere(full_path):
    script_name = os.path.basename(full_path)
    script_dir = os.path.dirname(full_path)
    
    print(f"\n{'='*70}")
    print(f"🚀 BAŞLATILIYOR: {script_name}")
    print(f"📂 Konum: {script_dir}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(full_path):
        print(f"❌ HATA: Dosya bulunamadı!\n   Yol: {full_path}")
        return

    start_time = time.time()
    
    try:
        # --- KRİTİK DÜZELTME ---
        # sys.executable, "-u" parametresi ile çağrılır.
        # "-u" (unbuffered binary stdout/stderr) çıktının anlık olarak terminale
        # akmasını sağlar, böylece tqdm takılmaz veya satır atlamaz.
        subprocess.run(
            [sys.executable, "-u", script_name], 
            cwd=script_dir, 
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ BAŞARILI: {script_name} tamamlandı.")
        print(f"⏱️  Süre: {elapsed/60:.2f} dakika")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ HATA: {script_name} çalışırken çöktü.")
        print(f"Hata Kodu: {e}")

if __name__ == "__main__":
    # TQDM'in terminal genişliğini doğru algılaması için ortam değişkeni ayarı (Opsiyonel ama faydalı)
    os.environ["TQDM_DISABLE"] = "0"
    
    print("🤖 Master Runner Başlıyor... (TQDM Düzeltildi)\n")
    
    total_start = time.time()
    
    for script_path in SCRIPTS:
        run_script_from_anywhere(script_path)
        
        print("💤 Sistem dinleniyor (60 sn)...")
        time.sleep(60)
        
    total_end = time.time()
    print(f"\n{'='*70}")
    print(f"🎉 TÜM ZİNCİR TAMAMLANDI!")
    print(f"⏱️  Toplam Geçen Süre: {(total_end - total_start)/60:.2f} dakika")
    print(f"{'='*70}")