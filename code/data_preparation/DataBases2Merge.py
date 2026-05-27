import sqlite3
import os

# ==========================================
# AYARLAR: Kaynak Dosyalar
# ==========================================
# Daha önce oluşturduğumuz DB dosyalarının yolları
SOURCE_DBS = {
    "FAD_Master":   "FallAllD_Master_ML_Ready.db",
    "FAD_40Hz":   "FallAllD_40Hz_ML_Ready.db",
    "SisFall":    "SisFall_ML_Ready_Detailed.db",
    "MobilFall":   "MobiFall_ML_Ready.db",
    "FD_Ori":     "FallDetection_WithOrientation.db"
}

# Hedef dosya adı
TARGET_DB = "All_Datasets_Container.db"

# ==========================================
# BİRLEŞTİRME KODU (ATTACH METHODU)
# ==========================================
def merge_databases_keeping_tables_separate():
    # Eğer hedef dosya varsa temizle (Sıfırdan başla)
    if os.path.exists(TARGET_DB):
        try:
            os.remove(TARGET_DB)
            print(f"Eski dosya silindi: {TARGET_DB}")
        except PermissionError:
            print("HATA: Dosya açık olduğu için silinemedi. Lütfen DB tarayıcılarını kapatın.")
            return

    # Hedef veritabanına bağlan
    tgt_conn = sqlite3.connect(TARGET_DB)
    tgt_cursor = tgt_conn.cursor()
    
    print(f"Hedef veritabanı oluşturuldu: {TARGET_DB}\n")

    for db_prefix, db_path in SOURCE_DBS.items():
        if not os.path.exists(db_path):
            print(f"⚠️ UYARI: Dosya bulunamadı, atlanıyor -> {db_path}")
            continue

        print(f"--- İşleniyor: {db_prefix} ({db_path}) ---")
        
        try:
            # 1. Kaynak veritabanını 'ATTACH' komutu ile bağla
            # Bu yöntem Python ile satır satır okumaktan 100 kat daha hızlıdır.
            attach_query = f"ATTACH DATABASE '{db_path}' AS source_db"
            tgt_cursor.execute(attach_query)
            
            # 2. Kaynak veritabanındaki tüm tabloların adını al
            tgt_cursor.execute("SELECT name FROM source_db.sqlite_master WHERE type='table'")
            tables = tgt_cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                
                # Sistem tablolarını atla
                if table_name.startswith("sqlite_"):
                    continue
                
                # 3. Yeni tablo adı oluştur (Örn: SisFall_sensor_data)
                new_table_name = f"{db_prefix}_{table_name}"
                
                print(f"   Transfer ediliyor: {table_name} -> {new_table_name}")
                
                # 4. Tabloyu ve verisini kopyala
                # "CREATE TABLE AS SELECT" komutu şemayı ve veriyi aynen kopyalar
                tgt_cursor.execute(f"CREATE TABLE {new_table_name} AS SELECT * FROM source_db.{table_name}")
            
            # 5. Kaynak veritabanını ayır (DETACH)
            tgt_cursor.execute("DETACH DATABASE source_db")
            print(f"✅ {db_prefix} başarıyla aktarıldı.\n")
            
        except sqlite3.OperationalError as e:
            print(f"❌ HATA: {e}")
        except Exception as e:
            print(f"❌ Beklenmeyen Hata: {e}")

    tgt_conn.commit()
    
    # --- SONUÇLARI LİSTELE ---
    print("="*40)
    print("BİRLEŞTİRME TAMAMLANDI! YENİ TABLO LİSTESİ:")
    print("="*40)
    tgt_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    final_tables = tgt_cursor.fetchall()
    for t in final_tables:
        print(f"📄 {t[0]}")
    
    tgt_conn.close()

if __name__ == "__main__":
    merge_databases_keeping_tables_separate()