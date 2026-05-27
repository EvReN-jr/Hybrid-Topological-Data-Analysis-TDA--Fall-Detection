import sqlite3
import os
import glob
from tqdm import tqdm  # progress bar için

db_folder = r"C:\Users\kbybt\Desktop\EVREN\MatMüh_İTÜ_ARŞ_GÖR\YL_TEZ\DataBases"
merged_db_path = r"C:\Users\kbybt\Desktop\EVREN\MatMüh_İTÜ_ARŞ_GÖR\YL_TEZ\DataBases\merged.db"


conn_merged = sqlite3.connect(merged_db_path)
cursor_merged = conn_merged.cursor()

db_files = glob.glob(os.path.join(db_folder, "*.db"))
total_files = len(db_files)
print(f"{total_files} dosya bulundu. Birleştirme başlıyor...\n")

for idx, db_file in enumerate(db_files, start=1):
    if db_file == merged_db_path:
        continue

    print(f"[{idx}/{total_files}] İşleniyor: {os.path.basename(db_file)}")
    
    conn_temp = sqlite3.connect(db_file)
    cursor_temp = conn_temp.cursor()

    cursor_temp.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor_temp.fetchall()

    for table_name_tuple in tables:
        table_name = table_name_tuple[0]

        # Sistem tablolarını atla
        if table_name.startswith("sqlite_"):
            continue

        cursor_temp.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor_temp.fetchall()
        column_defs = ", ".join([f"{col[1]} {col[2]}" for col in columns_info])
        cursor_merged.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({column_defs})")

        # Verileri ekle, chunk'lar halinde ve progress bar ile
        cursor_temp.execute(f"SELECT * FROM {table_name}")
        rows = cursor_temp.fetchall()
        chunk_size = 5000

        if rows:
            placeholders = ", ".join(["?"] * len(columns_info))
            print(f"    Tablo '{table_name}' satır ekleniyor: {len(rows)} satır")
            for i in tqdm(range(0, len(rows), chunk_size), desc=f"        {table_name}", unit="chunk"):
                cursor_merged.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", rows[i:i+chunk_size])

    conn_temp.close()

conn_merged.commit()
conn_merged.close()
print("\nBirleştirme tamamlandı!")
