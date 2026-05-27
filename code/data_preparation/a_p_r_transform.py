# -*- coding: utf-8 -*-
"""
MobiFall Sensor Verilerini İşleme + Madgwick + Yaw Drift Düzeltme
"""
import pandas as pd
import numpy as np
import sqlite3
import os
from tqdm import tqdm

# -------------------------
# 1️⃣ Madgwick Filter Sınıfı
# -------------------------
class MadgwickAHRS:
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def update_imu(self, gx, gy, gz, ax, ay, az, dt):
        q = self.q.copy()
        norm = np.linalg.norm([ax, ay, az])
        if norm == 0:
            return
        ax, ay, az = ax / norm, ay / norm, az / norm

        f = np.array([
            2*(q[1]*q[3]-q[0]*q[2]) - ax,
            2*(q[0]*q[1]+q[2]*q[3]) - ay,
            2*(0.5-q[1]**2-q[2]**2) - az
        ])
        J = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        step = J.T @ f
        step /= np.linalg.norm(step)

        q_dot = 0.5 * self.quat_multiply(q, np.array([0, gx, gy, gz])) - self.beta * step
        self.q += q_dot * dt
        self.q /= np.linalg.norm(self.q)

    @staticmethod
    def quat_multiply(q, r):
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = r
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def get_euler(self):
        q = self.q
        roll = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]**2+q[2]**2))
        pitch = np.arcsin(2*(q[0]*q[2]-q[3]*q[1]))
        yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
        return roll, pitch, yaw

# -------------------------
# 2️⃣ Yaw drift düzeltme
# -------------------------
def update_yaw_drift(yaw_list, last_smoothed=0.0, alpha=0.005):
    smoothed = [last_smoothed]
    corrected = []
    for yaw in yaw_list:
        smoothed_value = alpha * yaw + (1-alpha)*smoothed[-1]
        smoothed.append(smoothed_value)
        corrected_value = yaw - smoothed_value + smoothed[0]
        corrected.append(corrected_value)
    return corrected, smoothed[-1]

# -------------------------
# 3️⃣ DB bağlantısı
# -------------------------
os.chdir(r"C:\Users\kbybt\Desktop\EVREN\MatMüh_İTÜ_ARŞ_GÖR\YL_TEZ\DataBases")
input_db = "MobiFallData.db"
chunksize = 100_000
conn = sqlite3.connect(input_db)

# İşlenmiş tabloyu oluştur
conn.execute("DROP TABLE IF EXISTS MFD_Data_Processed")
conn.commit()

query = "SELECT * FROM MFD_Data"

# Trial + trial_no bazlı state saklama
trial_states = {}

# -------------------------
# 4️⃣ Chunk bazlı okuma
# -------------------------
for chunk_index, chunk in enumerate(
        tqdm(pd.read_sql_query(query, conn, chunksize=chunksize),
             desc="Chunks işleniyor")):

    processed_trials = []

    # trial_id + trial_no bazlı gruplama
    for (trial_id, trial_no), df_trial in tqdm(
            chunk.groupby(['trial_id','trial_no']),
            desc="Trials", leave=False):

        # Trial state oluştur
        state_key = (trial_id, trial_no)
        if state_key not in trial_states:
            trial_states[state_key] = {"madgwick": MadgwickAHRS(beta=0.1),
                                       "yaw_smoothed": 0.0}

        madgwick = trial_states[state_key]["madgwick"]
        last_smoothed = trial_states[state_key]["yaw_smoothed"]

        # Zaman
        time_acc = df_trial['timestamp_ns_acc'].values * 1e-9
        time_gyro = df_trial['timestamp_ns_gyro'].values * 1e-9

        # Accelerometer interpolate
        acc_x = np.interp(time_gyro, time_acc, df_trial['acc_x'].values)
        acc_y = np.interp(time_gyro, time_acc, df_trial['acc_y'].values)
        acc_z = np.interp(time_gyro, time_acc, df_trial['acc_z'].values)

        gyro_x = df_trial['gyro_x'].values
        gyro_y = df_trial['gyro_y'].values
        gyro_z = df_trial['gyro_z'].values

        # Madgwick filtre
        roll_list, pitch_list, yaw_list = [], [], []
        for i in range(len(time_gyro)):
            dt = 0.0 if i==0 else time_gyro[i]-time_gyro[i-1]
            madgwick.update_imu(gyro_x[i], gyro_y[i], gyro_z[i],
                                acc_x[i], acc_y[i], acc_z[i], dt)
            roll, pitch, yaw = madgwick.get_euler()
            roll_list.append(np.degrees(roll))
            pitch_list.append(np.degrees(pitch))
            yaw_list.append(np.degrees(yaw))

        # Yaw drift düzeltme
        yaw_corrected, new_smoothed = update_yaw_drift(
            yaw_list, last_smoothed, alpha=0.005)

        # İlk satırı 2. ile eşitle (kayma problemi)
        if len(roll_list)>1:
            roll_list[0] = roll_list[1]
            pitch_list[0] = pitch_list[1]
            yaw_list[0] = yaw_list[1]
            yaw_corrected[0] = yaw_corrected[1]

        # State güncelle
        trial_states[state_key]["yaw_smoothed"] = new_smoothed

        # Sonuçları df_trial’e ekle
        df_trial = df_trial.copy()
        df_trial['madgwick_roll'] = roll_list
        df_trial['madgwick_pitch'] = pitch_list
        df_trial['madgwick_yaw'] = yaw_list
        df_trial['madgwick_yaw_corrected'] = yaw_corrected

        processed_trials.append(df_trial)

    # Chunk içindeki tüm trial’ları birleştir
    chunk_processed = pd.concat(processed_trials).sort_index()

    # Veritabanına ekle
    chunk_processed.to_sql("MFD_Sensor_Data_Processed",
                           conn, if_exists='append', index=False)

conn.close()
print("✅ İşlem tamamlandı! MFD_Sensor_Data_Processed tablosuna yazıldı.")

#%%
# -*- coding: utf-8 -*-
"""
FAD Sensor Verilerini İşleme: Accelerometer ile Roll/Pitch, Yaw = 0
"""
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm

# -------------------------
# 1️⃣ Accelerometer ile Roll/Pitch Hesaplama
# -------------------------
def compute_roll_pitch_yaw(acc_x, acc_y, acc_z):
    roll = np.degrees(np.arctan2(acc_y, acc_z))
    pitch = np.degrees(np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2)))
    yaw = 0.0  # Gyro olmadığı için yaw = 0
    return roll, pitch, yaw

# -------------------------
# 2️⃣ DB bağlantısı
# -------------------------
input_db = "FAD.db"
chunksize = 100_000
conn = sqlite3.connect(input_db)

# İşlenmiş tabloyu oluştur
conn.execute("DROP TABLE IF EXISTS FAD_Data_Processed")
conn.commit()

query = "SELECT * FROM FAD_Data"

# -------------------------
# 3️⃣ Chunk bazlı okuma ve işleme
# -------------------------
for chunk in tqdm(pd.read_sql_query(query, conn, chunksize=chunksize), desc="Chunks işleniyor"):

    processed_trials = []

    # experimental_id + activity_id bazlı gruplama
    for (experimental_id, activity_id), df_trial in chunk.groupby(['experimental_id','activity_id']):

        # Accelerometer ile roll/pitch/yaw hesapla
        roll_list, pitch_list, yaw_list = [], [], []
        for ax, ay, az in zip(df_trial['acc_x'], df_trial['acc_y'], df_trial['acc_z']):
            roll, pitch, yaw = compute_roll_pitch_yaw(ax, ay, az)
            roll_list.append(roll)
            pitch_list.append(pitch)
            yaw_list.append(yaw)

        # Sonuçları df_trial’e ekle
        df_trial = df_trial.copy()
        df_trial['roll'] = roll_list
        df_trial['pitch'] = pitch_list
        df_trial['yaw'] = yaw_list  # hep 0

        processed_trials.append(df_trial)

    # Chunk içindeki tüm trial’ları birleştir
    chunk_processed = pd.concat(processed_trials).sort_index()

    # Veritabanına ekle
    chunk_processed.to_sql("FAD_Data_Processed", conn, if_exists='append', index=False)

conn.close()
print("✅ İşlem tamamlandı! FAD_Data_Processed tablosuna yazıldı.")

#%%
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
SFD Sensor Dataset İşleme: Hem raw hem converted, accelerometer ve gyro ile
"""
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm

# -------------------------
# 1️⃣ Madgwick Filter
# -------------------------
class MadgwickAHRS:
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)

    def update_imu(self, gx, gy, gz, ax, ay, az, dt):
        q = self.q.copy()
        norm = np.linalg.norm([ax, ay, az])
        if norm == 0:
            return
        ax, ay, az = ax/norm, ay/norm, az/norm
        f = np.array([
            2*(q[1]*q[3]-q[0]*q[2])-ax,
            2*(q[0]*q[1]+q[2]*q[3])-ay,
            2*(0.5-q[1]**2-q[2]**2)-az
        ])
        J = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        step = J.T @ f
        step /= np.linalg.norm(step)
        q_dot = 0.5 * self.quat_multiply(q, np.array([0,gx,gy,gz])) - self.beta*step
        self.q += q_dot*dt
        self.q /= np.linalg.norm(self.q)

    @staticmethod
    def quat_multiply(q,r):
        w1,x1,y1,z1 = q
        w2,x2,y2,z2 = r
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def get_euler(self):
        q = self.q
        roll = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]**2+q[2]**2))
        pitch = np.arcsin(2*(q[0]*q[2]-q[3]*q[1]))
        yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
        return roll,pitch,yaw

def update_yaw_drift(yaw_list, last_smoothed=0.0, alpha=0.005):
    smoothed = [last_smoothed]
    corrected = []
    for yaw in yaw_list:
        smoothed_value = alpha*yaw + (1-alpha)*smoothed[-1]
        smoothed.append(smoothed_value)
        corrected_value = yaw - smoothed_value + smoothed[0]
        corrected.append(corrected_value)
    return corrected, smoothed[-1]

# -------------------------
# 2️⃣ DB bağlantısı
# -------------------------
input_db = "SisFallData.db"
conn = sqlite3.connect(input_db)
chunksize = 100_000

# İşlenmiş tablolar
conn.execute("DROP TABLE IF EXISTS SFD_sensor_dataset_raw_processed")
conn.execute("DROP TABLE IF EXISTS SFD_sensor_dataset_converted_processed")
conn.commit()

tables_to_process = [
    ("SFD_sensor_dataset_raw", "_raw"),
    ("SFD_sensor_dataset_converted", "_converted")
]

for table_name, suffix in tables_to_process:
    query = f"SELECT * FROM {table_name}"
    trial_states = {}
    for chunk in tqdm(pd.read_sql_query(query, conn, chunksize=chunksize), desc=f"{table_name} işleniyor"):
        processed_trials = []
        for (subject_id, activity_code, trial_number), df_trial in chunk.groupby(
            ['subject_id','activity_code','trial_number']
        ):
            state_key = (subject_id, activity_code, trial_number)
            if state_key not in trial_states:
                trial_states[state_key] = {"madgwick": MadgwickAHRS(beta=0.1),
                                           "yaw_smoothed": 0.0}
            madgwick = trial_states[state_key]["madgwick"]
            last_smoothed = trial_states[state_key]["yaw_smoothed"]

            # sample_index → dt (ms → s)
            time_sec = df_trial['sample_index'].values*1e-3

            # Accelerometer seçimi
            acc_cols_candidates = [
                ['mma8451q_x','mma8451q_y','mma8451q_z'],
                ['mma8451q_x_g','mma8451q_y_g','mma8451q_z_g'],
                ['adxl345_x','adxl345_y','adxl345_z'],
                ['adxl345_x_g','adxl345_y_g','adxl345_z_g']
            ]
            for cols in acc_cols_candidates:
                if all(col in df_trial.columns for col in cols):
                    acc_x, acc_y, acc_z = df_trial[cols[0]].values, df_trial[cols[1]].values, df_trial[cols[2]].values
                    break
            else:
                raise ValueError(f"Trial {subject_id}-{activity_code}-{trial_number}: Accelerometer kolonları bulunamadı.")

            # Gyro seçimi
            if all(col in df_trial.columns for col in ['itg3200_x_dps','itg3200_y_dps','itg3200_z_dps']):
                gyro_x = df_trial['itg3200_x_dps'].values
                gyro_y = df_trial['itg3200_y_dps'].values
                gyro_z = df_trial['itg3200_z_dps'].values
            elif all(col in df_trial.columns for col in ['itg3200_x','itg3200_y','itg3200_z']):
                gyro_x = df_trial['itg3200_x'].values
                gyro_y = df_trial['itg3200_y'].values
                gyro_z = df_trial['itg3200_z'].values
            else:
                gyro_x = np.zeros_like(acc_x)
                gyro_y = np.zeros_like(acc_y)
                gyro_z = np.zeros_like(acc_z)

            # Madgwick filtre
            roll_list, pitch_list, yaw_list = [], [], []
            for i in range(len(time_sec)):
                dt = 0.0 if i==0 else time_sec[i]-time_sec[i-1]
                madgwick.update_imu(gyro_x[i], gyro_y[i], gyro_z[i],
                                    acc_x[i], acc_y[i], acc_z[i], dt)
                roll, pitch, yaw = madgwick.get_euler()
                roll_list.append(np.degrees(roll))
                pitch_list.append(np.degrees(pitch))
                yaw_list.append(np.degrees(yaw))

            # Yaw drift düzeltme
            yaw_corrected, new_smoothed = update_yaw_drift(yaw_list, last_smoothed, alpha=0.005)

            if len(roll_list)>1:
                roll_list[0]=roll_list[1]
                pitch_list[0]=pitch_list[1]
                yaw_list[0]=yaw_list[1]
                yaw_corrected[0]=yaw_corrected[1]

            trial_states[state_key]["yaw_smoothed"] = new_smoothed

            df_trial = df_trial.copy()
            df_trial['madgwick_roll'] = roll_list
            df_trial['madgwick_pitch'] = pitch_list
            df_trial['madgwick_yaw'] = yaw_list
            df_trial['madgwick_yaw_corrected'] = yaw_corrected

            processed_trials.append(df_trial)

        chunk_processed = pd.concat(processed_trials).sort_index()
        chunk_processed.to_sql(f"{table_name}_processed", conn, if_exists='append', index=False)

conn.close()
print("✅ İşlem tamamlandı! Hem raw hem converted tablolar işlendi.")

#%%
# -*- coding: utf-8 -*-
"""
PLD_Data İşleme: Madgwick (acc+gyro+mag) + Barometer saklama
"""
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm

# -------------------------
# 1️⃣ Madgwick Filter (full AHRS: acc + gyro + mag)
# -------------------------
class MadgwickAHRS:
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def update(self, gx, gy, gz, ax, ay, az, mx, my, mz, dt):
        q = self.q.copy()

        # Normalize accelerometer
        norm = np.linalg.norm([ax, ay, az])
        if norm == 0:
            return
        ax, ay, az = ax / norm, ay / norm, az / norm

        # Normalize magnetometer
        norm = np.linalg.norm([mx, my, mz])
        if norm == 0:
            return
        mx, my, mz = mx / norm, my / norm, mz / norm

        # Reference direction of Earth's magnetic field
        q1, q2, q3, q4 = q
        hx = 2*mx*(0.5 - q3**2 - q4**2) + 2*my*(q2*q3 - q1*q4) + 2*mz*(q2*q4 + q1*q3)
        hy = 2*mx*(q2*q3 + q1*q4) + 2*my*(0.5 - q2**2 - q4**2) + 2*mz*(q3*q4 - q1*q2)
        bx = np.sqrt(hx*hx + hy*hy)
        bz = 2*mx*(q2*q4 - q1*q3) + 2*my*(q3*q4 + q1*q2) + 2*mz*(0.5 - q2**2 - q3**2)

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q2*q4 - q1*q3) - ax,
            2*(q1*q2 + q3*q4) - ay,
            2*(0.5 - q2**2 - q3**2) - az,
            2*bx*(0.5 - q3**2 - q4**2) + 2*bz*(q2*q4 - q1*q3) - mx,
            2*bx*(q2*q3 - q1*q4) + 2*bz*(q1*q2 + q3*q4) - my,
            2*bx*(q1*q3 + q2*q4) + 2*bz*(0.5 - q2**2 - q3**2) - mz
        ])

        J = np.array([
            [-2*q3, 2*q4, -2*q1, 2*q2],
            [2*q2, 2*q1, 2*q4, 2*q3],
            [0, -4*q2, -4*q3, 0],
            [-2*bz*q3, 2*bz*q4, -4*bx*q3-2*bz*q1, -4*bx*q4+2*bz*q2],
            [-2*bx*q4+2*bz*q2, 2*bx*q3+2*bz*q1, 2*bx*q2+2*bz*q4, -2*bx*q1+2*bz*q3],
            [2*bx*q3, 2*bx*q4-4*bz*q2, 2*bx*q1-4*bz*q3, 2*bx*q2]
        ])

        step = J.T @ f
        step /= np.linalg.norm(step)

        q_dot = 0.5 * self.quat_multiply(q, np.array([0,gx,gy,gz])) - self.beta*step
        self.q += q_dot*dt
        self.q /= np.linalg.norm(self.q)

    @staticmethod
    def quat_multiply(q, r):
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = r
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def get_euler(self):
        q = self.q
        roll = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]**2+q[2]**2))
        pitch = np.arcsin(2*(q[0]*q[2]-q[3]*q[1]))
        yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
        return roll, pitch, yaw

# -------------------------
# 2️⃣ DB bağlantısı
# -------------------------
import os
os.chdir(r"C:\Users\kbybt\Desktop\EVREN\MatMüh_İTÜ_ARŞ_GÖR\YL_TEZ\DataBases")
input_db = "FallAllD.db"   # senin dosya adı
conn = sqlite3.connect(input_db)
chunksize = 100_000

# Yeni tablo oluştur
conn.execute("DROP TABLE IF EXISTS PLD_Data_Processed")
conn.commit()

query = "SELECT * FROM PLD_Data"
trial_states = {}

for chunk in tqdm(pd.read_sql_query(query, conn, chunksize=chunksize), desc="PLD_Data işleniyor"):
    processed_trials = []

    for (exp_id, act_id), df_trial in chunk.groupby(['experimental_id','activity_id']):
        state_key = (exp_id, act_id)
        if state_key not in trial_states:
            trial_states[state_key] = {"madgwick": MadgwickAHRS(beta=0.1)}

        madgwick = trial_states[state_key]["madgwick"]

        # timestamp → dt
        time_sec = df_trial['timestamp'].values  # zaten float
        acc_x, acc_y, acc_z = df_trial['acc_x'].values, df_trial['acc_y'].values, df_trial['acc_z'].values
        gyr_x, gyr_y, gyr_z = df_trial['gyr_x'].values, df_trial['gyr_y'].values, df_trial['gyr_z'].values
        mag_x, mag_y, mag_z = df_trial['mag_x'].values, df_trial['mag_y'].values, df_trial['mag_z'].values

        roll_list, pitch_list, yaw_list = [], [], []
        for i in range(len(time_sec)):
            dt = 0.0 if i == 0 else time_sec[i] - time_sec[i-1]
            madgwick.update(gyr_x[i], gyr_y[i], gyr_z[i],
                            acc_x[i], acc_y[i], acc_z[i],
                            mag_x[i], mag_y[i], mag_z[i], dt)
            roll, pitch, yaw = madgwick.get_euler()
            roll_list.append(np.degrees(roll))
            pitch_list.append(np.degrees(pitch))
            yaw_list.append(np.degrees(yaw))

        if len(roll_list) > 1:
            roll_list[0] = roll_list[1]
            pitch_list[0] = pitch_list[1]
            yaw_list[0] = yaw_list[1]

        df_trial = df_trial.copy()
        df_trial['madgwick_roll'] = roll_list
        df_trial['madgwick_pitch'] = pitch_list
        df_trial['madgwick_yaw'] = yaw_list

        processed_trials.append(df_trial)

    chunk_processed = pd.concat(processed_trials).sort_index()
    chunk_processed.to_sql("PLD_Data_Processed", conn, if_exists='append', index=False)

conn.close()
print("✅ İşlem tamamlandı! PLD_Data_Processed tablosu oluşturuldu.")

#%%
# -*- coding: utf-8 -*-
"""
PLSD_Data İşleme + Madgwick + Yaw Drift Düzeltme
"""
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm

# -------------------------
# 1️⃣ Madgwick Filter Sınıfı
# -------------------------
class MadgwickAHRS:
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def update_imu(self, gx, gy, gz, ax, ay, az, dt):
        q = self.q.copy()
        norm = np.linalg.norm([ax, ay, az])
        if norm == 0:
            return
        ax, ay, az = ax / norm, ay / norm, az / norm

        f = np.array([
            2*(q[1]*q[3]-q[0]*q[2]) - ax,
            2*(q[0]*q[1]+q[2]*q[3]) - ay,
            2*(0.5-q[1]**2-q[2]**2) - az
        ])
        J = np.array([
            [-2*q[2],  2*q[3], -2*q[0],  2*q[1]],
            [ 2*q[1],  2*q[0],  2*q[3],  2*q[2]],
            [ 0,      -4*q[1], -4*q[2],  0     ]
        ])
        step = J.T @ f
        step /= np.linalg.norm(step)

        q_dot = 0.5 * self.quat_multiply(q, np.array([0, gx, gy, gz])) - self.beta * step
        self.q += q_dot * dt
        self.q /= np.linalg.norm(self.q)

    @staticmethod
    def quat_multiply(q, r):
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = r
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def get_euler(self):
        q = self.q
        roll = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]),
                          1-2*(q[1]**2+q[2]**2))
        pitch = np.arcsin(2*(q[0]*q[2]-q[3]*q[1]))
        yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),
                         1-2*(q[2]**2+q[3]**2))
        return roll, pitch, yaw

# -------------------------
# 2️⃣ Yaw drift düzeltme
# -------------------------
def update_yaw_drift(yaw_list, last_smoothed=0.0, alpha=0.005):
    smoothed = [last_smoothed]
    corrected = []
    for yaw in yaw_list:
        smoothed_value = alpha * yaw + (1-alpha)*smoothed[-1]
        smoothed.append(smoothed_value)
        corrected_value = yaw - smoothed_value + smoothed[0]
        corrected.append(corrected_value)
    return corrected, smoothed[-1]

# -------------------------
# 3️⃣ DB bağlantısı
# -------------------------
import os
os.chdir(r"C:\Users\kbybt\Desktop\EVREN\MatMüh_İTÜ_ARŞ_GÖR\YL_TEZ\DataBases")
input_db = "FallAllD_40Hz.db"   # kendi DB dosya yolun
conn = sqlite3.connect(input_db)
chunksize = 100_000

# Yeni tablo
conn.execute("DROP TABLE IF EXISTS PLSD_Data_Processed")
conn.commit()

query = "SELECT * FROM PLSD_Data"

# Trial state: her trial için Madgwick + yaw düzeltme
trial_states = {}

# -------------------------
# 4️⃣ Chunk bazlı işleme
# -------------------------
for chunk in tqdm(pd.read_sql_query(query, conn, chunksize=chunksize), desc="PLSD_Data işleniyor"):
    processed_trials = []

    for (exp_id, act_id), df_trial in chunk.groupby(['experimental_id','activity_id']):
        state_key = (exp_id, act_id)
        if state_key not in trial_states:
            trial_states[state_key] = {
                "madgwick": MadgwickAHRS(beta=0.1),
                "yaw_smoothed": 0.0
            }

        madgwick = trial_states[state_key]["madgwick"]
        last_smoothed = trial_states[state_key]["yaw_smoothed"]

        # Veriler
        time_sec = df_trial['timestamp'].values
        acc_x, acc_y, acc_z = df_trial['acc_x'].values, df_trial['acc_y'].values, df_trial['acc_z'].values
        gyr_x, gyr_y, gyr_z = df_trial['gyr_x'].values, df_trial['gyr_y'].values, df_trial['gyr_z'].values

        # Madgwick çıktıları
        roll_list, pitch_list, yaw_list = [], [], []
        for i in range(len(time_sec)):
            dt = 0.0 if i == 0 else time_sec[i] - time_sec[i-1]
            madgwick.update_imu(gyr_x[i], gyr_y[i], gyr_z[i],
                                acc_x[i], acc_y[i], acc_z[i], dt)
            roll, pitch, yaw = madgwick.get_euler()
            roll_list.append(np.degrees(roll))
            pitch_list.append(np.degrees(pitch))
            yaw_list.append(np.degrees(yaw))

        # Yaw drift düzeltme
        yaw_corrected, new_smoothed = update_yaw_drift(yaw_list, last_smoothed, alpha=0.005)

        # İlk satırı 2. ile eşitle (kayma önleme)
        if len(roll_list) > 1:
            roll_list[0] = roll_list[1]
            pitch_list[0] = pitch_list[1]
            yaw_list[0] = yaw_list[1]
            yaw_corrected[0] = yaw_corrected[1]

        # State güncelle
        trial_states[state_key]["yaw_smoothed"] = new_smoothed

        # Sonuçları ekle
        df_trial = df_trial.copy()
        df_trial['madgwick_roll'] = roll_list
        df_trial['madgwick_pitch'] = pitch_list
        df_trial['madgwick_yaw'] = yaw_list
        df_trial['madgwick_yaw_corrected'] = yaw_corrected

        processed_trials.append(df_trial)

    chunk_processed = pd.concat(processed_trials).sort_index()
    chunk_processed.to_sql("PLSD_Data_Processed", conn, if_exists='append', index=False)

conn.close()
print("✅ İşlem tamamlandı! PLSD_Data_Processed tablosuna yazıldı.")
