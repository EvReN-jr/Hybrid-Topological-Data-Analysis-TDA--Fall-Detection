
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

import os
os.chdir(r"C:\Users\kbybt\Desktop\EVREN\MatMüh_İTÜ_ARŞ_GÖR\YL_TEZ\DataBases")

import sqlite3

# -------------------------
# 1️⃣ DB’den veriyi oku
# -------------------------
input_db = "MobiFallData.db"
trial_id = 1  # Görselleştirmek istediğin trial
conn = sqlite3.connect(input_db)
df = pd.read_sql_query(
    f"SELECT * FROM MFD_Sensor_Data_Processed WHERE trial_id={trial_id} ORDER BY timestamp_ns_gyro",
    conn
)
conn.close()

# Euler açılarını radiana çevir
roll = np.radians(df['madgwick_roll'].values)
pitch = np.radians(df['madgwick_pitch'].values)
yaw = np.radians(df['madgwick_yaw_corrected'].values)
N = len(roll)

# -------------------------
# 2️⃣ Telefon modelini tanımla (cube)
# -------------------------
def create_cube_vertices(center=[0,0,0], size=1.0):
    x, y, z = center
    d = size / 2
    corners = np.array([
        [x-d, y-d, z-d],
        [x+d, y-d, z-d],
        [x+d, y+d, z-d],
        [x-d, y+d, z-d],
        [x-d, y-d, z+d],
        [x+d, y-d, z+d],
        [x+d, y+d, z+d],
        [x-d, y+d, z+d],
    ])
    faces = [
        [corners[j] for j in [0,1,2,3]],
        [corners[j] for j in [4,5,6,7]],
        [corners[j] for j in [0,1,5,4]],
        [corners[j] for j in [2,3,7,6]],
        [corners[j] for j in [1,2,6,5]],
        [corners[j] for j in [4,7,3,0]]
    ]
    return corners, faces

# -------------------------
# 3️⃣ Rotasyon matrisi
# -------------------------
def rotation_matrix(roll, pitch, yaw):
    Rx = np.array([[1,0,0],
                   [0,np.cos(roll), -np.sin(roll)],
                   [0,np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],
                   [0,1,0],
                   [-np.sin(pitch),0,np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw),0],
                   [np.sin(yaw), np.cos(yaw),0],
                   [0,0,1]])
    return Rz @ Ry @ Rx

# -------------------------
# 4️⃣ Animasyon setup
# -------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)
ax.set_box_aspect([1,1,1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=30, azim=45)

# Cube
vertices, faces = create_cube_vertices(size=1.0)
poly = Poly3DCollection(faces, facecolors='cyan', edgecolors='k', linewidths=1, alpha=0.3)
ax.add_collection3d(poly)

# Köşe noktalarını renklendir
corner_colors = ['red','green','blue','yellow','magenta','cyan','orange','purple']
corner_scatters = []
for i, corner in enumerate(vertices):
    sc = ax.scatter(*corner, color=corner_colors[i], s=50)
    corner_scatters.append(sc)

# -------------------------
# 5️⃣ Güncelleme fonksiyonu
# -------------------------
def update(frame):
    r, p, y = roll[frame], pitch[frame], yaw[frame]
    R = rotation_matrix(r, p, y)
    
    # Cube yüzlerini güncelle
    new_vertices = np.array([R @ v for v in vertices])
    new_faces = [
        [new_vertices[j] for j in [0,1,2,3]],
        [new_vertices[j] for j in [4,5,6,7]],
        [new_vertices[j] for j in [0,1,5,4]],
        [new_vertices[j] for j in [2,3,7,6]],
        [new_vertices[j] for j in [1,2,6,5]],
        [new_vertices[j] for j in [4,7,3,0]]
    ]
    poly.set_verts(new_faces)
    
    # Köşeleri güncelle
    for i, sc in enumerate(corner_scatters):
        sc._offsets3d = (new_vertices[i,0:1], new_vertices[i,1:2], new_vertices[i,2:3])
    
    return [poly] + corner_scatters

# -------------------------
# 6️⃣ Animasyon başlat
# -------------------------# Animasyon oluştur
ani = FuncAnimation(fig, update, frames=N, interval=50, blit=False)

# GIF olarak kaydet
gif_filename = f"trial{trial_id}_animation_corners.gif"
ani.save(gif_filename, writer=PillowWriter(fps=20))
print(f"✅ GIF kaydedildi: {gif_filename}")

# Animasyonu ekranda göster
plt.show()

# ani objesini globalde tutmak için (Spyder/IDE içinde)
global anim
anim = ani
