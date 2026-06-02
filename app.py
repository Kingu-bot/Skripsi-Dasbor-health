import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# ==========================================
# 1. KONFIGURASI API & FIREBASE
# ==========================================
FIREBASE_URL = "https://knn-kardio-project-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_SECRET = "hioYtRxCiFwow18j8HNNsojBqJcLRuiflrt9mvHV"

TELEGRAM_TOKEN = "8775738096:AAEH1D5qooF09FQPORt9bwiIkScHIP4YgBM"  # Masukkan Token dari BotFather
CHAT_ID = "-5237787277"       # Masukkan Chat ID Grup (lengkap dengan tanda minus)

# ==========================================
# 2. MELATIH MODEL KNN (Dokter Virtual)
# ==========================================
@st.cache_resource
def latih_model_knn():
    # Ini adalah Data Latih (Dataset) standar JNC 8 / WHO untuk Lansia
    data = {
        'SBP':  [110, 120, 125, 128,  # Normal
                 135, 138, 132, 139,  # Hipertensi Ringan
                 150, 160, 145, 170], # Hipertensi Parah
        'DBP':  [70,  75,  80,  78,
                 85,  88,  82,  89,
                 95,  100, 92,  110],
        'HR':   [75,  80,  72,  78,
                 85,  90,  88,  95,
                 110, 115, 105, 120],
        'SpO2': [98,  99,  97,  98,
                 97,  96,  98,  97,
                 93,  92,  94,  90],
        'Suhu': [36.5, 36.6, 36.7, 36.5,
                 36.8, 36.9, 36.7, 37.0,
                 37.5, 37.2, 37.1, 38.0],
        'Label': ['Normal', 'Normal', 'Normal', 'Normal',
                  'Hipertensi Ringan', 'Hipertensi Ringan', 'Hipertensi Ringan', 'Hipertensi Ringan',
                  'Hipertensi Parah', 'Hipertensi Parah', 'Hipertensi Parah', 'Hipertensi Parah']
    }
    df = pd.DataFrame(data)
    X = df[['SBP', 'DBP', 'HR', 'SpO2', 'Suhu']]
    y = df['Label']
    
    # K=3 berarti melihat 3 tetangga data terdekat
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

knn_model = latih_model_knn()

# ==========================================
# 3. FUNGSI KIRIM TELEGRAM (LENGKAP)
# ==========================================
def kirim_notif_telegram(status, sbp, dbp, hr, spo2, suhu, is_manual=False):
    # Jika normal dan sistem berjalan otomatis, jangan kirim pesan agar tidak spam.
    # Tapi JIKA ditekan manual lewat tombol (is_manual=True), tetap kirimkan.
    if status == "Normal" and not is_manual:
        return 

    if status == "Normal":
        simbol = "🟢"
        header = "LAPORAN HARIAN KESEHATAN LANSIA"
    elif status == "Hipertensi Ringan":
        simbol = "🟡"
        header = "PERINGATAN: KONDISI WASPADA"
    else:
        simbol = "🔴"
        header = "BAHAYA: KONDISI DARURAT!"

    pesan = f"{simbol} *{header}* {simbol}\n\n"
    pesan += f"Status Diagnosa KNN: *{status}*\n\n"
    pesan += f"Tensi (SBP/DBP): {sbp}/{dbp} mmHg\n"
    pesan += f"Heart Rate: {hr} bpm\n"
    pesan += f"SpO2 (Oksigen): {spo2}%\n"
    pesan += f"Suhu Tubuh: {float(suhu):.1f} °C\n\n"
    
    # Catatan Kustom Berdasarkan Status
    if status == "Normal":
        pesan += "Catatan: Kondisi Kakek/Nenek stabil malam ini. Lanjutkan istirahat."
    elif status == "Hipertensi Ringan":
        pesan += "Catatan: Tensi sedikit tinggi. Pastikan obat malam sudah diminum."
    else:
        pesan += "Catatan: Mohon segera cek kondisi fisik Kakek/Nenek sekarang juga!"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": pesan, "parse_mode": "Markdown"}
    
    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"Gagal mengirim Telegram: {e}")

# ==========================================
# 4. FUNGSI BACA DATABASE ESP32
# ==========================================
def ambil_data_terbaru():
    url = f"{FIREBASE_URL}/Penelitian.json?auth={FIREBASE_SECRET}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            # Ambil sesi/pasien urutan paling akhir
            id_terakhir = list(data.keys())[-1]
            history = data[id_terakhir].get("History", {})
            if history:
                # Ambil push log paling terakhir dari sesi tersebut
                id_log = list(history.keys())[-1]
                data_log = history[id_log]
                return id_terakhir, data_log, history
    return None, None, None

# ==========================================
# 5. ANTARMUKA WEB (UI DASHBOARD)
# ==========================================
st.set_page_config(page_title="Monitor Kardio", layout="centered")

st.title("Dasbor Pemantauan Lansia")
st.write("Sistem Analisis Cerdas *K-Nearest Neighbors* (KNN)")
st.write("---")

pasien_id, log_terbaru, riwayat_lengkap = ambil_data_terbaru()

if log_terbaru:
    sbp = log_terbaru.get("SBP", 0)
    dbp = log_terbaru.get("DBP", 0)
    hr = log_terbaru.get("HR", 0)
    spo2 = log_terbaru.get("SpO2", 0)
    suhu = log_terbaru.get("Suhu", 0.0)
    waktu = log_terbaru.get("Waktu", "-")

    # Masukkan data ke otak KNN untuk ditebak statusnya
    input_sensor = np.array([[sbp, dbp, hr, spo2, suhu]])
    prediksi_status = knn_model.predict(input_sensor)[0]

    # Beri warna sesuai status
    if prediksi_status == "Normal":
        st.success(f"Status Terkini: {prediksi_status} 🟢")
    elif prediksi_status == "Hipertensi Ringan":
        st.warning(f"Status Terkini: {prediksi_status} 🟡")
    else:
        st.error(f"Status Terkini: {prediksi_status} 🔴")

    st.caption(f"Update terakhir: {waktu} (Sesi: {pasien_id})")

# Tampilkan Kotak Angka Besar (Dibagi jadi 4 kolom agar Suhu masuk)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tensi (SBP/DBP)", f"{sbp}/{dbp}", "mmHg")
    col2.metric("Heart Rate", f"{hr}", "bpm")
    col3.metric("SpO2 (Oksigen)", f"{spo2}", "%")
    col4.metric("Suhu Tubuh", f"{float(suhu):.1f}", "°C")
    
    # Tombol Eksekusi Manual
    if st.button("Kirim Laporan ke Grup Telegram Keluarga"):
        # Kita panggil fungsinya dengan tambahan parameter suhu dan status is_manual=True
        kirim_notif_telegram(prediksi_status, sbp, dbp, hr, spo2, suhu, is_manual=True)
        st.success("Laporan harian lengkap berhasil dikirim ke Telegram!")
# Tampilkan Grafik Tren Sesi Ini
    if riwayat_lengkap:
        st.subheader("📈 Grafik Sesi Pemantauan Malam Ini")
        df_hist = pd.DataFrame.from_dict(riwayat_lengkap, orient='index')
        
        # TRIK AMAN: Hanya ambil kolom yang benar-benar ada di database saat ini
        kolom_target = ['SBP', 'DBP', 'HR']
        kolom_tersedia = [kolom for kolom in kolom_target if kolom in df_hist.columns]
        
        df_hist = df_hist.tail(30) # Tampilkan 30 data terakhir
        
        if len(kolom_tersedia) > 0:
            # Pastikan datanya diubah ke format angka (numeric) agar grafik tidak error
            for col in kolom_tersedia:
                df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')
                
            st.line_chart(df_hist[kolom_tersedia])
        else:
            st.info("Menunggu data Tensi dan HR terekam penuh untuk membuat grafik...")
