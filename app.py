import streamlit as st
import pandas as pd
import joblib
import json

# Konfigurasi Halaman (Lebih baik pakai wide/centered sesuai selera, di sini kita percantik centered)
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_model():
    return joblib.load("laptop_price_model.pkl")

@st.cache_data
def load_meta():
    with open("meta.json") as f:
        return json.load(f)

@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        return json.load(f)

model = load_model()
meta = load_meta()
metrics = load_metrics()

# --- HEADER SECTION ---
st.title("💻 AI Laptop Price Predictor")
st.markdown("Dapatkan estimasi harga pasar laptop idamanmu secara instan. Model AI ini telah dilatih menggunakan **1,2 Juta data laptop** dari seluruh dunia! 🌍")
st.divider()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3173/3173239.png", width=100) # Opsional: Tambah logo ilustrasi
    st.header("📊 Akurasi Model AI")
    
    # Menampilkan metrik dengan style card
    st.metric("Tingkat Akurasi (R²)", f"{metrics['r2'] * 100:.1f}%")
    st.metric("Rata-rata Margin Error (MAE)", f"± ${metrics['mae']:.2f}")
    
    st.divider()
    st.caption("⚙️ **Informasi Dataset:**")
    st.caption(f"📚 Data Latih: {metrics['n_train']:,} baris")
    st.caption(f"🧪 Data Uji: {metrics['n_test']:,} baris")

# --- MAIN INPUT SECTION ---

# 1. Kelompok Informasi Dasar (Gunakan container dengan border agar terlihat seperti 'Card')
with st.container(border=True):
    st.subheader("🏷️ Identitas & Penggunaan")
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Merek Laptop", meta['uniques']['Laptop_Brand'])
        model_name = st.selectbox("Seri/Model", meta['uniques']['Laptop_Model'])
    with col2:
        usage_type = st.selectbox("Tujuan Penggunaan", meta['uniques']['Usage_Type'])
        country = st.selectbox("Negara Pasar", meta['uniques']['Country'])

# 2. Kelompok Spesifikasi Utama
with st.container(border=True):
    st.subheader("⚡ Spesifikasi Inti")
    col3, col4 = st.columns(2)
    with col3:
        cpu_brand = st.selectbox("Merek CPU", meta['uniques']['CPU_Brand'])
        gpu_brand = st.selectbox("Merek GPU", meta['uniques']['GPU_Brand'])
    with col4:
        ram = st.select_slider("Kapasitas RAM (GB)", options=[4, 8, 16, 32, 64], value=16)
        storage = st.select_slider("Kapasitas Storage (GB)", options=[128, 256, 512, 1024, 2048], value=512)

# 3. Kelompok Spesifikasi Teknis (Disembunyikan dalam Expander agar UI tidak sumpek)
with st.expander("🛠️ Spesifikasi Teknis Lanjutan (Klik untuk membuka)", expanded=False):
    st.info("Ubah bagian ini jika kamu mengetahui detail teknis prosesor dan kartu grafis laptopmu.")
    gpu_model = st.selectbox("Tipe Spesifik GPU", meta['uniques']['GPU_Model'])
    
    col5, col6 = st.columns(2)
    with col5:
        cores = st.slider("Jumlah Core CPU", int(meta['ranges']['Cores']['min']), int(meta['ranges']['Cores']['max']), int(meta['ranges']['Cores']['median']))
        base_clock = st.slider("Base Clock (GHz)", float(meta['ranges']['Base_Clock']['min']), float(meta['ranges']['Base_Clock']['max']), float(meta['ranges']['Base_Clock']['median']))
        tdp = st.slider("TDP (Watt)", int(meta['ranges']['TDP']['min']), int(meta['ranges']['TDP']['max']), int(meta['ranges']['TDP']['median']), help="Thermal Design Power (Konsumsi daya)")
    with col6:
        threads = st.slider("Jumlah Thread CPU", int(meta['ranges']['Threads']['min']), int(meta['ranges']['Threads']['max']), int(meta['ranges']['Threads']['median']))
        boost_clock = st.slider("Boost Clock (GHz)", float(meta['ranges']['Boost_Clock']['min']), float(meta['ranges']['Boost_Clock']['max']), float(meta['ranges']['Boost_Clock']['median']))

# 4. Kelompok Skor Benchmark
with st.expander("📈 Skor Performa (Benchmark)", expanded=False):
    col7, col8, col9 = st.columns(3)
    with col7:
        cpu_perf = st.number_input("Skor CPU", value=int(meta['ranges']['CPU_Performance']['median']), step=500)
    with col8:
        gpu_perf = st.number_input("Skor GPU", value=int(meta['ranges']['GPU_Performance']['median']), step=500)
    with col9:
        total_perf = st.number_input("Skor Total", value=cpu_perf + gpu_perf)

st.write("") # Memberi sedikit jarak (spacing)

# --- PREDICTION BUTTON & OUTPUT ---
if st.button("🔮 Hitung Estimasi Harga", type="primary", use_container_width=True):
    # Menyusun DataFrame
    input_df = pd.DataFrame([{
        'Country': country, 'Laptop_Brand': brand, 'Laptop_Model': model_name,
        'CPU_Brand': cpu_brand, 'GPU_Brand': gpu_brand, 'GPU_Model': gpu_model,
        'RAM_GB': ram, 'Storage_GB': storage, 'Cores': cores, 'Threads': threads,
        'Base_Clock': base_clock, 'Boost_Clock': boost_clock, 'TDP': tdp,
        'CPU_Performance': cpu_perf, 'GPU_Performance': gpu_perf,
        'Total_Performance': total_perf, 'Usage_Type': usage_type
    }])
    
    # Melakukan Prediksi
    with st.spinner("AI sedang menganalisis harga pasar..."):
        pred = model.predict(input_df)[0]
    
    # Menampilkan Hasil dengan UI yang memukau
    st.balloons() # Animasi balon saat berhasil
    st.success("Analisis Selesai!")
    
    # Menggunakan container untuk highlight hasil
    with st.container(border=True):
        st.metric(label="💰 Estimasi Harga Wajar", value=f"${pred:,.2f}", delta=f"Rekomendasi Terbaik", delta_color="normal")
        st.caption(f"*Perkiraan rentang harga di pasaran: **${pred - metrics['mae']:,.2f} - ${pred + metrics['mae']:,.2f}***")
