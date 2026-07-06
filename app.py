import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

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

st.title("💻 Laptop Price Predictor")
st.caption("Prediksi harga laptop (USD) berdasarkan spesifikasi, menggunakan model LightGBM yang dilatih dari 1.2 juta data laptop global.")

with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("R² Score", f"{metrics['r2']:.4f}")
    st.metric("MAE", f"${metrics['mae']:.2f}")
    st.metric("RMSE", f"${metrics['rmse']:.2f}")
    st.caption(f"Trained on {metrics['n_train']:,} rows, tested on {metrics['n_test']:,} rows.")

st.subheader("Spesifikasi Laptop")

col1, col2 = st.columns(2)

with col1:
    country = st.selectbox("Country", meta['uniques']['Country'])
    brand = st.selectbox("Laptop Brand", meta['uniques']['Laptop_Brand'])
    model_name = st.selectbox("Laptop Model", meta['uniques']['Laptop_Model'])
    cpu_brand = st.selectbox("CPU Brand", meta['uniques']['CPU_Brand'])
    gpu_brand = st.selectbox("GPU Brand", meta['uniques']['GPU_Brand'])
    gpu_model = st.selectbox("GPU Model", meta['uniques']['GPU_Model'])
    usage_type = st.selectbox("Usage Type", meta['uniques']['Usage_Type'])

with col2:
    ram = st.select_slider("RAM (GB)", options=[4,8,16,32,64], value=16)
    storage = st.select_slider("Storage (GB)", options=[128,256,512,1024,2048], value=512)
    cores = st.slider("CPU Cores", int(meta['ranges']['Cores']['min']), int(meta['ranges']['Cores']['max']), int(meta['ranges']['Cores']['median']))
    threads = st.slider("CPU Threads", int(meta['ranges']['Threads']['min']), int(meta['ranges']['Threads']['max']), int(meta['ranges']['Threads']['median']))
    base_clock = st.slider("Base Clock (GHz)", float(meta['ranges']['Base_Clock']['min']), float(meta['ranges']['Base_Clock']['max']), float(meta['ranges']['Base_Clock']['median']))
    boost_clock = st.slider("Boost Clock (GHz)", float(meta['ranges']['Boost_Clock']['min']), float(meta['ranges']['Boost_Clock']['max']), float(meta['ranges']['Boost_Clock']['median']))
    tdp = st.slider("TDP (W)", int(meta['ranges']['TDP']['min']), int(meta['ranges']['TDP']['max']), int(meta['ranges']['TDP']['median']))

st.subheader("Performance Scores")
col3, col4, col5 = st.columns(3)
with col3:
    cpu_perf = st.number_input("CPU Performance", value=int(meta['ranges']['CPU_Performance']['median']))
with col4:
    gpu_perf = st.number_input("GPU Performance", value=int(meta['ranges']['GPU_Performance']['median']))
with col5:
    total_perf = st.number_input("Total Performance", value=cpu_perf + gpu_perf)

if st.button("🔮 Prediksi Harga", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        'Country': country, 'Laptop_Brand': brand, 'Laptop_Model': model_name,
        'CPU_Brand': cpu_brand, 'GPU_Brand': gpu_brand, 'GPU_Model': gpu_model,
        'RAM_GB': ram, 'Storage_GB': storage, 'Cores': cores, 'Threads': threads,
        'Base_Clock': base_clock, 'Boost_Clock': boost_clock, 'TDP': tdp,
        'CPU_Performance': cpu_perf, 'GPU_Performance': gpu_perf,
        'Total_Performance': total_perf, 'Usage_Type': usage_type
    }])
    pred = model.predict(input_df)[0]
    st.success(f"### Estimasi Harga: **${pred:,.2f}**")
    st.caption(f"Margin error rata-rata model: ± ${metrics['mae']:.2f}")

st.divider()
st.caption("Dibuat dengan LightGBM + Streamlit • Dataset: Global Laptop Dataset (1.2M rows)")
