# 💻 Laptop Price Predictor

Prediksi harga laptop berbasis Machine Learning (LightGBM), dilatih dari dataset 1.2 juta baris spesifikasi laptop global.

## Performa Model
- **R²**: 0.9963
- **MAE**: ~$63
- **RMSE**: ~$73

## Fitur yang digunakan
- Kategorikal: Country, Laptop_Brand, Laptop_Model, CPU_Brand, GPU_Brand, GPU_Model, Usage_Type
- Numerik: RAM_GB, Storage_GB, Cores, Threads, Base_Clock, Boost_Clock, TDP, CPU_Performance, GPU_Performance, Total_Performance

## Struktur Project
```
├── app.py                    # Streamlit app
├── train.py                  # Script training model
├── laptop_price_model.pkl    # Model terlatih (LightGBM pipeline)
├── meta.json                 # Metadata untuk dropdown UI
├── metrics.json              # Metrik evaluasi model
├── requirements.txt
└── runtime.txt                # Pin Python 3.11 utk Streamlit Cloud
```

## Menjalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Push repo ini ke GitHub.
2. Buka https://share.streamlit.io, connect ke repo ini.
3. Set main file: `app.py`.
4. Deploy.

## Retrain model
```bash
python train.py
```
Pastikan `global_laptop_dataset_1_2M.csv` tersedia di path yang sesuai (ubah path di `train.py` jika perlu).
