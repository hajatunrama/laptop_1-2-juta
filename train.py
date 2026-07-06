import pandas as pd, numpy as np, joblib, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import lightgbm as lgb

df = pd.read_csv('/mnt/user-data/uploads/global_laptop_dataset_1_2M.csv')

cat_cols = ['Country','Laptop_Brand','Laptop_Model','CPU_Brand','GPU_Brand','GPU_Model','Usage_Type']
num_cols = ['RAM_GB','Storage_GB','Cores','Threads','Base_Clock','Boost_Clock','TDP','CPU_Performance','GPU_Performance','Total_Performance']

X = df[cat_cols+num_cols]
y = df['Price_USD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

pre = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
], remainder='passthrough')

model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=64, random_state=42, n_jobs=-1)

pipe = Pipeline([('pre', pre), ('model', model)])
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.4f}")

joblib.dump(pipe, 'laptop_price_model.pkl')

metrics = {'mae': mae, 'rmse': rmse, 'r2': r2, 'n_train': len(X_train), 'n_test': len(X_test)}
with open('metrics.json','w') as f:
    json.dump(metrics, f, indent=2)

# Save unique values for dropdowns in streamlit
uniques = {c: sorted(df[c].unique().tolist()) for c in cat_cols}
ranges = {c: {'min': float(df[c].min()), 'max': float(df[c].max()), 'median': float(df[c].median())} for c in num_cols}
with open('meta.json','w') as f:
    json.dump({'uniques': uniques, 'ranges': ranges}, f, indent=2)

print("Done.")
