import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import shap

# =====================================================================
# 0. CỐ ĐỊNH HẠT GIỐNG NGẪU NHIÊN (REPRODUCIBILITY)
# =====================================================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs('figures', exist_ok=True)

# Danh sách các ngày nghỉ lễ chính thức tại Việt Nam (2024 - 2026)
VN_HOLIDAYS = pd.to_datetime([
    '2024-01-01', '2024-02-08', '2024-02-09', '2024-02-10', '2024-02-11',
    '2024-02-12', '2024-02-13', '2024-02-14', '2024-04-18', '2024-04-29',
    '2024-04-30', '2024-05-01', '2024-09-02', '2024-09-03',
    '2025-01-01', '2025-01-25', '2025-01-26', '2025-01-27', '2025-01-28',
    '2025-01-29', '2025-01-30', '2025-01-31', '2025-02-01', '2025-02-02',
    '2025-04-07', '2025-04-30', '2025-05-01', '2025-05-02', '2025-09-01',
    '2025-09-02',
    '2026-01-01', '2026-02-13', '2026-02-14', '2026-02-15', '2026-02-16',
    '2026-02-17', '2026-02-18', '2026-02-19', '2026-02-20', '2026-04-26',
    '2026-04-27', '2026-04-30', '2026-05-01'
])

# =====================================================================
# BƯỚC 1: LÀM SẠCH DỮ LIỆU THÔ VÀ TẠO ĐẶC TRƯNG ĐA BIẾN (17 BIẾN + CENSUS)
# =====================================================================
print("BƯỚC 1: Đang nạp và xử lý dữ liệu thô từ raw_his_data.csv...")
raw_df = pd.read_csv('raw_his_data.csv', dtype={'Patient_ID_Hash': str}, low_memory=False)
raw_df.columns = raw_df.columns.str.strip()

# Chuẩn hóa thời gian vào/ra viện
raw_df['Admission_Time'] = pd.to_datetime(raw_df['Admission_Time'], format='mixed', dayfirst=True, errors='coerce')
raw_df['Discharge_Time'] = pd.to_datetime(raw_df['Discharge_Time'], format='mixed', dayfirst=True, errors='coerce')
raw_df['Age'] = raw_df['Admission_Time'].dt.year - raw_df['DoB_Year']

# 1.1. Trích xuất nhóm đặc trưng lâm sàng và nhân khẩu học theo ngày nhập viện
raw_df['Date'] = raw_df['Admission_Time'].dt.normalize()
raw_df['Is_Elderly'] = (raw_df['Age'] >= 60).astype(int)
raw_df['ICD_Main_Letter'] = raw_df['ICD_Code_Main'].astype(str).str[0].str.upper()
raw_df['Is_Respiratory'] = (raw_df['ICD_Main_Letter'] == 'J').astype(int)
raw_df['Is_Cardiovascular'] = (raw_df['ICD_Main_Letter'] == 'I').astype(int)
raw_df['Is_Infectious'] = (raw_df['ICD_Main_Letter'].isin(['A', 'B'])).astype(int)

daily_clinical_features = raw_df.groupby('Date').agg(
    Avg_ICD_Sub_Count=('ICD_Sub_Count', 'mean'),
    Total_Elderly_Admitted=('Is_Elderly', 'sum'),
    Total_Respiratory=('Is_Respiratory', 'sum'),
    Total_Cardiovascular=('Is_Cardiovascular', 'sum'),
    Total_Infectious=('Is_Infectious', 'sum')
).reset_index()

# 1.2. Lọc hồ sơ ảo (< 15 phút) và tổng hợp chuỗi thời gian công suất giường (Census)
raw_df['Duration_Hours'] = (raw_df['Discharge_Time'] - raw_df['Admission_Time']).dt.total_seconds() / 3600.0

# Lưu nhật ký hồ sơ ảo (< 15 phút) để đối chiếu vận hành
df_removed = raw_df[raw_df['Duration_Hours'] < 0.25].copy()
df_removed.to_csv('removed_cases_log.csv', index=False, encoding='utf-8-sig')

# Giữ lại các ca hợp lệ (>= 15 phút, chốt sổ xuất viện tại thời điểm trích xuất 28/08/2026)
df_clean = raw_df[(raw_df['Duration_Hours'] >= 0.25) & (raw_df['Discharge_Time'] <= '2026-08-28 23:59:59')].copy()

min_date = df_clean['Admission_Time'].min().normalize()
max_date = pd.to_datetime('2026-08-27')
date_range = pd.date_range(start=min_date, end=max_date, freq='D')

adm_norm = df_clean['Admission_Time'].dt.normalize()
dis_norm = df_clean['Discharge_Time'].dt.normalize()

daily_data = []
for current_date in date_range:
    next_date = current_date + pd.Timedelta(days=1)
    active_patients = df_clean[
        (df_clean['Admission_Time'] < next_date) &
        ((df_clean['Discharge_Time'] >= next_date) | (df_clean['Discharge_Time'].isna()))
    ]
    daily_data.append({
        'Date': current_date,
        'Census': int(len(active_patients)),
        'Admissions': int((adm_norm == current_date).sum()),
        'Discharges': int((dis_norm == current_date).sum()),
        'Avg_Age': float(active_patients['Age'].mean()),
        'Emergency_Count': int(active_patients['Admission_Source'].sum())
    })

df_timeseries = pd.DataFrame(daily_data)

# Cắt 30 ngày đầu tiên để loại bỏ hiệu ứng khởi động lạnh (Cold-start)
df_timeseries = df_timeseries.iloc[30:].reset_index(drop=True)

# Tạo đặc trưng chu kỳ lịch và độ trễ thời gian
df_timeseries['DayOfWeek'] = df_timeseries['Date'].dt.dayofweek
df_timeseries['Is_Weekend'] = (df_timeseries['DayOfWeek'] >= 5).astype(int)
df_timeseries['Month'] = df_timeseries['Date'].dt.month
df_timeseries['Lag_1'] = df_timeseries['Census'].shift(1)
df_timeseries['Lag_2'] = df_timeseries['Census'].shift(2)
df_timeseries['Lag_7'] = df_timeseries['Census'].shift(7)
df_timeseries['Rolling_Mean_7'] = df_timeseries['Census'].rolling(window=7).mean()
df_timeseries['Is_Holiday'] = df_timeseries['Date'].isin(VN_HOLIDAYS).astype(int)

# Loại bỏ các dòng NaN sinh ra do trượt độ trễ và xuất features.csv
features_df = df_timeseries.dropna().reset_index(drop=True)
features_df.to_csv('features.csv', index=False)
print(f"Đã tạo và lưu 'features.csv' thành công ({len(features_df)} ngày).")

# Hợp nhất với nhóm đặc trưng lâm sàng và khóa sổ tại 31/07/2026
df_final = pd.merge(features_df, daily_clinical_features, on='Date', how='left')
df_final = df_final.fillna(0).sort_values(by='Date').reset_index(drop=True)

cutoff_date = pd.to_datetime('2026-07-31')
df_final = df_final[df_final['Date'] <= cutoff_date].reset_index(drop=True)
print(f"Tổng số ngày đưa vào thực nghiệm (đến 31/07/2026): {len(df_final)} ngày.")

# =====================================================================
# BƯỚC 2: CHIA TẬP DỮ LIỆU (70/10/20) & CHUẨN HÓA CHỐNG RÒ RỈ
# =====================================================================
print("\nBƯỚC 2: Chia tập dữ liệu (Train 70% / Val 10% / Test 20%) & Chuẩn hóa...")
X_columns = df_final.drop(columns=['Census', 'Date']).columns.tolist()

n = len(df_final)
train_end = int(n * 0.70)
val_end   = int(n * 0.80)

train_df = df_final.iloc[:train_end]
val_df   = df_final.iloc[train_end:val_end]
test_df  = df_final.iloc[val_end:]

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

# Chỉ fit bộ chuẩn hóa trên tập Train
scaler_X.fit(train_df[X_columns])
scaler_Y.fit(train_df[['Census']])

X_train_scaled = scaler_X.transform(train_df[X_columns])
X_val_scaled   = scaler_X.transform(val_df[X_columns])
X_test_scaled  = scaler_X.transform(test_df[X_columns])

Y_train_scaled = scaler_Y.transform(train_df[['Census']])
Y_val_scaled   = scaler_Y.transform(val_df[['Census']])
Y_test_scaled  = scaler_Y.transform(test_df[['Census']])

def make_sequences(X, y, window=7):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

window = 7
X_train_lstm, Y_train_seq = make_sequences(X_train_scaled, Y_train_scaled, window=window)
X_val_lstm,   Y_val_seq   = make_sequences(X_val_scaled,   Y_val_scaled,   window=window)
X_test_lstm,  Y_test_seq  = make_sequences(X_test_scaled,  Y_test_scaled,  window=window)

dates_test = test_df['Date'].values[window:]

# =====================================================================
# BƯỚC 3: HUẤN LUYỆN MẠNG HỌC SÂU (LSTM) & XUẤT HÌNH 1, HÌNH 2
# =====================================================================
print("\nBƯỚC 3: Huấn luyện mạng học sâu LSTM...")
time_steps = window
features_count = X_train_lstm.shape[2]

model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(time_steps, features_count)),
    Dropout(0.2),
    LSTM(units=32, return_sequences=False),
    Dropout(0.2),
    Dense(units=16, activation='relu'),
    Dense(units=1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train_lstm, Y_train_seq,
    epochs=150,
    batch_size=16,
    validation_data=(X_val_lstm, Y_val_seq),
    callbacks=[early_stop],
    shuffle=False,
    verbose=1
)

predictions_scaled = model.predict(X_test_lstm, verbose=0)
predictions_real = scaler_Y.inverse_transform(predictions_scaled)
Y_test_real = scaler_Y.inverse_transform(Y_test_seq)

rmse = np.sqrt(mean_squared_error(Y_test_real, predictions_real))
mae = mean_absolute_error(Y_test_real, predictions_real)
print(f"--> KẾT QUẢ LSTM TRÊN TẬP TEST: MAE = {mae:.2f} giường | RMSE = {rmse:.2f} giường")

# --- Xuất Hình 1: Biểu đồ đường cong mất mát (Loss Curve) ---
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Sai số tập Train')
plt.plot(history.history['val_loss'], label='Sai số tập Validation')
plt.title('Quá trình huấn luyện LSTM')
plt.xlabel('Vòng lặp (Epochs)')
plt.ylabel('Sai số (MSE Scale)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/Hinh_1_LSTM_Loss_Curve.png', dpi=300)
plt.close()

# --- Xuất Hình 2: Dự báo vs Thực tế trên tập kiểm thử ---
plt.figure(figsize=(8, 6))
plt.plot(dates_test[-90:], Y_test_real[-90:], label='Thực tế', marker='o', linestyle='dashed')
plt.plot(dates_test[-90:], predictions_real[-90:], label='Dự báo (LSTM)', marker='x', alpha=0.8)
plt.title('Dự báo vs Thực tế')
plt.xlabel('Thời gian')
plt.ylabel('Số lượng giường bệnh')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
plt.xticks(rotation=45)
plt.gca().set_xlim(dates_test[-90], dates_test[-1])
plt.legend()
plt.tight_layout()
plt.savefig('figures/Hinh_2_LSTM_Forecast_vs_Actual.png', dpi=300)
plt.close()

# =====================================================================
# BƯỚC 4: ĐỐI SÁNH MÔ HÌNH CƠ SỞ ARIMA(7,1,1) & XUẤT HÌNH 3
# =====================================================================
print("\nBƯỚC 4: Huấn luyện mô hình cơ sở ARIMA(7,1,1)...")
Y_train_arima = train_df['Census'].values
Y_test_arima = test_df['Census'].values

arima_model = ARIMA(Y_train_arima, order=(7, 1, 1))
arima_result = arima_model.fit()
arima_preds_real = arima_result.forecast(steps=len(Y_test_arima))[window:]
Y_test_arima_eval = Y_test_arima[window:]

rmse_arima = np.sqrt(mean_squared_error(Y_test_arima_eval, arima_preds_real))
mae_arima = mean_absolute_error(Y_test_arima_eval, arima_preds_real)
print(f"--> KẾT QUẢ ARIMA TRÊN TẬP TEST: MAE = {mae_arima:.2f} giường | RMSE = {rmse_arima:.2f} giường")

# --- Xuất Hình 3: Biểu đồ so sánh sai số MAE và RMSE giữa ARIMA và LSTM ---
models_name = ['ARIMA', 'LSTM Đa biến']
mae_scores = [mae_arima, mae]
rmse_scores = [rmse_arima, rmse]

x = np.arange(len(models_name))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, mae_scores, width, label='MAE (Sai số tuyệt đối trung bình)', color='#5CB85C')
rects2 = ax.bar(x + width/2, rmse_scores, width, label='RMSE (Sai số toàn phương trung bình)', color='#D9534F')
ax.set_ylabel('Sai số (Số lượng giường bệnh)')
ax.set_title('So sánh Hiệu năng Dự báo giữa 2 Thuật toán')
ax.set_xticks(x)
ax.set_xticklabels(models_name, fontweight='bold')
ax.legend()
ax.bar_label(rects1, fmt='%.2f', padding=3)
ax.bar_label(rects2, fmt='%.2f', padding=3)
plt.tight_layout()
plt.savefig('figures/Hinh_3_Compare_ARIMA_LSTM.png', dpi=300)
plt.close()

# =====================================================================
# BƯỚC 5: MÔ HÌNH ĐẠI DIỆN XGBOOST (SURROGATE MODEL) & XUẤT HÌNH 4
# =====================================================================
print("\nBƯỚC 5: Huấn luyện mô hình đại diện toàn cục XGBoost (Chống rò rỉ dữ liệu)...")

# 1. Sinh nhãn giả (Pseudo-labels) từ mạng LSTM cho cả 3 tập Train, Validation và Test
Y_train_pseudo = model.predict(X_train_lstm, verbose=0).ravel()
Y_val_pseudo   = model.predict(X_val_lstm,   verbose=0).ravel()
Y_test_pseudo  = model.predict(X_test_lstm,  verbose=0).ravel()

# 2. Cắt bỏ 7 ngày đầu (window = 7) của dữ liệu 2D để đồng bộ kích thước với chuỗi LSTM
X_train_xgb = X_train_scaled[window:]
X_val_xgb   = X_val_scaled[window:]
X_test_xgb  = X_test_scaled[window:]

# 3. Khởi tạo XGBRegressor (Chuẩn XGBoost >= 2.0: đặt early_stopping_rounds trong constructor)
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=30,
    eval_metric='rmse',
    random_state=SEED
)

# 4. Huấn luyện và giám sát Dừng sớm độc lập trên tập Validation (X_val_xgb, Y_val_pseudo)
xgb_model.fit(
    X_train_xgb, Y_train_pseudo,
    eval_set=[(X_train_xgb, Y_train_pseudo), (X_val_xgb, Y_val_pseudo)],
    verbose=False
)

# 5. Đánh giá Độ trung thành (Fidelity R2) trên cả 3 tập Train, Validation, Test
xgb_pred_train = xgb_model.predict(X_train_xgb)
xgb_pred_val   = xgb_model.predict(X_val_xgb)
xgb_pred_test  = xgb_model.predict(X_test_xgb)

fidelity_train = r2_score(Y_train_pseudo, xgb_pred_train)
fidelity_val   = r2_score(Y_val_pseudo,   xgb_pred_val)
fidelity_test  = r2_score(Y_test_pseudo,  xgb_pred_test)

print("--- ĐÁNH GIÁ ĐỘ TRUNG THÀNH (FIDELITY R2) CỦA MÔ HÌNH ĐẠI DIỆN XGBOOST ---")
print(f"Số cây tối ưu (best_iteration): {xgb_model.best_iteration}")
print(f"1. Tập Huấn luyện (Train set)      : R2 = {fidelity_train:.4f}")
print(f"2. Tập Kiểm định (Validation set)  : R2 = {fidelity_val:.4f}")
print(f"3. Tập Kiểm thử (Test set - Độc lập): R2 = {fidelity_test:.4f}")

# --- Xuất Hình 4: Biểu đồ phân tán (Scatter Plot) đánh giá độ trung thành ---
plt.figure(figsize=(6.5, 6.5))
plt.scatter(Y_test_pseudo, xgb_predictions, color='#9B59B6', edgecolor='white', s=45, alpha=0.85)
min_val = min(Y_test_pseudo.min(), xgb_predictions.min())
max_val = max(Y_test_pseudo.max(), xgb_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Đường chuẩn y=x')
plt.title(f'Độ trung thành của XGBoost so với LSTM ($R^2 = {fidelity_score:.4f}$)')
plt.xlabel('Dự báo của mạng học sâu LSTM (Pseudo-labels)')
plt.ylabel('Dự báo của XGBoost (Surrogate Model)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('figures/Hinh_4_XGBoost_Fidelity_Scatter.png', dpi=300)
plt.close()

# =====================================================================
# BƯỚC 6: GIẢI THÍCH MÔ HÌNH VỚI TREESHAP & XUẤT HÌNH 5, 6, 7, 8, 9
# =====================================================================
print("\nBƯỚC 6: Trích xuất giá trị SHAP và vẽ biểu đồ giải thích (Hình 5 -> Hình 9)...")
feature_names = X_columns
X_test_df = pd.DataFrame(X_test_xgb, columns=feature_names)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_df)

# --- Xuất Hình 5: SHAP Feature Importance (Bar Plot) ---
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
ax = plt.gca()
for patch in ax.patches:
    w = patch.get_width()
    if w > 0:
        ax.text(w + 0.001, patch.get_y() + patch.get_height() / 2, f'{w:.4f}',
                ha='left', va='center', fontsize=10, color='black')
plt.title("Tầm quan trọng của các yếu tố ảnh hưởng đến nhu cầu giường bệnh", pad=20)
plt.margins(y=0.02)
plt.xlim(right=ax.get_xlim()[1] * 1.12)
plt.tight_layout()
plt.savefig('figures/Hinh_5_SHAP_Feature_Importance.png', dpi=300)
plt.close()

# --- Xuất Hình 6: SHAP Beeswarm Plot ---
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_df, show=False)
plt.title("Biểu đồ bầy đàn SHAP: Phân tích chiều tác động (Tăng/Giảm)")
plt.tight_layout()
plt.savefig('figures/Hinh_6_SHAP_Beeswarm.png', dpi=300)
plt.close()

# --- Xuất Hình 7, 8, 9: SHAP Waterfall cho 3 ngày điển hình trong tuần ---
dates_test_dt = pd.to_datetime(dates_test)
waterfall_configs = [
    ('2026-07-12', 'Chủ nhật', 'Hinh_7_SHAP_Waterfall_2026_07_12.png'),
    ('2026-07-13', 'Thứ Hai',  'Hinh_8_SHAP_Waterfall_2026_07_13.png'),
    ('2026-07-17', 'Thứ Sáu',  'Hinh_9_SHAP_Waterfall_2026_07_17.png')
]

base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = float(np.squeeze(base_val))

for date_str, day_name, fig_name in waterfall_configs:
    target_dt = pd.to_datetime(date_str)
    idx_array = np.where(dates_test_dt == target_dt)[0]
    if len(idx_array) > 0:
        idx = idx_array[0]
        plt.figure(figsize=(10, 6))
        explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=base_val,
            data=X_test_df.iloc[idx],
            feature_names=feature_names
        )
        shap.waterfall_plot(explanation, show=False)
        plt.title(f"Giải thích cục bộ: {day_name} - Ngày {target_dt.strftime('%d/%m/%Y')}")
        plt.tight_layout()
        plt.savefig(f'figures/{fig_name}', dpi=300)
        plt.close()

print("\nHOÀN TẤT TOÀN BỘ PIPELINE! Tất cả 9 hình ảnh đã được lưu trong thư mục 'figures/'.")