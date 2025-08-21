import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ========== Step 1: 加载数据并重命名列 ==========
@st.cache_data
def load_data():
    data1 = pd.read_csv(r"D:\A运动生物力学\新疆师范大学博士资料\小论文\MTSS\数据\MTSS-zixuan.csv", encoding='gbk')
    data1.dropna(inplace=True)

    data1.columns = [
        'Number',
        'Target variable',
        'Ankle plantar/dorsiflexion angle',
        'Ankle in/eversion angle',
        'Hip ad/abduction angle',
        'Knee flex/extension angle',
        'Knee ad/abduction angle',
        'Knee in/external rotation angle',
        'M/L pelvic lean',
        'Vertical pelvic lean',
        'Vertical trunk lean',
        'M/L COM position',
        'Vertical COM position'
    ]
    
    return data1

data = load_data()

# ========== Step 2: 特征和标签 ==========
features = data.columns[2:]  # 从第3列开始为特征
X = data[features]
y = data['Target variable']

# ========== Step 3: 拆分训练集与测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Step 4: 模型训练 ==========
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=1,
    min_child_weight=3,
    learning_rate=0.04,
    n_estimators=200,
    subsample=0.7,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# ========== Step 5: Streamlit 前端界面 ==========
st.title("Tibial Load Prediction Based on Joint Angles and Pelvic Lean")
st.write("""
    Please enter your joint angles and pelvic lean values below (in degrees).
""")

st.sidebar.header("Input Parameters")
input_values = []
for col in features:
    val = st.sidebar.slider(col, -180.0, 180.0, 0.0)
    input_values.append(val)

user_input_df = pd.DataFrame([input_values], columns=features)

# 预测
predicted_load = model.predict(user_input_df)

# 显示结果
st.subheader("Prediction Result")
st.write(f"**Predicted Tibial Load:** `{predicted_load[0]:.2f}` N/kg")
