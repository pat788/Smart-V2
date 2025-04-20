
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# สร้างและฝึกโมเดลแบบฝังในแอป (ฝึกครั้งเดียว)
@st.cache_resource
def train_model():
    import pandas as pd
    np.random.seed(42)
    rounds = 3000
    results = np.random.choice(['P', 'B', 'T'], size=rounds, p=[0.45, 0.45, 0.10])
    df = pd.DataFrame({'result': results})
    result_map = {'P': 0, 'B': 1, 'T': 2}
    df['result_code'] = df['result'].map(result_map)
    for i in range(1, 6):
        df[f'prev_{i}'] = df['result_code'].shift(i)

    def get_streak(data):
        streaks = []
        current = None
        count = 0
        for val in data:
            if val == current:
                count += 1
            else:
                count = 1
                current = val
            streaks.append(count)
        return streaks

    df['streak'] = get_streak(df['result_code'])

    def detect_pattern(p1, p2, p3, p4, p5):
        pattern = [p1, p2, p3, p4, p5]
        if pattern == pattern[::-1]: return 'mirror'
        if all(p == pattern[0] for p in pattern): return 'dragon'
        if len(set(pattern)) == 2 and pattern[::2] == pattern[::2][::-1]: return 'pingpong'
        return 'mixed'

    df['pattern_type'] = df.apply(lambda row: detect_pattern(row['prev_1'], row['prev_2'], row['prev_3'], row['prev_4'], row['prev_5']), axis=1)
    df.dropna(inplace=True)

    X = df[['prev_1', 'prev_2', 'prev_3', 'prev_4', 'prev_5', 'streak']]
    y = df['result_code']
    patterns = df['pattern_type']

    le = LabelEncoder()
    X['pattern_type'] = le.fit_transform(patterns)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)

    return model, le

model, encoder = train_model()

# Mapping
option_map = {"Player (P)": 0, "Banker (B)": 1, "Tie (T)": 2}
reverse_map = {0: "Player (P)", 1: "Banker (B)", 2: "Tie (T)"}

# ตรวจจับ pattern type
def detect_pattern(p):
    if p == p[::-1]:
        return 'mirror'
    if all(i == p[0] for i in p):
        return 'dragon'
    if len(set(p)) == 2 and p[::2] == p[::2][::-1]:
        return 'pingpong'
    return 'mixed'

# UI
st.title("Baccarat Predictor (Single File)")
st.write("ทำนายผลโดยวิเคราะห์ 5 ตาหลังสุด พร้อมจับ pattern อัตโนมัติ")

cols = st.columns(5)
inputs = [cols[i].selectbox(f"ผลตาก่อนหน้า {i+1}", list(option_map.keys())) for i in range(5)]
streak = st.number_input("จำนวน streak ล่าสุด", min_value=1, max_value=20, value=1)

if st.button("ทำนายผล"):
    values = [option_map[i] for i in inputs]
    pattern = detect_pattern(values)
    pattern_code = encoder.transform([pattern])[0]
    features = values + [streak, pattern_code]
    prediction = model.predict([features])[0]
    st.success(f"ระบบคาดการณ์ว่า: **{reverse_map[prediction]}**")
