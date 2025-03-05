import streamlit as st
import numpy as np
import joblib
import os
import sklearn

# ตั้งค่า Streamlit ต้องเป็นคำสั่งแรกสุด
st.set_page_config(page_title="Cardiovascular Risk Assessment", page_icon="❤️", layout="centered")

st.write("scikit-learn version:", sklearn.__version__)

# ตรวจสอบเวอร์ชันของ scikit-learn เพื่อป้องกันปัญหาความเข้ากันไม่ได้ของโมเดล
if sklearn.__version__ != "1.5.2":
    st.warning("⚠️ คำเตือน: scikit-learn เวอร์ชันอาจไม่ตรงกับที่ใช้ตอนฝึกโมเดล อาจทำให้เกิดข้อผิดพลาด")

# ตั้งค่าธีม UI
st.markdown(
    """
    <style>
    .stApp { background-color: #FFFFFF; color: #000000; }
    h1, h2, h3, h4, h5, h6, p, label, span { color: #000000 !important; }
    .stAlert[data-baseweb="alert"] { color: #FFFFFF !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ส่วนหัวของแอป
st.markdown(
    """
    <h1 style='text-align: center; color: #2E8B57;'>Cardiovascular Disease Prediction</h1>
    <p style='text-align: center; color: #555;'>กรอกข้อมูลสุขภาพของคุณเพื่อประเมินความเสี่ยงของโรคหัวใจ</p>
    """, unsafe_allow_html=True
)

# ใช้คอลัมน์เพื่อจัด UI
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🕒 อายุ (ปี)", 29, 64, 40)
    gender = st.radio("🚻 เพศ", ["ชาย", "หญิง"], horizontal=True)
    height = st.number_input("📏 ส่วนสูง (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("⚖️ น้ำหนัก (kg)", min_value=30, max_value=200, value=70)

with col2:
    ap_hi = st.number_input("🩸 ความดันโลหิต Systolic", min_value=50, max_value=250, value=120)
    ap_lo = st.number_input("🩸 ความดันโลหิต Diastolic", min_value=30, max_value=200, value=80)

    st.markdown("")
    
    cholesterol = st.selectbox("🧪 ระดับคอเลสเตอรอล", [1, 2, 3], format_func=lambda x: ["ปกติ", "สูง", "สูงมาก"][x-1])
    gluc = st.selectbox("🍬 ระดับน้ำตาลในเลือด", [1, 2, 3], format_func=lambda x: ["ปกติ", "สูง", "สูงมาก"][x-1])

# พฤติกรรมสุขภาพ
st.markdown("### 🏃‍♂️ พฤติกรรมสุขภาพ")
col3, col4, col5 = st.columns(3)

with col3:
    smoke = st.checkbox("🚬 สูบบุหรี่")
with col4:
    alco = st.checkbox("🍷 ดื่มแอลกอฮอล์")
with col5:
    active = st.checkbox("💪 ออกกำลังกาย")

# แปลงข้อมูลให้เป็นตัวเลข
gender = 1 if gender == "ชาย" else 2
smoke, alco, active = int(smoke), int(alco), int(active)
input_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])

# โหลดโมเดล
model_path = "voting_classifier.pkl"
model = None

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
else:
    st.error("❌ ไม่พบไฟล์โมเดล 'voting_classifier.pkl' กรุณาตรวจสอบว่าไฟล์อยู่ใน directory ที่ถูกต้อง")

# ทำนายผล
if st.button("🔍 ทำนายผลลัพธ์", use_container_width=True) and model is not None:
    try:
        prediction = model.predict(input_data)[0]
        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ คุณมีความเสี่ยงต่อโรคหัวใจ โปรดปรึกษาแพทย์")
        else:
            st.success("✅ คุณมีสุขภาพดี ไม่มีความเสี่ยงต่อโรคหัวใจ")
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดระหว่างการทำนาย: {e}")
