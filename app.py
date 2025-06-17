import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import time
from data_preparation import (
    data_preparation,
    encoder_Daytime_evening_attendance,
    encoder_Debtor,
    encoder_Displaced,
    encoder_Gender,
    encoder_Scholarship_holder,
    encoder_Tuition_fees_up_to_date,
    scaler_Admission_grade, 
    scaler_Curricular_units_1st_sem_approved, 
    scaler_Curricular_units_1st_sem_credited, 
    scaler_Curricular_units_1st_sem_enrolled, 
    scaler_Curricular_units_1st_sem_grade, 
    scaler_Curricular_units_2nd_sem_approved, 
    scaler_Curricular_units_2nd_sem_credited, 
    scaler_Curricular_units_2nd_sem_enrolled, 
    scaler_Curricular_units_2nd_sem_grade, 
    scaler_Previous_qualification_grade
)
from prediction import prediction


st.title("🎓 Prediction Status Students")

# Buat dictionary kosong untuk input pengguna
user_inputs = {}

# Ubah dictionary input pengguna menjadi DataFrame
df_user_inputs = pd.DataFrame(user_inputs, index=[0])

st.markdown("### 📋 Student Information")
cols1 = st.columns(3)
with cols1[0]:
    tuition_status = st.selectbox('Tuition fees', ['No', 'Yes'], index=1)
    user_inputs['Tuition_fees_up_to_date'] = [tuition_status]
with cols1[1]:
    scholar_status = st.selectbox('Scholarship holder', ['Non Scholarship', 'Scholarship'], index=0)
    user_inputs['Scholarship_holder'] = [scholar_status]
with cols1[2]:
    debtor_status = st.selectbox('Debtor', ['Non Debtor', 'Debtor'], index=1)
    user_inputs['Debtor'] = [debtor_status]

cols2 = st.columns(3)
with cols2[0]:
    displaced_status = st.selectbox('Displaced', ['Non Displaced', 'Displaced'], index=0)
    user_inputs['Displaced'] = [displaced_status]
with cols2[1]:
    attendance_status = st.selectbox('Attendance', ['Daytime', 'Evening'], index=0)
    user_inputs['Daytime_evening_attendance'] = [attendance_status]
with cols2[2]:
    gender_status = st.selectbox('Gender', ['Female', 'Male'], index=1)
    user_inputs['Gender'] = [gender_status]

st.markdown("### 📊 Academic Scores")
cols3 = st.columns(2)
with cols3[0]:
    admission_score = st.slider('Admission Grade', 0, 200, 100)
    user_inputs['Admission_grade'] = [admission_score]
with cols3[1]:
    prev_qual_score = st.slider('Previous Qualification Grade', 0, 200, 100)
    user_inputs['Previous_qualification_grade'] = [prev_qual_score]

st.markdown("#### 📚 Curricular Units 1st Semester")
cols4 = st.columns(4)
with cols4[0]:
    first_approved = st.number_input('1st Sem Approved', value=5)
    user_inputs['Curricular_units_1st_sem_approved'] = [first_approved]
with cols4[1]:
    first_grade = st.number_input('1st Sem Grade', value=12)
    user_inputs['Curricular_units_1st_sem_grade'] = [first_grade]
with cols4[2]:
    first_enrolled = st.number_input('1st Sem Enrolled', value=6)
    user_inputs['Curricular_units_1st_sem_enrolled'] = [first_enrolled]
with cols4[3]:
    first_credited = st.number_input('1st Sem Credited', value=0)
    user_inputs['Curricular_units_1st_sem_credited'] = [first_credited]

st.markdown("#### 📚 Curricular Units 2nd Semester")
cols5 = st.columns(4)
with cols5[0]:
    second_approved = st.number_input('2nd Sem Approved', value=5)
    user_inputs['Curricular_units_2nd_sem_approved'] = [second_approved]
with cols5[1]:
    second_grade = st.number_input('2nd Sem Grade', value=12)
    user_inputs['Curricular_units_2nd_sem_grade'] = [second_grade]
with cols5[2]:
    second_enrolled = st.number_input('2nd Sem Enrolled', value=6)
    user_inputs['Curricular_units_2nd_sem_enrolled'] = [second_enrolled]
with cols5[3]:
    second_credited = st.number_input('2nd Sem Credited', value=0)
    user_inputs['Curricular_units_2nd_sem_credited'] = [second_credited]

# Ubah input ke DataFrame
df_user_inputs = pd.DataFrame(user_inputs, index=[0])

# Tampilkan dataset
with st.expander("Raw Dataset"):
    st.dataframe(data=df_user_inputs, width=1200, height=20)

# Proses prediksi
if st.button('Click Here to Predict'):
    processed_data = data_preparation(data=df_user_inputs)
    with st.spinner("Predicting..."):
        time.sleep(2)
        result = prediction(processed_data)
        st.toast("Prediction completed!")
        st.success(f"## 🎯 Prediction Result: {result}")

