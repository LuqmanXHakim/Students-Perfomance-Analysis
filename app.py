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

# Initialize empty input storage
student_data = {}

# Convert to DataFrame
input_df = pd.DataFrame(student_data, index=[0])

st.markdown("### 📋 Student Information")
section1, section2, section3 = st.columns(3)
with section1:
    tf_encoder = LabelEncoder()
    tf_encoder.fit(['Not Update', 'Update'])
    tf_value = st.selectbox('Tuition fees', ['Not Update', 'Update'], index=1)
    student_data['Tuition_fees_up_to_date'] = [tf_encoder.transform([tf_value])[0]]
with section2:
    sh_encoder = LabelEncoder()
    sh_encoder.fit(['Non Scholarship', 'Scholarship'])
    sh_value = st.selectbox('Scholarship holder', ['Non Scholarship', 'Scholarship'], index=0)
    student_data['Scholarship_holder'] = [sh_encoder.transform([sh_value])[0]]
with section3:
    db_encoder = LabelEncoder()
    db_encoder.fit(['Non Debtor', 'Debtor'])
    db_value = st.selectbox('Debtor', ['Non Debtor', 'Debtor'], index=1)
    student_data['Debtor'] = [db_encoder.transform([db_value])[0]]

section4, section5, section6 = st.columns(3)
with section4:
    dp_encoder = LabelEncoder()
    dp_encoder.fit(['Non Displaced', 'Displaced'])
    dp_value = st.selectbox('Displaced', ['Non Displaced', 'Displaced'], index=0)
    student_data['Displaced'] = [dp_encoder.transform([dp_value])[0]]
with section5:
    at_encoder = LabelEncoder()
    at_encoder.fit(['Daytime', 'Evening'])
    at_value = st.selectbox('Attendance', ['Daytime', 'Evening'], index=0)
    student_data['Daytime_evening_attendance'] = [at_encoder.transform([at_value])[0]]
with section6:
    gd_encoder = LabelEncoder()
    gd_encoder.fit(['Female', 'Male'])
    gd_value = st.selectbox('Gender', ['Female', 'Male'], index=1)
    student_data['Gender'] = [gd_encoder.transform([gd_value])[0]]

st.markdown("### 📊 Academic Scores")
sec_a, sec_b = st.columns(2)
with sec_a:
    grade1 = st.slider('Admission Grade', 0, 200, 100)
    student_data['Admission_grade'] = [grade1]
with sec_b:
    grade2 = st.slider('Previous Qualification Grade', 0, 200, 100)
    student_data['Previous_qualification_grade'] = [grade2]

st.markdown("#### 📚 Curricular Units 1st Semester")
g1, g2, g3, g4 = st.columns(4)
with g1:
    approved1 = st.number_input('1st Sem Approved', value=5)
    student_data['Curricular_units_1st_sem_approved'] = [approved1]
with g2:
    grade_1st = st.number_input('1st Sem Grade', value=12)
    student_data['Curricular_units_1st_sem_grade'] = [grade_1st]
with g3:
    enrolled1 = st.number_input('1st Sem Enrolled', value=6)
    student_data['Curricular_units_1st_sem_enrolled'] = [enrolled1]
with g4:
    credited1 = st.number_input('1st Sem Credited', value=0)
    student_data['Curricular_units_1st_sem_credited'] = [credited1]

st.markdown("#### 📚 Curricular Units 2nd Semester")
h1, h2, h3, h4 = st.columns(4)
with h1:
    approved2 = st.number_input('2nd Sem Approved', value=5)
    student_data['Curricular_units_2nd_sem_approved'] = [approved2]
with h2:
    grade_2nd = st.number_input('2nd Sem Grade', value=12)
    student_data['Curricular_units_2nd_sem_grade'] = [grade_2nd]
with h3:
    enrolled2 = st.number_input('2nd Sem Enrolled', value=6)
    student_data['Curricular_units_2nd_sem_enrolled'] = [enrolled2]
with h4:
    credited2 = st.number_input('2nd Sem Credited', value=0)
    student_data['Curricular_units_2nd_sem_credited'] = [credited2]

# Convert again to DataFrame
input_df = pd.DataFrame(student_data, index=[0])

# Show in expander
with st.expander("Raw Dataset"):
    st.dataframe(input_df, width=1200, height=20)

# Predict on button click
if st.button('Click Here to Predict'):
    processed_data = data_preparation(data=student_data)
    with st.spinner("Predicting..."):
        time.sleep(2)
        prediction_result = prediction(processed_data)
        st.toast("Prediction completed!")
        st.success(f"## 🎯 Prediction Result: {prediction_result}")


