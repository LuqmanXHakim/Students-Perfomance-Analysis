import joblib
import numpy as np
import pandas as pd

# Direktori model
MODEL_PATH = "model"

# Fungsi bantu untuk load encoder/scaler
def load_model(file_name):
    return joblib.load(f"{MODEL_PATH}/{file_name}")

# Encoder
encoder_Tuition_fees_up_to_date = load_model("encoder_Tuition_fees_up_to_date.joblib")
encoder_Scholarship_holder = load_model("encoder_Scholarship_holder.joblib")
encoder_Debtor = load_model("encoder_Debtor.joblib")
encoder_Displaced = load_model("encoder_Displaced.joblib")
encoder_Daytime_evening_attendance = load_model("encoder_Daytime_evening_attendance.joblib")
encoder_Gender = load_model("encoder_Gender.joblib")

# Scaler
scaler_Admission_grade = load_model("scaler_Admission_grade.joblib")
scaler_Curricular_units_1st_sem_approved = load_model("scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_credited = load_model("scaler_Curricular_units_1st_sem_credited.joblib")
scaler_Curricular_units_1st_sem_enrolled = load_model("scaler_Curricular_units_1st_sem_enrolled.joblib")
scaler_Curricular_units_1st_sem_grade = load_model("scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_approved = load_model("scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_credited = load_model("scaler_Curricular_units_2nd_sem_credited.joblib")
scaler_Curricular_units_2nd_sem_enrolled = load_model("scaler_Curricular_units_2nd_sem_enrolled.joblib")
scaler_Curricular_units_2nd_sem_grade = load_model("scaler_Curricular_units_2nd_sem_grade.joblib")
scaler_Previous_qualification_grade = load_model("scaler_Previous_qualification_grade.joblib")

def data_preparation(df: pd.DataFrame) -> pd.DataFrame:
    """Transform input dataframe using pre-fitted encoders and scalers."""
    output = pd.DataFrame()

    # Encode categorical features
    output["Tuition_fees_up_to_date"] = encoder_Tuition_fees_up_to_date.transform(df["Tuition_fees_up_to_date"])
    output["Scholarship_holder"] = encoder_Scholarship_holder.transform(df["Scholarship_holder"])
    output["Debtor"] = encoder_Debtor.transform(df["Debtor"])
    output["Displaced"] = encoder_Displaced.transform(df["Displaced"])
    output["Daytime_evening_attendance"] = encoder_Daytime_evening_attendance.transform(df["Daytime_evening_attendance"])
    output["Gender"] = encoder_Gender.transform(df["Gender"])

    # Scale numerical features
    output["Admission_grade"] = scaler_Admission_grade.transform(df[["Admission_grade"]])
    output["Curricular_units_1st_sem_approved"] = scaler_Curricular_units_1st_sem_approved.transform(df[["Curricular_units_1st_sem_approved"]])
    output["Curricular_units_1st_sem_credited"] = scaler_Curricular_units_1st_sem_credited.transform(df[["Curricular_units_1st_sem_credited"]])
    output["Curricular_units_1st_sem_enrolled"] = scaler_Curricular_units_1st_sem_enrolled.transform(df[["Curricular_units_1st_sem_enrolled"]])
    output["Curricular_units_1st_sem_grade"] = scaler_Curricular_units_1st_sem_grade.transform(df[["Curricular_units_1st_sem_grade"]])
    output["Curricular_units_2nd_sem_approved"] = scaler_Curricular_units_2nd_sem_approved.transform(df[["Curricular_units_2nd_sem_approved"]])
    output["Curricular_units_2nd_sem_credited"] = scaler_Curricular_units_2nd_sem_credited.transform(df[["Curricular_units_2nd_sem_credited"]])
    output["Curricular_units_2nd_sem_enrolled"] = scaler_Curricular_units_2nd_sem_enrolled.transform(df[["Curricular_units_2nd_sem_enrolled"]])
    output["Curricular_units_2nd_sem_grade"] = scaler_Curricular_units_2nd_sem_grade.transform(df[["Curricular_units_2nd_sem_grade"]])
    output["Previous_qualification_grade"] = scaler_Previous_qualification_grade.transform(df[["Previous_qualification_grade"]])

    return output