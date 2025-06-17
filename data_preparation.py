import joblib
import numpy as np
import pandas as pd

# Load encoder dan scaler sebagai variabel global
encoder_Tuition_fees_up_to_date = joblib.load("./model/encoder_Tuition_fees_up_to_date.joblib")
encoder_Scholarship_holder = joblib.load("./model/encoder_Scholarship_holder.joblib")
encoder_Debtor = joblib.load("./model/encoder_Debtor.joblib")
encoder_Displaced = joblib.load("./model/encoder_Displaced.joblib")
encoder_Daytime_evening_attendance = joblib.load("./model/encoder_Daytime_evening_attendance.joblib")
encoder_Gender = joblib.load("./model/encoder_Gender.joblib")

scaler_Admission_grade = joblib.load("./model/scaler_Admission_grade.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("./model/scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_credited = joblib.load("./model/scaler_Curricular_units_1st_sem_credited.joblib")
scaler_Curricular_units_1st_sem_enrolled = joblib.load("./model/scaler_Curricular_units_1st_sem_enrolled.joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("./model/scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("./model/scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_credited = joblib.load("./model/scaler_Curricular_units_2nd_sem_credited.joblib")
scaler_Curricular_units_2nd_sem_enrolled = joblib.load("./model/scaler_Curricular_units_2nd_sem_enrolled.joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("./model/scaler_Curricular_units_2nd_sem_grade.joblib")
scaler_Previous_qualification_grade = joblib.load("./model/scaler_Previous_qualification_grade.joblib")

def data_preparation(data):
    """
    Melakukan encoding dan scaling terhadap input dataframe untuk prediksi.
    """
    df_input = data.copy()
    df_output = pd.DataFrame()

    # Encoding kategorikal
    df_output["Tuition_fees_up_to_date"] = encoder_Tuition_fees_up_to_date.transform(df_input["Tuition_fees_up_to_date"])
    df_output["Scholarship_holder"] = encoder_Scholarship_holder.transform(df_input["Scholarship_holder"])
    df_output["Debtor"] = encoder_Debtor.transform(df_input["Debtor"])
    df_output["Displaced"] = encoder_Displaced.transform(df_input["Displaced"])
    df_output["Daytime_evening_attendance"] = encoder_Daytime_evening_attendance.transform(df_input["Daytime_evening_attendance"])
    df_output["Gender"] = encoder_Gender.transform(df_input["Gender"])

    # Scaling numerikal
    df_output["Admission_grade"] = scaler_Admission_grade.transform(np.array(df_input["Admission_grade"]).reshape(-1, 1))
    df_output["Curricular_units_1st_sem_approved"] = scaler_Curricular_units_1st_sem_approved.transform(np.array(df_input["Curricular_units_1st_sem_approved"]).reshape(-1, 1))
    df_output["Curricular_units_1st_sem_credited"] = scaler_Curricular_units_1st_sem_credited.transform(np.array(df_input["Curricular_units_1st_sem_credited"]).reshape(-1, 1))
    df_output["Curricular_units_1st_sem_enrolled"] = scaler_Curricular_units_1st_sem_enrolled.transform(np.array(df_input["Curricular_units_1st_sem_enrolled"]).reshape(-1, 1))
    df_output["Curricular_units_1st_sem_grade"] = scaler_Curricular_units_1st_sem_grade.transform(np.array(df_input["Curricular_units_1st_sem_grade"]).reshape(-1, 1))
    df_output["Curricular_units_2nd_sem_approved"] = scaler_Curricular_units_2nd_sem_approved.transform(np.array(df_input["Curricular_units_2nd_sem_approved"]).reshape(-1, 1))
    df_output["Curricular_units_2nd_sem_credited"] = scaler_Curricular_units_2nd_sem_credited.transform(np.array(df_input["Curricular_units_2nd_sem_credited"]).reshape(-1, 1))
    df_output["Curricular_units_2nd_sem_enrolled"] = scaler_Curricular_units_2nd_sem_enrolled.transform(np.array(df_input["Curricular_units_2nd_sem_enrolled"]).reshape(-1, 1))
    df_output["Curricular_units_2nd_sem_grade"] = scaler_Curricular_units_2nd_sem_grade.transform(np.array(df_input["Curricular_units_2nd_sem_grade"]).reshape(-1, 1))
    df_output["Previous_qualification_grade"] = scaler_Previous_qualification_grade.transform(np.array(df_input["Previous_qualification_grade"]).reshape(-1, 1))

    return df_output

