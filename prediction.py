import joblib

# Load the trained model and label encoder
logreg_model = joblib.load("./logistic_regression_best_model.joblib")
label_decoder = joblib.load("./model/encoder_target.joblib")

def prediction(input_df):
    """
    Perform prediction using the trained Logistic Regression model.

    Parameters:
        input_df (pd.DataFrame): The preprocessed input features.

    Returns:
        np.ndarray: Human-readable prediction labels (e.g., 'Drop Out' or 'Graduate').
    """
    predicted = logreg_model.predict(input_df)
    decoded_result = label_decoder.inverse_transform(predicted)
    return decoded_result