import numpy as np
import logging
import pandas as pd
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from flask import Flask, render_template, send_file
from tensorflow.keras.models import load_model
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Initialize Flask app
app = Flask(__name__)
baseline_df = pd.read_csv('train_data.csv')  # Replace with your file path
predictions_df = pd.read_csv('prediction_data.csv')
predictions_df = pd.concat([predictions_df] * 3, ignore_index=True)

# Set up logging
log_filename = 'prediction_logs.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Load test data
def load_test_data():
    logging.info("Loading test data...")
    input_sequences_test = np.load('input_sequences_test.npy')
    next_step_targets_test = np.load('next_step_targets_test.npy')
    logging.info("Test data loaded successfully.")
    return input_sequences_test, next_step_targets_test

# Load trained models
def load_models_mae():
    logging.info("Loading MAE models...")
    lstm_model = load_model('best_lstm_model_mae.keras')
    gru_model = load_model('best_gru_model_mae.keras')
    logging.info("MAE models loaded successfully.")
    return lstm_model, gru_model

def load_models_rmse():
    logging.info("Loading RMSE models...")
    lstm_model = load_model('best_lstm_model_rmse.keras')
    gru_model = load_model('best_gru_model_rmse.keras')
    logging.info("RMSE models loaded successfully.")
    return lstm_model, gru_model

# Make predictions
def make_predictions_mae(input_sequences_test, lstm_model_mae, gru_model_mae):
    logging.info("Making predictions using MAE models...")
    lstm_predictions_mae = lstm_model_mae.predict(input_sequences_test)
    gru_predictions_mae = gru_model_mae.predict(input_sequences_test)
    logging.info("Predictions made successfully using MAE models.")
    return lstm_predictions_mae, gru_predictions_mae

def make_predictions_rmse(input_sequences_test, lstm_model_rmse, gru_model_rmse):
    logging.info("Making predictions using RMSE models...")
    lstm_predictions_rmse = lstm_model_rmse.predict(input_sequences_test)
    gru_predictions_rmse = gru_model_rmse.predict(input_sequences_test)
    logging.info("Predictions made successfully using RMSE models.")
    return lstm_predictions_rmse, gru_predictions_rmse

# Save predictions to file
def save_predictions_to_file(lstm_predictions_mae, gru_predictions_mae, lstm_predictions_rmse, gru_predictions_rmse):
    predictions_df = pd.DataFrame({
        'LSTM_MAE': lstm_predictions_mae.flatten(),
        'GRU_MAE': gru_predictions_mae.flatten(),
        'LSTM_RMSE': lstm_predictions_rmse.flatten(),
        'GRU_RMSE': gru_predictions_rmse.flatten()
    })
    
    predictions_df.to_csv('predictions.csv', index=False)
    logging.info("Predictions saved to predictions.csv")
    
def send_email_alert(drift_score):
    message = Mail(
        from_email='ayushit@andrew.cmu.edu',  # Replace with your SendGrid verified sender email
        to_emails='pmuller@andrew.cmu.edu',
        subject='Data Drift Alert',
        plain_text_content=f'Data drift detected! The drift score is {drift_score}, which exceeds the threshold of 0.5.'
    )
    
    try:
        sg = SendGridAPIClient('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')  # Replace with your SendGrid API Key
        response = sg.send(message)
        logging.info(f"Email alert sent successfully. Status code: {response.status_code}")
    except Exception as e:
        logging.error(f"Error sending email alert: {e}")
        
# Flask Routes
@app.route('/')
def home():
    return "<h1>Welcome to the Prediction API</h1><p>Visit <a href='/predict'>/predict</a> for predictions and <a href='/report'>/report</a> for the Evidently report.</p>"

@app.route('/predict', methods=['GET'])
def predict():
    input_sequences_test, _ = load_test_data()
    
    # Load and make predictions using MAE models
    lstm_model_mae, gru_model_mae = load_models_mae()
    lstm_predictions_mae, gru_predictions_mae = make_predictions_mae(input_sequences_test, lstm_model_mae, gru_model_mae)
    
    # Load and make predictions using RMSE models
    lstm_model_rmse, gru_model_rmse = load_models_rmse()
    lstm_predictions_rmse, gru_predictions_rmse = make_predictions_rmse(input_sequences_test, lstm_model_rmse, gru_model_rmse)
    
    # Save predictions
    save_predictions_to_file(lstm_predictions_mae, gru_predictions_mae, lstm_predictions_rmse, gru_predictions_rmse)

    return "Predictions generated and saved successfully!"


@app.route('/report')
def generate_report():
    try:
        # Load the predictions and ground truth data from separate CSVs
        # pulling from the predictions_data due to no new input file (Not reading actual traffic data as it occurs.)
        predictions_df = pd.read_csv('prediction_data.csv')
        multiplied_predictions_df = pd.concat([predictions_df] * 3, ignore_index=True)
        ground_truth_df = pd.read_csv('train_data.csv')

        # Create the report object and specify the preset for Data Drift
        report = Report(metrics=[DataDriftPreset()])

        # Run the report to detect drift
        report.run(reference_data=ground_truth_df, current_data=multiplied_predictions_df)

        # Extract the drift results as a dictionary
        drift_results = report.as_dict()

        # Extract the 'dataset_drift' value, which indicates the level of drift in the entire dataset
        dataset_drift = drift_results['metrics'][0]['result']['dataset_drift']  # Access the drift score from the results
        print(dataset_drift)
        # Define a threshold for triggering the alert (e.g., if drift exceeds 0.5 or 50%)
        drift_threshold = .2

        # Check if the detected drift exceeds the defined threshold
        if dataset_drift > drift_threshold:
            # If drift is significant, print a warning message
            logging.warning(f"Data drift detected! Drift score: {dataset_drift}")
            send_email_alert(dataset_drift)
            print(f"Data drift detected! Drift score: {dataset_drift}")  # You might want to replace this with your alerting logic
        else:
            # If no significant drift is detected, print a message to indicate the system is stable
            logging.info(f"No significant data drift detected. Drift score: {dataset_drift}")
            print(f"No significant data drift detected. Drift score: {dataset_drift}")

        # Save the report to an HTML file
        report.save_html('templates/report.html')
        logging.info("Drift report generated successfully.")

    except Exception as e:
        logging.error(f"Error generating report: {e}")
        return "Error generating report", 500

    return render_template('report.html')

# Run Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
