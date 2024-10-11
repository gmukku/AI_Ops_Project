
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import logging

# Set up logging
log_filename = 'ground_truth_logs.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # Log to a file
        logging.StreamHandler()             # Log to console
    ]
)

# Load preprocessed test data (replace this with actual test data)
def load_test_data():
    logging.info("Loading test data...")
    input_sequences_test = np.load('input_sequences_test.npy')  # Update with actual path
    logging.info("Test data loaded successfully.")
    return input_sequences_test

# Load models
def load_models():
    logging.info("Loading models...")
    lstm_model_mae = load_model('best_lstm_model_mae.keras')  # Replace with actual MAE LSTM model path
    gru_model_mae = load_model('best_gru_model_mae.keras')    # Replace with actual MAE GRU model path
    lstm_model_rmse = load_model('best_lstm_model_rmse.keras')  # Replace with actual RMSE LSTM model path
    gru_model_rmse = load_model('best_gru_model_rmse.keras')    # Replace with actual RMSE GRU model path
    logging.info("Models loaded successfully.")
    return lstm_model_mae, gru_model_mae, lstm_model_rmse, gru_model_rmse

# Make predictions
def make_predictions(input_sequences_test, lstm_model, gru_model):
    logging.info(f"Making predictions...")
    lstm_predictions = lstm_model.predict(input_sequences_test)
    gru_predictions = gru_model.predict(input_sequences_test)
    logging.info("Predictions made successfully.")
    return lstm_predictions, gru_predictions

# Calculate average predictions
def calculate_average(lstm_predictions, gru_predictions):
    logging.info("Calculating average predictions...")
    avg_lstm = np.mean(lstm_predictions)
    avg_gru = np.mean(gru_predictions)
    logging.info(f"Average LSTM: {avg_lstm}, Average GRU: {avg_gru}")
    return avg_lstm, avg_gru

# Save averages to a CSV file
def save_averages_to_csv(avg_lstm_mae, avg_gru_mae, avg_lstm_rmse, avg_gru_rmse):
    df = pd.DataFrame({
        'LSTM_MAE': [avg_lstm_mae],
        'GRU_MAE': [avg_gru_mae],
        'LSTM_RMSE': [avg_lstm_rmse],
        'GRU_RMSE': [avg_gru_rmse]
    })
    df.to_csv('ground_truth.csv', index=False)
    logging.info("Averages saved to ground_truth.csv.")

# Main execution
if __name__ == '__main__':
    input_sequences_test = load_test_data()

    lstm_model_mae, gru_model_mae, lstm_model_rmse, gru_model_rmse = load_models()

    lstm_predictions_mae, gru_predictions_mae = make_predictions(input_sequences_test, lstm_model_mae, gru_model_mae)
    lstm_predictions_rmse, gru_predictions_rmse = make_predictions(input_sequences_test, lstm_model_rmse, gru_model_rmse)

    avg_lstm_mae, avg_gru_mae = calculate_average(lstm_predictions_mae, gru_predictions_mae)
    avg_lstm_rmse, avg_gru_rmse = calculate_average(lstm_predictions_rmse, gru_predictions_rmse)

    save_averages_to_csv(avg_lstm_mae, avg_gru_mae, avg_lstm_rmse, avg_gru_rmse)

    logging.info("Processing complete.")
