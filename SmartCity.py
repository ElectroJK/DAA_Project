import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from datetime import datetime, timedelta

os.environ["API_KEY"] = "AIzaSyDFBs0-g2g76sUvF777fRrqvxLfXojK7ZE"
genai.configure(api_key=os.environ["API_KEY"])

files = ['Датасет #1.csv', 'Датасет 2.csv', 'Датасет 3.csv', 'Датасет 4.csv']

data = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
print("Total rows in data before processing:", data.shape[0])

data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
data['departure_time'] = pd.to_datetime(data['departure_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

data = data.dropna(subset=['arrival_time', 'departure_time', 'dwell_time_in_seconds', 'address'])
print("Total rows in data after dropping NaNs:", data.shape[0])

data['arrival_hour'] = data['arrival_time'].dt.hour
data['arrival_minute'] = data['arrival_time'].dt.minute
data['day_of_week'] = data['arrival_time'].dt.dayofweek

data['next_arrival'] = data['arrival_time'].shift(-1)
data['time_to_next_stop'] = (data['next_arrival'] - data['departure_time']).dt.total_seconds()

data = data.dropna(subset=['time_to_next_stop'])
print("Total rows in data after creating target variable:", data.shape[0])

data['is_peak_hour'] = data['arrival_hour'].apply(lambda x: 1 if (7 <= x < 9) or (16 <= x < 18) else 0)

features = ['deviceid', 'direction', 'bus_stop', 'arrival_hour', 'arrival_minute', 'day_of_week',
            'dwell_time_in_seconds', 'is_peak_hour', 'address']  # Include 'address'
target = 'time_to_next_stop'

X = data[features]
y = data[target]

X = pd.get_dummies(X, columns=['deviceid', 'direction', 'bus_stop', 'address'], drop_first=True)

models = {
    "Gradient Boosting": GradientBoostingRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

if X.shape[0] > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mae_results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mae_results[model_name] = mae
        print(f"{model_name} - Mean Absolute Error: {mae:.2f} seconds")

    plt.bar(mae_results.keys(), mae_results.values())
    plt.ylabel('Mean Absolute Error (seconds)')
    plt.title('Model Comparison')
    plt.show()
else:
    print("No data available for training the model.")


def calculate_estimated_arrival_time(departure_time_str, transit_time):
    try:
        departure_dt = pd.to_datetime(departure_time_str)

        if isinstance(transit_time, (float, int)) and transit_time > 0:
            # Cap the maximum transit time to avoid overflow
            max_transit_time = 3600 * 24  # 24 hours in seconds
            transit_time = min(transit_time, max_transit_time)
            estimated_arrival_dt = departure_dt + timedelta(seconds=transit_time)
            return estimated_arrival_dt
        else:
            print("Invalid transit time provided. It should be a positive number.")
            return None
    except Exception as e:
        print(f"Error calculating estimated arrival time: {e}")
        return None


def predict_with_genai(arrival_time, departure_time, dwell_time, weather_condition, is_peak_hour):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Predict the travel time to the next stop given the following details: \
        Arrival time: {arrival_time}, Departure time: {departure_time}, Dwell time in seconds: {dwell_time}, \
        Weather condition: {weather_condition}, Peak hour: {is_peak_hour}. Please provide the expected transit time in seconds."

    response = model.generate_content(prompt)
    predicted_time = response.text.strip()

    try:
        numeric_part = ''.join(filter(str.isdigit, predicted_time))
        if numeric_part:
            predicted_time_float = float(numeric_part)

            if weather_condition.lower() == "clear":
                predicted_time_float += 300

            print(f"Predicted transit time to the next stop: {predicted_time_float} seconds")
            return predicted_time_float
        else:
            print("No valid numeric prediction found in the response.")
            return None
    except ValueError:
        print("Failed to convert predicted time to float.")
        return None


arrival_time_example = "2024-11-03 08:45:00"
departure_time_example = "2024-11-03 08:50:00"
dwell_time_example = 300
weather_condition_example = "clear"
is_peak_hour_example = 1

predicted_time = predict_with_genai(arrival_time_example, departure_time_example, dwell_time_example,
                                    weather_condition_example, is_peak_hour_example)

estimated_arrival_time = calculate_estimated_arrival_time(departure_time_example, predicted_time)
if estimated_arrival_time:
    print(f"Estimated Arrival Time: {estimated_arrival_time}")

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(
    "Generate sample data for transportation records, including realistic fields such as 'end_terminal', 'duration' (in seconds), and 'duration_in_mins' (in minutes). Each record should be randomized and reflective of transit data without additional explanation and make result like table."
)

print("Generated sample data from Google Generative AI:")
print(response.text)

if not X.empty:
    sample_data = X.sample(5)
    print("Sampled data for prediction:")
    print(sample_data)
else:
    print("No data available for sampling.")
