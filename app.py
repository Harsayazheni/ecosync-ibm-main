import joblib
import plotly
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from Elec_consumption import create_dashboard_plots  # Import your plotting functions
import os  # To get the port number

app = Flask(__name__)

# Load the model and necessary preprocessing objects
model = load_model('gru_co2_model.h5')
data = pd.read_csv('climate_change_data.csv')

# Create and fit the encoders and scaler
location_encoder = LabelEncoder()
country_encoder = LabelEncoder()
scaler = StandardScaler()

data['Location_encoded'] = location_encoder.fit_transform(data['Location'])
data['Country_encoded'] = country_encoder.fit_transform(data['Country'])

# Fit the scaler on the encoded data
scaler.fit(data[['Temperature', 'Location_encoded', 'Country_encoded']])

# Energy
# Load and preprocess data
dataEC = pd.read_csv('energy-consumption.csv')

def categorize_consumption(consumption):
    if consumption < 200:
        return 'Very Low'
    elif 200 <= consumption < 400:
        return 'Low'
    elif 400 <= consumption < 600:
        return 'Medium'
    else:
        return 'High'

dataEC['Consumption Category'] = dataEC['energy_consumption'].apply(categorize_consumption)

features = ['temperature', 'humidity', 'hour', 'day', 'month']
target = 'energy_consumption'

X = dataEC[features]
y = dataEC[target]

# # Get min and max values for temperature and humidity
temp_min, temp_max = X['temperature'].min(), X['temperature'].max()
humidity_min, humidity_max = X['humidity'].min(), X['humidity'].max()

# Scale the features
scalerEC = StandardScaler()
X_scaledEC = scalerEC.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaledEC, y, test_size=0.2, random_state=42)

# Train model (you might want to do this separately and save the model)
def train_model():
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'energy_model.joblib')
    joblib.dump(scalerEC, 'scaler.joblib')
    return best_model

# Load or train the model
try:
    modelEC = joblib.load('energy_model.joblib')
    scalerEC = joblib.load('scaler.joblib')
except:
    modelEC = train_model()
    scalerEC = joblib.load('scaler.joblib')

@app.route('/')
def index():
    countries = sorted(data['Country'].unique())
    locations = sorted(data['Location'].unique())

    # Insights
    # Generate the plots
    fig1, fig2, fig3, fig4 = create_dashboard_plots()  # This function should return your Plotly figures
    
    # Convert the plots to JSON
    plot1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    plot2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    plot3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    plot4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html', countries=countries, locations=locations, temp_min=temp_min, temp_max=temp_max, humidity_min=humidity_min, humidity_max=humidity_max, plot1JSON=plot1JSON, plot2JSON=plot2JSON, 
                           plot3JSON=plot3JSON, plot4JSON=plot4JSON)

@app.route('/predict', methods=['POST'])
def predict():
    temperature = float(request.form['temperature'])
    country = request.form['country']
    location = request.form['location']

    # Encode the input
    country_encoded = country_encoder.transform([country])[0]
    location_encoded = location_encoder.transform([location])[0]

    # Scale the input
    input_data = np.array([[temperature, location_encoded, country_encoded]])
    input_scaled = scaler.transform(input_data)
    input_reshaped = np.reshape(input_scaled, (1, 1, 3))

    # Make prediction
    prediction = model.predict(input_reshaped)[0][0]

    return jsonify({
        'prediction': float(prediction)
    })

@app.route('/get_world_data')
def get_world_data():
    # This is a placeholder. You'll need to implement this function
    # to return the actual data from your dataset.
    data = {
        'countries': ['United States', 'China', 'India', 'Russia', 'Japan'],
        'emissions': [5000000, 10000000, 2500000, 1750000, 1250000]
    }
    return jsonify(data)

# Energy
@app.route('/predictenergy', methods=['POST'])
def predictenergy():
    data = request.json
    temperature = data['temperature']
    humidity = data['humidity']
    hour = data['hour']
    day = data['day']
    month = data['month']
    
    # Generate predictions for each month
    monthly_data = []
    for m in range(1, 13):
        input_data = [temperature, humidity, hour, day, m]
        scaled_input = scalerEC.transform([input_data])
        prediction = modelEC.predict(scaled_input)[0]
        monthly_data.append(prediction)
    
    # Use the prediction for the current month
    current_prediction = monthly_data[month - 1]
    category = categorize_consumption(current_prediction)
    
    return jsonify({
        'predicted_consumption': current_prediction,
        'category': category,
        'monthly_data': monthly_data
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
