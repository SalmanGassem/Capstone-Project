from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# Define your Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS


def map_user_input_to_df(user_input):
    '''
    Map the user input to the same features of the model
    '''
    mapping = {
        'Car_Brands': {
            'Lexus': 'Car_Brands_Brand:Lexus',
            'Mercedes': 'Car_Brands_Brand:Mercedes',
            'Range Rover': 'Car_Brands_Brand:Range Rover'
        },
        'Car_Models': {
            'LX': 'Car_Models_LX',
            'Land Cruiser': 'Car_Models_Land Cruiser',
            'Range Rover': 'Car_Models_Range Rover',
            'S Class': 'Car_Models_S Class'
        },
        'Car_Gear_Types': {
            'Automatic': 'Car_Gear_Types_Automatic',
            'CVT': 'Car_Gear_Types_CVT'
        },
        'Car_Drivetrains': {
            'AWD': 'Car_Drivetrains_AWD',
            'Double (4x4)': 'Car_Drivetrains_Double (4x4)',
            'FWD': 'Car_Drivetrains_FWD'
        },
        'Car_Extensions': {
            '500': 'Car_Extensions_500'
        },
        'Car_Exterior_Colors': {
            'Black': 'Car_Exterior_Colors_Black'
        },
        'Car_Interior_Colors': {
            'Camel': 'Car_Interior_Colors_Camel',
            'Grey': 'Car_Interior_Colors_Grey'
        },
        'Car_Origins': {
            'Saudi': 'Car_Origins_Saudi'
        }
    }

    result = pd.DataFrame({
        'Car_Kilometers': [user_input['Car_Kilometers']],
        'Car_Engine_Sizes': [user_input['Car_Engine_Sizes']],
        'Car_Seat_Numbers': [user_input['Car_Seat_Numbers']],
        'Car_Models_LX': [False],
        'Car_Models_Land Cruiser': [False],
        'Car_Models_Range Rover': [False],
        'Car_Models_S Class': [False],
        'Car_Brands_Brand:Lexus': [False],
        'Car_Brands_Brand:Mercedes': [False],
        'Car_Brands_Brand:Range Rover': [False],
        'Car_Gear_Types_Automatic': [False],
        'Car_Gear_Types_CVT': [False],
        'Car_Drivetrains_AWD': [False],
        'Car_Drivetrains_Double (4x4)': [False],
        'Car_Drivetrains_FWD': [False],
        'Car_Extensions_500': [False],
        'Car_Exterior_Colors_Black': [False],
        'Car_Interior_Colors_Camel': [False],
        'Car_Interior_Colors_Grey': [False],
        'Car_Origins_Saudi': [False]
    })

    for key, value in user_input.items():
        if key in mapping:
            if value in mapping[key]:
                result[mapping[key][value]] = True

    return result


def scale(df):
    '''
    Scale numeric data only!
    '''
    scaler = joblib.load('Scaler_SA.joblib')
    numeric_columns = ['Car_Kilometers', 'Car_Engine_Sizes', 'Car_Seat_Numbers']
    scaled_numeric = scaler.transform(df[numeric_columns])
    scaled_df = pd.DataFrame(scaled_numeric, columns=numeric_columns,
                             index=df.index)
    df[numeric_columns] = scaled_df
    return df


def predict_ksa(df):
    model = joblib.load('Knn_SA.joblib')
    scaled_data = pd.read_csv('Data.csv')
    if 'Unnamed: 0' in scaled_data.columns:
        scaled_data.drop(columns='Unnamed: 0', inplace=True)
    values = scaled_data.values
    distance = cosine_distances(df, values)
    return model.predict(distance)


@app.route('/predict/ksa', methods=['POST'])
def predict_car_price():
    data = request.json

    # Check for missing data
    required_fields = [
        'Car_Brands', 'Car_Models', 'Car_Years', 'Car_Kilometers',
        'Car_Fuel_Types', 'Car_Gear_Types', 'Car_Engine_Sizes',
        'Car_Drivetrains', 'Car_Extensions', 'Car_Exterior_Colors',
        'Car_Interior_Colors', 'Car_Seat_Numbers', 'Car_Origins'
    ]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'error': f'Missing fields: {missing_fields}'}), 400

    try:
        user_input = data
        df = map_user_input_to_df(user_input)
        df = scale(df)
        prediction = predict_ksa(df)
        return jsonify({'Predicted_Price': prediction[0]})
    except Exception as e:
        print(f"Error processing request: {e}")  # Log the error for debugging
        return jsonify({'error': str(e)}), 500


# Test endpoint to ensure API is running
@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({"message": "API is working!"})


if __name__ == '__main__':
    app.run(debug=True)
