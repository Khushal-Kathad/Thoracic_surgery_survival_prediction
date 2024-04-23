from flask import Flask, render_template, request
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = load_model("C:\\Users\\91982\\Desktop\\Projects\\LIFE EXPECTANCY PREDICTION FOR POST THORACIC SURGERY - AI\\thoracic+surgery+data\\Thoracic_surgery_survival_prediction\\my_model.h5")

# Load the scaler for preprocessing input data
scaler = StandardScaler()
scaler.fit(pd.read_csv(
    "C:\\Users\\91982\\Desktop\\Projects\\LIFE EXPECTANCY PREDICTION FOR POST THORACIC SURGERY - AI\\thoracic+surgery+data\\Thoracic_surgery_survival_prediction\\Data Preprocessing\\thoracic_surgery.csv").drop(['Death_In_1yr', 'MI_6mo', 'Asthma'], axis=1))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [float(request.form[feat]) for feat in request.form]

    # Scale the features
    scaled_features = scaler.transform([features])

    # Make prediction
    prediction = model.predict(scaled_features)[0][0]
    predicted_class = 'Death' if prediction >= 0.5 else 'Live'

    return render_template('index.html', prediction_text='Predicted class: {}'.format(predicted_class))


if __name__ == "__main__":
    app.run(debug=True)
