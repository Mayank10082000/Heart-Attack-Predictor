from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

app = Flask(__name__)

# Load the trained model
with open('C:/Users/Mayank/Desktop/(Done) Heart Attack Predictor Website/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Preprocessing functions
def preprocess_input(df):
    ohe = OneHotEncoder()
    scaler = MinMaxScaler()

    categorical_columns = ['Gender', 'prevalentStroke', 'prevalentHyp', 'currentSmoker', 'BPMeds', 'diabetes']
    numerical_columns = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate']

    for column in categorical_columns:
        df[column] = ohe.fit_transform(df[[column]]).toarray()

    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve user input from the form
    gender = request.form['gender']
    age = int(request.form['age'])
    currentsmoker = request.form['currentsmoker']
    cigsperday = float(request.form['cigsperday'])
    bpmeds = request.form['bpmeds']
    prevalentstroke = request.form['prevalentstroke']
    prevalenthyp = request.form['prevalenthyp']
    diabetes = request.form['diabetes']
    totchol = float(request.form['totcol'])
    sysbp = float(request.form['sysbp'])
    diabp = float(request.form['diabp'])
    BMI = float(request.form['BMI'])
    heartrate = float(request.form['heartrate'])

    # Create a dataframe from the user input
    data = {
        'Gender': [gender],
        'age': [age],
        'currentSmoker': [currentsmoker],
        'cigsPerDay': [cigsperday],
        'BPMeds': [bpmeds],
        'prevalentStroke': [prevalentstroke],
        'prevalentHyp': [prevalenthyp],
        'diabetes': [diabetes],
        'totChol': [totchol],
        'sysBP': [sysbp],
        'diaBP': [diabp],
        'BMI': [BMI],
        'heartRate': [heartrate]
    }

    df = pd.DataFrame(data)

    # Preprocess the input data
    df_preprocessed = preprocess_input(df)

    # Perform prediction
    prediction = model.predict(df_preprocessed)

    # Convert prediction to a meaningful result
    if prediction == 0:
        result = 'No risk of heart stroke'
    else:
        result = 'Risk of heart stroke'

    # Return the prediction result to the user
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)