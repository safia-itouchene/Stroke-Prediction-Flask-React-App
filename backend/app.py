from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from flask_cors import CORS


# Load the trained model
model = load('./stroke_prediction_model.joblib')

#initialize the flask app
app = Flask(__name__)
CORS(app)

#POSt mean we are wating somthing from the frontend
#the API will get the data that we POST from the fronend
@app.route('/predict',methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data) 
        df = pd.DataFrame([data]) # our ml model use the data as a datafreme
        prediction = model.predict(df)[0]
        print(f'prediction: {prediction}')
        return jsonify({"stroke":int(prediction)}), 200 #return the predction result as jison file to the frontend
    except Exception as e:
        print("Error:", str(e)) 
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return 'WellCome'

if __name__ =="__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)