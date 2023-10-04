from flask import Flask, render_template, request, app
from flask_cors import cross_origin
import pickle

application = Flask(__name__)
app = application


scaler = pickle.load(open('D:/Diabetes_Detection/Models/Scaler.pkl', 'rb'))
regressor = pickle.load(open('D:/Diabetes_Detection/Models/Regressor.pkl', 'rb'))

result = ''
@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        BloodPressure = float(request.form.get('BloodPressure'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        
        scaled_data = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,
                                         Insulin,BMI,DiabetesPedigreeFunction,Age]])
        prediction = regressor.predict(scaled_data)
        
        print(prediction)
        if prediction[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'
            
        return render_template('output.html',result=result)
    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run(port=4040)