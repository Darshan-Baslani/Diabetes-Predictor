from flask import Flask, render_template, request, app
from flask_cors import cross_origin
import pickle

appliction = Flask(__name__)
app = appliction

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        try:
            return pickle.load(f)
        except EOFError:
            # The pickle file is empty or corrupt
            return None

scaler = pickle.load(open('D:/Diabetes_Detection/Models/Scaler.pkl', 'rb'))
regressor = load_pickle('D:/Diabetes_Detection/Models/Regressor.pkl')


@app.route('/')
@cross_origin
def homepage():
    return render_template('index.html')


@app.route('/input', methods=['GET', 'POST'])
@cross_origin(endpoint='predict')
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
        regressor.close()
        # scaler.close()
        if prediction[0] == 0:
            return render_template('output.html', result='Non-Diabetic')
        else:
            return render_template('output.html', result='Diabetic')
    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run(port=4040)