# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('main.html')


@app.route('/info')
def info():
    return render_template('Information.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        sex = request.form.get('sex')
        age = int(request.form['age'])
        education = request.form.get('education')
        smoker = request.form.get('smoker')
        cpd = int(request.form['cpd'])
        bpm = request.form.get('bpm')
        prevalentstroke = request.form.get('prevalentStroke')
        prevalentHyperextention = request.form.get('prevalentHyperextention')
        diabetes = request.form.get('diabetes')
        chol = int(request.form['chol'])
        sysBP = int(request.form['sysBP'])
        diaBP = int(request.form['diaBP'])
        
        bmi = float(request.form['BMI'])
        heartrate = int(request.form['HR'])
        glucose = int(request.form['glucose'])
        print(sex,age,education,smoker,cpd,bpm,prevalentstroke,prevalentHyperextention,diabetes,chol,sysBP,diaBP,bmi,heartrate,glucose)
        data = np.array([[sex,age,education,smoker,cpd,bpm,prevalentstroke,prevalentHyperextention,diabetes,chol,sysBP,diaBP,bmi,heartrate,glucose]])
        my_prediction =  model.predict_proba(data)
        print(str(my_prediction))
        
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=True)

