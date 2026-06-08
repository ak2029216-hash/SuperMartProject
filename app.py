from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the newly trained Machine Learning model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/result', methods=['POST'])
def result():
    # 1. Capture all the data from the HTML form
    input_data = {
        'weight': [float(request.form['weight'])],
        'mrp': [float(request.form['mrp'])],
        'visibility': [float(request.form['visibility'])],
        'fat_content': [request.form['fat_content']],
        'item_type': [request.form['item_type']],
        'year': [int(request.form['year'])],
        'outlet_size': [request.form['outlet_size']],
        'location_type': [request.form['location_type']],
        'outlet_type': [request.form['outlet_type']]
    }

    # 2. Convert it into a Pandas DataFrame (which the model expects)
    df = pd.DataFrame(input_data)

    # 3. Ask the model to predict the sales
    prediction = model.predict(df)[0]

    # Ensure prediction is positive and rounded
    final_sales = max(0, round(prediction, 2))

    return render_template('result.html', prediction=final_sales)

if __name__ == '__main__':
    app.run(debug=True)
    