from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model, label_encoder = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil input dari form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Prediksi
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    output = label_encoder.inverse_transform(prediction)[0]

    # Kirim hasil dan input ke HTML
    return render_template(
        'index.html',
        prediction_text=f"Hasil Prediksi : {output}",
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width
    )

if __name__ == '__main__':
    app.run(debug=True)