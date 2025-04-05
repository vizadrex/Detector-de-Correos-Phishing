from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Cargar modelo y vectorizador
model = joblib.load('model/spam_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

def predict_spam(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector)
    proba = model.predict_proba(vector)[0][1] if prediction[0] == 1 else model.predict_proba(vector)[0][0]
    return 'PHISHING' if prediction[0] == 1 else 'LEGÍTIMO', round(proba*100, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form.get('text', '')
        if text:
            prediction, confidence = predict_spam(text)
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'text_preview': text[:500] + '...' if len(text) > 500 else text
            }
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)