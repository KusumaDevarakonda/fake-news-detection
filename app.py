from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)
history = []

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']

    cleaned_news = clean_text(news)

    vector_input = vectorizer.transform([cleaned_news])

    prediction = model.predict(vector_input)
    probability = model.predict_proba(vector_input)
    confidence = round(max(probability[0]) * 100, 2)
    
    if prediction[0] == 0:
        result = "Fake News"
    else:
        result = "Real News"
    history.append({
    'news': news,
    'result': result
    })

    return render_template(
    'index.html',
    prediction_text=result,
    confidence=confidence,
    history=history[-5:]
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)