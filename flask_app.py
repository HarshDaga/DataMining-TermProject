import predictor
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    review_text = request.form['review']
    cleaned = predictor.clean_str(review_text)
    rating = predictor.predict(review_text)

    return render_template('prediction.html', review=review_text, cleaned=cleaned, rating=rating)


if __name__ == "__main__":
    app.run()