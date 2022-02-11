from flask import Flask, render_template, request

import pickle

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

nltk.download("stopwords")

ps = PorterStemmer()

model = pickle.load(open("xgb_fake_news_predictor.pkl", 'rb'))

def preprocess_news(news):
    p_news = re.sub('[^a-zA-Z]', ' ', news)
    p_news = p_news.lower().split()
    p_news = [ps.stem(word) for word in p_news if word not in stopwords.words('english')]
    p_news = ' '.join(p_news)
    return p_news

@app.route("/", methods = ['GET', 'POST'])
def home():
    fake_flag = False
    non_fake_flag = False
    danger = False
    message = ""
    try:
        if request.method == 'POST':
            dic = request.form.to_dict()
            news = dic['news']
            if len(news) == 0:
                raise Exception
            news = preprocess_news(news)
            prediction = model.predict([news])
            probability = model.predict_proba([news])
            if prediction[0] == 1:
                fake_flag = True
                message = f"This NEWS is predicted as FAKE NEWS with {round(max(probability[0])*100, 2)}% accuracy"
            else:
                non_fake_flag = True
                message = f"This NEWS is predicted as REAL NEWS with {round(max(probability[0])*100, 2)}% accuracy"

    except:
        danger = True
        message = "Please enter some text"
    return render_template("home.html", fake_flag = fake_flag, non_fake_flag = non_fake_flag, message = message, danger = danger)

if __name__ == '__main__':
    app.run(debug = True)