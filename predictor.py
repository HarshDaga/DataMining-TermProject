import string

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords')

model_svm = joblib.load(r'svm.pkl')
model_svm_balanced = joblib.load(r'svm_balanced.pkl')
model_final = joblib.load(r'decision_tree.pkl')

stopwords_set = set(stopwords.words('english'))
stopwords_set = stopwords_set.union((
                                    'game', 'play', 'played', 'players', 'player', 'people', 'really', 'board', 'games',
                                    'one', 'plays', 'cards', 'would'))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df['cleaned'] = df['comment'].str.lower().apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
    df['cleaned'] = df['cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords_set]))
    return df


def clean_str(review: str) -> str:
    cleaned = ''.join([i for i in review if i not in string.punctuation])
    cleaned = ' '.join([word for word in cleaned.split() if word not in stopwords_set])
    return cleaned


def predict(review: str):
    cleaned = clean_str(review)
    balanced = model_svm_balanced.predict([cleaned])[0]
    unbalanced = model_svm.predict([cleaned])[0]
    y_pred = model_final.predict([(balanced, unbalanced)])[0]
    return y_pred
