from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from data_prep import create_df_text
from model_prep import prep_text_data_TFIDF

def get_X_y(df):
    df = df
    X = df['Consumer complaint narrative'].tolist()
    y = df['Company response to consumer'].tolist()
    return X, y

def lemmatize_descriptions(X):
    lem = WordNetLemmatizer()
    lemmatize = lambda d: " ".join(lem.lemmatize(word) for word in d.split())
    return [lemmatize(desc) for desc in X]


def get_vectorizer(X, num_features=5000):
    vect = TfidfVectorizer(max_features=num_features, stop_words='english')
    return vect.fit(X)

def compare_models(X, y, models):
    desc_train, desc_test, y_train, y_test = train_test_split(X, y)

def run_model(Model, X_train, X_test, y_train, y_test):
    m = Model()
    m.fit(X_train, y_train)
    y_predict = m.predict(X_test)
    return accuracy_score(y_test, y_predict), \
        f1_score(y_test, y_predict, average='weighted'), \
        precision_score(y_test, y_predict, average='weighted'), \
        recall_score(y_test, y_predict, average='weighted')



def compare_models(descriptions, labels, models):
    desc_train, desc_test, y_train, y_test = \
        train_test_split(descriptions, labels)

    print "-----------------------------"
    print "Without Lemmatization:"
    run_test(models, desc_train, desc_test, y_train, y_test)

    print "-----------------------------"
    print "With Lemmatization:"
    run_test(models, lemmatize_descriptions(desc_train),
             lemmatize_descriptions(desc_test), y_train, y_test)

    print "-----------------------------"


def run_test(models, desc_train, desc_test, y_train, y_test):
    vect = get_vectorizer(desc_train)
    X_train = vect.transform(desc_train).toarray()
    X_test = vect.transform(desc_test).toarray()

    print "acc\tf1\tprec\trecall"
    for Model in models:
        name = Model.__name__
        acc, f1, prec, rec = run_model(Model, X_train, X_test, y_train, y_test)
        print "%.4f\t%.4f\t%.4f\t%.4f\t%s" % (acc, f1, prec, rec, name)


if __name__ == '__main__':
    df = pd.read_csv('../data/Consumer_Complaints_with_Consumer_Complaint_Narratives.csv')
    df = create_df_text(df)                      # For text data
    print df
    X, y = get_X_y(df)
    # print len(X)
    # print len(y)

    print "distribution of labels:"
    for i, count in enumerate(np.bincount(y)):
        print "%d: %d" % (i, count)

    models = [LogisticRegression, KNeighborsClassifier, MultinomialNB,
              RandomForestClassifier]
    compare_models(X, y, models)
