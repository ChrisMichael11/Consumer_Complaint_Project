#  For TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
#  Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
#  Crossvalidation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.metrics import recall_score, roc_curve, auc, roc_auc_score
#  Data Wrangling
import pandas as pd
import numpy as np

#  Plotting
import matplotlib.pyplot as plt

#  Import from data processing script
from data_prep import create_df_text
from model_prep import prep_text_data_TFIDF


def get_X_y(df):
    """
    Split dataframe into X (text) and y (labels)
    """
    X = df['Consumer complaint narrative'].tolist()
    y = df['Company response to consumer'].tolist()

    return X, y

def get_vectorizer(X, num_features=1000):
    """
    Vectorize text
    """
    vect = TfidfVectorizer(max_features=num_features, stop_words='english')
    return vect.fit(X)

def run_model(Model, X_train, X_test, y_train, y_test):
    """
    Run each model, produce scores and other data
    INPUT - model name, train/test data
    OUTPUT - accuracy, F1, precision, recall, AUC, predict_probs (for ROC plotting)
    """
    m = Model()
    m.fit(X_train, y_train)
    y_predict = m.predict(X_test)
    predict_probs = m.predict_proba(X_test)[:,1]
    # 0 = False, 1 = True.  If choose 0, ROC will be inverted
    roc_curve(y_test, y_predict)
    return accuracy_score(y_test, y_predict), \
           f1_score(y_test, y_predict), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict), \
           roc_auc_score(y_test, predict_probs),\
           predict_probs

def compare_models(descriptions, labels, models):
    """
    Run models (with run_model function), produce scores (Accuracy, precision, recall, F1, AUC) and
    plot ROC Curve.

    Function produces train/test split data.  Test size and random state set for repeatability
    """
    desc_train, desc_test, y_train, y_test = train_test_split(descriptions,
                                                              labels,
                                                              test_size=0.3,
                                                              random_state=11)

    print "=========================================================================="
    print "RUN SOME MODELS!!!"
    run_test(models, desc_train, desc_test, y_train, y_test)

    print "=========================================================================="

def run_test(models, desc_train, desc_test, y_train, y_test):
    """
    Run models in "main" statement and plot ROC curve for all models run.
    INPUT = model name, X_train, X_test, y_train, y_test
    OUTPUT = ROC Curve
    """
    vect = get_vectorizer(desc_train)
    X_train = vect.transform(desc_train).toarray()
    X_test = vect.transform(desc_test).toarray()

    print "acc\tf1(weighted)\tprec\trecall\tAUC_score"

    #  Empty lists and dicts for ROC curve creation
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fig = plt.figure(figsize=(10, 8))

    for Model in models:
        """Run model, calculate:
            Accuracy, F1 Score, Precision, Recall, AUC, Probas
        """
        name = Model.__name__
        acc, f1, prec, rec, roc, probs = run_model(Model,
                                                   X_train,
                                                   X_test,
                                                   y_train,
                                                   y_test)
        #  Print good looking format
        print "%.4f\t%.4f\t\t%.4f\t%.4f\t%.4f\t%s" % (acc,
                                                      f1,
                                                      prec,
                                                      rec,
                                                      roc,
                                                      name)

        # Plot ROC Curve
        fpr[name], tpr[name], _ = roc_curve(y_test,probs,drop_intermediate=False)
        roc_auc[name] = auc(fpr[name], tpr[name])

        plt.plot(fpr[name], tpr[name], label= '{0} (AUC = {1:0.2f})'
                                            ''.format(name, roc_auc[name]))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity, Recall)')
    plt.title('ROC Using Text Features')
    plt.legend(loc="lower right")

    plt.show()


if __name__ == '__main__':
    #  Import data from "data" folder
    df = pd.read_csv('../data/Consumer_Complaints_with_Consumer_Complaint_Narratives.csv')
    #  Create DataFrame
    df = create_df_text(df)
    #  Create X, y for modeling
    X, y = get_X_y(df)

    #  Print out some general info
    print "Distribution of Labels (No Relief Provided = 0, Relief Provided = 1):"
    for i, count in enumerate(np.bincount(y)):
        print "%d: %d" % (i, count)

    #  Models to run
    models = [LogisticRegression,
              KNeighborsClassifier,
              MultinomialNB,
              RandomForestClassifier,
              AdaBoostClassifier,
              GradientBoostingClassifier]
    #  Compare above models by producing scores and plotting ROC
    compare_models(X, y, models)
