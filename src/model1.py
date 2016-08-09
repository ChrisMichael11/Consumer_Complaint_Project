from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
    recall_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import ggplot

from data_prep import create_df_text
from model_prep import prep_text_data_TFIDF

from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

def get_X_y(df):
    df = df
    X = df['Consumer complaint narrative'].tolist()
    y = df['Company response to consumer'].tolist()

    # print "distribution of labels:"
    # for i, count in enumerate(np.bincount(y)):
    #     print "%d: %d" % (i, count)
    # US = RandomUnderSampler()
    # X, y = US.fit_sample(X.reshape(len(X),1), y.reshape(len(y),1))

    return X, y


# def balanced_sample_maker(X, y, sample_size, random_seed=11):
#     """ return a balanced data set by sampling all classes with sample_size
#     current version is developed on assumption that the positive
#     class is the minority.
#
#     Parameters:
#     ===========
#     X: {numpy.ndarrray}
#     y: {numpy.ndarray}
#     """
#
#     uniq_levels = np.unique(y)
#     uniq_counts = {level: sum(y == level) for level in uniq_levels}
#
#     if not random_seed is None:
#         np.random.seed(random_seed)
#
#     # find observation index of each class levels
#     groupby_levels = {}
#     for ii, level in enumerate(uniq_levels):
#         obs_idx = [idx for idx, val in enumerate(y) if val == level]
#         groupby_levels[level] = obs_idx
#     # oversampling on observations of each label
#     balanced_copy_idx = []
#     for gb_level, gb_idx in groupby_levels.iteritems():
#         over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
#         balanced_copy_idx+=over_sample_idx
#     np.random.shuffle(balanced_copy_idx)
#
#     return (X[balanced_copy_idx, :], y[balanced_copy_idx], balanced_copy_idx)


def lemmatize_descriptions(X):
    lem = WordNetLemmatizer()
    lemmatize = lambda d: " ".join(lem.lemmatize(word) for word in d.split())
    return [lemmatize(desc) for desc in X]


def get_vectorizer(X, num_features=5000):
    vect = TfidfVectorizer(max_features=num_features, stop_words='english')
    return vect.fit(X)

# def compare_models(X, y, models):
#     desc_train, desc_test, y_train, y_test = train_test_split(X, y,
#                                                               test_size=0.3,
#                                                               random_state=11)
#
#     US = RandomUnderSampler()
#     desc_train, y_train = US.fit_sample(desc_train, y_train)


def run_model(Model, X_train, X_test, y_train, y_test):
    m = Model()
    m.fit(X_train, y_train)
    y_predict = m.predict(X_test)
    roc_curve(y_test, y_predict)
    return accuracy_score(y_test, y_predict), \
           f1_score(y_test, y_predict), \
           precision_score(y_test, y_predict), \
           recall_score(y_test, y_predict), \
           roc_auc_score(y_test, y_predict)

def compare_models(descriptions, labels, models):
    desc_train, desc_test, y_train, y_test = train_test_split(descriptions,
                                                              labels,
                                                              test_size=0.3,
                                                              random_state=11)


    # print "-----------------------------"
    # print "Without Lemmatization:"
    # run_test(models, desc_train, desc_test, y_train, y_test)

    print "-----------------------------"
    print "With Lemmatization:"
    run_test(models, lemmatize_descriptions(desc_train),
             lemmatize_descriptions(desc_test), y_train, y_test)

    print "-----------------------------"

    # Lemmatization doesn't seem to have any appreciable effect.

def run_test(models, desc_train, desc_test, y_train, y_test):
    vect = get_vectorizer(desc_train)
    X_train = vect.transform(desc_train).toarray()
    X_test = vect.transform(desc_test).toarray()

    print "acc\tf1(weighted)\tprec\trecall\tAUC_score"
    for Model in models:
        name = Model.__name__
        acc, f1, prec, rec, roc = run_model(Model, X_train, X_test, y_train, y_test)
        print "%.4f\t%.4f\t\t%.4f\t%.4f\t%.4f\t%s" % (acc, f1, prec, rec, roc, name)


        # # ROC Curve
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # for i in range(len(models)):
        #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i],y_predict[:, i],drop_intermediate=False)
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        #
        # fig = plt.figure(figsize=(10, 8))
        #
        # label = ['Closed w/o Relief', 'Closed w/relief']  #'Closed w/explaination',
        # for i,v in enumerate(label):
        #     plt.plot(fpr[i], tpr[i], label= v + ' (auc_area = {1:0.2f})'
        #                                    ''.format(i, roc_auc[i]))
        #
        # plt.plot([0, 1], [0, 1], 'k--')
        #
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate (1 - Specificity)')
        # plt.ylabel('True Positive Rate (Sensitivity, Recall)')
        # plt.title('ROC with non-text Features')
        # plt.legend(loc="lower right")
        #
        #
        # plt.show()




if __name__ == '__main__':
    df = pd.read_csv('../data/Consumer_Complaints_with_Consumer_Complaint_Narratives.csv')
    df = create_df_text(df)                      # For text data
    print df.head()
    X, y = get_X_y(df)
    # X , y = balanced_sample_maker(X, y, 2500)
    # print len(X)
    # print len(y)

    print "distribution of labels:"
    for i, count in enumerate(np.bincount(y)):
        print "%d: %d" % (i, count)

    models = [LogisticRegression, KNeighborsClassifier, MultinomialNB,
              RandomForestClassifier]
    compare_models(X, y, models)
