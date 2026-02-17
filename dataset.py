import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import pickle


df = pd.read_csv(
    "Phishing_Legitimate_updated.csv",
)
df.drop(columns=["id"], inplace=True)
cols = df.columns.tolist()

print(df.head())
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y

train, X_train, y_train = scale_dataset(train)
valid, X_valid, y_valid = scale_dataset(valid)
test, X_test, y_test = scale_dataset(test)

def predict_and_evaluate(model): 
    print(model.__class__.__name__)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    with open(f"{model.__class__.__name__}.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

lg_model = predict_and_evaluate(LogisticRegression())
knn_model = predict_and_evaluate(KNeighborsClassifier())
rf_model = predict_and_evaluate(RandomForestClassifier())
svc_model = predict_and_evaluate(SVC())
dt_model = predict_and_evaluate(tree.DecisionTreeClassifier())
gb_model = predict_and_evaluate(GradientBoostingClassifier())
adb_model = predict_and_evaluate(AdaBoostClassifier())
bg_model = predict_and_evaluate(BaggingClassifier())
et_model = predict_and_evaluate(ExtraTreesClassifier())
cb_model = predict_and_evaluate(CatBoostClassifier(verbose=0))
hgb_model = predict_and_evaluate(HistGradientBoostingClassifier())
gnb_model = predict_and_evaluate(GaussianNB())
bnb_model = predict_and_evaluate(BernoulliNB())
rc_model = predict_and_evaluate(RidgeClassifier())
pa_model = predict_and_evaluate(PassiveAggressiveClassifier())
p_model = predict_and_evaluate(Perceptron())
sgd_model = predict_and_evaluate(SGDClassifier())


    