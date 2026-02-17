import re
from urllib.parse import urlparse
import pandas as pd
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt


df1 = pd.read_csv("malicious_phish.csv")
print(df1.head())
df1 = df1[df1['type'].isin(['phishing', 'benign'])]
df1['type'].replace({'phishing': 0, 'benign': 1}, inplace=True)
print('count', df1['type'].value_counts())
print(df1.head())

def extract_features(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    return {
        "NumDots": url.count('.'),
        "SubdomainLevel": hostname.count('.') - 1,
        "PathLevel": path.count('/'),
        "UrlLength": len(url),
        "NumDash": url.count('-'),
        "NumDashInHostname": hostname.count('-'),
        "AtSymbol": '@' in url,
        "TildeSymbol": '~' in url,
        "NumUnderscore": url.count('_'),
        "NumPercent": url.count('%'),
        "NumQueryComponents": query.count('='),
        "NumAmpersand": url.count('&'),
        "NumHash": url.count('#'),
        "NumNumericChars": sum(c.isdigit() for c in url),
        "NoHttps": not url.startswith("https"),
        "RandomString": bool(re.search(r"[a-zA-Z]{5,}", hostname)),
        "RandomString": bool(re.search(r"[a-zA-Z]{10,}", hostname.replace('.', ''))),
        "IpAddress": bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname)),
        "DomainInSubdomain": hostname.count('.') > 2,
        "DomainInPath": hostname.split('.')[0] in path,
        "HttpsInHostname": 'https' in hostname,
        "HostnameLength": len(hostname),
        "PathLength": len(path),
        "QueryLength": len(query),
        "DoubleSlashInPath": '//' in path,
        "NumSensitiveWords": sum(word in url.lower() for word in ["login", "signin", "bank", "update", "free", "lucky", "click"]),
        "EmbeddedBrandName": any(brand in url.lower() for brand in ["paypal", "apple", "google", "amazon", "facebook"]),
    }

features = df1['url'].apply(extract_features)
df_input = pd.DataFrame(features.tolist())

results = []

def predict_and_evaluate(model):
    model_name = f"{model.__class__.__name__}.pkl"
    model = joblib.load(model_name)
    predictions = model.predict(df_input)

    count_phishing_true = sum(pred == 0 and df == 0 for pred, df in zip(predictions, df1['type']))
    count_phishing_false = sum(pred == 0 and df == 1 for pred, df in zip(predictions, df1['type']))
    count_legitimate_true = sum(pred == 1 and df == 1 for pred, df in zip(predictions, df1['type']))
    count_legitimate_false = sum(pred == 1 and df == 0 for pred, df in zip(predictions, df1['type']))

    accuracy = (count_phishing_true + count_legitimate_true) / len(predictions)
    precision_phish = count_phishing_true / (count_phishing_true + count_phishing_false) if (count_phishing_true + count_phishing_false) else 0
    recall_phish = count_phishing_true / (count_phishing_true + count_legitimate_false) if (count_phishing_true + count_legitimate_false) else 0
    precision_legit = count_legitimate_true / (count_legitimate_true + count_legitimate_false) if (count_legitimate_true + count_legitimate_false) else 0
    recall_legit = count_legitimate_true / (count_legitimate_true + count_phishing_false) if (count_legitimate_true + count_phishing_false) else 0

    results.append({
        "Model": model.__class__.__name__,
        "Accuracy": round(accuracy, 4),
        "Precision (Phishing)": round(precision_phish, 4),
        "Recall (Phishing)": round(recall_phish, 4),
        "Precision (Legitimate)": round(precision_legit, 4),
        "Recall (Legitimate)": round(recall_legit, 4)
    })

for m in [
    LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier(),
    SVC(), DecisionTreeClassifier(), GradientBoostingClassifier(),
    AdaBoostClassifier(), BaggingClassifier(), ExtraTreesClassifier(),
    CatBoostClassifier(verbose=0), HistGradientBoostingClassifier(),
    GaussianNB(), BernoulliNB(), RidgeClassifier(),
    PassiveAggressiveClassifier(), Perceptron(), SGDClassifier()
]:
    predict_and_evaluate(m)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)
print(results_df)

results_df.plot(x="Model", y=["Accuracy", "Precision (Phishing)", "Recall (Phishing)", "Precision (Legitimate)", "Recall (Legitimate)"], kind="bar", figsize=(12,6))
plt.title("Model Performance Comparison")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()