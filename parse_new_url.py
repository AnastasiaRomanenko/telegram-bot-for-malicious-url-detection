import re
from urllib.parse import urlparse
import pandas as pd
import joblib
import pandas as pd


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
    
def main(url):

    url = url
    features = extract_features(url)
    df_input = pd.DataFrame(features, index=[0])
    model_name = "DecisionTreeClassifier.pkl"
    model = joblib.load(model_name)

    predictions = model.predict(df_input)

    return f"The URL is classified as: {'Benign' if predictions[0] == 1 else 'Phishing'}"