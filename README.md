# telegram-bot-for-malicious-url-detection

## Technologies

Python-based ML + bot stack:
* aiogram — Telegram bot framework and user interaction.
* scikit-learn — model training, testing, classification.
* joblib — saving/loading trained .pkl models.
* pandas & numpy — dataset loading, feature processing.

Architecture: modular client-server. Telegram = client interface, Python app = backend with ML logic.

## Implementation

### Dataset and training
Dataset is cleaned, shuffled, and split: train 60%, validation 20%, test 20%.
StandardScaler normalizes features for stable training.
Step-by-step procedure:
* Train model on scaled data.
* Predict on test set.
* Evaluate (precision, recall, F1).
* Export trained model to .pkl.

### Model selection
Multiple ML models were trained and evaluated in the same way.
Saved models were tested on a separate phishing/benign dataset using the same feature extraction.

Metrics calculated:
* accuracy
* precision/recall (phishing and legitimate)
Results are stored, ranked by accuracy, and visualized.
DecisionTreeClassifier was selected due to:
* strong accuracy-recall balance for phishing detection
* fewer false negatives
* fast and lightweight for real-time bot use

### Feature extraction and prediction
The extract_features() function parses URLs and calculates 25 engineered features including structural elements (dots, dashes, path length), special characters, security indicators, and content analysis.

Prediction workflow:
* extract features
* convert to DataFrame
* load trained model
* return classification

The trained phishing detection model was integrated into a Telegram bot using the aiogram framework. The main handler processes incoming URLs by integrating with the prediction module. It alls the main() function from the prediction module, which loads the DecisionTreeClassifier model. It also catches type errors with a clear message about expected input format and ses asynchronous architecture for responsive user experience.
