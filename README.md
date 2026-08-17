# Online Shopper Purchase Intention Prediction - ML Assignment 2

## a. Problem Statement

E-commerce businesses need to understand which browsing sessions are likely to result in a purchase. This project frames that requirement as a **binary classification problem**: using the behavioral and technical attributes of a website session, predict whether it will end in a purchase (`Revenue = True`) or no purchase (`Revenue = False`).

Five machine learning classification models are trained on the same dataset, evaluated using six performance metrics, and demonstrated through an interactive Streamlit web application called **ShopperSense**.

## b. Dataset Description

- **Name:** Online Shoppers Purchasing Intention Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Citation:** Sakar, C. & Kastro, Y. (2018), DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)
- **Instances:** 12,330 website sessions
- **Predictor features:** 17
- **Target variable:** `Revenue` (`True` or `False`)
- **Classification type:** Binary classification
- **Missing values:** None

The UCI source describes the complete dataset as containing 10 numerical and 8 categorical attributes. Because Revenue is one of the eight categorical attributes and is used as the class label, the model uses 10 numerical and 7 categorical/Boolean predictors, for a total of 17 predictor features.

- **Numerical:** `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`
- **Categorical/Boolean:** `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`

The target distribution consists of 10,422 non-purchasing sessions and 1,908 purchasing sessions. Approximately 15.5% of the sessions belong to the positive purchase class, making the dataset moderately imbalanced.

### Data Preprocessing

- Numerical features are standardized using `StandardScaler`.
- Categorical features are transformed using `OneHotEncoder`.
- Preprocessing and classification are combined in a scikit-learn `Pipeline` for every model.
- The dataset is divided into 80% training data and 20% testing data.
- The split is stratified on `Revenue` and uses `random_state=42` for reproducibility.
- The held-out test set contains 2,466 rows and is saved as `test_data.csv`.

## c. GitHub Repository Link

[View the GitHub repository](https://github.com/Adisha-13/online-shopper-purchase-prediction)

## Live Streamlit Application

[Open the deployed ShopperSense application](https://adisha-13-online-shopper-purchase-prediction-app-q5kyfy.streamlit.app/)

## d. Models Used

The following five classification models were trained using the same training data and evaluated using the same held-out test data:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8808 | 0.8881 | **0.7418** | 0.3534 | 0.4787 | 0.4579 |
| Decision Tree | 0.8917 | 0.8697 | 0.6849 | 0.5576 | 0.6147 | 0.5565 |
| kNN | 0.8751 | 0.8184 | 0.6850 | 0.3586 | 0.4708 | 0.4353 |
| Naive Bayes | 0.6719 | 0.7944 | 0.2917 | **0.7827** | 0.4250 | 0.3189 |
| Random Forest (Ensemble) | **0.8982** | **0.9197** | 0.7266 | 0.5497 | **0.6259** | **0.5757** |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression achieved a strong AUC of 0.8881 and the highest precision of 0.7418. However, its low recall of 0.3534 indicates that it missed many actual purchasing sessions. It provides a useful linear baseline but does not capture all the non-linear relationships in the data. |
| Decision Tree | The Decision Tree achieved good accuracy of 0.8917, recall of 0.5576, F1 of 0.6147, and MCC of 0.5565. Its recall was slightly higher than Random Forest, showing that it identified more actual purchasing sessions, although its overall AUC and F1 were lower. |
| kNN | kNN achieved reasonable accuracy and precision but relatively low recall of 0.3586. Its performance may be affected by the combination of numerical and one-hot encoded categorical features, which makes distance-based classification more challenging. |
| Naive Bayes | Naive Bayes achieved the highest recall of 0.7827, meaning that it identified most purchasing sessions. However, its low precision of 0.2917 and accuracy of 0.6719 indicate that it also produced many false-positive purchase predictions. |
| Random Forest (Ensemble) | Random Forest achieved the highest accuracy of 0.8982, AUC of 0.9197, F1 of 0.6259, and MCC of 0.5757. It provided the strongest overall classification performance and the best balance across the evaluation metrics. |
| **Overall Winner for the Dataset** | **Random Forest (Ensemble)** is the overall winner because it achieved the highest Accuracy, AUC, F1, and MCC among all five models. |

The metrics were generated by `model/train_models.py` using the same stratified test split for every model. AUC was calculated using predicted class probabilities.

## Repository Structure

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- data/
|   `-- online_shoppers_intention.csv
`-- model/
    |-- train_models.py
    |-- metrics_summary.csv
    `-- saved_models/
        |-- logistic_regression.pkl
        |-- decision_tree.pkl
        |-- knn.pkl
        |-- naive_bayes.pkl
        `-- random_forest_ensemble.pkl
```

## How to Run Locally

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 2. Install the required packages

```bash
python -m pip install -r requirements.txt
```

### 3. Optionally retrain the models

```bash
python model/train_models.py
```

This recreates the saved model pipelines, `metrics_summary.csv`, and `test_data.csv`.

### 4. Start the Streamlit application

```bash
python -m streamlit run app.py
```

## Streamlit Application Features

- CSV test-data upload facility
- Bundled `test_data.csv` option
- Model-selection dropdown containing all five classifiers
- Display of Accuracy, AUC, Precision, Recall, F1, and MCC
- Confusion matrix for the selected model
- Classification report
- Row-level actual and predicted results
- Purchase probabilities
- Customized animated shopping interface

## Deployment on Streamlit Community Cloud

1. Push the complete project to a public GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using GitHub.
4. Create a new application and select the repository.
5. Select the `main` branch.
6. Set the main file path to `app.py`.
7. Deploy the application.

## Author

**Aditi Vithob Shanbhag**  
M.Tech Artificial Intelligence and Machine Learning  
BITS Pilani WILP
