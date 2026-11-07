# Titanic Survival Prediction using Python and SVC

A machine learning project that predicts whether a passenger survived the Titanic disaster using a Support Vector Classifier (SVC). The project covers data loading, exploration, categorical encoding, model training, and evaluation in a Jupyter Notebook.

## Project Overview

Using passenger details such as class, sex, age, family size, fare, and port of embarkation, this project builds a binary classification model that predicts survival (1) or non-survival (0).

## Dataset

- File: `titanic.csv`
- Target column: `Survived`
- Features used: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`

## Workflow

1. Load the dataset and check its shape and first rows
2. Encode the categorical columns (`Sex`, `Embarked`) using `LabelEncoder`
3. Select the features and the target (`Survived`)
4. Split the data into 80% training and 20% testing (`random_state=42`)
5. Train an SVC model with an RBF kernel
6. Predict on the test set and evaluate the model

## Model

- Algorithm: Support Vector Classifier (`SVC`)
- Kernel: RBF
- Library: scikit-learn

## Results

Evaluated on the 20% test set (84 passengers):

| Metric | Score |
|--------|-------|
| Accuracy | 60.71% |
| Precision | 66.67% |
| Recall | 5.88% |
| F1-Score | 10.81% |

Confusion matrix:

|  | Predicted: Not Survived | Predicted: Survived |
|--|--|--|
| **Actual: Not Survived** | 49 | 1 |
| **Actual: Survived** | 32 | 2 |

**Observation:** the model predicts "Not Survived" for almost every passenger, so accuracy looks acceptable but recall for survivors is very low. SVC is sensitive to feature scale, and the features here were not scaled.

**Possible improvements:** apply feature scaling (for example `StandardScaler`), tune hyperparameters (`C`, `gamma`) with `GridSearchCV`, handle class imbalance, and compare with other models such as Logistic Regression or Random Forest.

## Tech Stack

- Python
- Pandas and NumPy
- scikit-learn (SVC, LabelEncoder, train_test_split, metrics)
- Matplotlib and Seaborn
- Jupyter Notebook

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/shahidlaiba899-ops/titanic-survival-svc.git
   cd titanic-survival-svc
   ```
2. Install the required libraries:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. Open `solution.ipynb` in Jupyter Notebook or VS Code and run all cells.

## Author

**Laiba Shahid**
GitHub: [shahidlaiba899-ops](https://github.com/shahidlaiba899-ops)
