# ML_Project-Titanic_Survival_Prediction 🌊🚢

This project explores the task of predicting the survival of passengers aboard the Titanic using a **Logistic Regression** model. By analyzing various passenger attributes like age, gender, and class, the model aims to predict whether a passenger survived or not.

## Data
This directory contains the dataset (`train.csv`) used for the project. The dataset includes the following features:

- **PassengerId**: Unique ID of the passenger.
- **Pclass**: Passenger class (1st, 2nd, or 3rd).
- **Name**: Passenger’s name.
- **Sex**: Gender of the passenger.
- **Age**: Age of the passenger.
- **SibSp**: Number of siblings/spouses aboard the Titanic.
- **Parch**: Number of parents/children aboard the Titanic.
- **Ticket**: Ticket number.
- **Fare**: Amount of money paid for the ticket.
- **Cabin**: Cabin number.
- **Embarked**: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).
- **Survived**: Target variable (1 indicates survived, 0 indicates did not survive).

> **Note:** You may need to adjust the dataset features based on your specific project requirements.

## Notebooks
This directory contains the Jupyter Notebook (`Titanic_Survival_Prediction_using_Machine_Learning.ipynb`) that guides you through the entire process of data exploration, preprocessing, model training, evaluation, and visualization.

## Running the Project
The Jupyter Notebook (`Titanic_Survival_Prediction_using_Machine_Learning.ipynb`) walks through the following steps:

### Data Loading and Exploration:
- Load the dataset and explore basic statistics.
- Visualize relationships between features and the target variable (`Survived`).

### Data Preprocessing:
- Handle missing values (e.g., for **Age** or **Cabin**).
- Scale numerical features like **Age** and **Fare**.
- Encode categorical variables like **Sex**, **Embarked**, and **Pclass**.

### Train-Test Split:
- The data is split into training and testing sets using `train_test_split` from the `sklearn` library, with a typical 80-20 or 70-30 ratio for training and testing, respectively.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Feature Engineering (Optional):
- Create additional features (e.g., family size by combining **SibSp** and **Parch**).
- Analyze correlations between features and the target variable.

### Model Training:
- Trains the model using **Logistic Regression**, potentially tuning hyperparameters for improved performance.

### Model Evaluation:
- Evaluates model performance using metrics like accuracy, precision, recall, and F1-score.

### Visualization of Results:
- Analyze the confusion matrix to understand model performance on different categories.
- Visualize feature importance to explore the impact of specific features on model predictions.

## Customization
Modify the Jupyter Notebook to:
- Experiment with different preprocessing techniques and feature engineering methods.
- Try other classification algorithms for comparison (e.g., **Random Forest**, **SVM**, **K-Nearest Neighbors**).
- Explore advanced techniques like ensemble methods or neural networks.

## Resources
- Sklearn Documentation: [https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Kaggle Titanic Dataset: [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)

## Further Contributions
Extend this project by:
- Incorporating additional data (e.g., passenger demographics) for improved prediction.
- Implementing a real-time Titanic survival prediction system using a trained model and an API.
- Exploring explainability techniques to understand the reasoning behind the model's predictions.

By leveraging machine learning models, specifically **Logistic Regression**, and Titanic passenger data, we aim to develop a reliable method for predicting survival. This project lays the foundation for further exploration into classification-based machine learning applications.
