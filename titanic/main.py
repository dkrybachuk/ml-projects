import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Configuration ---
INPUT_DIR = Path("input/")
OUTPUT_DIR = Path("output/")
TRAIN_FILE = INPUT_DIR / "train.csv"
TEST_FILE = INPUT_DIR / "test.csv"
SUBMISSION_FILE = OUTPUT_DIR / "predictions.csv"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Custom Transformers for Feature Engineering ---

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selects specified columns from a DataFrame."""
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.feature_names]

class TitleExtractor(BaseEstimator, TransformerMixin):
    """Extracts titles from the 'Name' column."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['Title'] = X_copy['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        # Replace rare titles with a common category 'Rare'
        X_copy['Title'] = X_copy['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                                 'Jonkheer', 'Dona'], 'Rare')
        X_copy['Title'] = X_copy['Title'].replace('Mlle', 'Miss')
        X_copy['Title'] = X_copy['Title'].replace('Ms', 'Miss')
        X_copy['Title'] = X_copy['Title'].replace('Mme', 'Mrs')
        return X_copy[['Title']]

class FamilySizeCreator(BaseEstimator, TransformerMixin):
    """Creates 'FamilySize' from 'SibSp' and 'Parch'."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy['FamilySize'] = X_copy['SibSp'] + X_copy['Parch'] + 1
        return X_copy[['FamilySize']]

# --- Main Data Processing and Modeling ---

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def create_preprocessing_pipeline():
    """Creates a full preprocessing pipeline for the Titanic dataset."""
    
    # Pipeline for numerical features: impute missing values, then scale.
    num_pipeline = Pipeline([
        ('selector', FeatureSelector(['Age', 'Fare', 'Pclass'])),
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features: impute, then one-hot encode.
    cat_pipeline = Pipeline([
        ('selector', FeatureSelector(['Embarked', 'Sex'])),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Pipeline for extracting titles from names
    title_pipeline = Pipeline([
        ('extractor', TitleExtractor()),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Pipeline for creating family size
    family_size_pipeline = Pipeline([
        ('creator', FamilySizeCreator()),
        ('scaler', StandardScaler())
    ])

    # Combine all preprocessing pipelines into a single FeatureUnion
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
        ("title_pipeline", title_pipeline),
        ("family_size_pipeline", family_size_pipeline)
    ])
    
    return full_pipeline

def main():
    """Main function to run the ML pipeline."""
    
    # 1. Load Data
    titanic_data = load_data(TRAIN_FILE)
    
    # 2. Split Data
    # Stratify on 'Survived' and 'Pclass' to ensure balanced splits
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_indices, test_indices in split.split(titanic_data, titanic_data[['Survived', 'Pclass']]):
        strat_train_set = titanic_data.loc[train_indices]
        strat_test_set = titanic_data.loc[test_indices]

    # Separate features and labels
    X_train = strat_train_set.drop("Survived", axis=1)
    y_train = strat_train_set["Survived"].copy()

    X_test = strat_test_set.drop("Survived", axis=1)
    y_test = strat_test_set["Survived"].copy()

    # 3. Create and Fit Preprocessing & Modeling Pipeline
    preprocessing_pipeline = create_preprocessing_pipeline()
    
    # Full pipeline with preprocessing and classifier
    full_pipeline_with_model = Pipeline([
        ("preprocessing", preprocessing_pipeline),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # 4. Hyperparameter Tuning with GridSearchCV
    param_grid = {
        'classifier__n_estimators': [100, 200, 500],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 3, 4],
        'classifier__max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(full_pipeline_with_model, param_grid, cv=5, 
                               scoring='accuracy', return_train_score=True)
    
    print("Starting GridSearchCV... (This may take a while)")
    grid_search.fit(X_train, y_train)
    print("GridSearchCV finished.")

    # The best model found by the grid search
    final_model = grid_search.best_estimator_
    
    print(f"\nBest Hyperparameters: {grid_search.best_params_}")
    
    # 5. Evaluate on the Test Set
    test_predictions = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"\nAccuracy on the stratified test set: {test_accuracy:.4f}")

    # 6. Train Final Model on Full Dataset and Generate Submission File
    print("\nTraining final model on the entire training dataset...")
    # We use the best parameters found, but train on all available data
    final_model.fit(titanic_data.drop("Survived", axis=1), titanic_data["Survived"])
    
    # Load Kaggle test data
    titanic_test_data = load_data(TEST_FILE)
    
    # Make predictions
    print("Making predictions on the Kaggle test set...")
    final_predictions = final_model.predict(titanic_test_data)

    # Create submission file
    submission_df = pd.DataFrame({'PassengerId': titanic_test_data['PassengerId'], 'Survived': final_predictions})
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    
    print(f"\nSubmission file created at: {SUBMISSION_FILE}")
    print("Done!")

if __name__ == "__main__":
    main()
