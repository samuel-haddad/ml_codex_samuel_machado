"""
Authos: Samuel Machado
Date: May, 2024

Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict
if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a
machine learning solution to predict if an item is new or used and then evaluate
the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines`
and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result
of 0.86 as minimum. Additionally, you will have to choose an appropiate
secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features,
  the proposed secondary metric and the performance achieved on that metrics.
  Optionally, you can deliver an EDA analysis with other formart like .ipynb
"""
import subprocess
import sys

def install_packages(package_list):
    for package in package_list:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages
packages = [
    "pandas==2.1.4",
    "numpy==1.26.4",
    "scikit-learn==1.4.2",
    "xgboost==2.0.3",
]

# Call the function with the list of packages
install_packages(packages)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, TargetEncoder
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = [json.loads(x) for x in open("MLA_100k_checked_v3.jsonlines")]
    def target(x): return x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()

    # Insert your code below this line:

    # CLASSES AND PARAMETERS
    logger.info("Creating classes and params...")

    # JsonNormalizer
    class JsonNormalizer(BaseEstimator, TransformerMixin):
        """
        Transformer that normalizes a list of dictionaries into a pandas DataFrame.
        """

        def __init__(self):
            """
            Initialize the JsonNormalizer.
            """
            pass

        def fit(self, X, y=None):
            """
            Fit the transformer.

            Parameters:
            X (array-like): Input data.
            y (array-like, optional): Target values (ignored).

            Returns:
            self: The fitted transformer.
            """
            return self

        def transform(self, X):
            """
            Transform the input data by normalizing JSON.

            Parameters:
            X (array-like): Input data.

            Returns:
            pandas.DataFrame: Transformed DataFrame.
            """
            X_df = pd.json_normalize(X)
            return X_df

    # Column Selector Class

    class ColumnSelector(BaseEstimator, TransformerMixin):
        """
        Transformer that selects specified columns from a DataFrame.
        """

        def __init__(self, columns):
            """
            Initialize the ColumnSelector.

            Parameters:
            columns (list of str): List of columns to select.
            """
            self.columns = columns

        def fit(self, X, y=None):
            """
            Fit the transformer.

            Parameters:
            X (pandas.DataFrame): Input data.
            y (array-like, optional): Target values (ignored).

            Returns:
            self: The fitted transformer.
            """
            return self

        def transform(self, X):
            """
            Transform the input data by selecting specified columns.

            Parameters:
            X (pandas.DataFrame): Input data.

            Returns:
            pandas.DataFrame: DataFrame with selected columns.
            """
            return X[self.columns]

    # Type Transformer Class

    class TypeTransformer(BaseEstimator, TransformerMixin):
        """
        Transformer that converts all columns in a DataFrame to a specified data type.
        """

        def __init__(self, type="object"):
            """
            Initialize the TypeTransformer.

            Parameters:
            type (str): The type to convert columns to. Default is "object".
            """
            self.type = type

        def fit(self, X, y=None):
            """
            Fit the transformer.

            Parameters:
            X (pandas.DataFrame): Input data.
            y (array-like, optional): Target values (ignored).

            Returns:
            self: The fitted transformer.
            """
            return self

        def transform(self, X):
            """
            Transform the input data by converting columns to the specified type.

            Parameters:
            X (pandas.DataFrame): Input data.

            Returns:
            pandas.DataFrame: DataFrame with columns converted to the specified type.
            """
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            X = X.astype(self.type)
            return X

    # Custom Imputer Class (to output a dataframe)

    class CustomImputer(BaseEstimator, TransformerMixin):
        """
        Transformer that imputes missing values and outputs a DataFrame.
        """

        def __init__(
                self,
                strategy='mean',
                missing_values=None,
                fill_value=None):
            """
            Initialize the CustomImputer.

            Parameters:
            strategy (str): The imputation strategy. Default is 'mean'.
            missing_values (scalar, str, np.nan, or None): The placeholder for the missing values. Default is None.
            fill_value (scalar, str, or None): The value to replace the missing values. Default is None.
            """
            self.strategy = strategy
            self.missing_values = missing_values
            self.fill_value = fill_value

        def fit(self, X, y=None):
            """
            Fit the imputer to the data.

            Parameters:
            X (pandas.DataFrame): Input data.
            y (array-like, optional): Target values (ignored).

            Returns:
            self: The fitted imputer.
            """
            self.imputer = SimpleImputer(
                strategy=self.strategy,
                missing_values=self.missing_values,
                fill_value=self.fill_value)
            self.imputer.fit(X)
            return self

        def transform(self, X):
            """
            Transform the input data by imputing missing values.

            Parameters:
            X (pandas.DataFrame): Input data.

            Returns:
            pandas.DataFrame: DataFrame with imputed values.
            """
            X_imputed = self.imputer.transform(X)
            X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)
            return X_imputed_df

    # Flag Transformer Class

    class FlagTransformer(BaseEstimator, TransformerMixin):
        """
        Convert specified columns into binary flags (1 if not null, 0 otherwise).

        Parameters:
        -----------
        flag_lst : list
            List of column names to be converted into binary flags.

        Methods:
        --------
        fit(self, X, y=None)
            Fit method required by scikit-learn's TransformerMixin. Returns self.

        transform(self, X)
            Transform method required by scikit-learn's TransformerMixin. 
            Converts specified columns into binary flags.

        Attributes:
        -----------
        flag_lst : list
            List of column names to be converted into binary flags.
        """

        def __init__(self, flag_lst):
            """
            Initialize the FlagTransformer instance.

            Parameters:
            -----------
            flag_lst : list
                List of column names to be converted into binary flags.
            """
            self.flag_lst = flag_lst

        def fit(self, X, y=None):
            """Fit method required by scikit-learn's TransformerMixin. 
            Returns self."""
            return self

        def transform(self, X):
            """Transform method required by scikit-learn's TransformerMixin. 
            Converts specified columns into binary flags."""
            return self.convert_to_flag(X)
        
        def convert_to_flag(self, X):
            """Convert specified columns into binary flags."""
            for col in self.flag_lst:
                X[col] = X[col].notnull().astype(int)
            return X
    
    # Cardinality Reduction Class

    class CardinalityTransformer(BaseEstimator, TransformerMixin):
        """
        Transformer that reduces the cardinality of categorical features based on specified parameters.
        """

        def __init__(self, params):
            """
            Initialize the CardinalityTransformer.

            Parameters:
            params (dict): Dictionary of cardinality reduction parameters for specific columns.
            """
            self.params = params

        def fit(self, X, y=None):
            """
            Fit the transformer.

            Parameters:
            X (pandas.DataFrame or numpy.ndarray): Input data.
            y (array-like, optional): Target values (ignored).

            Returns:
            self: The fitted transformer.
            """
            return self

        def transform(self, X):
            """
            Transform the input data by reducing the cardinality of specified columns.

            Parameters:
            X (pandas.DataFrame or numpy.ndarray): Input data.

            Returns:
            pandas.DataFrame: DataFrame with reduced cardinality.
            """
            return self.cardinality(X)

        def cardinality(self, X):
            """
            Reduce the cardinality of the specified columns.

            Parameters:
            X (pandas.DataFrame or numpy.ndarray): Input data.

            Returns:
            pandas.DataFrame: DataFrame with reduced cardinality.
            """
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)

            params = self.params

            for col in params.keys():
                column = col
                threshold = params[col]["threshold"]
                others_label = params[col]["others_label"]

                x_pareto = X.groupby(column)[[column]].count().rename(
                    columns={column: "count"}).reset_index()
                x_pareto = x_pareto.sort_values(by='count', ascending=False)
                x_pareto["cumpercentage"] = x_pareto["count"].cumsum(
                ) / x_pareto["count"].sum() * 100
                pareto_list = x_pareto.loc[x_pareto["cumpercentage"]
                                           < threshold][column].unique()

                for label in pareto_list:
                    X.loc[X[column] == label, column] = label
                X[column] = np.where(
                    X[column].isin(pareto_list), X[column], others_label)

                X = pd.DataFrame(X)

            return X

    supported_cols = [
        "warranty",
        "listing_type_id",
        "buying_mode",
        "currency_id",
        "status",
        "seller_address.state.name",
        "shipping.mode",
        "category_id",
        "seller_id",
        "seller_address.city.name",
        "price",
        "sold_quantity",
        "available_quantity",
        "accepts_mercadopago",
        "automatic_relist",
        "shipping.local_pick_up",
        "shipping.free_shipping",
    ]

    categorical = [
        "listing_type_id",
        "buying_mode",
        "currency_id",
        "status",
        "seller_address.state.name",
        "shipping.mode",
        "category_id",
        "seller_id",
        "seller_address.city.name",
    ]

    params = {
        "category_id": {
            "threshold": 81,
            "others_label": "other_category",
        },
        "seller_id": {
            "threshold": 40,
            "others_label": "other_seller",
        },
        "seller_address.city.name": {
            "threshold": 50,
            "others_label": "other_city",
        },
    }

    numerical_standard_scaler = ['price']
    numerical_min_max_scaler = ["sold_quantity", 'available_quantity']

    flag_lst = ["warranty"]

    boolean = [
        "warranty",
        "accepts_mercadopago",
        "automatic_relist",
        "shipping.local_pick_up",
        "shipping.free_shipping",
    ]
    
    # CATEGORICAL FEATURES
    logger.info("Creating categorical pipeline...")

    categorical_imputers = []
    categorical_imputers.append((
        "impute_most_frequent",
        SimpleImputer(strategy='most_frequent', missing_values=pd.NA,),
        categorical
    ))

    categorical_pipeline = Pipeline(steps=[
        ('string_transform', TypeTransformer(type='string')),
        ('imputer', CustomImputer(strategy='most_frequent')),
        ('cardinality', CardinalityTransformer(params)),
        ("target", TargetEncoder(target_type="binary")),
    ])

    categorical_transformers = [
        ("categorical", categorical_pipeline, categorical)
    ]

    # NUMERIC FEATURES
    logger.info("Creating numerical pipeline...")

    # Define the pipelines for each list of features
    standard_scaler_transformer = Pipeline(steps=[
        ('impute_mean', SimpleImputer(strategy='mean')),
        ('standard_scaler', StandardScaler())
    ])

    min_max_scaler_transformer = Pipeline(steps=[
        ('impute_mean', SimpleImputer(strategy='mean')),
        ('min_max_scaler', MinMaxScaler())
    ])

    # Combine the transformers using a ColumnTransformer
    numerical_transformers = [
            ('num_standard_scaler', standard_scaler_transformer, numerical_standard_scaler),
            ('num_min_max_scaler', min_max_scaler_transformer, numerical_min_max_scaler)
        ]

    # BOOL FEATURES
    logger.info("Creating boolean pipeline...")

    bool_imputers = []
    bool_imputers.append((
        "impute_most_frequent",
        SimpleImputer(strategy='most_frequent', missing_values=pd.NA,),
        boolean
    ))

    bool_pipeline = Pipeline(steps=[
        ('string_transform', TypeTransformer(type='string')),
        ('flag', FlagTransformer(flag_lst)),
        ("imputer", ColumnTransformer(bool_imputers, remainder='passthrough')),
        ("encoder", OrdinalEncoder()),
    ])

    bool_transformers = [("boolean", bool_pipeline, boolean)]

    # MODEL
    logger.info("Creating the classifier...")

    space = {
        'learning_rate': 0.049462279298388846,
        'max_depth': 10,
        'n_estimators': 143,
        'subsample': 0.7677084973211453
    }

    # XGBClassifier with the parameters from the space dictionary
    xgbc_classifier = XGBClassifier(**space)

    # PIPELINE
    logger.info("Creating the Pipeline...")

    column_selector = ColumnSelector(columns=supported_cols)
    transformers = categorical_transformers + \
        numerical_transformers + bool_transformers
    preprocessor = ColumnTransformer(transformers, remainder="passthrough")

    model = Pipeline([
        ("json_normalizer", JsonNormalizer()),
        ("column_selector", column_selector),
        ("preprocessor", preprocessor),
        ("classifier", xgbc_classifier),
    ])

    # TRAINING
    logger.info("Preprocessing...")

    label_encoder = LabelEncoder()
    y_processed = label_encoder.fit_transform(y_train)

    model.fit(X_train, y_processed)

    # PREDICT
    logger.info("Predicting...")

    y_pred = model.predict(X_test)

    # SCORE
    logger.info("Scoring...")

    y_test_processed = label_encoder.transform(y_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test_processed, y_pred)

    # Print the accuracy
    print(f"Accuracy: {np.round(accuracy, 4) * 100}%")

    # Calculate the ROC AUC
    roc_auc = roc_auc_score(y_test_processed, y_pred)

    # Print the ROC AUC
    print(f"ROC AUC: {np.round(roc_auc, 4) * 100}%")

    logger.info("END")
