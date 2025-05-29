import xgboost as xgb
import pandas as pd
import os
import numpy as np

def load_data(path):
    """
    Load data from a parquet file.

    Args:
        path (str): Path to the parquet file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """

    df = pd.read_parquet('train.parquet')
    print(len(df))

    return df

def feature_selection(df, select_feats):
    """
    Select features from the DataFrame based on the provided list.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        select_feats (list): List of features to select.

    Returns:
        pd.DataFrame: DataFrame containing only the selected features.
        pd.Series: Series containing the labels.
    """

    X = df[select_feats]
    y = df["label"]

    return X, y

def cross_validation(X, y, model, folds=3):
    """
    Perform cross-validation on the model.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
        model: XGBoost model to evaluate.
        folds (int): Number of folds for cross-validation.

    Returns:
        model: Trained XGBoost model to predict.
    """

    for f in range(folds):
        # Slicing the training data
        n_data_per_flow = int(len(X) / folds)
        cur_X, cur_Y = X.iloc[f*n_data_per_flow: (f+1)*n_data_per_flow], y.iloc[f*n_data_per_flow: (f+1)*n_data_per_flow]
        cur_X.replace([np.inf, -np.inf], 0, inplace=True)

        model.fit(cur_X, cur_Y)

    return model

def predict(model, test_X):
    """
    Predict using the trained model.

    Args:
        model: Trained XGBoost model.
        test_X (pd.DataFrame): Features for prediction.

    Returns:
        np.ndarray: Predictions from the model.
    """

    test_X.replace([np.inf, -np.inf], 0, inplace=True)
    preds = model.predict(test_X)

    return preds

def save_predictions(preds, output_path='submission.csv'):
    """
    Save predictions to a CSV file.

    Args:
        preds (np.ndarray): Predictions to save.
        output_path (str): Path to save the predictions.
    """

    with open(output_path, 'w') as f:
        f.write("ID,prediction\n")
        for idx, pred in enumerate(preds):
            f.write(f"{idx+1},{pred}\n")

if __name__ == "__main__":
    # Load data
    df = load_data('train.parquet')

    # Define features to select
    select_feats = ['X784', 'X785', 'X515', 'X466', 'X522', 'X507', 'X292', 'X284', 'X272', 'X841']  # Replace with actual feature names
    select_feats += ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']  # General features
    
    # Feature selection
    X, y = feature_selection(df, select_feats)

    # Initialize model
    model = xgb.XGBRegressor(n_estimators=200, max_depth=40, learning_rate=0.01, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss', verbose=1)

    # Cross-validation
    model = cross_validation(X, y, model, folds=5)

    # Load test data
    test_df = pd.read_parquet('test.parquet')
    test_X = test_df[select_feats]

    # Predict
    preds = predict(model, test_X)
    print(preds)
    # Save predictions
    save_predictions(preds)