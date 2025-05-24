import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter

def load_file(path='train.parquet'):
    """
    Load the parquet file and return a DataFrame.

    Args:
        path (str): Path to the parquet file. Default is 'train.parquet'.
    Returns:
        pd.DataFrame: DataFrame containing the data from the parquet file.
    """
    df = pd.read_parquet()
    print(len(df))

    return df

def convert(df):
    """
    Convert the DataFrame to a format suitable for training.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
    Returns:
        pd.DataFrame: Converted DataFrame with 'X1~X890' column.
        pd.DataFrame: Converted DataFrame with 'label' column.
    """
    feature_cols = [col for col in df.columns if col.startswith("X")]
    X = df[feature_cols]
    y = df["label"]

    return X, y

def selection(X, y, epochs, folds, output_dir="analysis"):
    """
    Perform feature selection using XGBoost.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Labels Series.
        epochs (int): Number of epochs for training.
        folds (int): Number of folds for cross-validation.
    """
    n_data_per_flow = int(len(X) / folds)
    print(f"{n_data_per_flow} per fold")

    for e in range(epochs):
        for f in range(folds):
            print("Fold: ", f)

            idx = np.arange(len(X))
            np.random.shuffle(idx)
            cur_X, cur_Y = X.iloc[idx[:n_data_per_flow]], y.iloc[idx[:n_data_per_flow]]
            cur_X.replace([np.inf, -np.inf], 0, inplace=True)
            
            # SVM
            X_train, X_test, y_train, y_test = train_test_split(cur_X, cur_Y, test_size=0.2, random_state=42)
        
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss', verbose=1)
            model.fit(X_train, y_train)
        
            # xgb.plot_importance(model, max_num_features=20, importance_type='gain', height=0.5)
            # plt.title("Top 20 Feature Importances (XGBoost)")
            # plt.tight_layout()
            # plt.show()
            # plt.clf()
        
            importance = pd.Series(model.feature_importances_, index=X.columns)
            importance_sorted = importance.sort_values(ascending=False)
            importance_sorted.to_csv(f"{output_dir}/epoch{e}_fold{f}.csv")

def statistical(output_dir="analysis"):
    """
    Perform statistical analysis on the feature importances.

    Args:
        output_dir (str): Directory containing the feature importance files.
    """
    files = os.listdir(output_dir)
    print("Number of files: ", len(files))

    candidates = []
    for f in files:
        df = pd.read_csv(os.path.join(path, f))
        candidates += df['Unnamed: 0'].iloc[:5].tolist()

    stat = Counter(candidates)
    filtered = {k: v for k, v in stat.items() if v >= 5}
    with open("useful_feats.txt", 'w') as f:
        f.write(",".join(filtered.keys()))

if __name__ == "__main__":
    path = "train.parquet"
    df = load_file(path)
    X, y = convert(df)

    epochs = 10
    folds = 5
    output_dir = "analysis"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    selection(X, y, epochs, folds, output_dir)
    statistical(output_dir)