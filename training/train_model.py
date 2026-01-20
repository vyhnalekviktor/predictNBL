import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import argparse
import os
import joblib

def prepare_rolling_features(df):
    print("Creating Rolling Averages (Player Form)...")

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['player', 'date'])
    else:
        df = df.sort_values(['player', 'match_id'])

    player_metrics = [
            'points', 'MIN_FL',
            'Adj_Points',
            'USG%',
            'Pace',
            'Consistency',
            'TS%', 'eFG%', 'Fouls_Per_Min',
            'ISO_Attempts',
            'Rim_PPS', 'Rim_Freq',
            'Corner3_Freq', 'Arc3_Freq',
            'Clutch_FG%'
        ]

    for col in player_metrics:
        if col not in df.columns: continue
        df[f'L5_{col}'] = df.groupby('player')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=2).mean())
        df[f'L10_{col}'] = df.groupby('player')[col].transform(
            lambda x: x.shift(1).rolling(window=10, min_periods=5).mean())

        df[f'Trend_{col}'] = df[f'L5_{col}'] - df[f'L10_{col}']

    original_len = len(df)
    df = df.dropna(subset=['L5_points'])
    print(f"   Dropped {original_len - len(df)} rows (insufficient history). Remaining: {len(df)}")

    return df


def train_model(data_path, model_output, plot_output):
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"❌ Error: File {data_path} does not exist.")
        return

    print(f"📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path, sep=';')

    # 2. Feature Engineering
    df = prepare_rolling_features(df)

    # 3. Define Features
    features = [c for c in df.columns if c.startswith(('L5_', 'L10_', 'Trend_'))]

    context_candidates = [
        'Is_Home',
        'Opp_Avg_Pts_Allowed',
        'DvP_Avg_Pts',
        'Opp_Avg_Pace',
        'Days_Rest'
    ]

    features += [c for c in context_candidates if c in df.columns]

    target = 'points'

    print(f"🧠 Training on {len(features)} features.")
    print(f"   Context Features: {[c for c in features if c in context_candidates]}")

    X = df[features]
    y = df[target]

    weights = df['Season_Weight'] if 'Season_Weight' in df.columns else None

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights if weights is not None else np.ones(len(X)),
        test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.7,
        n_jobs=-1,
        objective='reg:absoluteerror',
        early_stopping_rounds=50
    )

    print("🚀 Starting XGBoost training...")

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )

    # 4. Evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n📊 MODEL RESULTS:")
    print(f"   MAE (Mean Absolute Error): {mae:.2f} points")
    print(f"   R2 Score: {r2:.4f}")

    # Save Model
    joblib.dump(model, model_output)
    print(f"✅ Model saved to: {model_output}")

    # 5. Feature Importance Plot
    plt.figure(figsize=(12, 10))
    sorted_idx = model.feature_importances_.argsort()[-25:]  # Top 25
    plt.barh(X.columns[sorted_idx], model.feature_importances_[sorted_idx])
    plt.title(f"XGBoost Feature Importance (MAE: {mae:.2f})")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(plot_output)
    print(f"🖼️ Feature Importance plot saved to: {plot_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost model for NBL Points Prediction")

    parser.add_argument('--input', type=str, default='data/final.csv',
                        help='Path to the enriched CSV file')
    parser.add_argument('--model', type=str, default='result/nbl_model.pkl',
                        help='Path to save the trained model')
    parser.add_argument('--plot', type=str, default='result/feature_importance.png',
                        help='Path to save the importance plot')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    train_model(args.input, args.model, args.plot)