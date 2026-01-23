import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import argparse
import sys
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

FEATURES_MIN = [
    'L1_MIN_FL', 'L5_MIN_FL', 'L10_MIN_FL', 'Trend_MIN_FL',
    'Smart_MIN_Baseline',
    'L5_Fouls_Per_Min', 'Fouls_Drawn',
    'L5_GT_Activity_Index',
    'Days_Rest', 'Is_Home',
    'Opp_Avg_Pace', 'Opp_Avg_Fouls',
    'Pos_Code'
]

FEATURES_PPM = [
    'Def_Suppressor', 'Mismatch_Adv', 'Opp_Avg_Pts_Allowed',
    'L5_Synergy_Score', 'L5_Dependency_Index', 'L5_Gravity',
    'L5_Shot_Quality_Avg', 'L5_Adj_PPM', 'L5_Tm_PTS',
    'Is_Home', 'Pos_Code'
]


def prepare_data(df):
    print("🛠️  Processing Features...")

    # 1. Konverze data s ošetřením chyb
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 2. FILL LOGIC: Doplnění chybějících dat z okolí
    df['date'] = df['date'].ffill()
    df['date'] = df['date'].bfill()

    # 3. Pojistka: Odstranění řádků, kde datum stále chybí
    df = df.dropna(subset=['date'])

    df = df.sort_values(['player', 'date'])

    if 'Pos_Est' in df.columns:
        pos_map = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5, 'F': 3}
        df['Pos_Code'] = df['Pos_Est'].map(pos_map).fillna(3)
    else:
        df['Pos_Code'] = 3

    df['Raw_PPM'] = df['points'] / df['MIN_FL'].replace(0, 1)

    if 'Season_Weight' not in df.columns:
        df['Season_Weight'] = 1.0

    metrics_to_roll = [
        'MIN_FL', 'points', 'efficiency',
        'Synergy_Score', 'Dependency_Index', 'Gravity',
        'Shot_Quality_Avg', 'GT_Activity_Index',
        'Q1_Activity', 'Q2_Activity', 'Q3_Activity', 'Q4_Activity',
        'Clutch_Events'
    ]

    for m in metrics_to_roll:
        if m not in df.columns: df[m] = 0

    for col in metrics_to_roll:
        grp = df.groupby('player')[col]
        df[f'L1_{col}'] = grp.shift(1)
        df[f'L5_{col}'] = grp.shift(1).rolling(window=5, min_periods=1).mean()
        df[f'L10_{col}'] = grp.shift(1).rolling(window=10, min_periods=1).mean()

    df['Trend_MIN_FL'] = df['L5_MIN_FL'] - df['L10_MIN_FL']
    df['Smart_MIN_Baseline'] = (df['L5_MIN_FL'] * 0.7) + (df['L10_MIN_FL'] * 0.3)
    df['L5_Adj_PPM'] = df.groupby('player')['Raw_PPM'].shift(1).rolling(5).mean()
    df['L5_Fouls_Per_Min'] = df.groupby('player')['fouls'].shift(1).rolling(5).mean() / df['L5_MIN_FL'].replace(0, 1)

    # 4. Nahrazení chybějících hodnot nulou (datum je už v pořádku)
    df.fillna(0, inplace=True)
    return df


def train_and_evaluate(data_path, output_dir='models'):
    print(f"🚀 Loading: {data_path}")
    df = pd.read_csv(data_path, sep=';')

    df = prepare_data(df)

    # Split
    # OPRAVA: Používáme np.sort místo .sort() pro kompatibilitu
    dates = np.sort(df['date'].unique())

    if len(dates) == 0:
        print("❌ Chyba: Žádná platná data k tréninku.")
        return

    split_date = dates[int(len(dates) * 0.8)]

    train = df[df['date'] < split_date]
    test = df[df['date'] >= split_date]

    print(f"📅 Split: {split_date} | Train: {len(train)} | Test: {len(test)}")

    w_train = train['MIN_FL'].clip(lower=1) * train['Season_Weight']

    # 1. Minutes Model
    print("⏳ Training Minutes...")
    valid_cols_min = [c for c in FEATURES_MIN if c in df.columns]
    model_min = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.02, max_depth=4,
        objective='reg:absoluteerror', n_jobs=-1
    )
    model_min.fit(train[valid_cols_min], train['MIN_FL'], sample_weight=train['Season_Weight'])
    preds_min = model_min.predict(test[valid_cols_min])
    print(f"   MAE Minutes: {mean_absolute_error(test['MIN_FL'], preds_min):.2f}")

    # 2. PPM Model
    print("🏀 Training PPM...")
    valid_cols_ppm = [c for c in FEATURES_PPM if c in df.columns]
    model_ppm = xgb.XGBRegressor(
        n_estimators=1500, learning_rate=0.015, max_depth=5,
        objective='reg:absoluteerror', n_jobs=-1
    )
    model_ppm.fit(train[valid_cols_ppm], train['Raw_PPM'], sample_weight=w_train)
    preds_ppm = model_ppm.predict(test[valid_cols_ppm])

    # 3. Total & Sigma
    final_preds_total = np.maximum(preds_min, 0) * np.maximum(preds_ppm, 0)
    residuals_total = test['points'] - final_preds_total
    sigma_total = np.std(residuals_total)

    print(f"🔥 TOTAL MAE: {mean_absolute_error(test['points'], final_preds_total):.2f} pts")
    print(f"📉 SIGMA: {sigma_total:.2f}")

    # 4. Quarters
    print("🍰 Quarter Breakdown:")
    total_activity = test['L5_Q1_Activity'] + test['L5_Q2_Activity'] + test['L5_Q3_Activity'] + test['L5_Q4_Activity']
    total_activity = total_activity.replace(0, 1)

    for q in [1, 2, 3, 4]:
        ratio = test[f'L5_Q{q}_Activity'] / total_activity
        q_pred = final_preds_total * ratio
        print(f"   Q{q} Avg Pred: {q_pred.mean():.1f}")

    # Export
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model_min, os.path.join(output_dir, 'model_minutes.pkl'))
    joblib.dump(model_ppm, os.path.join(output_dir, 'model_ppm.pkl'))
    joblib.dump(valid_cols_min, os.path.join(output_dir, 'features_min.pkl'))
    joblib.dump(valid_cols_ppm, os.path.join(output_dir, 'features_ppm.pkl'))

    with open(os.path.join(output_dir, 'sigma.txt'), 'w') as f:
        f.write(str(sigma_total))

    print(f"✅ Done. Saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/final_enhanced.csv')
    args = parser.parse_args()

    if os.path.exists(args.input):
        train_and_evaluate(args.input)
    else:
        print(f"❌ Input file not found: {args.input}")