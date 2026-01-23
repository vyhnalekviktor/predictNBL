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

# ==========================================
# 1. DEFINICE FEATURE LISTŮ (VČETNĚ NOVÝCH PBP HROZEB)
# ==========================================

FEATURES_MIN = [
    'L1_MIN_FL', 'L3_MIN_FL', 'L5_MIN_FL', 'Trend_MIN_FL',
    'Smart_MIN_Baseline',
    'L5_Fouls_Per_Min', 'Fouls_Drawn',
    'L5_GT_Activity_Index',
    'Days_Rest', 'Is_Home',
    'Opp_Avg_Pace', 'Opp_Avg_Fouls',
    'Pos_Code',
    'Month',
    'L1_Opp_MIN',
    'Avg_Opp_MIN',
    'Opp_Pos_Allowed_MIN',
    'L5_Starter_Rate',
    'L1_Starter',
    'Spread_Est',
    'Abs_Spread',
    'Game_Importance',
    # --- PBP RELACE (MINUTY) ---
    'Opp_Max_Foul_Drawer'  # Riziko vyfaulování proti soupeři
]

FEATURES_PPM = [
    'Def_Suppressor', 'Mismatch_Adv', 'Opp_Avg_Pts_Allowed',
    'L5_Synergy_Score', 'L5_Dependency_Index', 'L5_Gravity',
    'L5_Shot_Quality_Avg', 'L5_Adj_PPM', 'L5_Tm_PTS',
    'Is_Home', 'Pos_Code',
    'Opp_Allowed_Pos_Avg',
    # --- PBP RELACE (BODY) ---
    'Opp_Best_Def_Synergy'  # Přítomnost "zlého psa" na hřišti
]


def prepare_data(df):
    print("🛠️  Processing Features (All Relational Logic Included)...")

    # 1. Čištění a Konverze
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # 2. Sjednocení jmen týmů
    name_map = {
        'bk jip pardubice': 'pardubice', 'bk kvis pardubice': 'pardubice',
        'sluneta usti nad labem': 'usti', 'sluneta  usti nad labem': 'usti',
        'bk armex energy decin': 'decin', 'bk opava': 'opava',
        'era basketball nymburk': 'nymburk', 'basket brno': 'brno',
        'pumpa basket brno': 'brno', 'bk olomoucko': 'olomoucko',
        'bk redstone olomoucko': 'olomoucko', 'nh ostrava': 'ostrava',
        'bk gapa hradec kralove': 'hradec', 'sokol hradec kralove': 'hradec',
        'usk praha': 'usk', 'sk slavia praha': 'slavia',
        'srsni photomate pisek': 'pisek', 'srsni pisek': 'pisek'
    }
    if 'team' in df.columns:
        df['team_norm'] = df['team'].str.lower().str.strip().replace(name_map)

    # 3. Minuty na float
    if 'MIN_FL' not in df.columns and 'minutes' in df.columns:
        df['MIN_FL'] = df['minutes'].apply(
            lambda x: float(str(x).split(':')[0]) + float(str(x).split(':')[1]) / 60 if isinstance(x,
                                                                                                   str) and ':' in x else 0)

    # 4. Pozice
    if 'Pos_Est' in df.columns:
        pos_map = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5, 'F': 3}
        df['Pos_Code'] = df['Pos_Est'].map(pos_map).fillna(3)
    else:
        df['Pos_Code'] = 3

    # --- DNY ODPOČINKU A DOMÁCÍ PROSTŘEDÍ (VRÁCENO) ---
    df = df.sort_values(['player', 'date'])
    df['Days_Rest'] = df.groupby('player')['date'].diff().dt.days.fillna(7).clip(upper=14)
    if 'Is_Home' not in df.columns: df['Is_Home'] = 0.5
    # ---------------------------------------------------

    # --- BEZPEČNOSTNÍ POJISTKA PRO NOVÉ PBP FEATURES ---
    if 'Opp_Max_Foul_Drawer' not in df.columns: df['Opp_Max_Foul_Drawer'] = 0
    if 'Opp_Best_Def_Synergy' not in df.columns: df['Opp_Best_Def_Synergy'] = 0

    # --- GAME IMPORTANCE & SPREAD ---
    if 'match_id' in df.columns and 'points' in df.columns:
        df['Tm_PTS_Game'] = df.groupby(['match_id', 'team'])['points'].transform('sum')
    else:
        df['Tm_PTS_Game'] = 80

    df = df.sort_values(['player', 'date'])
    df['L5_Tm_PTS'] = df.groupby('player')['Tm_PTS_Game'].shift(1).rolling(5).mean().fillna(80)

    df['Month'] = df['date'].dt.month
    if 'match_id' in df.columns and 'team_norm' in df.columns:
        match_teams = df.groupby('match_id')['team_norm'].unique()

        def get_opponent(row):
            teams = match_teams.get(row['match_id'])
            if teams is None or len(teams) < 2: return 'unknown'
            for t in teams:
                if t != row['team_norm']: return t
            return 'unknown'

        df['Opponent'] = df.apply(get_opponent, axis=1)
    else:
        df['Opponent'] = 'unknown'

    if 'Opp_Avg_Pts_Allowed' not in df.columns: df['Opp_Avg_Pts_Allowed'] = 80
    df['Spread_Est'] = df['L5_Tm_PTS'] - df['Opp_Avg_Pts_Allowed']
    df['Abs_Spread'] = df['Spread_Est'].abs()
    df['Game_Importance'] = 1 / (df['Abs_Spread'] * 0.5 + 1)

    # --- STARTER LOGIC ---
    if 'starter' in df.columns:
        df['starter'] = df['starter'].fillna(0).astype(int)
        df['L5_Starter_Rate'] = df.groupby('player')['starter'].shift(1).rolling(window=5, min_periods=1).mean().fillna(
            0)
        df['L1_Starter'] = df.groupby('player')['starter'].shift(1).fillna(0)
    else:
        df['L5_Starter_Rate'] = 0
        df['L1_Starter'] = 0

    # --- DvP (Defense vs Position) ---
    print("   -> Calculating DvP Minutes...")
    df = df.sort_values(['date'])
    dvp_min = df.groupby(['Opponent', 'Pos_Code'])['MIN_FL'] \
        .apply(lambda x: x.expanding().mean().shift(1)) \
        .reset_index(level=[0, 1], drop=True)
    df['Opp_Pos_Allowed_MIN'] = dvp_min

    league_pos_min = df.groupby('Pos_Code')['MIN_FL'].mean()
    for pos in df['Pos_Code'].unique():
        mask = (df['Pos_Code'] == pos)
        if mask.any():
            fill_val = league_pos_min.get(pos, 20)
            df.loc[mask, 'Opp_Pos_Allowed_MIN'] = df.loc[mask, 'Opp_Pos_Allowed_MIN'].fillna(fill_val)

    # --- HISTORIE PROTI SOUPEŘI ---
    print("   -> Calculating Player vs Opponent History...")
    if 'Opponent' in df.columns:
        df = df.sort_values(['player', 'Opponent', 'date'])
        df['L1_Opp_MIN'] = df.groupby(['player', 'Opponent'])['MIN_FL'].shift(1)

        avg_opp_min = df.groupby(['player', 'Opponent'])['MIN_FL'] \
            .apply(lambda x: x.expanding().mean().shift(1)) \
            .reset_index(level=[0, 1], drop=True)
        df['Avg_Opp_MIN'] = avg_opp_min

        df = df.sort_values(['player', 'date'])
        player_avg = df.groupby('player')['MIN_FL'].apply(lambda x: x.expanding().mean().shift(1)).reset_index(level=0,
                                                                                                               drop=True)
        df['L1_Opp_MIN'] = df['L1_Opp_MIN'].fillna(player_avg).fillna(0)
        df['Avg_Opp_MIN'] = df['Avg_Opp_MIN'].fillna(player_avg).fillna(0)

    # --- ROLLING METRICS ---
    df['Raw_PPM'] = df['points'] / df['MIN_FL'].replace(0, 1)
    if 'Opp_Allowed_Pos_Avg' not in df.columns: df['Opp_Allowed_Pos_Avg'] = 0
    if 'Season_Weight' not in df.columns: df['Season_Weight'] = 1.0

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
        df[f'L3_{col}'] = grp.shift(1).rolling(window=3, min_periods=1).mean()
        df[f'L5_{col}'] = grp.shift(1).rolling(window=5, min_periods=1).mean()
        df[f'L10_{col}'] = grp.shift(1).rolling(window=10, min_periods=1).mean()

    # Derived
    df['Trend_MIN_FL'] = df['L5_MIN_FL'] - df['L10_MIN_FL']
    df['Smart_MIN_Baseline'] = (df['L3_MIN_FL'] * 0.5) + (df['L5_MIN_FL'] * 0.3) + (df['L10_MIN_FL'] * 0.2)
    df['L5_Adj_PPM'] = df.groupby('player')['Raw_PPM'].shift(1).rolling(5).mean()
    df['L5_Fouls_Per_Min'] = df.groupby('player')['fouls'].shift(1).rolling(5).mean() / df['L5_MIN_FL'].replace(0, 1)

    df.fillna(0, inplace=True)
    return df


def train_and_evaluate(data_path, output_dir='models'):
    print(f"🚀 Loading: {data_path}")
    df = pd.read_csv(data_path, sep=';')

    df = prepare_data(df)

    dates = np.sort(df['date'].unique())
    if len(dates) == 0:
        print("❌ Chyba: Žádná platná data k tréninku.")
        return

    split_date = dates[int(len(dates) * 0.85)]
    train = df[df['date'] < split_date]
    test = df[df['date'] >= split_date]

    print(f"📅 Split: {split_date} | Train: {len(train)} | Test: {len(test)}")

    w_train = train['MIN_FL'].clip(lower=1) * train['Season_Weight']

    # 1. Minutes Model
    print("⏳ Training Minutes...")
    valid_cols_min = [c for c in FEATURES_MIN if c in df.columns]

    model_min = xgb.XGBRegressor(
        n_estimators=1200, learning_rate=0.015, max_depth=5,
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
    if 'L5_Q1_Activity' in test.columns:
        total_activity = test['L5_Q1_Activity'] + test['L5_Q2_Activity'] + test['L5_Q3_Activity'] + test[
            'L5_Q4_Activity']
        total_activity = total_activity.replace(0, 1)

        for q in [1, 2, 3, 4]:
            col = f'L5_Q{q}_Activity'
            if col in test.columns:
                ratio = test[col] / total_activity
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

    df.to_csv(os.path.join(output_dir, 'last_training_data.csv'), sep=';', index=False)
    print(f"✅ Done. Saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/final_enhanced.csv')
    args = parser.parse_args()
    if os.path.exists(args.input):
        train_and_evaluate(args.input)
    else:
        print(f"❌ Input file not found: {args.input}")