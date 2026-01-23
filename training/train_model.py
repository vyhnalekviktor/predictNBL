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
# 1. DEFINICE FEATURE LISTŮ (MASTER LIST)
# ==========================================

FEATURES_MIN = [
    'L1_MIN_FL', 'L3_MIN_FL', 'L5_MIN_FL', 'Trend_MIN_FL',
    'L1_Fouls', 'Blowout_Risk',
    'Smart_MIN_Baseline', 'Season_MIN_Avg',
    'L5_Fouls_Per_Min', 'Fouls_Drawn',
    'L5_GT_Activity_Index',
    'Days_Rest', 'Is_Home',
    'Opp_Avg_Pace', 'Opp_Avg_Fouls',
    'Pos_Code', 'Month',
    'L5_Starter_Rate', 'L1_Starter',
    'Spread_Est', 'Abs_Spread', 'Game_Importance',
    'Opp_Max_Foul_Drawer',
    'Pace_Clash'
]

FEATURES_PPM = [
    'Smoothed_PPM',  # <-- Úspěšné Bayesovské PPM
    'Opp_Allowed_Pos_Avg',  # <-- Úspěšné DvP
    'Def_Suppressor', 'Mismatch_Adv', 'Opp_Avg_Pts_Allowed',
    'L5_Synergy_Score', 'L5_Dependency_Index', 'L5_Gravity',
    'Shot_Quality_Avg', 'Pace_Est', 'Size_Index',
    'Opp_Avg_Pace', 'Opp_Best_Def_Synergy', 'Opp_Avg_Size',
    'Pos_Code', 'Month'
]


def load_data(filepath):
    print(f"📥 Načítám data z: {filepath}")
    df = pd.read_csv(filepath, sep=';', encoding='utf-8')
    df['date'] = pd.to_datetime(df['date'])
    return df


def convert_minutes(min_str):
    if isinstance(min_str, str) and ':' in min_str:
        try:
            m, s = map(int, min_str.split(':'))
            return m + s / 60.0
        except ValueError:
            return 0.0
    return float(min_str) if pd.notna(min_str) else 0.0


def engineer_features(df):
    print("⚙️ Generuji features (Rolling průměry + Bayesovské vyhlazení PPM)...")

    df['MIN_FL'] = df['minutes'].apply(convert_minutes)

    # --- 1. ČASOVÉ ŘAZENÍ (Kritické pro prevenci leakage) ---
    df = df.sort_values(by=['player', 'date'])

    # --- 2. RESTAUROVANÉ VÝPOČTY MINUT (Oprava chyby 4.90 MAE) ---
    # Základní historické minuty (posunuto o 1 zápas zpět)
    grp = df.groupby('player')['MIN_FL']
    df['L1_MIN_FL'] = grp.shift(1).fillna(0)
    df['L3_MIN_FL'] = grp.shift(1).rolling(window=3, min_periods=1).mean().fillna(0)
    df['L5_MIN_FL'] = grp.shift(1).rolling(window=5, min_periods=1).mean().fillna(0)
    df['L10_MIN_FL'] = grp.shift(1).rolling(window=10, min_periods=1).mean().fillna(0)

    # Odvozené metriky minut
    df['Trend_MIN_FL'] = df['L5_MIN_FL'] - df['L10_MIN_FL']
    df['Smart_MIN_Baseline'] = (df['L3_MIN_FL'] * 0.5) + (df['L5_MIN_FL'] * 0.3) + (df['L10_MIN_FL'] * 0.2)
    df['Season_MIN_Avg'] = grp.apply(lambda x: x.expanding().mean().shift(1)).reset_index(level=0, drop=True).fillna(0)

    # Historie Faulů
    df['fouls'] = pd.to_numeric(df['fouls'], errors='coerce').fillna(0)
    df['L1_Fouls'] = df.groupby('player')['fouls'].shift(1).fillna(0)
    df['L5_Fouls_Per_Min'] = df.groupby('player')['fouls'].shift(1).rolling(5, min_periods=1).mean() / df[
        'L5_MIN_FL'].replace(0, 1)

    # --- 3. BAYESOVSKÉ VYHLAZENÍ PPM (Důvod zlepšení na 2.34 MAE) ---
    # Spočítat kumulativní součet minut a bodů (shift(1) proti leakage)
    df['Hist_MIN'] = df.groupby('player')['MIN_FL'].transform(lambda x: x.expanding().sum().shift(1)).fillna(0)
    df['Hist_PTS'] = df.groupby('player')['points'].transform(lambda x: x.expanding().sum().shift(1)).fillna(0)

    # Výpočet historického PPM hráče DO DNEŠNÍHO DNE
    df['Hist_Raw_PPM'] = df['Hist_PTS'] / np.maximum(df['Hist_MIN'], 1)

    # Celoligový průměr PPM (pro vyhlazení)
    global_ppm_mean = df['points'].sum() / np.maximum(df['MIN_FL'].sum(), 1)
    smoothing_factor = 30.0  # Faktor 30 minut

    # Výpočet Bayesovské váhy a finální Smoothed_PPM
    df['Weight'] = df['Hist_MIN'] / (df['Hist_MIN'] + smoothing_factor)
    df['Smoothed_PPM'] = (df['Weight'] * df['Hist_Raw_PPM']) + ((1 - df['Weight']) * global_ppm_mean)

    # --- 4. OSTATNÍ FEATURES ---
    df['plus_minus'] = pd.to_numeric(df['plus_minus'], errors='coerce').fillna(0)
    df['Blowout_Risk'] = np.where(df['plus_minus'].abs() > 15, 1, 0)

    if 'Pos_Est' in df.columns:
        pos_map = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5, 'F': 3}
        df['Pos_Code'] = df['Pos_Est'].map(pos_map).fillna(3)
    elif 'Pos_Code' not in df.columns:
        df['Pos_Code'] = 3

    df['Month'] = df['date'].dt.month

    # Zajištění, že všechny FEATURES_MIN/PPM sloupce existují
    for col in set(FEATURES_MIN + FEATURES_PPM):
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Návrat k časovému řazení pro správný Train/Test split
    df = df.sort_values(by='date').reset_index(drop=True)

    return df


def train_and_evaluate(input_csv, output_dir):
    df = load_data(input_csv)
    df = engineer_features(df)

    # Split 60 dní
    cutoff_date = df['date'].max() - pd.Timedelta(days=60)
    train = df[df['date'] < cutoff_date].copy()
    test = df[df['date'] >= cutoff_date].copy()

    print(f"✅ Data rozdělena: TRAIN={len(train)} řádků, TEST={len(test)} řádků (od {cutoff_date.date()})")

    target_min = train['MIN_FL']
    target_ppm = train['points'] / np.maximum(train['MIN_FL'], 1)

    # --- MODEL MINUT ---
    print("\n🧠 Trénuji model: MINUTY")
    dtrain_min = xgb.DMatrix(train[FEATURES_MIN], label=target_min)
    params_min = {
        'objective': 'reg:absoluteerror',
        'max_depth': 4,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist'
    }
    model_min = xgb.train(params_min, dtrain_min, num_boost_round=400)

    # --- MODEL PPM ---
    print("🧠 Trénuji model: PPM (Bayesovsky vyhlazený)")
    mask_train = train['MIN_FL'] >= 3.0
    dtrain_ppm = xgb.DMatrix(train.loc[mask_train, FEATURES_PPM], label=target_ppm[mask_train])
    params_ppm = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'learning_rate': 0.015,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'tree_method': 'hist',
        'reg_alpha': 1.0
    }
    model_ppm = xgb.train(params_ppm, dtrain_ppm, num_boost_round=300)

    # --- EVALUACE ---
    print("\n📊 EVALUACE NA SLEPÉM TESTOVACÍM VZORKU (Posledních 60 dní):")
    dtest_min = xgb.DMatrix(test[FEATURES_MIN])
    dtest_ppm = xgb.DMatrix(test[FEATURES_PPM])

    pred_min = model_min.predict(dtest_min)
    pred_ppm = model_ppm.predict(dtest_ppm)
    final_preds_total = np.maximum(pred_min, 0) * np.maximum(pred_ppm, 0)

    test_min_actual = test['MIN_FL']
    test_pts_actual = test['points']

    mae_min = mean_absolute_error(test_min_actual, pred_min)
    mae_pts = mean_absolute_error(test_pts_actual, final_preds_total)

    print(f"   ⏱️ MAE Minuty: {mae_min:.2f} (Cíl < 3.0)")
    print(f"   🏀 MAE Body:   {mae_pts:.2f}")

    errors = np.abs(test_pts_actual - final_preds_total)
    sigma_total = np.mean(errors) + np.std(errors)
    print(f"   📉 SIGMA: {sigma_total:.2f}")

    # --- Q1-Q4 Rozpad ---
    print("\n🍰 Quarter Breakdown:")
    q_prefix = 'L5_Q' if 'L5_Q1_Activity' in test.columns else 'Q'
    if f'{q_prefix}1_Activity' in test.columns:
        total_activity = test[f'{q_prefix}1_Activity'] + test[f'{q_prefix}2_Activity'] + test[f'{q_prefix}3_Activity'] + \
                         test[f'{q_prefix}4_Activity']
        total_activity = total_activity.replace(0, 1)

        for q in [1, 2, 3, 4]:
            col = f'{q_prefix}{q}_Activity'
            if col in test.columns:
                ratio = test[col] / total_activity
                q_pred = final_preds_total * ratio
                print(f"   Q{q} Avg Pred: {q_pred.mean():.1f}")

    # --- EXPORT ---
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model_min, os.path.join(output_dir, 'model_minutes.pkl'))
    joblib.dump(model_ppm, os.path.join(output_dir, 'model_ppm.pkl'))
    joblib.dump(FEATURES_MIN, os.path.join(output_dir, 'features_min.pkl'))
    joblib.dump(FEATURES_PPM, os.path.join(output_dir, 'features_ppm.pkl'))
    with open(os.path.join(output_dir, 'sigma.txt'), 'w') as f:
        f.write(str(sigma_total))

    print(f"\n✅ Modely a features úspěšně uloženy do: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trénování basketbalového modelu')
    parser.add_argument('--input', type=str, default='FINAL.csv', help='Cesta ke vstupnímu CSV souboru')
    parser.add_argument('--output', type=str, default='models', help='Složka pro uložení modelů')
    args = parser.parse_args()
    train_and_evaluate(args.input, args.output)