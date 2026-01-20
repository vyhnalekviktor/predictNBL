import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import argparse
import sys
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings

# Potlačení varování o fragmentaci (kosmetická úprava pro čistý výpis)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore')

# ==========================================
# 1. FEATURE DEFINITIONS (LOGICKÉ ROZDĚLENÍ)
# ==========================================

# A) MODEL MINUT: Vidí jen to, co ovlivňuje střídání (ne střelbu)
FEATURES_MIN = [
    # Historie minut (to nejdůležitější)
    'L1_MIN_FL', 'L5_MIN_FL', 'L10_MIN_FL', 'Trend_MIN_FL',
    'Smart_MIN_Baseline',

    # Důvody střídání
    'Fouls_Per_Min', 'L5_Fouls_Per_Min',  # Faulové problémy
    'Days_Rest',  # Únava
    'Is_Home',  # Domácí rotace bývá jiná
    'Def_Rating',  # Trenéři nechávají na hřišti obránce
    'USG%',  # Hvězdy hrají déle

    # Kontext zápasu
    'Opp_Avg_Pace',  # Rychlé tempo = více střídání?
    'Opp_Avg_Pts_Allowed'  # Síla soupeře (blowout riziko)
]

# B) MODEL PPM (EFEKTIVITA): Vidí kompletní "Skillset" hráče
FEATURES_PPM = [
    # Forma (Body a upravené body)
    'L5_points', 'L10_points', 'Trend_points',
    'L5_Adj_Points', 'L10_Adj_Points',  # Body bez Garbage Time
    'L5_GT_Points',  # Historie levných bodů

    # Historie efektivity
    'L5_PPM', 'L10_PPM', 'Trend_PPM',

    # --- ZDE JSOU VRÁCENY CHYBĚJÍCÍ METRIKY ---
    'L5_TS%', 'L10_TS%',  # True Shooting (zohledňuje šestky a trojky)
    'L5_eFG%', 'L10_eFG%',  # Effective FG% (zohledňuje trojky)

    # Moneyball Stats
    'FTA', 'L5_FTA',  # Nájezdy / Šestky
    'OREB', 'L5_OREB',  # Útočné doskoky
    'turnovers', 'L5_turnovers',  # Ztráty

    # Pokročilá střelba (SPATIAL & ISO - Tvoje data)
    'L5_ISO_Attempts', 'L5_ISO_Makes',
    'L5_Rim_PPS', 'L5_Rim_Freq',
    'L5_MidRange_PPS', 'L5_MidRange_Freq',
    'L5_Arc3_PPS', 'L5_Arc3_Freq',
    'L5_Corner3_PPS', 'L5_Corner3_Freq',

    # Clutch & Consistency
    'L5_Clutch_FG%', 'Consistency',

    # Matchup (Klíčové pro efektivitu)
    'Roster_Matchup_Strength', 'DvP_Avg_Pts', 'Opp_Avg_Pts_Allowed'
]


# ==========================================
# 2. DATA PREP
# ==========================================
def prepare_data(data_path):
    print("🔄 Loading Data & Calculating Full History...")
    df = pd.read_csv(data_path, sep=';')

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['player', 'date'])

    # Pomocný sloupec pro PPM
    df['PPM'] = df['points'] / df['MIN_FL'].replace(0, 1)

    # SEZNAM VŠECH METRIK Z UPGRADE_DATA.PY (Nic nechybí)
    all_raw_metrics = [
        'points', 'MIN_FL', 'Adj_Points', 'GT_Points',
        'USG%', 'Pace', 'PPM',
        'FTA', 'OREB', 'TS%', 'eFG%', 'turnovers', 'Fouls_Per_Min',
        'ISO_Attempts', 'ISO_Makes', 'Def_Rating',
        'Rim_PPS', 'Rim_Freq', 'MidRange_PPS', 'MidRange_Freq',
        'Corner3_PPS', 'Corner3_Freq', 'Arc3_PPS', 'Arc3_Freq',
        'Clutch_FG%'
    ]

    print(f"   Rolling stats for {len(all_raw_metrics)} metrics...")

    # Výpočet L1, L5, L10, Trend pro všechno
    for col in all_raw_metrics:
        if col not in df.columns: continue

        grp = df.groupby('player')[col]

        # L1 (Last Game)
        df[f'L1_{col}'] = grp.transform(lambda x: x.shift(1))
        # L5 (Short term)
        df[f'L5_{col}'] = grp.transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
        # L10 (Long term)
        df[f'L10_{col}'] = grp.transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
        # Trend
        df[f'Trend_{col}'] = df[f'L5_{col}'] - df[f'L10_{col}']

    # Smart Baseline pro Minuty (Vážený průměr)
    if 'L1_MIN_FL' in df.columns:
        df['Smart_MIN_Baseline'] = (0.5 * df['L1_MIN_FL']) + (0.3 * df['L5_MIN_FL']) + (0.2 * df['L10_MIN_FL'])

    # Consistency
    df['Consistency'] = df.groupby('player')['points'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).std()
    )

    # One-Hot Pozice (Přidáme dynamicky do obou feature listů)
    pos_cols = []
    if 'Pos_Est' in df.columns:
        df = pd.get_dummies(df, columns=['Pos_Est'], prefix='Pos')
        pos_cols = [c for c in df.columns if c.startswith('Pos_')]

    # Přidáme pozice do definic features (aby to model viděl)
    global FEATURES_MIN, FEATURES_PPM
    FEATURES_MIN += pos_cols
    FEATURES_PPM += pos_cols

    # TARGETS
    df['Target_PPM'] = df['points'] / df['MIN_FL'].replace(0, 1)
    df['Target_MIN'] = df['MIN_FL']

    # Filter: Musíme mít data pro trénink (historie)
    df = df.dropna(subset=['L5_points', 'Target_PPM', 'Smart_MIN_Baseline'])

    # Validace pro trénink PPM (jen relevantní minuty)
    df['Valid_PPM_Train'] = df['MIN_FL'] >= 5

    # Defragmentace (zkopírování dat pro odstranění varování)
    df = df.copy()

    return df


# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def train_specialized_model(X, y, feature_names, name, w=None):
    print(f"\n⚙️ Training Optimized {name} Model...")

    # Filtrace features, které reálně existují v DF
    valid_features = [f for f in feature_names if f in X.columns]

    missing = set(feature_names) - set(valid_features)
    # Tiché ignorování chybějících features (např. L10 u nováčků)

    X_subset = X[valid_features]

    # Time Series Split (85/15)
    split = int(len(X) * 0.85)
    X_train, X_test = X_subset.iloc[:split], X_subset.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Váhy (Season Weight)
    w_train = w.iloc[:split] if w is not None else None

    # XGBoost Config - OPRAVENÝ PRO VERZI 2.0+
    model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.015,
        max_depth=4,  # Menší hloubka = méně overfittingu
        subsample=0.7,
        colsample_bytree=0.5,  # Feature sampling
        reg_alpha=2,  # L1 Regularizace (Anti-Noise)
        reg_lambda=3,
        objective='reg:absoluteerror',
        n_jobs=-1,
        early_stopping_rounds=50  # <--- ZDE SPRÁVNĚ
    )

    model.fit(
        X_train, y_train,
        sample_weight=w_train,  # <--- POUŽITÍ VAH
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"   👉 {name} MAE: {mae:.4f}")

    return model, preds, valid_features


# ==========================================
# 4. MAIN RUN
# ==========================================
def run_training(input_file, output_dir):
    df = prepare_data(input_file)

    # Načtení vah sezóny (aby nová data měla přednost)
    weights = df['Season_Weight'] if 'Season_Weight' in df.columns else None
    if weights is None: print("⚠️ Warning: Season_Weight column missing!")

    # --- MODEL 1: MINUTES (Lobotomized) ---
    # Učí se minuty bez šumu střelby
    model_min, preds_min, feats_min = train_specialized_model(
        df, df['Target_MIN'], FEATURES_MIN, "MINUTES", weights
    )

    # --- MODEL 2: PPM (Full Skillset) ---
    # Učí se efektivitu se vším všudy (včetně TS% a eFG%)
    print("\n⚙️ Training Optimized PPM Model (Full Features)...")

    # Features check
    valid_ppm_feats = [f for f in FEATURES_PPM if f in df.columns]

    # PPM trénujeme jen na datech, kde hráč hrál > 5 min
    mask = df['Valid_PPM_Train']

    X_ppm_all = df[valid_ppm_feats]
    y_ppm_all = df['Target_PPM']

    split = int(len(df) * 0.85)

    # Train set (Filtrovaný maskou)
    X_train_ppm = X_ppm_all.iloc[:split][mask[:split]]
    y_train_ppm = y_ppm_all.iloc[:split][mask[:split]]
    w_train_ppm = weights.iloc[:split][mask[:split]] if weights is not None else None

    # Test set (Kompletní - testujeme realitu)
    X_test_ppm = X_ppm_all.iloc[split:]
    y_test_ppm = y_ppm_all.iloc[split:]

    model_ppm = xgb.XGBRegressor(
        n_estimators=1200, learning_rate=0.015, max_depth=5,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1,
        objective='reg:absoluteerror', n_jobs=-1,
        early_stopping_rounds=50  # <--- ZDE SPRÁVNĚ
    )

    model_ppm.fit(
        X_train_ppm, y_train_ppm,
        sample_weight=w_train_ppm,
        eval_set=[(X_test_ppm, y_test_ppm)],
        verbose=False
    )

    preds_ppm = model_ppm.predict(X_test_ppm)
    print(f"   👉 PPM MAE: {mean_absolute_error(y_test_ppm, preds_ppm):.4f}")

    # --- FINAL EVALUATION ---
    print("\n🔗 COMBINING RESULTS...")

    # Oříznutí záporných hodnot
    final_mins = np.maximum(preds_min, 0)
    final_ppm = np.maximum(preds_ppm, 0)

    final_preds = final_mins * final_ppm
    actual_pts = df.iloc[split:]['points']

    # GLOBAL MAE
    mae_total = mean_absolute_error(actual_pts, final_preds)

    print("-" * 40)
    print(f"📊 GLOBAL RESULTS (All Players):")
    print(f"   MAE: {mae_total:.4f}")

    # SNIPER EVALUATION (Jen pro hráče > 15 min v testovací sadě)
    test_indices = df.iloc[split:].index
    sniper_mask = df.loc[test_indices, 'L5_MIN_FL'] >= 15

    if sniper_mask.sum() > 0:
        mae_sniper = mean_absolute_error(actual_pts[sniper_mask], final_preds[sniper_mask])
        print(f"🎯 SNIPER RESULTS (Rotation Players >15m):")
        print(f"   MAE: {mae_sniper:.4f} 🔥 (Sázkařská přesnost)")
    else:
        print("   (Not enough rotation players in test set for sniper eval)")

    print("-" * 40)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model_min, os.path.join(output_dir, 'model_min.pkl'))
    joblib.dump(model_ppm, os.path.join(output_dir, 'model_ppm.pkl'))

    # Metadata pro prediktor
    meta = {
        'features_min': feats_min,
        'features_ppm': valid_ppm_feats,
        'mae': mae_total
    }
    joblib.dump(meta, os.path.join(output_dir, 'model_meta.pkl'))
    print(f"✅ Saved to {output_dir}")

    # Plot Importance
    plt.figure(figsize=(10, 8))
    if hasattr(model_ppm, 'feature_importances_'):
        sorted_idx = model_ppm.feature_importances_.argsort()[-20:]
        plt.barh(np.array(valid_ppm_feats)[sorted_idx], model_ppm.feature_importances_[sorted_idx])
        plt.title("PPM Model - Key Skill Factors")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'importance_ppm.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/final.csv')
    parser.add_argument('--output', default='result_optimized')
    args = parser.parse_args()

    run_training(args.input, args.output)