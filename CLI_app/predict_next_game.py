import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import argparse
import os
from datetime import timedelta
from scipy.stats import norm


def prepare_latest_form(df):
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['player', 'date'])
    else:
        df = df.sort_values(['player', 'match_id'])

    player_metrics = [
        'points', 'MIN_FL', 'USG%', 'Rim_PPS', 'Rim_Freq',
        'Corner3_Freq', 'Clutch_FG%', 'Fouls_Per_Min', 'TS%'
    ]

    for col in player_metrics:
        if col not in df.columns: continue
        df[f'L5_{col}'] = df.groupby('player')[col].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        df[f'L10_{col}'] = df.groupby('player')[col].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        df[f'Trend_{col}'] = df[f'L5_{col}'] - df[f'L10_{col}']

    df['last_played_date'] = df['date']
    return df.groupby('player').tail(1).copy()


def get_defensive_context(df, opponent_name):
    opp_data = df[df['opponent'] == opponent_name.lower().strip()]
    if opp_data.empty:
        return df['Opp_Avg_Pts_Allowed'].mean(), df['Opp_FG_Allowed'].mean()
    last_game = opp_data.sort_values('match_id').iloc[-1]
    return last_game['Opp_Avg_Pts_Allowed'], last_game['Opp_FG_Allowed']


def calc_over_prob(pred, line, mae=3.61):
    # Odhad směrodatné odchylky z MAE (Normal distribution assumption)
    # sigma = MAE * sqrt(pi/2) approx 1.2533 * MAE
    sigma = mae * 1.2533
    prob_over = 1 - norm.cdf(line, loc=pred, scale=sigma)
    return prob_over * 100


def run_prediction(team_name, opponent_name, is_home, data_path, model_path, target_surname=None, line=None):
    df = pd.read_csv(data_path, sep=';')
    df['date'] = pd.to_datetime(df['date'])
    model = joblib.load(model_path)
    model_features = model.get_booster().feature_names

    t1_name = team_name.lower().strip()
    t2_name = opponent_name.lower().strip()

    latest_stats = prepare_latest_form(df)
    mae_val = 3.61

    def predict_for_team(current_team, current_opp, playing_home):
        team_last_game_date = df[df['team'] == current_team]['date'].max()
        roster = latest_stats[latest_stats['team'] == current_team].copy()
        if roster.empty: return pd.DataFrame()

        # 60-denní filtr aktivních hráčů
        threshold_date = team_last_game_date - timedelta(days=60)
        roster = roster[roster['last_played_date'] >= threshold_date]

        if roster.empty: return pd.DataFrame()

        opp_data = df[df['opponent'] == current_opp.lower().strip()]
        opp_pts, opp_fg = get_defensive_context(df, current_opp)

        roster['Is_Home'] = 1 if playing_home else 0
        roster['Opp_Avg_Pts_Allowed'] = opp_pts
        roster['Opp_FG_Allowed'] = opp_fg
        roster['Days_Rest'] = 4.0

        for col in model_features:
            if col not in roster.columns: roster[col] = 0

        roster['predicted_pts'] = model.predict(roster[model_features])
        return roster[['player', 'predicted_pts', 'L5_points', 'MIN_FL', 'last_played_date']]

    team1_results = predict_for_team(t1_name, t2_name, is_home)
    team2_results = predict_for_team(t2_name, t1_name, not is_home)

    # REŽIM 1: Kontrola konkrétní hranice (check)
    if target_surname and line is not None:
        all_players = pd.concat([team1_results, team2_results])
        all_players['surname_key'] = all_players['player'].apply(lambda x: str(x).split()[-1].lower())
        match = all_players[all_players['surname_key'] == target_surname.lower()]

        if match.empty:
            print(f"❌ Hráč s příjmením '{target_surname}' nebyl nalezen v aktivních soupiskách.")
            return

        p = match.iloc[0]
        prob = calc_over_prob(p['predicted_pts'], line, mae_val)
        print(f"\n🎯 KONTROLA HRANICE: {p['player'].upper()}")
        print(f"Zápas: {t1_name.upper()} vs {t2_name.upper()}")
        print(f"Vypsaná hranice: {line} bodů")
        print(f"Model predikuje: {p['predicted_pts']:.1f}")
        print("-" * 35)
        print(f"PRAVDĚPODOBNOST OVER:  {prob:.1f}%")
        print(f"PRAVDĚPODOBNOST UNDER: {100 - prob:.1f}%")
        return

    # REŽIM 2: Obecný seznam predikcí (predict)
    print(f"\n🏀 MATCHUP PREDICTION: {t1_name.upper()} vs {t2_name.upper()}")
    display_order = [
        ("HOME", team1_results if is_home else team2_results, t1_name if is_home else t2_name),
        ("AWAY", team2_results if is_home else team1_results, t2_name if is_home else t1_name)
    ]

    for label, res, name in display_order:
        print(f"\n>>> {label} TEAM: {name.upper()}")
        print("-" * 85)
        print(f"{'PLAYER (Surname Sort)':<22} | {'PRED':<6} | {'FORM':<6} | {'% > FORM':<8} | {'LAST PLAYED':<12}")
        print("-" * 85)

        if res.empty:
            print(f"Žádní aktivní hráči pro tým: {name}")
        else:
            res['surname'] = res['player'].apply(lambda x: str(x).split()[-1] if pd.notnull(x) else "")
            res = res.sort_values('surname')

            for _, row in res.iterrows():
                if row['MIN_FL'] < 2: continue
                # Pravděpodobnost, že překoná svou formu (L5)
                prob_v_form = calc_over_prob(row['predicted_pts'], row['L5_points'], mae_val)
                last_date_str = row['last_played_date'].strftime('%Y-%m-%d')
                print(
                    f"{row['player']:<22} | {row['predicted_pts']:<6.1f} | {row['L5_points']:<6.1f} | {prob_v_form:>7.1f}% | {last_date_str:<12}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--team', type=str, required=True)
    parser.add_argument('--opponent', type=str, required=True)
    parser.add_argument('--home', action='store_true')
    parser.add_argument('--player', type=str)
    parser.add_argument('--line', type=float)
    parser.add_argument('--data', type=str, default='../data/FINALjan25.csv')
    parser.add_argument('--model', type=str, default='nbl_modelv1.pkl')
    args = parser.parse_args()
    run_prediction(args.team, args.opponent, args.home, args.data, args.model, args.player, args.line)