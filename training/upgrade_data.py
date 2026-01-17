import os
import glob
import pandas as pd
import numpy as np
import warnings
import argparse
import sys

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# volej: python upgrade_data.py --data-dir "../sample_nbl_season_data" --output "sample.csv" --current-season "2025-08-06"
# current season je start aktualni sezony

# ==========================================
# 1. LOAD AND PREPARE DATA (RECURSIVE)
# ==========================================
def load_data(root_folder):
    """
    Recursively loads data from the specified folder.
    """
    if not os.path.exists(root_folder):
        print(f"❌ Error: Folder '{root_folder}' does not exist.")
        return None, None, None

    print(f"Searching folder '{root_folder}' and all subfolders...")

    def fetch_and_concat(keyword):
        # Pattern: data/**/*keyword*.csv (recursive search)
        search_pattern = os.path.join(root_folder, '**', f'*{keyword}*.csv')
        files = glob.glob(search_pattern, recursive=True)

        if not files:
            print(f"   ⚠️ Warning: No files containing '{keyword}' found.")
            return None

        print(f"Found {len(files)} files for '{keyword}'. Merging...")

        df_list = []
        for f in files:
            try:
                # Load CSV (assuming semicolon delimiter)
                df = pd.read_csv(f, sep=';')
                df_list.append(df)
            except Exception as e:
                print(f"   ❌ Error reading file {f}: {e}")

        if df_list:
            return pd.concat(df_list, ignore_index=True)
        else:
            return None

    # 1. Box Scores
    df_box = fetch_and_concat('box_score')
    # 2. Play-by-Play
    df_pbp = fetch_and_concat('pbp')
    # 3. Shots Spatial
    df_shots = fetch_and_concat('shots')

    # Verify we have all data types
    if df_box is None or df_pbp is None or df_shots is None:
        print("❌ Critical Error: Missing data types. Script cannot continue.")
        return None, None, None

    # --- DATA PREPARATION ---
    print("Data loaded. Performing basic cleaning...")

    def parse_min(t):
        try:
            if pd.isna(t): return 0.0
            t_str = str(t)
            if ':' in t_str:
                parts = t_str.split(':')
                return int(parts[0]) + int(parts[1]) / 60
            return float(t_str)
        except:
            return 0.0

    if 'minutes' in df_box.columns:
        df_box['MIN_FL'] = df_box['minutes'].apply(parse_min)

    for df in [df_box, df_pbp, df_shots]:
        if 'player' in df.columns:
            df['player'] = df['player'].astype(str).str.lower().str.strip()
        if 'team' in df.columns:
            df['team'] = df['team'].astype(str).str.lower().str.strip()

    return df_box, df_pbp, df_shots


# ==========================================
# 2. HOME TEAM DETECTION
# ==========================================
def identify_home_team(df_pbp):
    df_pbp = df_pbp.sort_values(['match_id', 'period', 'clock'], ascending=[True, True, False])
    df_pbp['prev_s1'] = df_pbp.groupby('match_id')['score_home'].shift(1).fillna(0)

    home_score_events = df_pbp[
        (df_pbp['score_home'] > df_pbp['prev_s1']) &
        (df_pbp['team'].notna())
        ]

    match_home_map = home_score_events.groupby('match_id')['team'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    ).to_dict()

    return match_home_map


def aggregate_raw_stats(df_box, df_pbp, df_shots):
    print("Calculating missing stats (FGA, FTA, OREB)...")

    # --- SHOTS ---
    df_shots['is_made'] = df_shots['result'].isin(['made', 'Make', 1])
    df_shots['is_3pt'] = df_shots['action'].astype(str).str.contains('3pt', case=False, na=False)
    df_shots['is_3pt_made'] = df_shots['is_made'] & df_shots['is_3pt']

    shots_agg = df_shots.groupby(['match_id', 'player']).agg(
        FGA=('result', 'count'),
        FGM=('is_made', 'sum'),
        FG3A=('is_3pt', 'sum'),
        FG3M=('is_3pt_made', 'sum')
    ).reset_index()

    # --- PBP ---
    is_ft = df_pbp['action_type'] == 'freethrow'
    is_ft_made = is_ft & (df_pbp['success'] == 1)
    is_reb = df_pbp['action_type'] == 'rebound'
    is_oreb = is_reb & (df_pbp['sub_type'] == 'offensive')
    is_dreb = is_reb & (df_pbp['sub_type'] == 'defensive')

    pbp_metrics = df_pbp[['match_id', 'player']].copy()
    pbp_metrics['FTA'] = is_ft.astype(int)
    pbp_metrics['FTM'] = is_ft_made.astype(int)
    pbp_metrics['OREB'] = is_oreb.astype(int)
    pbp_metrics['DREB'] = is_dreb.astype(int)

    pbp_agg = pbp_metrics.groupby(['match_id', 'player']).sum().reset_index()

    # --- MERGE ---
    df_full = pd.merge(df_box, shots_agg, on=['match_id', 'player'], how='left')
    df_full = pd.merge(df_full, pbp_agg, on=['match_id', 'player'], how='left')

    cols = ['FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 'OREB', 'DREB']
    df_full[cols] = df_full[cols].fillna(0)

    home_map = identify_home_team(df_pbp)
    df_full['Home_Team_Name'] = df_full['match_id'].map(home_map)
    df_full['Is_Home'] = (df_full['team'] == df_full['Home_Team_Name']).astype(int)

    return df_full


# ==========================================
# 3. SPATIAL & SHOOTING
# ==========================================
def calculate_spatial(df_shots):
    print("Calculating Spatial Shooting...")

    def get_zone(row):
        sub = str(row.get('sub_type', '')).lower()
        act = str(row.get('action', '')).lower()
        if any(x in sub for x in ['layup', 'dunk', 'driving']): return 'Rim'
        if '3pt' in act:
            y = row.get('y', 50)
            if y < 10 or y > 90: return 'Corner3'
            return 'Arc3'
        return 'MidRange'

    df_shots['Zone'] = df_shots.apply(get_zone, axis=1)
    df_shots['Pts'] = df_shots.apply(lambda x: (3 if '3pt' in str(x['action']) else 2) if x['result'] == 'made' else 0,
                                     axis=1)

    stats = df_shots.groupby(['match_id', 'player', 'Zone']).agg(
        Attempts=('result', 'count'),
        Made=('result', lambda x: (x == 'made').sum()),
        Points=('Pts', 'sum')
    ).unstack(fill_value=0)

    stats.columns = [f'{c[1]}_{c[0]}' for c in stats.columns]
    stats = stats.reset_index()
    stats['Total_Shots'] = stats[[c for c in stats.columns if 'Attempts' in c]].sum(axis=1)

    for z in ['Rim', 'MidRange', 'Corner3', 'Arc3']:
        att_col = f'{z}_Attempts'
        pts_col = f'{z}_Points'
        if att_col in stats.columns:
            stats[f'{z}_Freq'] = stats[att_col] / stats['Total_Shots'].replace(0, np.nan)
            stats[f'{z}_PPS'] = stats[pts_col] / stats[att_col].replace(0, np.nan)

    return stats.drop(columns=['Total_Shots'])


# ==========================================
# 4. ADVANCED MATH
# ==========================================
def calculate_advanced(df):
    print("Calculating Advanced Metrics...")

    team_grp = df.groupby(['match_id', 'team'])
    team_stats = team_grp.agg(
        Tm_MIN=('MIN_FL', 'sum'),
        Tm_FGA=('FGA', 'sum'),
        Tm_FTA=('FTA', 'sum'),
        Tm_TOV=('turnovers', 'sum'),
        Tm_REB=('rebounds', 'sum'),
        Tm_OREB=('OREB', 'sum'),
        Tm_PTS=('points', 'sum')
    ).reset_index()

    df = pd.merge(df, team_stats, on=['match_id', 'team'])

    df['Poss'] = df['FGA'] + 0.44 * df['FTA'] - df['OREB'] + df['turnovers']

    term1 = (df['FGA'] + 0.44 * df['FTA'] + df['turnovers']) * (df['Tm_MIN'] / 5)
    term2 = df['MIN_FL'] * (df['Tm_FGA'] + 0.44 * df['Tm_FTA'] + df['Tm_TOV'])
    df['USG%'] = 100 * (term1 / term2.replace(0, np.nan))

    df['ORtg'] = 100 * (df['points'] / df['Poss'].replace(0, np.nan))
    df['TS%'] = df['points'] / (2 * (df['FGA'] + 0.44 * df['FTA'])).replace(0, np.nan)
    df['eFG%'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, np.nan)

    df['Reb_Share'] = df['rebounds'] / df['Tm_REB'].replace(0, np.nan)
    df['Scoring_Share'] = df['points'] / df['Tm_PTS'].replace(0, np.nan)
    df['Fouls_Per_Min'] = df['fouls'] / df['MIN_FL'].replace(0, np.nan)
    df['Blowout_Factor'] = df['plus_minus'] / df['MIN_FL'].replace(0, np.nan)

    return df


# ==========================================
# 5. CLUTCH STATS
# ==========================================
def calculate_clutch(df_pbp):
    print("Calculating Clutch Statistics...")

    def clock_to_sec(c):
        try:
            p = str(c).split(':')
            return int(p[0]) * 60 + int(p[1])
        except:
            return 999

    df_pbp['Sec_Rem'] = df_pbp['clock'].apply(clock_to_sec)
    df_pbp['Score_Diff'] = abs(df_pbp['score_home'] - df_pbp['score_away'])

    clutch_mask = (df_pbp['period'] >= 4) & (df_pbp['Score_Diff'] <= 5) & (df_pbp['Sec_Rem'] <= 300)
    clutch = df_pbp[clutch_mask]

    shots = clutch[clutch['action_type'].isin(['2pt', '3pt'])]
    clutch_stats = shots.groupby(['match_id', 'player']).agg(
        Clutch_FGA=('success', 'count'),
        Clutch_FGM=('success', 'sum')
    ).reset_index()

    clutch_stats['Clutch_FG%'] = clutch_stats['Clutch_FGM'] / clutch_stats['Clutch_FGA'].replace(0, np.nan)
    return clutch_stats


# ==========================================
# 6. CONTEXT & OPPONENT (With Args)
# ==========================================
def add_context_and_opponent(df, current_season_start, last_season_start):
    print("Analyzing Context (Opponent, Dates, Weights)...")

    # A) Opponent Stats
    matches = df[['match_id', 'team']].drop_duplicates()
    matches_merged = pd.merge(matches, matches, on='match_id')
    opp_map = matches_merged[matches_merged['team_x'] != matches_merged['team_y']]
    opp_map = opp_map.rename(columns={'team_x': 'team', 'team_y': 'opponent'})

    df = pd.merge(df, opp_map[['match_id', 'team', 'opponent']], on=['match_id', 'team'], how='left')

    match_team_stats = df.groupby(['match_id', 'team']).agg(
        Pts_Scored=('points', 'sum'),
        FG_Pct=('FGM', lambda x: x.sum() / df.loc[x.index, 'FGA'].sum() if df.loc[x.index, 'FGA'].sum() > 0 else 0)
    ).reset_index()

    match_team_stats = pd.merge(match_team_stats, opp_map, on=['match_id', 'team'])

    defense_stats = match_team_stats.groupby('opponent').agg(
        Opp_Avg_Pts_Allowed=('Pts_Scored', 'mean'),
        Opp_FG_Allowed=('FG_Pct', 'mean')
    ).reset_index()

    df = pd.merge(df, defense_stats, left_on='opponent', right_on='opponent', how='left')

    # --- FIX MISSING DATES (FFILL/BFILL) ---
    # 1. Nahradíme "unknown" za NaN
    df['date'] = df['date'].replace(['unknown', 'None', ''], np.nan)

    # 2. Seřadíme, aby doplnění dávalo smysl (podle ID)
    df = df.sort_values(['match_id', 'team', 'player'])

    # 3. Doplníme chybějící data z okolních řádků
    df['date'] = df['date'].ffill().bfill()
    # ---------------------------------------

    # C) Time Series Features
    df['Date_Parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['player', 'Date_Parsed'])

    df['Consistency_Index'] = df.groupby('player')['points'].transform(lambda x: x.rolling(5, min_periods=2).std())
    df['Prev_Date'] = df.groupby('player')['Date_Parsed'].shift(1)
    df['Days_Rest'] = (df['Date_Parsed'] - df['Prev_Date']).dt.days.fillna(7)

    # Convert args to Timestamps
    cur_season_ts = pd.Timestamp(current_season_start)
    last_season_ts = pd.Timestamp(last_season_start)

    def get_weight(d):
        if pd.isna(d): return 0.5
        if d > cur_season_ts: return 1.0
        if d > last_season_ts: return 0.7
        return 0.4

    df['Season_Weight'] = df['Date_Parsed'].apply(get_weight)

    return df.drop(columns=['Date_Parsed', 'Prev_Date'])


# ==========================================
# MAIN LOOP
# ==========================================
def main():
    # --- ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Moneyball Feature Engineering Script")

    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Path to the folder containing data subfolders (default: ../data)')

    parser.add_argument('--output', type=str, default='data/final.csv',
                        help='Path where the final CSV will be saved (default: data/final.csv)')

    parser.add_argument('--current-season', type=str, default='2024-08-01',
                        help='Start date of current season for weighting (YYYY-MM-DD)')

    parser.add_argument('--last-season', type=str, default='2023-08-01',
                        help='Start date of previous season for weighting (YYYY-MM-DD)')

    args = parser.parse_args()

    # -----------------------

    print(f"STARTING ENRICHMENT")
    print(f"   Input Dir: {args.data_dir}")
    print(f"   Output File: {args.output}")
    print(f"   Seasons: Current > {args.current_season}, Last > {args.last_season}\n")

    # 1. Load
    df_box, df_pbp, df_shots = load_data(args.data_dir)
    if df_box is None: return

    # 2. Basic Stats
    df_main = aggregate_raw_stats(df_box, df_pbp, df_shots)

    # 3. Spatial Features
    spatial = calculate_spatial(df_shots)
    df_main = pd.merge(df_main, spatial, on=['match_id', 'player'], how='left')

    # 4. Advanced Metrics
    df_main = calculate_advanced(df_main)

    # 5. Clutch
    clutch = calculate_clutch(df_pbp)
    df_main = pd.merge(df_main, clutch, on=['match_id', 'player'], how='left')

    # 6. Context & Opponent (Passing args for seasons)
    df_final = add_context_and_opponent(df_main, args.current_season, args.last_season)

    # 7. Clean & Save
    df_final.fillna(0, inplace=True)

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df_final.to_csv(args.output, index=False, sep=';')
    print(f"\nDONE! Results saved to: {args.output}")
    print(f"Matrix dimensions: {df_final.shape}")

    ''' Preview
    print("\nTop Players Preview (by Points):")
    if 'points' in df_final.columns:
        print(df_final.sort_values('points', ascending=False).head(2).to_string(index=False))
    '''

if __name__ == "__main__":
    main()