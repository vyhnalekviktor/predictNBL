import os
import glob
import pandas as pd
import numpy as np
import warnings
import argparse

warnings.filterwarnings('ignore')


def load_data(root_folder):
    if not os.path.exists(root_folder):
        print(f"❌ Error: Folder '{root_folder}' does not exist.")
        return None, None, None

    print(f"Searching folder '{root_folder}'...")

    def fetch_and_concat(keyword):
        search_pattern = os.path.join(root_folder, '**', f'*{keyword}*.csv')
        files = glob.glob(search_pattern, recursive=True)
        if not files: return None

        df_list = []
        for f in files:
            try:
                df = pd.read_csv(f, sep=';')
                df_list.append(df)
            except:
                pass
        return pd.concat(df_list, ignore_index=True) if df_list else None

    df_box = fetch_and_concat('box_score')
    df_pbp = fetch_and_concat('pbp')
    df_shots = fetch_and_concat('shots')

    if df_box is None or df_pbp is None:
        print("❌ Critical Error: Missing box score or pbp data.")
        return None, None, None

    for df in [df_box, df_pbp, df_shots]:
        if df is not None:
            if 'player' in df.columns:
                df['player'] = df['player'].astype(str).str.lower().str.strip()
            if 'team' in df.columns:
                df['team'] = df['team'].astype(str).str.lower().str.strip()

    def parse_min(t):
        try:
            if pd.isna(t): return 0.0
            t_str = str(t)
            if ':' in t_str:
                p = t_str.split(':')
                return int(p[0]) + int(p[1]) / 60
            return float(t_str)
        except:
            return 0.0

    if 'minutes' in df_box.columns:
        df_box['MIN_FL'] = df_box['minutes'].apply(parse_min)

    return df_box, df_pbp, df_shots

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

def estimate_positions(df_box):
    print("🕵️  Estimating Player Positions (G/F/C)...")
    stats = df_box.groupby('player').agg({
        'assists': 'mean',
        'rebounds': 'mean',
        'blocks': 'mean',
        'match_id': 'count'
    })

    def assign_pos(row):
        if row['assists'] > 2.5 and row['assists'] > row['rebounds']: return 'G'
        if row['rebounds'] > 4.5 and row['blocks'] > 0.3: return 'C'
        if row['rebounds'] > row['assists'] * 2: return 'C'
        return 'F'

    stats['Pos_Est'] = stats.apply(assign_pos, axis=1)
    return stats[['Pos_Est']]

def calculate_garbage_time(df_pbp):
    print("Filtering Garbage Time (Score Diff > 20 in 4Q)...")
    df_pbp['score_diff'] = abs(df_pbp['score_home'] - df_pbp['score_away'])


    is_gt = (df_pbp['period'] >= 4) & (df_pbp['score_diff'] >= 20)
    gt_events = df_pbp[is_gt].copy()

    gt_events['pts_val'] = 0
    gt_events.loc[(gt_events['action_type'].str.contains('2pt')) & (gt_events['success'] == 1), 'pts_val'] = 2
    gt_events.loc[(gt_events['action_type'].str.contains('3pt')) & (gt_events['success'] == 1), 'pts_val'] = 3
    gt_events.loc[(gt_events['action_type'] == 'freethrow') & (gt_events['success'] == 1), 'pts_val'] = 1

    gt_stats = gt_events.groupby(['match_id', 'player'])['pts_val'].sum().reset_index()
    gt_stats.rename(columns={'pts_val': 'GT_Points'}, inplace=True)
    return gt_stats

def calculate_spatial_and_quality(df_shots):
    if df_shots is None: return None
    print("Calculating Spatial Stats & Shot Quality...")

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
    df_shots['Pts'] = df_shots.apply(
        lambda x: (3 if '3pt' in str(x['action']) else 2) if x['result'] in ['made', 'Make', 1] else 0, axis=1)
    df_shots['is_made'] = df_shots['result'].isin(['made', 'Make', 1])

    iso_keywords = ['driving', 'turnaround', 'step back', 'pullup', 'dunk']
    df_shots['is_iso'] = df_shots['sub_type'].astype(str).str.lower().apply(lambda x: any(k in x for k in iso_keywords))

    stats = df_shots.groupby(['match_id', 'player']).agg(
        ISO_Attempts=('is_iso', 'sum'),
        ISO_Makes=('is_iso', lambda x: x[df_shots.loc[x.index, 'is_made']].sum())
    )

    zone_stats = df_shots.groupby(['match_id', 'player', 'Zone']).agg(
        Attempts=('result', 'count'),
        Points=('Pts', 'sum')
    ).unstack(fill_value=0)
    zone_stats.columns = [f'{c[1]}_{c[0]}' for c in zone_stats.columns]

    final_spatial = pd.merge(stats, zone_stats, on=['match_id', 'player'], how='left')

    final_spatial['Total_Shots'] = final_spatial[
        [c for c in final_spatial.columns if 'Attempts' in c and 'ISO' not in c]].sum(axis=1)
    for z in ['Rim', 'MidRange', 'Corner3', 'Arc3']:
        if f'{z}_Attempts' in final_spatial.columns:
            final_spatial[f'{z}_Freq'] = final_spatial[f'{z}_Attempts'] / final_spatial['Total_Shots'].replace(0,
                                                                                                               np.nan)
            final_spatial[f'{z}_PPS'] = final_spatial[f'{z}_Points'] / final_spatial[f'{z}_Attempts'].replace(0, np.nan)

    return final_spatial.drop(columns=['Total_Shots'])


def calculate_clutch(df_pbp):
    print("🔥 Calculating Clutch Statistics...")

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

def aggregate_and_calculate_advanced(df_box, df_pbp, df_shots):
    print("📊 Aggregating and Calculating Advanced Metrics...")

    if df_shots is not None:
        if 'is_made' not in df_shots.columns:
            df_shots['is_made'] = df_shots['result'].isin(['made', 'Make', 1])

        shots_agg = df_shots.groupby(['match_id', 'player']).agg(
            FGA=('result', 'count'),
            FGM=('is_made', 'sum')
        ).reset_index()
    else:
        shots_agg = pd.DataFrame()


    is_ft = df_pbp['action_type'] == 'freethrow'
    is_ft_made = is_ft & (df_pbp['success'] == 1)
    is_reb = df_pbp['action_type'] == 'rebound'
    is_oreb = is_reb & (df_pbp['sub_type'] == 'offensive')

    pbp_agg = df_pbp.groupby(['match_id', 'player']).agg(
        FTA=('action_type', lambda x: (x == 'freethrow').sum()),
        FTM=('success', lambda x: x[is_ft_made].sum()),
        OREB=('sub_type', lambda x: x[is_oreb].sum())
    ).reset_index()

    df = pd.merge(df_box, shots_agg, on=['match_id', 'player'], how='left')
    df = pd.merge(df, pbp_agg, on=['match_id', 'player'], how='left')

    for c in ['FGA', 'FGM', 'FTA', 'FTM', 'OREB']:
        if c in df.columns: df[c] = df[c].fillna(0)

    home_map = identify_home_team(df_pbp)
    df['Home_Team_Name'] = df['match_id'].map(home_map)
    df['Is_Home'] = (df['team'] == df['Home_Team_Name']).astype(int)

    team_grp = df.groupby(['match_id', 'team'])
    team_stats = team_grp.agg(
        Tm_MIN=('MIN_FL', 'sum'),
        Tm_FGA=('FGA', 'sum'),
        Tm_FTA=('FTA', 'sum'),
        Tm_TOV=('turnovers', 'sum'),
        Tm_REB=('rebounds', 'sum'),
        Tm_PTS=('points', 'sum')
    ).reset_index()

    df = pd.merge(df, team_stats, on=['match_id', 'team'])

    df['Tm_Poss'] = df['Tm_FGA'] + 0.44 * df['Tm_FTA'] + df['Tm_TOV']  # Basic Formula
    df['Pace'] = 40 * (df['Tm_Poss'] / df['Tm_MIN'].replace(0, np.nan))

    term1 = (df['FGA'] + 0.44 * df['FTA'] + df['turnovers']) * (df['Tm_MIN'] / 5)
    term2 = df['MIN_FL'] * df['Tm_Poss']
    df['USG%'] = 100 * (term1 / term2.replace(0, np.nan))

    df['TS%'] = df['points'] / (2 * (df['FGA'] + 0.44 * df['FTA'])).replace(0, np.nan)
    df['eFG%'] = (df['FGM'] + 0.5 * (df['points'] - df['FGM'] * 2)) / df['FGA'].replace(0, np.nan)  # Approx 3PM
    df['Fouls_Per_Min'] = df['fouls'] / df['MIN_FL'].replace(0, np.nan)

    return df

def add_context_complete(df, current_season_start):
    print("Calculating Context (Opponent, DvP, Days Rest)...")

    # 1. Id opponent
    matches = df[['match_id', 'team']].drop_duplicates()
    matches_merged = pd.merge(matches, matches, on='match_id')
    opp_map = matches_merged[matches_merged['team_x'] != matches_merged['team_y']].rename(
        columns={'team_x': 'team', 'team_y': 'opponent'})
    df = pd.merge(df, opp_map[['match_id', 'team', 'opponent']], on=['match_id', 'team'], how='left')

    # 2. Opponent Stats (Global Defense)
    opp_stats = df.groupby('opponent').agg(
        Opp_Avg_Pts_Allowed=('points', 'sum'),
        Opp_Avg_Pace=('Pace', 'mean')
    ).reset_index()
    # Normalize SUM to MEAN
    opp_games = df.groupby('opponent')['match_id'].nunique().reset_index().rename(columns={'match_id': 'Games'})
    opp_stats = pd.merge(opp_stats, opp_games, on='opponent')
    opp_stats['Opp_Avg_Pts_Allowed'] = opp_stats['Opp_Avg_Pts_Allowed'] / opp_stats['Games']

    df = pd.merge(df, opp_stats[['opponent', 'Opp_Avg_Pts_Allowed', 'Opp_Avg_Pace']], on='opponent', how='left')

    # 3. DvP (Defense vs Position)
    dvp_base = df.groupby(['match_id', 'opponent', 'Pos_Est'])['points'].sum().reset_index()
    dvp_stats = dvp_base.groupby(['opponent', 'Pos_Est'])['points'].mean().reset_index().rename(
        columns={'points': 'DvP_Avg_Pts'})
    df = pd.merge(df, dvp_stats, on=['opponent', 'Pos_Est'], how='left')

    # 4. Days Rest
    df['date'] = pd.to_datetime(df['date'], errors='coerce').ffill().bfill()
    df['Date_Parsed'] = df['date']
    df = df.sort_values(['player', 'Date_Parsed'])
    df['Prev_Date'] = df.groupby('player')['Date_Parsed'].shift(1)
    df['Days_Rest'] = (df['Date_Parsed'] - df['Prev_Date']).dt.days.fillna(7)

    # 5. Season Weight
    season_ts = pd.Timestamp(current_season_start)
    df['Season_Weight'] = df['Date_Parsed'].apply(lambda d: 1.0 if d > season_ts else 0.5)

    return df.drop(columns=['Date_Parsed', 'Prev_Date'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--output', type=str, default='data/final.csv')
    parser.add_argument('--current-season', type=str, default='2024-08-01')
    args = parser.parse_args()

    print("STARTING FULL DATA ENRICHMENT")

    # 1. Load
    df_box, df_pbp, df_shots = load_data(args.data_dir)
    if df_box is None: return

    # 2. Position Estimate
    pos_map = estimate_positions(df_box)
    df_box = pd.merge(df_box, pos_map, on='player', how='left')
    df_box['Pos_Est'] = df_box['Pos_Est'].fillna('F')

    # 3. Garbage Time
    gt_data = calculate_garbage_time(df_pbp)
    df_box = pd.merge(df_box, gt_data, on=['match_id', 'player'], how='left')
    df_box['GT_Points'] = df_box['GT_Points'].fillna(0)
    df_box['Adj_Points'] = df_box['points'] - df_box['GT_Points']

    # 4. Main Stats + Advanced
    df = aggregate_and_calculate_advanced(df_box, df_pbp, df_shots)

    # 5. Spatial & Clutch
    spatial = calculate_spatial_and_quality(df_shots)
    if spatial is not None:
        df = pd.merge(df, spatial, on=['match_id', 'player'], how='left')

    clutch = calculate_clutch(df_pbp)
    if clutch is not None:
        df = pd.merge(df, clutch, on=['match_id', 'player'], how='left')

    # 6. Context (DvP, Pace, Rest)
    df = add_context_complete(df, args.current_season)

    # 7. Final Cleanup
    df.fillna(0, inplace=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False, sep=';')
    print(f"\n✅ DONE! Saved to: {args.output}")
    print(f"   Matrix: {df.shape}. Includes: Pace, DvP, Spatial, Clutch, GarbageTime.")


if __name__ == "__main__":
    main()