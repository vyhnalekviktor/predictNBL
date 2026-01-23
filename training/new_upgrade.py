import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
import argparse

warnings.filterwarnings('ignore')


# ==========================================
# POMOCNÉ FUNKCE
# ==========================================

def parse_clock(t_str):
    if pd.isna(t_str): return 0.0
    try:
        p = str(t_str).split(':')
        return int(p[0]) * 60 + float(p[1]) if len(p) >= 2 else float(t_str)
    except:
        return 0.0


def parse_min(t_str):
    if pd.isna(t_str): return 0.0
    try:
        p = str(t_str).split(':')
        return int(p[0]) + int(p[1]) / 60
    except:
        return 0.0


def load_data(root):
    if not os.path.exists(root): return None, None, None
    print(f"📂 Načítám data z: {root}")

    def get_df(k):
        f = glob.glob(os.path.join(root, '**', k), recursive=True)
        return pd.concat([pd.read_csv(x, sep=';') for x in f], ignore_index=True) if f else None

    df_b, df_p, df_s = get_df('*box_score*.csv'), get_df('*pbp*.csv'), get_df('*shots*.csv')

    for d in [df_b, df_p, df_s]:
        if d is not None:
            if 'player' in d.columns: d['player'] = d['player'].astype(str).str.lower().str.strip()
            if 'team' in d.columns: d['team'] = d['team'].astype(str).str.lower().str.strip()

    if df_b is not None and 'minutes' in df_b.columns: df_b['MIN_FL'] = df_b['minutes'].apply(parse_min)
    if df_p is not None and 'clock' in df_p.columns: df_p['SEC'] = df_p['clock'].apply(parse_clock)

    return df_b, df_p, df_s


# ==========================================
# MODUL 1: LINEUPY & SYNERGIE
# ==========================================

def process_lineups(df_pbp, df_box):
    print("🤝 Modul 1: Synergie & Lineup rekonstrukce...")
    if df_pbp is None:
        df_box['Synergy_Score'] = 0
        df_box['Dependency_Index'] = 0
        return df_box

    df_pbp = df_pbp.sort_values(['match_id', 'period', 'SEC'], ascending=[True, True, False])
    synergy_ratings = {}

    for mid, group in df_pbp.groupby('match_id'):
        match_box = df_box[df_box['match_id'] == mid]
        starters = match_box[match_box['starter'] == 1]['player'].tolist()
        on_court = set(starters)
        last_score_sum = 0
        pm_log = {p: [] for p in match_box['player'].unique()}

        for idx, row in group.iterrows():
            current_score_sum = row['score_home'] + row['score_away']
            diff = current_score_sum - last_score_sum

            if diff != 0:
                for p in on_court:
                    if p in pm_log:
                        pm_log[p].append(diff)

            if row['action_type'] == 'substitution':
                p = row['player']
                if row['sub_type'] == 'in':
                    on_court.add(p)
                elif row['sub_type'] == 'out':
                    if p in on_court: on_court.remove(p)

            last_score_sum = current_score_sum

        for p, deltas in pm_log.items():
            synergy_ratings[(mid, p)] = sum(deltas) / len(deltas) if deltas else 0

    syn_df = pd.DataFrame(list(synergy_ratings.items()), columns=['keys', 'Synergy_Score'])
    syn_df[['match_id', 'player']] = pd.DataFrame(syn_df['keys'].tolist(), index=syn_df.index)
    syn_df = syn_df.drop(columns=['keys'])

    df_box = pd.merge(df_box, syn_df, on=['match_id', 'player'], how='left')

    df_pbp['next_player'] = df_pbp['player'].shift(-1)
    assists = df_pbp[(df_pbp['action_type'] == 'assist')]
    dep_map = assists.groupby(['match_id', 'next_player']).size().reset_index(name='Assisted_Baskets')
    df_box = pd.merge(df_box, dep_map, left_on=['match_id', 'player'], right_on=['match_id', 'next_player'], how='left')
    df_box['Dependency_Index'] = df_box['Assisted_Baskets'].fillna(0) / df_box['points'].replace(0, 1)

    df_box['Gravity'] = (df_box['points'] * 0.4) + (df_box['efficiency'] * 0.6)

    return df_box


# ==========================================
# MODUL 2: STYL
# ==========================================

def calculate_style(df_main, df_pbp, df_shots):
    print("🎨 Modul 2: Styl, Pace & Shot Quality...")
    df_main['Pace_Est'] = (df_main['points'] + df_main['rebounds'] + df_main['turnovers']) / df_main['MIN_FL'].replace(
        0, 1)
    df_main['Raw_PPM'] = df_main['points'] / df_main['MIN_FL'].replace(0, 1)

    if df_pbp is not None:
        fd = df_pbp[df_pbp['action_type'] == 'foulon'].groupby(['match_id', 'player']).size().reset_index(
            name='Fouls_Drawn')
        df_main = pd.merge(df_main, fd, on=['match_id', 'player'], how='left')
        df_main['Fouls_Drawn'] = df_main['Fouls_Drawn'].fillna(0)
    else:
        df_main['Fouls_Drawn'] = 0

    if df_shots is not None:
        df_shots['dist'] = np.minimum(
            np.sqrt((df_shots['x'] - 5) ** 2 + (df_shots['y'] - 50) ** 2),
            np.sqrt((df_shots['x'] - 95) ** 2 + (df_shots['y'] - 50) ** 2)
        )
        conds = [(df_shots['dist'] < 8), ((df_shots['dist'] > 22) & (abs(df_shots['y'] - 50) > 35))]
        df_shots['Quality_Shot'] = np.select(conds, [1, 1], default=0)
        sq = df_shots.groupby(['match_id', 'player'])['Quality_Shot'].mean().reset_index(name='Shot_Quality_Avg')
        df_main = pd.merge(df_main, sq, on=['match_id', 'player'], how='left')
    else:
        df_main['Shot_Quality_Avg'] = 0.5

    return df_main


# ==========================================
# MODUL 3: MATCHUPS & ADVANCED OPPONENT (PBP)
# ==========================================

def calculate_matchups(df_box, df_main):
    print("🛡️ Modul 3: Pokročilé Matchupy a Detekce Hrozeb (PBP)...")

    # 1. Pozice
    stats = df_box.groupby('player').agg({'rebounds': 'mean', 'blocks': 'mean', 'assists': 'mean', 'points': 'mean'})

    def get_pos(r):
        if r['rebounds'] > 6 and r['blocks'] > 0.5: return 'C'
        if r['assists'] > 3.5: return 'PG'
        if r['rebounds'] > 4.5: return 'PF'
        if r['points'] > 12 and r['assists'] < 2.5: return 'SG'
        return 'SF'

    stats['Size_Index'] = (stats['rebounds'] * 2 + stats['blocks'] * 5)
    stats['Pos_Est'] = stats.apply(get_pos, axis=1)
    df_main = pd.merge(df_main, stats[['Pos_Est', 'Size_Index']], on='player', how='left')

    # 2. AGREGACE SOUPEŘE VČETNĚ PBP HROZEB
    if 'opponent' in df_main.columns:
        team_stats = df_main.groupby(['match_id', 'team']).agg({
            'points': 'sum',
            'fouls': 'sum',
            'Pace_Est': 'mean',
            'Fouls_Drawn': 'max',  # NEW: PBP Faul magnet
            'Synergy_Score': 'max'  # NEW: PBP Elitní obránce
        }).reset_index()

        opp_stats = team_stats.rename(columns={
            'team': 'opponent',
            'points': 'Opp_Avg_Pts_Allowed',
            'fouls': 'Opp_Avg_Fouls',
            'Pace_Est': 'Opp_Avg_Pace',
            'Fouls_Drawn': 'Opp_Max_Foul_Drawer',
            'Synergy_Score': 'Opp_Best_Def_Synergy'
        })

        df_main = pd.merge(df_main, opp_stats, on=['match_id', 'opponent'], how='left')

        # VÝPOČET OBRANY PROTI POZICI (Opp_Allowed_Pos_Avg) - TOTO CHYBĚLO!
        opp_def = df_main.groupby(['opponent', 'Pos_Est'])['points'].mean().reset_index(name='Opp_Allowed_Pos_Avg')
        df_main = pd.merge(df_main, opp_def, on=['opponent', 'Pos_Est'], how='left')

        # Opponent Size & Mismatch
        opp_size = df_main.groupby(['opponent', 'Pos_Est'])['Size_Index'].mean().reset_index(name='Opp_Avg_Size')
        df_main = pd.merge(df_main, opp_size, on=['opponent', 'Pos_Est'], how='left')
        df_main['Mismatch_Adv'] = df_main['Size_Index'] - df_main['Opp_Avg_Size']

        # Def_Suppressor calculation
        player_avgs = df_main.groupby('player')['points'].transform('mean')
        df_main['Def_Suppressor'] = df_main['Opp_Allowed_Pos_Avg'] - player_avgs

    return df_main


# ==========================================
# MODUL 4: QUARTERS & ODDS
# ==========================================

def calculate_quarters(df_box, df_pbp):
    if df_pbp is None: return df_box
    scorers = df_pbp[df_pbp['action_type'].str.contains('2pt|3pt|freethrow', na=False) & (df_pbp['success'] == 1)]
    q_stats = scorers.pivot_table(index=['match_id', 'player'], columns='period', values='action_type', aggfunc='count',
                                  fill_value=0)
    q_stats.columns = [f'Q{c}_Activity' for c in q_stats.columns]
    df_box = pd.merge(df_box, q_stats, on=['match_id', 'player'], how='left')
    return df_box


def calculate_clutch(df_pbp):
    if df_pbp is None: return None
    if 'score_home' in df_pbp.columns:
        df_pbp['diff'] = abs(df_pbp['score_home'] - df_pbp['score_away'])
        mask = (df_pbp['period'] == 4) & (df_pbp['diff'] <= 5) & (df_pbp['SEC'] <= 300)
        return df_pbp[mask].groupby(['match_id', 'player']).size().reset_index(name='Clutch_Events')
    return None


def calculate_garbage_time(df_pbp):
    if df_pbp is None or 'score_home' not in df_pbp.columns: return None
    df_pbp['diff'] = abs(df_pbp['score_home'] - df_pbp['score_away'])
    is_gt = (df_pbp['period'] == 4) & (df_pbp['diff'] >= 20)
    return df_pbp[is_gt].groupby(['match_id', 'player']).size().reset_index(name='GT_Activity_Index')


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='sample_nbl_season_data')
    parser.add_argument('--output', default='data/final_enhanced.csv')
    parser.add_argument('--current-season', type=str, default=None,
                        help='Start year of current season (YYYY) or date (YYYY-MM-DD)')
    args = parser.parse_args()

    print("🚀 START: Advanced NBL Processor (PBP THREAT DETECTION + SEASON WEIGHTS)")

    df_box, df_pbp, df_shots = load_data(args.data_dir)
    if df_box is None: return

    matches = df_box[['match_id', 'team']].drop_duplicates()
    merged = pd.merge(matches, matches, on='match_id')
    opp_map = merged[merged['team_x'] != merged['team_y']].rename(columns={'team_x': 'team', 'team_y': 'opponent'})
    df = pd.merge(df_box, opp_map[['match_id', 'team', 'opponent']], on=['match_id', 'team'], how='left')

    if args.current_season:
        try:
            cy_str = str(args.current_season).strip().split('-')[0]
            current_year = int(cy_str)

            print(f"⚖️  Aplikuji váhy sezóny (Start aktuální: 6.8.{current_year})...")

            cutoff_current = pd.Timestamp(year=current_year, month=8, day=6)
            cutoff_prev = pd.Timestamp(year=current_year - 1, month=8, day=6)

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

                def calc_weight(d):
                    if pd.isna(d): return 0.4
                    if d >= cutoff_current: return 1.0
                    if d >= cutoff_prev: return 0.7
                    return 0.4

                df['Season_Weight'] = df['date'].apply(calc_weight)
                print(f"   ✅ Váhy přidány do sloupce 'Season_Weight': 1.0 (Aktuální), 0.7 (Minulá), 0.4 (Starší)")
            else:
                df['Season_Weight'] = 1.0
        except Exception as e:
            print(f"❌ Chyba při výpočtu vah: {e}")
            df['Season_Weight'] = 1.0
    else:
        df['Season_Weight'] = 1.0

    # Moduly
    df = process_lineups(df_pbp, df)
    df = calculate_style(df, df_pbp, df_shots)
    df = calculate_matchups(df_box, df)

    df = calculate_quarters(df, df_pbp)
    clutch = calculate_clutch(df_pbp)
    if clutch is not None: df = pd.merge(df, clutch, on=['match_id', 'player'], how='left')

    gt = calculate_garbage_time(df_pbp)
    if gt is not None: df = pd.merge(df, gt, on=['match_id', 'player'], how='left')

    nums = df.select_dtypes(include=[np.number]).columns
    df[nums] = df[nums].fillna(0)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, sep=';', index=False)

    print(f"✅ Uloženo: {args.output}")
    print(f"   NOVÉ SLOUPCE: 'Opp_Max_Foul_Drawer', 'Opp_Best_Def_Synergy', 'Raw_PPM', 'Opp_Allowed_Pos_Avg'")


if __name__ == "__main__":
    main()