import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# 1. KONFIGURACE A NAČTENÍ DAT
# ==========================================
def load_and_prep_data():
    print("Načítám data...")
    # Načtení s oddělovačem středník
    df_box = pd.read_csv('match_box_score.csv', sep=';')
    df_pbp = pd.read_csv('pbp_events.csv', sep=';')
    df_shots = pd.read_csv('shots_spatial.csv', sep=';')

    # Konverze času (mm:ss) na minuty (float) v box score
    def time_to_min(t_str):
        try:
            if pd.isna(t_str): return 0.0
            parts = str(t_str).split(':')
            return int(parts[0]) + int(parts[1]) / 60
        except:
            return 0.0

    df_box['MIN_FL'] = df_box['minutes'].apply(time_to_min)

    # Vyčištění jmen (lowercase pro jistotu joinů)
    for df in [df_box, df_pbp, df_shots]:
        if 'player' in df.columns:
            df['player'] = df['player'].str.lower().str.strip()
        if 'team' in df.columns:
            df['team'] = df['team'].str.lower().str.strip()

    return df_box, df_pbp, df_shots


# ==========================================
# 2. AGREGACE CHYBĚJÍCÍCH STATISTIK (FGA, FTA, OREB)
# ==========================================
def aggregate_basic_stats(df_box, df_pbp, df_shots):
    print("Dopočítávám chybějící statistiky (FGA, FTA, OREB) z PBP a Shots...")

    # 1. Střelba z pole (FGA, FGM, 3PM) ze shots_spatial
    shots_agg = df_shots.groupby(['match_id', 'player']).agg(
        FGA=('result', 'count'),
        FGM=('result', lambda x: (x == 'made').sum()),
        FG3A=('action', lambda x: (x == '3pt').sum()),
        FG3M=('result', lambda x: ((x == 'made') & (df_shots.loc[x.index, 'action'] == '3pt')).sum())
    ).reset_index()

    # 2. Trestné hody (FTA, FTM) a Doskoky (OREB, DREB) z PBP
    # Filtry pro PBP
    ft_mask = df_pbp['action_type'] == 'freethrow'
    reb_mask = df_pbp['action_type'] == 'rebound'

    pbp_agg = df_pbp.groupby(['match_id', 'player']).agg(
        FTA=('action_type', lambda x: (x == 'freethrow').sum()),
        FTM=('success', lambda x: x[ft_mask].sum() if ft_mask.any() else 0),
        OREB=('sub_type', lambda x: (x == 'offensive').sum()),
        DREB=('sub_type', lambda x: (x == 'defensive').sum())
    ).reset_index()

    # 3. Merge do hlavního Box Score
    df_full = pd.merge(df_box, shots_agg, on=['match_id', 'player'], how='left')
    df_full = pd.merge(df_full, pbp_agg, on=['match_id', 'player'], how='left')

    # Vyplnění NaN nulami (pokud hráč nestřílel, má NaN -> 0)
    cols_to_fix = ['FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 'OREB', 'DREB']
    df_full[cols_to_fix] = df_full[cols_to_fix].fillna(0)

    return df_full


# ==========================================
# 3. SPATIAL ENGINEERING (Zóny & PPS)
# ==========================================
def calculate_spatial_features(df_shots):
    print("Počítám Spatial Features (Zóny, PPS)...")

    # Definice zón (zjednodušená logika na základě souřadnic)
    # Předpoklad: Koš je cca [5, 50] nebo [95, 50] podle hřiště,
    # ale zde použijeme logiku vzdálenosti, pokud x/y jsou v nějakých jednotkách.
    # PRO TYTO DATA: Předpokládám standardizované souřadnice, použiju jednoduchou logiku "Distance".
    # Protože neznáme přesné měřítko hřiště v datech, odvodíme typ střely z 'action' (3pt vs 2pt)
    # a pro "Rim" použijeme "layup/dunk" v sub_type.

    def assign_zone(row):
        sub = str(row['sub_type']).lower()
        act = str(row['action']).lower()

        if 'layup' in sub or 'dunk' in sub:
            return 'Rim'
        if '3pt' in act:
            # Zde bychom ideálně dělili na Corner3/Arc3 podle souřadnic Y
            # Pokud Y je 0-100, rohy jsou <10 a >90 (zhruba)
            if row['y'] < 10 or row['y'] > 90: return 'Corner3'
            return 'Arc3'
        return 'MidRange'  # Vše ostatní (jumpshot 2pt)

    df_shots['Zone'] = df_shots.apply(assign_zone, axis=1)

    # Hodnota střely
    df_shots['ShotVal'] = df_shots['action'].apply(lambda x: 3 if '3pt' in str(x) else 2)
    df_shots['Points'] = df_shots.apply(lambda x: x['ShotVal'] if x['result'] == 'made' else 0, axis=1)

    # Pivot tabulky
    spatial_pivot = df_shots.groupby(['match_id', 'player', 'Zone']).agg(
        FGA=('result', 'count'),
        FGM=('result', lambda x: (x == 'made').sum()),
        Points=('Points', 'sum')
    ).unstack(fill_value=0)

    # Zploštění sloupců (Rim_FGA, Rim_PPS...)
    spatial_pivot.columns = [f'{col[1]}_{col[0]}' for col in spatial_pivot.columns]
    spatial_pivot = spatial_pivot.reset_index()

    # Dopočet relativních metrik
    # Potřebujeme celkové FGA pro hráče v zápase pro výpočet Freq
    spatial_pivot['Total_FGA_Calc'] = spatial_pivot[[c for c in spatial_pivot.columns if 'FGA' in c]].sum(axis=1)

    for zone in ['Rim', 'MidRange', 'Corner3', 'Arc3']:
        if f'{zone}_FGA' in spatial_pivot.columns:
            # Frequency (% střel z této zóny)
            spatial_pivot[f'{zone}_Freq'] = spatial_pivot[f'{zone}_FGA'] / spatial_pivot['Total_FGA_Calc'].replace(0,
                                                                                                                   np.nan)
            # Efficiency (PPS - Points Per Shot)
            spatial_pivot[f'{zone}_PPS'] = spatial_pivot[f'{zone}_Points'] / spatial_pivot[f'{zone}_FGA'].replace(0,
                                                                                                                  np.nan)

    return spatial_pivot.drop(columns=['Total_FGA_Calc'])


# ==========================================
# 4. ADVANCED STATS (Four Factors)
# ==========================================
def calculate_advanced_stats(df_full):
    print("Počítám Advanced Stats (ORtg, DRtg, USG%)...")

    # Agregace týmových statistik pro USG% a TRB%
    # Potřebujeme sumu minut, FGA, atd. za celý tým v zápase
    team_cols = ['MIN_FL', 'FGA', 'FTA', 'turnovers', 'rebounds', 'points', 'OREB']
    df_team = df_full.groupby(['match_id', 'team'])[team_cols].sum().reset_index()
    df_team = df_team.rename(columns={c: f'Team_{c}' for c in team_cols})

    # Merge týmových dat zpět k hráčům
    df_merged = pd.merge(df_full, df_team, on=['match_id', 'team'])

    # --- VZORCE ---
    # 1. Possessions (Individuální odhad pro ORtg)
    # Poss = FGA + 0.44*FTA - OREB + TOV
    df_merged['Poss'] = df_merged['FGA'] + 0.44 * df_merged['FTA'] - df_merged['OREB'] + df_merged['turnovers']

    # 2. Usage Rate (USG%) - Kolik % útoků týmu hráč zakončil
    # USG% = 100 * ((FGA + 0.44*FTA + TOV) * (Team_MIN / 5)) / (MIN * (Team_FGA + 0.44*Team_FTA + Team_TOV))
    usg_num = (df_merged['FGA'] + 0.44 * df_merged['FTA'] + df_merged['turnovers']) * (df_merged['Team_MIN_FL'] / 5)
    usg_denom = df_merged['MIN_FL'] * (
                df_merged['Team_FGA'] + 0.44 * df_merged['Team_FTA'] + df_merged['Team_turnovers'])
    df_merged['USG%'] = 100 * (usg_num / usg_denom.replace(0, np.nan))

    # 3. Offensive Rating (ORtg) - Body na 100 útoků
    df_merged['ORtg'] = 100 * (df_merged['points'] / df_merged['Poss'].replace(0, np.nan))

    # 4. True Shooting (TS%)
    ts_denom = 2 * (df_merged['FGA'] + 0.44 * df_merged['FTA'])
    df_merged['TS%'] = df_merged['points'] / ts_denom.replace(0, np.nan)

    # Clean up infinite/NaN
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_merged


# ==========================================
# 5. CLUTCH STATS (Koncovky)
# ==========================================
def calculate_clutch(df_pbp):
    print("Počítám Clutch Stats z PBP...")

    # Definice Clutch: 4. čtvrtina (period >= 4), rozdíl skóre <= 5, posledních 5 min
    # Převedeme 'clock' (mm:ss) na sekundy. Format v PBP je např. 08:34:00 (v minutách do konce?) nebo 00:00:00
    # Podle snippetu: '00:11:60' ? Vypadá to divně, zkusíme bezpečný parse.
    # Předpoklad: Clock je čas DO KONCE čtvrtiny.

    def parse_clock(x):
        try:
            parts = str(x).split(':')
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return 9999  # Fail safe

    df_pbp['Sec_Rem'] = df_pbp['clock'].apply(parse_clock)

    # Score diff
    df_pbp['Score_Diff'] = abs(df_pbp['score_home'] - df_pbp['score_away'])

    clutch_mask = (df_pbp['period'] >= 4) & (df_pbp['Score_Diff'] <= 5) & (df_pbp['Sec_Rem'] <= 300)
    clutch_events = df_pbp[clutch_mask]

    # Spočítáme střelbu v clutch time
    # Filtrujeme jen střelecké pokusy (action_type 2pt/3pt)
    shots_mask = clutch_events['action_type'].isin(['2pt', '3pt'])

    clutch_stats = clutch_events[shots_mask].groupby(['match_id', 'player']).agg(
        Clutch_FGA=('success', 'count'),
        Clutch_FGM=('success', 'sum')  # success je 1/0
    ).reset_index()

    clutch_stats['Clutch_FG%'] = clutch_stats['Clutch_FGM'] / clutch_stats['Clutch_FGA'].replace(0, np.nan)

    return clutch_stats


# ==========================================
# 6. OPPONENT DEFENSIVE METRICS (DvP)
# ==========================================
def calculate_opponent_metrics(df_full):
    print("Počítám statistiky obrany soupeře (DvP)...")

    # Cíl: Pro každého hráče připojit statistiky TÝMU, proti kterému hraje.
    # 1. Spočítáme statistiky týmu v každém zápase
    team_stats = df_full.groupby(['match_id', 'team']).agg(
        Team_PTS_Allowed=('points', 'sum'),  # Kolik bodů tým dal (pro soupeře je to Allowed)
        Team_FG_Pct=('FGM', 'sum'),  # Pomocné, musíme vydělit FGA
        Team_FGA=('FGA', 'sum'),
        Team_3P_Allowed=('FG3M', 'sum')
    ).reset_index()

    team_stats['Team_FG_Eff'] = team_stats['Team_FG_Pct'] / team_stats['Team_FGA'].replace(0, np.nan)

    # 2. Vytvoříme mapping zápas -> soupeř
    # Předpoklad: V zápase jsou přesně 2 týmy.
    matches = df_full[['match_id', 'team']].drop_duplicates()
    matches_merged = pd.merge(matches, matches, on='match_id')
    # Vyfiltrujeme řádky kde team_x == team_y (to je ten samý tým)
    opponents = matches_merged[matches_merged['team_x'] != matches_merged['team_y']].rename(
        columns={'team_x': 'team', 'team_y': 'opponent'}
    )

    # 3. Připojíme k hráčům jméno soupeře
    df_with_opp = pd.merge(df_full, opponents, on=['match_id', 'team'], how='left')

    # 4. Připojíme statistiky soupeře (jako 'opponent' join na 'team' v team_stats)
    # Co nás zajímá: Jak soupeř brání. Tzn. vezmeme statistiky 'opponent' týmu (např. kolik bodů dostal).
    # Ale pozor: 'Team_PTS_Allowed' výše jsem spočítal jako body co tým DAL.
    # Takže 'Defensive Rating' soupeře je odvozen z toho, kolik bodů jsme mu MY dali? Ne.
    # Uděláme to jednodušeji: Opp_Points_Allowed = Kolik bodů soupeř v průměru dostává.
    # Zde v rámci jednoho zápasu to je tautologie (Opp_Allowed = My_Points).
    # Pro model potřebujeme průměr ze sezóny.
    # Spočítáme "Global Defensive Stats" pro každý tým (průměr přes všechny zápasy v datasetu)

    global_def_stats = team_stats.groupby('team').agg(
        Avg_Pts_Allowed=('Team_PTS_Allowed', 'mean'),  # Zde je chyba v logice názvu, toto jsou body co tým dal.
        # Oprava: Chceme vědět, kolik bodů tým DOSTÁVÁ.
        # Musíme se podívat na zápasy, kde tým hrál, a sečíst body soupeře.
    ).reset_index()

    # Pro tento dataset (jeden zápas nebo pár) to uděláme přesně:
    # 1. Spočítáme body každého týmu v zápase.
    match_scores = df_full.groupby(['match_id', 'team'])['points'].sum().reset_index()

    # 2. Join na soupeře
    match_scores = pd.merge(match_scores, opponents, on=['match_id', 'team'])

    # 3. Teď víme: match_id, team, opponent, points (co tým dal).
    # Body co 'opponent' DOSTAL = points (co 'team' dal).

    opp_def_stats = match_scores.groupby('opponent')['points'].mean().reset_index(name='Opp_Avg_Points_Allowed')

    # Připojíme k hlavní tabulce podle sloupce 'opponent' (který jsme získali v kroku 3)
    df_final = pd.merge(df_with_opp, opp_def_stats, left_on='opponent', right_on='opponent', how='left')

    return df_final


# ==========================================
# MAIN EXECUTION
# ==========================================
# 1. Load
df_box, df_pbp, df_shots = load_and_prep_data()

# 2. Basic Stats Aggregate
df_full_stats = aggregate_basic_stats(df_box, df_pbp, df_shots)

# 3. Spatial
spatial_feats = calculate_spatial_features(df_shots)
# Merge Spatial
df_master = pd.merge(df_full_stats, spatial_feats, on=['match_id', 'player'], how='left')

# 4. Advanced Stats
df_master = calculate_advanced_stats(df_master)

# 5. Clutch
clutch_feats = calculate_clutch(df_pbp)
df_master = pd.merge(df_master, clutch_feats, on=['match_id', 'player'], how='left')

# 6. Opponent Context
df_master = calculate_opponent_metrics(df_master)

# 7. Consistency (Rolling Std)
# Seřadíme podle ID zápasu jako proxy pro čas
df_master = df_master.sort_values(['player', 'match_id'])
df_master['Consistency_PTS'] = df_master.groupby('player')['points'].transform(
    lambda x: x.rolling(5, min_periods=1).std())

# Final Cleanup
df_master = df_master.fillna(0)

# Uložení
output_filename = 'moneyball_features_final.csv'
df_master.to_csv(output_filename, index=False)
print(f"Hotovo! Soubor uložen jako: {output_filename}")
print(df_master[['player', 'points', 'USG%', 'ORtg', 'Rim_PPS', 'Clutch_FG%']].head(10))