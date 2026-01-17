import requests
import pandas as pd
import os
import json
import unicodedata

# === CONFIGURATION ===
MATCH_ID = 2705712
OUTPUT_DIR = "nbl_complete_data"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://fibalivestats.dcd.shared.geniussports.com/",
    "Origin": "https://fibalivestats.dcd.shared.geniussports.com",
    "Accept": "application/json, text/plain, */*"
}


def normalize_text(text):
    """
    Advanced normalization: lowercase, remove accents, strip whitespace.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    nfkd_form = unicodedata.normalize('NFKD', text)
    only_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return only_ascii.lower().strip()


def get_match_json(match_id):
    url = f"https://fibalivestats.dcd.shared.geniussports.com/data/{match_id}/data.json"
    print(f"Downloading match ID {match_id}...", end="")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            print(" OK")
            return r.json()
        print(f" ❌ Error {r.status_code}")
    except Exception as e:
        print(f" ❌ Exception: {e}")
    return None


def build_player_map(data):
    """
    Builds a lookup map: {(TeamID, PNO): "normalized_name"}
    """
    player_map = {}
    teams = data.get('tm', {})

    for team_id_str, team_data in teams.items():
        try:
            tid = int(team_id_str)
        except ValueError:
            continue

        players = team_data.get('pl', {})
        for pid_str, p in players.items():
            pno = p.get('pno')
            if pno is None:
                try:
                    pno = int(pid_str)
                except:
                    pass

            if pno is not None:
                raw_name = f"{p.get('name', '')} {p.get('sName', '')}"
                if not raw_name.strip():
                    raw_name = p.get('scoreboardName', 'unknown')

                normalized_name = normalize_text(raw_name)

                try:
                    player_map[(tid, int(pno))] = normalized_name
                except ValueError:
                    continue
    return player_map


def parse_match_data(data, match_id):
    # 1. Build Lookup Maps
    lookup = build_player_map(data)

    teams_names = {}
    for k, v in data.get('tm', {}).items():
        try:
            teams_names[int(k)] = normalize_text(v.get('name', 'unknown'))
        except:
            pass

    # --- A) PBP (Play by Play) & Time Mapping ---
    pbp_rows = []
    # Novinka: Vytvoříme si mapu ActionNumber -> Time
    action_time_map = {}

    for event in data.get('pbp', []):
        tno = event.get('tno')
        pno = event.get('pno')
        action_num = event.get('actionNumber')
        clock = event.get('clock')  # Formát MM:SS:MS

        # Uložení času pro pozdější použití u střel
        if action_num is not None:
            action_time_map[action_num] = clock

        player_name = None
        if tno is not None and pno is not None:
            player_name = lookup.get((tno, pno))

        pbp_rows.append({
            'period': event.get('period'),
            'clock': clock,
            'gt': event.get('gt'),
            'team': teams_names.get(tno),
            'player': player_name,
            'action_type': normalize_text(event.get('actionType')),
            'sub_type': normalize_text(event.get('subType')),
            'success': event.get('success'),
            'score_home': event.get('s1'),
            'score_away': event.get('s2')
        })

    # --- B) SHOTS (Now with Time!) ---
    shot_rows = []
    for tid_str, team_data in data.get('tm', {}).items():
        try:
            tid = int(tid_str)
        except:
            continue

        team_name = normalize_text(team_data.get('name'))

        for s in team_data.get('shot', []):
            pno = s.get('pno')
            player_name = None
            if pno is not None:
                player_name = lookup.get((tid, int(pno)))
            if not player_name:
                player_name = normalize_text(s.get('person'))

            # Získání času pomocí actionNumber
            shot_action_num = s.get('actionNumber')
            shot_time = action_time_map.get(shot_action_num)  # Tady se děje to kouzlo

            shot_rows.append({
                'match_id': match_id,
                'team': team_name,
                'player': player_name,
                'action': normalize_text(s.get('actionType')),
                'sub_type': normalize_text(s.get('subType')),
                'x': s.get('x'),
                'y': s.get('y'),
                'result': 'made' if s.get('r') == 1 else 'missed',
                'period': s.get('per'),
                'time': shot_time  # Teď už by to mělo být vyplněné
            })

    # --- C) BOX SCORE ---
    box_rows = []
    for tid_str, team_data in data.get('tm', {}).items():
        try:
            tid = int(tid_str)
        except:
            continue

        team_name = normalize_text(team_data.get('name'))

        for pid, p in team_data.get('pl', {}).items():
            pno = p.get('pno')
            if pno is None:
                try:
                    pno = int(pid)
                except:
                    pass

            if pno is None: continue

            player_name = lookup.get((tid, int(pno)))

            # 1. Base Stats
            pts = p.get('sPoints', 0)
            reb = p.get('sReboundsTotal', 0)
            ast = p.get('sAssists', 0)
            stl = p.get('sSteals', 0)
            blk = p.get('sBlocks', 0)
            to = p.get('sTurnovers', 0)

            # 2. Efficiency Calculation
            fga = p.get('sFieldGoalsAttempted', 0)
            fgm = p.get('sFieldGoalsMade', 0)
            missed_fg = fga - fgm

            fta = p.get('sFreeThrowsAttempted', 0)
            ftm = p.get('sFreeThrowsMade', 0)
            missed_ft = fta - ftm

            efficiency = (pts + reb + ast + stl + blk) - (missed_fg + missed_ft + to)

            box_rows.append({
                'match_id': match_id,
                'team': team_name,
                'player': player_name,
                'jersey': p.get('shirtNumber'),
                'starter': p.get('starter'),
                'minutes': p.get('sMinutes'),
                'points': pts,
                'rebounds': reb,
                'assists': ast,
                'fouls': p.get('sFoulsPersonal'),
                'turnovers': to,
                'steals': stl,
                'blocks': blk,
                'plus_minus': p.get('sPlusMinusPoints'),
                'efficiency': efficiency
            })

    return pbp_rows, shot_rows, box_rows


# === MAIN EXECUTION ===
if __name__ == "__main__":
    json_data = get_match_json(MATCH_ID)

    if json_data:
        pbp, shots, box = parse_match_data(json_data, MATCH_ID)

        print(f"\nSaving data to '{OUTPUT_DIR}'...")

        if pbp:
            pd.DataFrame(pbp).to_csv(f"{OUTPUT_DIR}/pbp_events.csv", index=False, sep=';', encoding='utf-8-sig')
            print(f"- pbp_events.csv ({len(pbp)} rows)")

        if shots:
            df_shots = pd.DataFrame(shots)
            df_shots.to_csv(f"{OUTPUT_DIR}/shots_spatial.csv", index=False, sep=';', encoding='utf-8-sig')
            print(f"- shots_spatial.csv ({len(shots)} rows)")

        if box:
            df_box = pd.DataFrame(box)
            df_box.to_csv(f"{OUTPUT_DIR}/match_box_score.csv", index=False, sep=';', encoding='utf-8-sig')
            print(f"- match_box_score.csv ({len(box)} players)")

    else:
        print("❌ Failed to download data.")