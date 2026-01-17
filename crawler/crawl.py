import requests
import pandas as pd
import os
import json
import unicodedata
import sys

# === CONFIGURATION ===
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://fibalivestats.dcd.shared.geniussports.com/",
    "Origin": "https://fibalivestats.dcd.shared.geniussports.com",
    "Accept": "application/json, text/plain, */*"
}


def normalize_text(text):
    if text is None: return None
    if not isinstance(text, str): text = str(text)
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip()


def get_match_json(match_id):
    url = f"https://fibalivestats.dcd.shared.geniussports.com/data/{match_id}/data.json"
    print(f"Processing ID {match_id}...", end="")
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            print(" Downloaded", end="")
            return r.json()
        print(f" ❌ API Error {r.status_code}")
    except Exception as e:
        print(f" ❌ Network Error: {e}")
    return None


def build_player_map(data):
    player_map = {}
    teams = data.get('tm', {})
    for tid_str, team_data in teams.items():
        try:
            tid = int(tid_str)
        except:
            continue
        for pid_str, p in team_data.get('pl', {}).items():
            pno = p.get('pno')
            if pno is None:
                try:
                    pno = int(pid_str)
                except:
                    pass
            if pno is not None:
                raw_name = f"{p.get('name', '')} {p.get('sName', '')}"
                if not raw_name.strip(): raw_name = p.get('scoreboardName', 'unknown')
                player_map[(tid, int(pno))] = normalize_text(raw_name)
    return player_map


def parse_match_data(data, match_id):
    lookup = build_player_map(data)
    teams_names = {int(k): normalize_text(v.get('name', 'unknown')) for k, v in data.get('tm', {}).items() if
                   k.isdigit()}

    # A) Play by play
    pbp_rows = []
    action_time_map = {}
    for event in data.get('pbp', []):
        tno, pno, action_num, clock = event.get('tno'), event.get('pno'), event.get('actionNumber'), event.get('clock')
        if action_num is not None: action_time_map[action_num] = clock
        player_name = lookup.get((tno, pno)) if tno is not None and pno is not None else None
        pbp_rows.append({
            'match_id': match_id, 'period': event.get('period'), 'clock': clock, 'gt': event.get('gt'),
            'team': teams_names.get(tno), 'player': player_name, 'action_type': normalize_text(event.get('actionType')),
            'sub_type': normalize_text(event.get('subType')), 'success': event.get('success'),
            'score_home': event.get('s1'), 'score_away': event.get('s2')
        })

    # B) SHOTS
    shot_rows = []
    for tid_str, team_data in data.get('tm', {}).items():
        try:
            tid = int(tid_str)
        except:
            continue
        team_name = normalize_text(team_data.get('name'))
        for s in team_data.get('shot', []):
            pno = s.get('pno')
            player_name = lookup.get((tid, int(pno))) if pno is not None else normalize_text(s.get('person'))
            shot_rows.append({
                'match_id': match_id, 'team': team_name, 'player': player_name,
                'action': normalize_text(s.get('actionType')), 'sub_type': normalize_text(s.get('subType')),
                'x': s.get('x'), 'y': s.get('y'), 'result': 'made' if s.get('r') == 1 else 'missed',
                'period': s.get('per'), 'time': action_time_map.get(s.get('actionNumber'))
            })

    # C) BOX SCORE
    box_rows = []
    for tid_str, team_data in data.get('tm', {}).items():
        try:
            tid = int(tid_str)
        except:
            continue
        team_name = normalize_text(team_data.get('name'))
        for pid, p in team_data.get('pl', {}).items():
            pno = p.get('pno') if p.get('pno') is not None else (int(pid) if pid.isdigit() else None)
            if pno is None: continue

            pts, reb, ast, stl, blk, to = p.get('sPoints', 0), p.get('sReboundsTotal', 0), p.get('sAssists', 0), p.get(
                'sSteals', 0), p.get('sBlocks', 0), p.get('sTurnovers', 0)
            missed_fg = p.get('sFieldGoalsAttempted', 0) - p.get('sFieldGoalsMade', 0)
            missed_ft = p.get('sFreeThrowsAttempted', 0) - p.get('sFreeThrowsMade', 0)
            eff = (pts + reb + ast + stl + blk) - (missed_fg + missed_ft + to)

            box_rows.append({
                'match_id': match_id, 'team': team_name, 'player': lookup.get((tid, int(pno))),
                'jersey': p.get('shirtNumber'), 'starter': p.get('starter'), 'minutes': p.get('sMinutes'),
                'points': pts, 'rebounds': reb, 'assists': ast, 'fouls': p.get('sFoulsPersonal'),
                'turnovers': to, 'steals': stl, 'blocks': blk, 'plus_minus': p.get('sPlusMinusPoints'),
                'efficiency': eff
            })
    return pbp_rows, shot_rows, box_rows


def parse_input_file(filename):
    """Reads the input config file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines()]

        output_folder = lines[0]
        raw_ids = "".join(lines[2:]).split(',')
        clean_ids = [x.strip() for x in raw_ids if x.strip().isdigit()]

        return output_folder, clean_ids
    except Exception as e:
        print(f"❌ Error reading input file: {e}")
        sys.exit(1)


def is_match_scraped(folder, match_id):
    """Checks if match_id already exists in the box_score file to prevent duplicates."""
    box_file = os.path.join(folder, "match_box_score.csv")
    if not os.path.exists(box_file):
        return False

    try:
        existing_ids = pd.read_csv(box_file, usecols=['match_id'], sep=';')['match_id'].unique()
        return match_id in existing_ids
    except:
        return False


def append_to_csv(data_list, folder, filename):
    """Appends data to CSV. Creates file with header if not exists."""
    if not data_list: return

    filepath = os.path.join(folder, filename)
    df = pd.DataFrame(data_list)

    if not os.path.isfile(filepath):
        df.to_csv(filepath, index=False, sep=';', encoding='utf-8-sig')
    else:
        df.to_csv(filepath, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')


# === MAIN LOGIC ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py input.txt")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_folder, match_ids = parse_input_file(input_filename)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    print(f"Target: {len(match_ids)} matches -> {output_folder}/ \n")

    count = 0
    for mid in match_ids:
        mid = int(mid)

        # 1. control duplicity
        if is_match_scraped(output_folder, mid):
            print(f" ⏩ ID {mid} already exists. Skipping.")
            continue

        # 2. Download and parse
        data = get_match_json(mid)
        if data:
            pbp, shots, box = parse_match_data(data, mid)

            append_to_csv(pbp, output_folder, "pbp_events.csv")
            append_to_csv(shots, output_folder, "shots_spatial.csv")
            append_to_csv(box, output_folder, "match_box_score.csv")

            print(" -> Saved & Appended.")
            count += 1
        else:
            print("... Failed.")

    print(f"\nDone! Added {count} new matches to {output_folder}.")