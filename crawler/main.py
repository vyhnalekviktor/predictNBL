import requests
import pandas as pd
import os
import json
import unicodedata

# === CONFIGURATION ===
MATCH_ID = 2705712
OUTPUT_DIR = "nbl_normalized_data"

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
    Advanced normalization:
    1. Converts to lowercase.
    2. Removes diacritics (accents) -> 'š' becomes 's'.
    3. Strips leading/trailing whitespace.
    """
    if text is None:
        return None

    # Convert non-string types to string just in case
    if not isinstance(text, str):
        text = str(text)

    # Normalize Unicode characters (decompose accents)
    nfkd_form = unicodedata.normalize('NFKD', text)

    # Filter out non-spacing mark characters (the accents)
    only_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    # Lowercase and strip whitespace
    return only_ascii.lower().strip()


def get_match_json(match_id):
    """Downloads the raw JSON data from the API."""
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
    Source of truth for linking player IDs to names.
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
            # 1. Try to get 'pno' from the object
            pno = p.get('pno')

            # 2. Fallback: If 'pno' is missing, try using the dictionary key
            if pno is None:
                try:
                    pno = int(pid_str)
                except:
                    pass

            if pno is not None:
                # Construct full name
                raw_name = f"{p.get('name', '')} {p.get('sName', '')}"

                # Fallback to scoreboard name if empty
                if not raw_name.strip():
                    raw_name = p.get('scoreboardName', 'unknown')

                # Apply normalization immediately
                normalized_name = normalize_text(raw_name)

                try:
                    # Key is tuple: (TeamID, PlayerNumberOrder)
                    player_map[(tid, int(pno))] = normalized_name
                except ValueError:
                    continue

    return player_map


def parse_match_data(data, match_id):
    # 1. Build Player Map (Normalized)
    lookup = build_player_map(data)

    # 2. Build Team Map (Normalized)
    teams_names = {}
    for k, v in data.get('tm', {}).items():
        try:
            teams_names[int(k)] = normalize_text(v.get('name', 'unknown'))
        except:
            pass

    # --- A) PBP (Play by Play) ---
    pbp_rows = []
    raw_pbp = data.get('pbp', [])

    for event in raw_pbp:
        tno = event.get('tno')
        pno = event.get('pno')

        player_name = None
        team_name = teams_names.get(tno)

        # Resolve player name from lookup
        if tno is not None and pno is not None:
            player_name = lookup.get((tno, pno))

        pbp_rows.append({
            'period': event.get('period'),
            'clock': event.get('clock'),
            'gt': event.get('gt'),
            'team': team_name,
            'player': player_name,
            'action_type': normalize_text(event.get('actionType')),
            'sub_type': normalize_text(event.get('subType')),
            'success': event.get('success'),
            'score_home': event.get('s1'),
            'score_away': event.get('s2')
        })

    # --- B) SHOTS ---
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

            # 1. Try Lookup Map (Preferred)
            if pno is not None:
                player_name = lookup.get((tid, int(pno)))

            # 2. Fallback to 'person' field (Normalized)
            if not player_name:
                player_name = normalize_text(s.get('person'))

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
                'time': s.get('time')
            })

    return pbp_rows, shot_rows


# === MAIN EXECUTION ===
if __name__ == "__main__":
    json_data = get_match_json(MATCH_ID)

    if json_data:
        pbp, shots = parse_match_data(json_data, MATCH_ID)

        print(f"\nSaving normalized data to '{OUTPUT_DIR}'...")

        if pbp:
            df_pbp = pd.DataFrame(pbp)
            df_pbp.to_csv(f"{OUTPUT_DIR}/pbp_events.csv", index=False, sep=';', encoding='utf-8-sig')
            print(f"- pbp_events.csv ({len(df_pbp)} rows)")

        if shots:
            df_shots = pd.DataFrame(shots)
            df_shots.to_csv(f"{OUTPUT_DIR}/shots_spatial.csv", index=False, sep=';', encoding='utf-8-sig')
            print(f"- shots_spatial.csv ({len(df_shots)} shots)")

    else:
        print("❌ Failed to download data.")