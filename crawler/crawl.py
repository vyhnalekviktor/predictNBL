import requests
import pandas as pd
import os
import json
import unicodedata
import sys
import re

# === CONFIGURATION ===
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://fibalivestats.dcd.shared.geniussports.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
}


def normalize_text(text):
    if text is None: return None
    if not isinstance(text, str): text = str(text)
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip()


def get_match_data_tuple(match_id):
    """
    Stáhne:
    1. JSON s daty (statistiky)
    2. HTML stránku (pro datum zápasu)
    Vrátí: (json_data, html_text)
    """
    # URL pro JSON (statistiky)
    json_url = f"https://fibalivestats.dcd.shared.geniussports.com/data/{match_id}/data.json"

    # URL pro HTML (webová stránka, kde je text "Tip off")
    # Zkoušíme obecný formát vieweru
    html_url = f"https://fibalivestats.dcd.shared.geniussports.com/u/CBFFE/{match_id}/index.html"

    print(f"Processing ID {match_id}...", end="")

    json_data = None
    html_text = ""

    # 1. Stáhneme JSON
    try:
        r_json = requests.get(json_url, headers=HEADERS, timeout=10)
        if r_json.status_code == 200:
            json_data = r_json.json()
            print(" [JSON OK]", end="")
        else:
            print(f" [JSON Error {r_json.status_code}]", end="")
    except Exception as e:
        print(f" [JSON Fail: {e}]", end="")

    # 2. Stáhneme HTML stránku (jen pokud máme JSON, jinak to nemá smysl)
    if json_data:
        try:
            r_html = requests.get(html_url, headers=HEADERS, timeout=10)
            if r_html.status_code == 200:
                html_text = r_html.text
                print(" [HTML OK]", end="")
            else:
                # Fallback: Někdy je URL trochu jiné, ale zkusíme aspoň tohle
                print(f" [HTML Error {r_html.status_code}]", end="")
        except Exception as e:
            print(f" [HTML Fail: {e}]", end="")

    return json_data, html_text


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


def clean_html_text(raw_html):
    """Odstraní HTML tagy pro snazší regex."""
    if not raw_html: return ""
    # Nahradí <br> mezerami
    text = raw_html.replace('<br>', ' ').replace('<p>', ' ').replace('</p>', ' ')
    # Odstraní ostatní tagy
    clean = re.sub(r'<[^>]+>', ' ', text)
    # Zredukuje mezery
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def parse_match_data(data, html_text, match_id):
    lookup = build_player_map(data)
    teams_names = {int(k): normalize_text(v.get('name', 'unknown')) for k, v in data.get('tm', {}).items() if
                   k.isdigit()}

    # --- 1. ZÍSKÁNÍ DATA (SCRAPING Z HTML) ---
    match_date = "unknown"

    if html_text:
        # Vyčistíme HTML -> získáme text
        clean_text = clean_html_text(html_text)

        # Hledáme: Tip off [cokoliv] číslo/číslo/číslo
        # Regex ignoruje velikost písmen (re.IGNORECASE)
        date_match = re.search(r"Tip off.*?(\d{1,2}/\d{1,2}/\d{2,4})", clean_text, re.IGNORECASE)

        if date_match:
            raw_d_str = date_match.group(1)  # Např. "23/9/23"
            try:
                day, month, year = raw_d_str.split('/')

                # Pokud je rok "23", uděláme "2023"
                if len(year) == 2: year = "20" + year

                # YYYY-MM-DD
                match_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            except:
                pass
        else:
            pass

    # ------------------------------------------

    # A) Play by play
    pbp_rows = []
    action_time_map = {}
    for event in data.get('pbp', []):
        tno, pno, action_num, clock = event.get('tno'), event.get('pno'), event.get('actionNumber'), event.get('clock')
        if action_num is not None: action_time_map[action_num] = clock
        player_name = lookup.get((tno, pno)) if tno is not None and pno is not None else None
        pbp_rows.append({
            'match_id': match_id,
            'period': event.get('period'), 'clock': clock, 'gt': event.get('gt'),
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
                'match_id': match_id,
                'team': team_name, 'player': player_name,
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
                'match_id': match_id,
                'date': match_date,  # <--- ZDE SE ULOŽÍ DATUM
                'team': team_name, 'player': lookup.get((tid, int(pno))),
                'jersey': p.get('shirtNumber'), 'starter': p.get('starter'), 'minutes': p.get('sMinutes'),
                'points': pts, 'rebounds': reb, 'assists': ast, 'fouls': p.get('sFoulsPersonal'),
                'turnovers': to, 'steals': stl, 'blocks': blk, 'plus_minus': p.get('sPlusMinusPoints'),
                'efficiency': eff
            })
    return pbp_rows, shot_rows, box_rows


def parse_input_file(filename):
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
    box_file = os.path.join(folder, "match_box_score.csv")
    if not os.path.exists(box_file):
        return False
    try:
        existing_ids = pd.read_csv(box_file, usecols=['match_id'], sep=';')['match_id'].unique()
        return match_id in existing_ids
    except:
        return False


def append_to_csv(data_list, folder, filename):
    if not data_list: return
    filepath = os.path.join(folder, filename)
    df = pd.DataFrame(data_list)
    if not os.path.isfile(filepath):
        df.to_csv(filepath, index=False, sep=';', encoding='utf-8-sig')
    else:
        df.to_csv(filepath, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crawl.py input.txt")
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

        if is_match_scraped(output_folder, mid):
            print(f" ⏩ ID {mid} already exists. Skipping.")
            continue

        # Voláme funkci, která stáhne JSON i HTML
        json_data, html_text = get_match_data_tuple(mid)

        if json_data:
            pbp, shots, box = parse_match_data(json_data, html_text, mid)

            append_to_csv(pbp, output_folder, "pbp_events.csv")
            append_to_csv(shots, output_folder, "shots_spatial.csv")
            append_to_csv(box, output_folder, "match_box_score.csv")

            print(" -> Saved.")
            count += 1
        else:
            print("... Failed.")

    print(f"\nDone! Added {count} new matches.")