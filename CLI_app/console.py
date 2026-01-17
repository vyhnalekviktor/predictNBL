import argparse
import subprocess
import sys
import os


def display_guide():
    guide = """
    === 🏀 NBL MONEYBALL: UŽIVATELSKÝ MANUÁL ===

    1. SEZNAM PREDIKCÍ (Celý tým):
       python console.py predict "Tým A" "Tým B" --home
       - Ukáže pravděpodobnost (% > FORM), že hráč překoná svůj průměr.

    2. KONTROLA KONKRÉTNÍ SÁZKY (Hranice):
       python console.py check "Tým A" "Tým B" "Příjmení" 15.5 --home
       - Vypočítá přesné šance na Over/Under pro zadanou hranici.

    3. PRAVIDLO 65%:
       Ziskové sázky se statisticky nacházejí tam, kde je 
       Pravděpodobnost Over (nebo Under) vyšší než 65%.
    ============================================
    """
    print(guide)


def display_stats():
    mae, r2, samples = 3.61, 0.4709, 19551
    print("\n📊 AKTUÁLNÍ VÝKONNOST MODELU")
    print("-" * 45)
    print(f"Počet vzorků: {samples:,} | MAE: {mae} | R2: {r2}")
    print("Vazena data od sezony 2022/23 do 17.ledna.2025")
    print("-" * 45)


def main():
    parser = argparse.ArgumentParser(prog='console', description='NBL Moneyball CLI')
    subparsers = parser.add_subparsers(dest='command', help='Příkazy')

    # PREDICT
    p_predict = subparsers.add_parser('predict', help='Seznam predikcí pro zápas')
    p_predict.add_argument('team', help='Název týmu')
    p_predict.add_argument('opponent', help='Název soupeře')
    p_predict.add_argument('--home', action='store_true', help='Hraje první tým doma?')

    # CHECK
    p_check = subparsers.add_parser('check', help='Kontrola konkrétního hráče a hranice')
    p_check.add_argument('team', help='Tým hráče')
    p_check.add_argument('opponent', help='Soupeř')
    p_check.add_argument('surname', help='Příjmení hráče')
    p_check.add_argument('line', type=float, help='Hranice bodů (např. 15.5)')
    p_check.add_argument('--home', action='store_true', help='Hraje první tým doma?')

    subparsers.add_parser('stats', help='Zobrazit statistiky modelu')
    subparsers.add_parser('guide', help='Zobrazit manuál')

    args = parser.parse_args()

    if args.command == 'predict':
        cmd = [sys.executable, "predict_next_game.py", "--team", args.team, "--opponent", args.opponent]
        if args.home: cmd.append("--home")
        subprocess.run(cmd)

    elif args.command == 'check':
        cmd = [sys.executable, "predict_next_game.py", "--team", args.team, "--opponent", args.opponent,
               "--player", args.surname, "--line", str(args.line)]
        if args.home: cmd.append("--home")
        subprocess.run(cmd)

    elif args.command == 'stats':
        display_stats()
    elif args.command == 'guide':
        display_guide()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()