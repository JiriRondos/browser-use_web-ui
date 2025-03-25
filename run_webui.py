#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Spustit webové rozhraní Shaman Browser Agent")
    parser.add_argument("--host", "--ip", type=str, default="127.0.0.1", help="Host, na kterém se spustí webové rozhraní")
    parser.add_argument("--port", type=int, default=7860, help="Port, na kterém se spustí webové rozhraní")
    parser.add_argument("--share", action="store_true", help="Sdílet webové rozhraní")
    parser.add_argument("--theme", type=str, default="Ocean", help="Téma pro webové rozhraní")
    parser.add_argument("--dark-mode", action="store_true", help="Tmavý režim")
    parser.add_argument("--lang", "--language", type=str, choices=["en", "cs"], default="en",
                        help="Jazyk rozhraní (en=angličtina, cs=čeština)")

    args = parser.parse_args()

    # Určení, který skript spustit
    if args.lang == "cs":
        script_path = os.path.join(os.path.dirname(__file__), "webui_czech.py")
        # Sestavení příkazové řádky pro českou verzi
        cmd = [sys.executable, script_path, "--host", args.host, "--port", str(args.port)]
        if args.share:
            cmd.append("--share")
    else:
        script_path = os.path.join(os.path.dirname(__file__), "webui.py")
        # Sestavení příkazové řádky pro anglickou verzi
        cmd = [sys.executable, script_path, "--ip", args.host, "--port", str(args.port)]
        if args.theme:
            cmd.extend(["--theme", args.theme])
        if args.dark_mode:
            cmd.append("--dark-mode")
        cmd.extend(["--language", "en"])  # Vždy použít angličtinu pro původní verzi

    print(f"Spouštím: {' '.join(cmd)}")

    # Spuštění příslušného skriptu
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Chyba při spuštění webového rozhraní: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Ukončeno uživatelem")
        sys.exit(0)

if __name__ == "__main__":
    main()