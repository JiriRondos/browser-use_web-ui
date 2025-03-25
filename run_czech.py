#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Spustit české webové rozhraní Shaman Browser Agent")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host, na kterém se spustí webové rozhraní")
    parser.add_argument("--port", type=int, default=7860, help="Port, na kterém se spustí webové rozhraní")
    parser.add_argument("--share", action="store_true", help="Sdílet webové rozhraní")
    
    args = parser.parse_args()
    
    # Cesta k českému skriptu
    script_path = os.path.join(os.path.dirname(__file__), "webui_czech.py")
    
    # Sestavení příkazové řádky
    cmd = [sys.executable, script_path, "--host", args.host, "--port", str(args.port)]
    if args.share:
        cmd.append("--share")
    
    print(f"Spouštím: {' '.join(cmd)}")
    
    # Spuštění skriptu
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