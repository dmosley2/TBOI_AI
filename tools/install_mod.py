"""Copy the bridge mod into the game's mods folder.

    python tools/install_mod.py
    python tools/install_mod.py --game-dir "D:\\SteamLibrary\\steamapps\\common\\The Binding of Isaac Rebirth"

Re-run it after editing mod/isaac_ai_bridge/main.lua. In game, the console
command `luamod isaac_ai_bridge` reloads the mod without restarting.
"""

import argparse
import os
import shutil

DEFAULT_GAME_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-dir", default=DEFAULT_GAME_DIR)
    args = ap.parse_args()

    mods = os.path.join(args.game_dir, "mods")
    if not os.path.isdir(mods):
        raise SystemExit(f"mods folder not found: {mods}")
    src = os.path.join(HERE, "mod", "isaac_ai_bridge")
    dst = os.path.join(mods, "isaac_ai_bridge")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"installed {src} -> {dst}")


if __name__ == "__main__":
    main()
