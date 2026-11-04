"""Switch between your normal mod list and a clean "AI only" profile.

The game skips any mod folder that contains a file named `disable.it`
(that is what the in-game Mods menu toggles).

    python tools/mod_profile.py status
    python tools/mod_profile.py ai        # disable everything except the bridge
    python tools/mod_profile.py restore   # undo exactly what `ai` changed

`ai` records which mods IT disabled in mods/.isaac_ai_profile.json, so
`restore` never re-enables a mod you had already turned off yourself.
"""

import argparse
import json
import os

DEFAULT_GAME_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth"
KEEP = {"isaac_ai_bridge"}
RECORD = ".isaac_ai_profile.json"


def mod_dirs(mods):
    return sorted(d for d in os.listdir(mods) if os.path.isdir(os.path.join(mods, d)))


def is_disabled(mods, d):
    return os.path.exists(os.path.join(mods, d, "disable.it"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["status", "ai", "restore"])
    ap.add_argument("--game-dir", default=DEFAULT_GAME_DIR)
    args = ap.parse_args()
    mods = os.path.join(args.game_dir, "mods")
    record_path = os.path.join(mods, RECORD)

    if args.action == "status":
        enabled = [d for d in mod_dirs(mods) if not is_disabled(mods, d)]
        print(f"{len(enabled)} enabled:")
        for d in enabled:
            print("  " + d)
        return

    if args.action == "ai":
        if os.path.exists(record_path):
            raise SystemExit("AI profile already active (run `restore` first).")
        changed = []
        for d in mod_dirs(mods):
            if d in KEEP or is_disabled(mods, d):
                continue
            open(os.path.join(mods, d, "disable.it"), "w").close()
            changed.append(d)
        with open(record_path, "w") as f:
            json.dump(changed, f, indent=1)
        print(f"disabled {len(changed)} mods; restore with `python tools/mod_profile.py restore`")
        return

    if not os.path.exists(record_path):
        raise SystemExit("no AI profile record found; nothing to restore")
    with open(record_path) as f:
        changed = json.load(f)
    for d in changed:
        flag = os.path.join(mods, d, "disable.it")
        if os.path.exists(flag):
            os.remove(flag)
    os.remove(record_path)
    print(f"re-enabled {len(changed)} mods")


if __name__ == "__main__":
    main()
