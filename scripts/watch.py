"""Step 1 check: connect to the game and print what the AI would see.

You keep control. Walk around, enter rooms, fight, and check the numbers.

    python scripts/watch.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaac_ai.bridge import GameDisconnected, IsaacBridge, as_list  # noqa: E402


def main():
    bridge = IsaacBridge()
    s = bridge.wait_for_game()
    try:
        while True:
            if s["frame"] % 30 == 0:  # about once a second
                p, room = s["player"], s["room"]
                print(f"frame {s['frame']:>6} | stage {s['stage']} room {room['index']:>3} "
                      f"clear={room['clear']!s:5} | pos ({p['pos'][0]:6.1f},{p['pos'][1]:6.1f}) "
                      f"hearts {p['hearts']}/{p['max_hearts']} soul {p['soul']} | "
                      f"enemies {len(as_list(s['enemies'])):>2} "
                      f"projectiles {len(as_list(s['projectiles'])):>2}", flush=True)
            ev = s.get("events") or {}
            if ev.get("hits"):
                print(f"  ouch: {ev['hits']} hit(s), {ev['damage_taken']} half-hearts")
            s = bridge.step({"control": False})
    except (KeyboardInterrupt, GameDisconnected) as e:
        print(f"stopped: {e!r}")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
