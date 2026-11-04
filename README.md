# Isaac AI

An AI that learns to play The Binding of Isaac: Repentance, with Dead God on a
fresh save file as the long-term goal.

- [docs/DESIGN.md](docs/DESIGN.md): how it works and why each design decision was made
- [ROADMAP.md](ROADMAP.md): the plan from room combat to Dead God

## How it fits together

```
 The game (Lua mod)                          Python
 ┌──────────────────────┐   state (JSON)   ┌───────────────────────────────┐
 │ mod/isaac_ai_bridge  │ ───────────────▶ │ isaac_ai/bridge.py  TCP link   │
 │  - snapshot / update │                  │ isaac_ai/features.py  → vector │
 │  - MC_INPUT_ACTION   │ ◀─────────────── │ isaac_ai/env.py   Gym env      │
 │  - arena resets      │  action (JSON)   │ scripts/train_ppo.py  PPO      │
 └──────────────────────┘                  └───────────────────────────────┘
```

The game and Python run in **lockstep**: the mod sends a snapshot every game
update (30/s) and waits for the reply. So the game pauses whenever Python is
busy, for example during PPO's learning phase. If Python isn't running, the mod
does nothing and you can play normally.

## One-time setup

1. **Use Repentance, not Repentance+.** In Steam, right-click the game →
   Properties → DLC and untick *Repentance+*. Repentance has its own save folder
   (`Documents/My Games/Binding of Isaac Repentance`), separate from your R+ saves.
   Check that the title screen says Repentance.

2. **Switch to a clean mod profile.** You have about 90 mods installed, and many
   change gameplay. Fiend Folio, rebalanced Greedier and random start item would
   distort training, and *recover completion marks* would directly undermine a
   fresh-file Dead God.
   ```bash
   python tools/mod_profile.py ai
   ```
   This disables every mod except the bridge, and
   `python tools/mod_profile.py restore` puts your setup back exactly as it was.

3. **Add the launch option `--luadebug`.** In Steam: Properties → General →
   Launch Options. This lets mods use sockets, which the bridge needs.
   ⚠ It also gives *every enabled mod* file and network access, which is
   another reason for step 2. Remove it when you play normally with your full mod list.

4. **Install the bridge mod**:
   ```bash
   python tools/install_mod.py
   ```

5. **Python environment** (already created in `.venv` if you used Claude):
   ```bash
   py -3.14 -m venv .venv
   ```
   ```bash
   .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
   ```bash
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

6. **Save slots.** Use one slot for training only (slot 3, say) and keep a
   separate, untouched slot for the real Dead God run. Training spawns enemies
   and refills health, which is not a fair run. Mods don't block achievements,
   so the training slot *will* collect unlocks.

## Running it, in order

| Step | Command | What it checks |
|---|---|---|
| 0 | `.venv\Scripts\python tests/smoke_test.py --ppo` | Python side works against a fake game |
| 1 | `.venv\Scripts\python scripts/watch.py` | Mod connects; state looks right while **you** play |
| 2 | `.venv\Scripts\python scripts/scripted_bot.py --level 1` | AI control, arena resets, rewards |
| 3 | `.venv\Scripts\python scripts/train_ppo.py --level 1` | Learning! |

For steps 2 and 3, start a new run as Isaac on the training slot and stand in
the Basement 1 starting room, then start the script. Each episode wipes the
room, heals you, and spawns enemies from `isaac_ai/curriculum.py`.

Watch training with `.venv\Scripts\tensorboard --logdir runs/ppo/tb`. The
number to watch is `rollout/ep_rew_mean`. Move up a level with
`--level 2 --resume runs/ppo/latest.zip` once the agent beats the scripted
bot's score on the current level.

## Things that can go wrong

- **"LuaSocket unavailable"** in the console (press `~`): `--luadebug` isn't set.
- **Nothing connects:** the mod retries every ~2.5 s while you're *in a run*. It
  doesn't run on the menu.
- **Game shows "Not Responding" during training:** that's the lockstep pause
  while PPO updates. It recovers on its own. Lower `--n-steps` to make the pauses shorter.
- **Doors:** arena mode keeps the doors shut and reopens them when Python
  disconnects or releases control. If you ever see `left_room` episodes, the
  door lock failed. Report it; the env still ends the episode as a backstop.
- **Stuck in a room with closed doors:** run any script and Ctrl+C it, or
  leave and re-enter the run. Either one resets the doors.
- **After editing `main.lua`:** re-run `tools/install_mod.py`, then `luamod isaac_ai_bridge`
  in the console. Don't use the console on the Dead God slot, because it disables
  achievements for that run.
