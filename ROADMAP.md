# Roadmap to Dead God

Dead God means unlocking all 637 achievements on one save file. Humans
typically take several hundred to over a thousand hours. As far as I know, no
AI has done it publicly. So this is a long research project, and each phase
below is a real milestone in its own right.

The core design idea is **hierarchy**. One neural network can't learn
"press left" and "which character should I unlock next" together. Instead
we stack layers, each solving one kind of problem, with the simplest tool
that works:

```
 Achievement manager   what to unlock next → which character, which ending   (planner + rules)
 Run strategy          which items, shops, deals, which path                 (rules → learned)
 Floor navigation      which room next, where's the boss                     (graph search)
 Room combat           move, shoot, dodge                                    (RL policy)
```

Lower layers get done first, because everything above depends on them.

---

## Phase 0: Bridge (code written; needs testing in the real game)
Lua mod ↔ Python lockstep link, structured observations, arena resets, a
scripted baseline and a PPO training script.

**Done when:** `watch.py` shows sensible numbers in the real game and
`scripted_bot.py` clears level-1 arenas.

## Phase 1: Room combat (RL)
Train the combat policy through the curriculum levels in `curriculum.py`.

- Beat the scripted bot at each level before moving on.
- Expected issues and their fixes:
  - **Agent hugs a corner:** the time penalty or hit penalty is too large relative to damage reward.
  - **Agent never shoots:** raise `ent_coef` or simplify level 1 further.
  - **Too slow to learn:** see "The speed problem" below.
- Add what's missing from the observation as we go: lasers, creep, explosive
  barrels, bombs, enemy *type* (embedding), and tear stats beyond damage.

**Done when:** above 90% clear rate on level 3 arenas with under 1 hit per room.

## Phase 2: Whole floors
Stop resetting rooms and play real runs: combat policy in rooms with enemies,
scripted navigation everywhere else.

- The mod exports the floor layout (`Level:GetRooms()`: room shapes, types,
  doors, cleared/visited).
- Navigation = shortest path (BFS) to the next uncleared room, then to the boss
  room, then the trapdoor. Pick up pickups by walking to them.
- New action: walk through doors, use keys and bombs for locked/secret rooms.

**Done when:** Isaac regularly beats Mom (Depths 2) on normal runs.

## Phase 3: Run decisions
Items, shops, devil/angel deals, pills, cards, active items, beggars.

- Start with a hand-made item tier list. That's quick, and decent.
- Later: learn item values from run outcomes (which items correlate with
  winning *for this agent*, which may differ from what humans think).
- Extend the action space: use active item, bomb, card/pill, drop.

**Done when:** consistent wins through Mom's Heart, Isaac, Satan.

## Phase 4: Late game and characters
Blue Baby, The Lamb, Mega Satan, Hush, Delirium, Mother, The Beast, Boss Rush,
Greed and Greedier mode, Hard mode.

- **Boss arenas:** the same `reset_arena` trick, spawning bosses on later floors.
- **Characters change the controls:** The Forgotten (melee), Azazel (short
  brimstone), The Lost (one hit), Keeper (coins as health), Jacob & Esau (two
  bodies), plus every Tainted character. The policy gets a "character" input,
  and some characters will need dedicated training.
- **Route quests:** Mother (knife pieces), The Beast (Home), Tainted unlocks via
  the Red Key at Home. These are long scripted sequences with combat in between.

## Phase 5: Achievement manager
The top layer. It reads which achievements the save has, keeps the full
637-entry checklist (target, character, mode, prerequisites), and plans the
next run. Many unlocks are odd one-offs (donation machine totals, Greed
machine, specific challenges, "do X N times"), so this layer is mostly a
dependency graph plus scripted objectives, not learning.

**Possible upgrade:** REPENTOGON (already installed on this PC) adds API calls
for reading unlock state and more. That's worth evaluating here.

## Phase 6: The real attempt
New save slot, AI-only, no console, no arena mode, no help. Log every run.

---

## The speed problem
The game runs at real-time speed (30 updates/s) and we can only run one copy
(maybe more; worth testing). RL usually wants millions of steps; 1M steps at
15 decisions/s is about 18 hours. Mitigations, in rough order of effort:

1. **Small, focused tasks** (arenas instead of whole runs) and compact observations. This is already done.
2. **Imitation learning:** record your own play with `watch.py`-style logging,
   pretrain the policy to copy you ("behavioral cloning"), then fine-tune with PPO.
   This often cuts training time a lot.
3. **A simplified simulator** in Python with similar dodge-and-shoot mechanics,
   running thousands of times faster. Pretrain there, fine-tune in the real game.
   The `tests/mock_game.py` toy is a seed for this.
4. **Multiple game instances**, if Steam allows it.

## Open questions to decide later
- **Fairness:** the API reveals things a player can't see, like the full map
  and secret room locations. Do we allow the AI to use that, or restrict it to
  what's on screen?
- **Fresh-file check:** confirm that a brand-new Repentance save stays at zero
  unlocks, and doesn't pick up achievements already on your Steam account. Do
  this before Phase 6.
- **Version pinning:** we're on Repentance v1.7.9b. If Steam ever updates
  it, re-run the smoke and watch checks.
