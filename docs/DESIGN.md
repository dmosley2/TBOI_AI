# Design and decision log

This document explains **how the system works and why it's built this way**.
Each decision records the context, what we chose, the alternatives, and what
the choice costs us. When a decision changes, add a new entry that supersedes
the old one, rather than editing history.

---

## 1. The problem

Train an agent to play *The Binding of Isaac: Repentance*, with the long-term
goal of earning every achievement (Dead God) on a fresh save file.

**Constraints that shape everything:**

| Constraint | Consequence |
|---|---|
| Closed-source commercial game, no RL interface | We must build the environment ourselves |
| Runs only in real time: 30 logic updates/s | Experience is expensive: ~54k agent steps per hour at most |
| One game instance (as far as we know) | No parallel environments, unlike most RL setups |
| The game has a Lua modding API, sandboxed by default | A mod can read game state and override input; networking needs `--luadebug` |
| Roguelike: random floors, about 700 items, 34 characters | One end-to-end policy is unrealistic, so the problem has to be decomposed |

---

## 2. Architecture

```
┌──────────────────────── Game process ────────────────────────┐
│  Lua mod (mod/isaac_ai_bridge/main.lua)                      │
│   MC_POST_UPDATE  → snapshot → JSON ──┐                      │
│   MC_INPUT_ACTION ← apply action  ◀───┼─┐                    │
│   MC_ENTITY_TAKE_DMG → event counters │ │                    │
└───────────────────────────────────────┼─┼────────────────────┘
                              TCP, one JSON line each way per update
┌───────────────────────────────────────┼─┼──── Python ────────┐
│  bridge.py      IsaacBridge.step(msg) ┘ │  lockstep transport │
│  features.py    snapshot → 351-float vector                   │
│  env.py         Gymnasium env: reset / step / reward          │
│  curriculum.py  which enemies to spawn (adaptive)             │
│  train_ppo.py   Stable-Baselines3 PPO                         │
│  eval_enemies.py, scripted_bot.py   measurement & baselines   │
└───────────────────────────────────────────────────────────────┘
```

**One step in time:** the game finishes an update → the mod serializes state
and blocks on the socket → Python encodes features, the policy picks an
action, and Python replies → the mod stores the action and returns → on the
next update, the game's input polling calls `MC_INPUT_ACTION`, and the mod
answers with the AI's buttons instead of the real controller's.

---

## 3. Decisions

### D1. Read state through the mod API, not the screen
- **Context:** the agent needs to know where things are.
- **Decision:** a Lua mod reads entity data directly (positions, velocities, HP, room grid).
- **Alternatives:** screen capture plus a CNN, which is how Atari agents and
  most "AI plays X" videos work.
- **Why:** pixel-based RL needs tens of millions of frames. At real-time speed
  that's months of play. Structured state removes the whole perception
  problem, so learning only has to solve control.
- **Cost:** the agent only knows what we export. Lasers, creep, explosions and
  bombs aren't exported yet, so it's blind to them until we add them. Code
  must also be written for every new kind of thing in the game.

### D2. Lockstep synchronization
- **Context:** the game and the learner run in different processes.
- **Decision:** the mod blocks after every update until Python replies.
- **Alternatives:** asynchronous, where the game runs freely and Python sends
  actions whenever it's ready.
- **Why:** with async, the delay between observation and action varies with
  CPU load, and during PPO's learning phase (seconds long) the game would keep
  running with nobody playing. Lockstep makes every transition exact: this
  state, this action, this result. That's an assumption RL relies on (the
  Markov property).
- **Cost:** the game freezes during network updates. Windows can show "Not
  Responding", which is cosmetic. Throughput is capped by Python's response time.

### D3. Python is the server, the game is the client
- **Decision:** Python listens on `127.0.0.1:28771`; the mod retries
  connecting every ~2.5 s.
- **Why:** the game is the long-lived process and scripts come and go. With
  this setup any script (watch, bot, training, eval) can attach and detach
  without restarting the game. Localhost only, so nothing is exposed to the network.

### D4. Newline-delimited JSON over TCP
- **Alternatives:** binary format (struct/msgpack), shared memory, files.
- **Why:** the game already ships `json.lua` and LuaSocket, so no native code is
  needed. The messages are human-readable, which mattered while debugging the
  first connection, and a 1–2 KB message per update is negligible on localhost.
- **Revisit if:** profiling shows serialization as the bottleneck.

### D5. Inject input through `MC_INPUT_ACTION`
- **Alternatives:** simulated keystrokes (SendInput, virtual gamepad), or
  setting the player's velocity directly.
- **Why:** the callback answers the game's own input queries, so the AI uses
  the exact same control path as a human. Movement acceleration, shooting
  rules and item behaviour all stay faithful. Simulated keys need the window
  focused and add timing jitter. Setting velocity directly would bypass the
  game's movement physics, so the learned skill wouldn't transfer.
- **Detail:** "triggered" (just pressed) is computed as an edge between
  consecutive actions, so one bomb request drops exactly one bomb.

### D6. Observation design (`features.py`)
A neural network needs a fixed-size input, but rooms hold a variable number of things.

| Choice | Reason |
|---|---|
| **K nearest** enemies (8) and projectiles (12), zero-padded | Fixed size. The nearest threats matter most |
| **Presence flag** on each slot | Lets the network tell "enemy at offset (0,0)" apart from padding |
| **Coordinates relative to the player** | "Bullet just to my left" looks the same anywhere in the room, so the policy generalizes |
| **Scaling to about [-1, 1]** | Large raw values (hundreds of pixels) make gradient descent unstable |
| **7×7 terrain window** around the player | Local obstacles and pits, independent of room size and shape |
| **One-hot enemy type** (18 slots, D12) | Lets the agent learn per-enemy behaviour; see below |

**Why one-hot instead of the type ID as a number:** a network reads numbers as
amounts, so ID 12 would mean "slightly more than ID 10". One-hot gives each
type its own input. For an MLP, the first-layer weights on those inputs work
exactly like a learned embedding.

**Cost:** the vector size depends on the vocabulary. Adding an enemy type
changes the input size, and saved models can't be loaded anymore.
**Alternative for later:** an attention or set encoder over all entities,
which removes the K limit and the sorting.

### D7. Action space: `MultiDiscrete([9, 5])` with frame skip 2
- **Decision:** move (8 directions + none) and shoot (4 directions + none),
  chosen together. Each action is held for 2 game updates (15 decisions/s).
- **Alternatives:** a single 45-way discrete action, continuous movement, or no frame skip.
- **Why:** Isaac's controls *are* 8-way movement and 4-way shooting, so we
  match the game. MultiDiscrete factors them, so the network learns "where to
  move" and "where to shoot" as two outputs instead of 45 unrelated combos.
  Frame skip halves the number of decisions per second of play, which makes
  credit assignment over time easier, and 15 Hz is still faster than human reaction time.
- **Not yet included:** bombs, active items, pills/cards. The mod supports
  them; they're left out of the action space until the agent needs them.

### D8. Reward design (`env.py`)
| Term | Value | Purpose |
|---|---|---|
| Damage dealt | +1 total for killing everything | Dense signal: progress inside a room |
| Clear bonus | +1 | The actual goal |
| Hit taken | −0.5 per hit | Teaches dodging. Health is the scarcest resource in a run |
| Death or leaving the room | −1 and episode ends | Hard failure |
| Time | −0.002 per step | Discourages stalling or hiding forever |

- **Damage is normalized by total enemy HP**, so a room of flies and a room
  with a 250-HP boss are both worth +1. Without this, reward scale would
  depend on what got spawned, which destabilizes learning.
- **Risk we watch for:** reward hacking, i.e. maximizing the number without
  doing the task. Example: if the hit penalty is too large relative to
  damage, the agent learns to hide in a corner. The scripted baseline and
  the per-enemy report card exist to catch this.

### D9. Arena episodes instead of full runs
- **Decision:** an episode is one fight. The mod wipes the current room,
  heals the player, spawns chosen enemies, locks the doors, and cancels the
  killing blow (reported as `lethal` instead).
- **Why:** full runs are minutes long, with sparse, delayed reward and
  constant non-combat activity. Arenas give hundreds of short, focused episodes
  per hour. Cancelling the killing blow avoids game-over screens and menu
  navigation, so training never stops.
- **Doors:** dodging could carry the agent through an open door. The mod
  re-closes doors every update in arena mode. Leaving arena mode is routed
  through one function (`setArena(false)`), which always reopens them, even
  on disconnect or a Python crash.
- **Cost:** arenas don't train navigation, pickups or room transitions; those
  belong to Phase 2 (ROADMAP). Spawned enemies also skip their normal
  room-entry timing.

### D10. PPO from Stable-Baselines3, on the CPU
- **Alternatives:** DQN, SAC, writing PPO from scratch.
- **Why PPO:** stable and forgiving with hyperparameters, handles
  MultiDiscrete natively, and is the standard first choice for game control.
  DQN doesn't support MultiDiscrete directly, and SAC is mainly for continuous actions.
- **Why a library:** a hand-written PPO is a classic source of silent bugs.
  Using SB3 lets the project focus on environment design, which is the novel part.
- **Why CPU:** a 2×256 MLP is tiny. GPU transfer overhead exceeds the
  compute saved, and the game is the bottleneck anyway.
- **Key hyperparameters:** `n_steps=2048` (≈4.5 min of play per update),
  `gamma=0.99` (≈7 s effective horizon at 15 Hz, about one room fight),
  `ent_coef=0.01` (keeps early exploration).

### D11. A scripted baseline before any learning
- **Decision:** a hand-written policy (line up, shoot, keep distance,
  sidestep bullets) runs through the same environment.
- **Why:** (1) it validated the entire pipeline in the real game before any
  training time was spent; (2) it's a reference score. If PPO can't beat a
  30-line heuristic, the problem is in rewards or features, not training time.
- **Result so far:** 10/10 clears, 0 hits on level 1 (one easy enemy) in the real game.

### D12. Curriculum: mixed enemy types with adaptive sampling
*(Replaced a fixed 4-level curriculum. Discussed when choosing whether to
train enemy by enemy.)*
- **Context:** should the agent master each enemy type in turn before seeing mixed rooms?
- **Decision:** levels increase in difficulty (1 easy enemy → 1 of any type →
  mixed groups → full rooms → bosses), but **within a level the enemy type is
  randomized every episode**. The sampler tracks a skill score per enemy
  (an exponential moving average of episode quality, α = 0.05) and samples
  in proportion to `max(0.15, 1 − skill)`.
- **Alternative rejected:** sequential per-enemy training. Neural networks
  suffer **catastrophic forgetting**: hours of Horf-only data push the weights
  away from what was learned about flies, because nothing in the data
  preserves that knowledge. It also invites enemy-specific tricks instead of
  general skills, and doesn't scale to hundreds of enemy types at real-time speed.
- **Why adaptive:** it keeps the benefit of focused practice on weak spots,
  without the forgetting. The 0.15 floor keeps mastered enemies in practice.
- **Known compromise:** in mixed rooms a hit can't be attributed to a
  specific enemy, so all types present share the score. The single-enemy
  levels supply clean data per type.
- **Persistence:** saved to `runs/<name>/skills.json` next to the model, so
  `--resume` continues both together.

### D13. Separate evaluation from training statistics
- **Decision:** `eval_enemies.py` fights each enemy alone N times with
  **deterministic** actions and reports clear rate, hits and score per enemy.
- **Why:** training reward is one average that hides failures ("1.6 mean"
  could be perfect against 12 enemies and hopeless against Horfs). Training
  also samples random actions on purpose (exploration), so its numbers
  underestimate the learned policy. Eval doesn't update the curriculum, so
  measurement can't change training.
- **Level-advance rule:** advance when every enemy in the pool scores well in eval.

### D14. A mock game for automated tests
- **Decision:** `tests/mock_game.py` speaks the exact protocol with toy
  physics, and `tests/smoke_test.py` runs env, bot, curriculum, eval and a
  short PPO run against it.
- **Why:** the real game needs a human to launch it and start a run. The mock
  catches Python bugs in seconds, and it already caught one: a numpy boolean
  subtraction error in the bot. It's explicitly **not** a training simulator.

### D15. Environment hygiene: Repentance, clean mods, separate saves
- **Pinned to Repentance v1.7.9b** (not Repentance+), which is stable and no
  longer updated, so API behaviour won't shift under us.
- **Clean mod profile** (`tools/mod_profile.py`): other mods change gameplay
  and one unlocks achievements. The tool records exactly what it disabled, so
  `restore` puts the user's setup back exactly.
- **`--luadebug` security:** it gives *every* enabled mod file and network
  access, which is another reason to run with only our mod enabled.
- **Separate save slots:** training spawns enemies and refills health, so it
  isn't a legitimate run. The Dead God attempt uses an untouched slot with no
  console commands.

### D16. Hierarchical decomposition for the long-term goal
- **Decision:** separate layers: combat (RL) → floor navigation (graph search)
  → run decisions (rules, then learned) → achievement planning (dependency
  graph and scripted objectives). See ROADMAP.md.
- **Why:** these problems have very different time scales (1/15 s per combat
  decision vs a whole run per achievement) and very different data. One
  network learning "press left" and "which character to unlock next" from
  one reward is far beyond the experience we can collect. Using the simplest
  tool that works at each layer keeps every part testable on its own.

---

## 4. Status and known limitations

| Area | Status |
|---|---|
| Bridge: state, input, events | Verified in the real game |
| Arena reset, scripted control | Verified in the real game (level 1) |
| Door locking | Implemented, **not yet verified in game** |
| Enemy-type features, adaptive curriculum, eval | Verified against the mock only |
| PPO training in the real game | Not started |

**Limitations:**
- Not observed yet: lasers, creep, bombs, explosions, pickups, tear stats beyond damage.
- Only player 0 is controlled. Jacob & Esau and co-op characters will need changes.
- One game instance at real time. See the "speed problem" in ROADMAP.md
  (imitation learning, a simulator, multiple instances).
- Spawned arena enemies skip natural room-entry patterns, so there's a
  domain gap with real rooms that Phase 2 has to close.

---

## 5. Metrics

| Metric | Where | Meaning |
|---|---|---|
| `rollout/ep_rew_mean` | TensorBoard | Average training return (includes exploration noise) |
| Per-enemy clear %, hits, score | `eval_enemies.py` | Actual capability of the deterministic policy |
| Skill table | `python -m isaac_ai.curriculum runs/ppo/skills.json` | What the sampler thinks the agent struggles with |
| Scripted-bot scores | `scripted_bot.py`, `eval_enemies.py --scripted` | The baseline that learning must beat |

---

## 6. Glossary
- **Episode:** one attempt at a task from reset to end (here: one arena fight).
- **Policy:** the network mapping observations to action probabilities.
- **Advantage:** how much better an action did than the value network expected. PPO pushes up the probability of actions with positive advantage.
- **GAE (λ = 0.95):** a way to estimate advantage that trades bias against variance.
- **Clipping (PPO):** limits how far one update can move the policy, which keeps training stable.
- **Entropy bonus:** reward for staying uncertain early on, so the agent keeps trying different actions.
- **Frame skip:** repeating each action for several game frames.
- **Curriculum:** ordering training tasks from easy to hard.
- **Catastrophic forgetting:** a network losing old skills when trained only on new data.
- **One-hot:** a vector of zeros with a single 1 marking a category.
- **Reward hacking:** maximizing the reward number without solving the intended task.
- **Domain gap:** differences between training conditions and real use.
