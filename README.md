# BS

A 3-player implementation of the bluffing card game BS (aka Cheat), with a
PettingZoo RL environment, PPO self-play training, rule-based bots, a
FastAPI websocket server, and a single-file HTML frontend.

## Project layout

```
core/
  Action.py         engine actions (playCards, challenge, passChallenge, ...)
  GameState.py      game state, cards, ranks, PlayerObservation
  BSEnv.py          PettingZoo AECEnv wrapping the engine
  utils.py
  verboseRun.py     Command line gameplay

RL/
  agents.py             Agent interface + naive bots + PolicyAgent
  trainWithPool.py       BSPolicy network + self-play PPO trainer with past pool
  train.py              Same as above without past pool
  utils.py    

web/server/
  main.py           FastAPI app, websocket endpoint
  game_manager.py   maps table id -> GameSession
  game_session.py   one BSEnv + connected players/bots
  player.py

bs_client.html      single-file frontend (this is the deliverable to open
                    in a browser — no build step)
```

## Requirements

- Python 3.10+
- `pip install fastapi uvicorn pettingzoo gymnasium numpy torch`

## Running the server

From the repo root:

```bash
uvicorn newWeb.server.main:app --reload
```

This starts the API at `http://localhost:8000`, with the game websocket at
`ws://localhost:8000/ws/{table_id}`. Any string works as `{table_id}` — it's
just a key for `GameManager` to look up (or create) a `GameSession`.

Check it's up:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## Running the frontend

`bs_client.html` needs no build step or server of its own — just open it
directly in a browser (double-click it, or `open bs_client.html`).

1. Open `bs_client.html`.
2. Leave **Server** as `ws://localhost:8000` (or point it at wherever
   `uvicorn` is running) and **Table** as any id you want, e.g. `demo`.
3. Click **Connect**. The first person to connect to a given table id
   resets the game and takes seat 0; the next two connections take seats
   1 and 2.
4. Open the same URL in two more tabs (or send it to two friends) and
   connect to the same table id to fill the other seats — or use **Add
   bot** in the "Table management & debug" panel to seat a bot instead of
   a person.
5. As of 9/11/26, this program cannot yet handle disconnects

To play against a bot solo: connect once, add a bot for each of the other
two seats, and play. Currently, adding a "policy" bot is hardcoded to use the best policy model currently available,
and anything else creates a RandomAgent.

## Playing a turn

- **Declare phase**: pick how many cards you're playing (1–4) and how many
  of those are actually the current rank (the rest are bluffed), then
  **Declare**. Only combinations your hand size and true-card count
  actually support are selectable.
- **Challenge phase**: **Call BS** to challenge the last claim, or **Pass**
  to let it stand.

## Loading a trained policy as a bot

```python
from RL.checkpoint_utils import load_policy_agent

bot_agent = load_policy_agent("checkpoints/policy_0050000.pt")
session.add_bot(player_id, bot_agent)
```

`hidden_dim` defaults to 128 (matching `trainWithPool.py`'s default) — pass
a different value if you trained with a different `--hidden-dim`.

## Training a new policy

```bash
python -m RL.trainWithPool --total-episodes 100000
```

Checkpoints land in `checkpoints/` every `--save-interval` episodes
(default 5,000). Resume with `--resume checkpoints/policy_XXXXXXX.pt`.

## Known gaps

- **`main.py`'s initial `playerObs` send is double-JSON-encoded** (a stray
  `json.dumps()` around a dict already going into `send_json`). The
  frontend parses defensively either way, but it's worth fixing at the
  source.
- **Game-over isn't signaled to clients.** `GameState.winner` is tracked
  server-side but never reaches `PlayerObservation` — the frontend has no
  way to show a "you won" screen yet.
- **Action legality isn't broadcast** — the frontend recomputes it
  client-side by mirroring `BSEnv._get_action_mask`'s logic, so if that
  logic changes server-side, keep the two in sync (or start sending
  `action_mask` alongside `playerObs`).
- **Bot agent selection isn't wired up.** `add_bot` accepts an `agent`
  name from the client but `game_session.py`/`main.py` don't yet map that
  string to a specific `Agent` (naive type or checkpoint path) — currently
  it's on you to instantiate the right agent server-side.
- `Action._retroactive_reward` / bluff-success reward shaping is a TODO in
  `BSEnv.py`.