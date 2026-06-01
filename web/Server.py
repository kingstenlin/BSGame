"""
BS Card Game – Multiplayer Server (web + terminal)
---------------------------------------------------
Usage:
    python server.py [--host HOST] [--port PORT]

Serves game.html at / and drives all game logic over Socket.IO.
"""
import sys
sys.path.append('../')
import argparse
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from flask import Flask, request, send_from_directory
from flask_socketio import SocketIO, emit, join_room

import core.BSEnv as BSEnv
from bots import RandomBot, HonestBot, AggressiveBot, ConservativeBot

# ── constants ────────────────────────────────────────────────────────────────
IND_TO_STRING = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
VERBOSE_ACTIONS: dict[int, str] = {
    i: f"{honest} true, {quantity - honest} false"
    for i, (quantity, honest) in enumerate(BSEnv.declareActions)
}
VERBOSE_ACTIONS[14] = "Challenge"
VERBOSE_ACTIONS[15] = "Pass"

VERBOSE_ACTIONS_LONG: dict[int, str] = {
    i: f"Declare {quantity} card(s): {honest} true + {quantity - honest} bluff"
    for i, (quantity, honest) in enumerate(BSEnv.declareActions)
}
VERBOSE_ACTIONS_LONG[14] = "Challenge the previous declaration"
VERBOSE_ACTIONS_LONG[15] = "Pass (do not challenge)"

DEFAULT_PLAYERS = 4

BOT_REGISTRY = {
    "random":       RandomBot,
    "honest":       HonestBot,
    "aggressive":   AggressiveBot,
    "conservative": ConservativeBot,
}

# ── helpers ──────────────────────────────────────────────────────────────────

def decode_rank(obs) -> int:
    a, b = obs[13], obs[14]
    angle = np.arccos(np.clip(2 * (b - 0.5), -1, 1))
    idx = int(np.round(13 * angle / (2 * np.pi)))
    if a <= 0:
        idx = 13 - idx
    return max(0, min(12, idx))


def hand_counts(obs) -> dict[str, int]:
    return {IND_TO_STRING[i]: int(obs[i] * 4)
            for i in range(13) if int(obs[i] * 4) > 0}


def build_game_state(agent: str, obs, action_mask) -> dict:
    rank_idx = decode_rank(obs)
    pile_size = int(np.round(obs[15] * 52))
    is_challenge = bool(action_mask[14] != 0)

    available = {
        str(i): {"short": VERBOSE_ACTIONS[i], "long": VERBOSE_ACTIONS_LONG[i]}
        for i in range(len(action_mask)) if action_mask[i] != 0
    }

    state = {
        "agent": agent,
        "phase": "CHALLENGE" if is_challenge else "DECLARE",
        "hand": hand_counts(obs),
        "pile_size": pile_size,
        "available_actions": available,
    }
    if is_challenge:
        prev_count = int(np.round(obs[16] * 4))
        state["previous_claim"] = {"rank": IND_TO_STRING[rank_idx], "count": prev_count}
    else:
        state["rank_to_play"] = IND_TO_STRING[rank_idx]

    return state


# ── room dataclass ────────────────────────────────────────────────────────────

@dataclass
class Room:
    room_id: str
    num_players: int
    bot_config: list
    seat_map: dict = field(default_factory=dict)
    sid_to_agent: dict = field(default_factory=dict)
    player_names: dict = field(default_factory=dict)
    env: Optional[object] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: bool = False
    finished: bool = False
    _bot_instances: dict = field(default_factory=dict)


# ── Flask / SocketIO ──────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))
app.config["SECRET_KEY"] = "bs-game-secret"
sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

rooms: dict[str, Room] = {}


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "game.html")


# ── broadcast helpers ─────────────────────────────────────────────────────────

def _player_list(room: Room) -> list:
    result = []
    for seat_idx in range(room.num_players):
        agent = f"player_{seat_idx}"
        seat = room.seat_map.get(agent, "")
        is_bot = seat.startswith("__bot")
        name = room.player_names.get(agent, agent)
        hand_size = 0
        if room.env and room.started:
            try:
                obs = room.env.observe(agent)
                if obs is not None:
                    hand_size = sum(int(obs[i] * 4) for i in range(13))
            except Exception:
                pass
        result.append({
            "agent": agent,
            "name": name,
            "is_bot": is_bot,
            "bot_type": seat.split("_")[2] if is_bot and len(seat.split("_")) > 2 else "",
            "hand_size": hand_size,
            "connected": bool(seat and not is_bot),
        })
    return result


def _broadcast_table(room: Room):
    sio.emit("table_update", {"players": _player_list(room)}, to=room.room_id)


def _broadcast_lobby(room: Room):
    human_slots = room.num_players - len(room.bot_config)
    seated = sum(1 for v in room.seat_map.values() if not v.startswith("__bot"))
    sio.emit("lobby_update", {
        "room_id": room.room_id,
        "seats_taken": seated,
        "seats_total": human_slots,
        "waiting_for": human_slots - seated,
        "players": [
            {"agent": a, "name": room.player_names.get(a, a)}
            for a, s in room.seat_map.items() if not s.startswith("__bot")
        ],
    }, to=room.room_id)

def _broadcast_hands(room: Room):
    """Push each human player their current hand via env.observe."""
    if not room.env or not room.started:
        return
    for agent, sid in room.seat_map.items():
        if sid.startswith("__bot"):
            continue
        try:
            obs = room.env.observe(agent)
            if obs is None:
                continue
            hand = {IND_TO_STRING[i]: int(obs[i] * 4)
                    for i in range(13) if int(obs[i] * 4) > 0}
            pile_size = int(np.round(obs[15] * 52))
            sio.emit("hand_update", {"hand": hand, "pile_size": pile_size}, to=sid)
        except Exception:
            pass

# ── socket events ─────────────────────────────────────────────────────────────

@sio.on("create_room")
def on_create_room(data):
    num_players = int(data.get("num_players", DEFAULT_PLAYERS))
    bot_names = data.get("bots", [])

    if len(bot_names) >= num_players:
        emit("error", {"msg": "Need at least one human player."})
        return
    for b in bot_names:
        if b not in BOT_REGISTRY:
            emit("error", {"msg": f"Unknown bot '{b}'. Options: {list(BOT_REGISTRY)}"})
            return

    room_id = uuid.uuid4().hex[:6].upper()
    room = Room(room_id=room_id, num_players=num_players, bot_config=bot_names)
    rooms[room_id] = room

    emit("room_created", {
        "room_id": room_id,
        "num_players": num_players,
        "bot_slots": len(bot_names),
        "human_slots": num_players - len(bot_names),
    })


@sio.on("join_room")
def on_join_room(data):
    room_id = data.get("room_id", "").upper().strip()
    player_name = (data.get("name") or request.sid[:6]).strip()[:20]

    if room_id not in rooms:
        emit("error", {"msg": f"Room {room_id!r} not found."})
        return

    room = rooms[room_id]
    with room.lock:
        if room.started:
            emit("error", {"msg": "Game already started."})
            return

        human_slots = room.num_players - len(room.bot_config)
        seated = sum(1 for v in room.seat_map.values() if not v.startswith("__bot"))
        if seated >= human_slots:
            emit("error", {"msg": "Room is full."})
            return

        taken = set(room.seat_map.keys())
        assigned = None
        for seat_idx in range(room.num_players):
            agent = seat_idx
            if agent not in taken:
                room.seat_map[agent] = request.sid
                room.sid_to_agent[request.sid] = agent
                room.player_names[agent] = player_name
                join_room(room_id)
                assigned = agent
                break

        emit("joined", {
            "room_id": room_id,
            "agent": assigned,
            "name": player_name,
            "num_players": room.num_players,
            "bot_config": room.bot_config,
        })

        _broadcast_lobby(room)

        seated_now = sum(1 for v in room.seat_map.values() if not v.startswith("__bot"))
        if seated_now == human_slots:
            _start_game(room)


@sio.on("submit_action")
def on_submit_action(data):
    room_id = data.get("room_id", "").upper()
    try:
        action = int(data.get("action", -1))
    except (TypeError, ValueError):
        emit("error", {"msg": "Invalid action."})
        return

    if room_id not in rooms:
        emit("error", {"msg": "Room not found."})
        return

    room = rooms[room_id]
    with room.lock:
        if not room.started or room.finished:
            emit("error", {"msg": "No game in progress."})
            return

        expected = room.sid_to_agent.get(request.sid)
        current = room.env.agent_selection
        if expected != current:
            emit("error", {"msg": "Not your turn."})
            return

        _, _, _, _, info = room.env.last()
        mask = info["action_mask"]
        if action < 0 or action >= len(mask) or mask[action] == 0:
            emit("error", {"msg": f"Action {action} is invalid."})
            return

        room.env.step(action)

    _broadcast_hands(room)
    _broadcast_table(room)
    _advance_game(room)


@sio.on("disconnect")
def on_disconnect():
    sid = request.sid
    for room in list(rooms.values()):
        if sid in room.sid_to_agent:
            agent = room.sid_to_agent[sid]
            name = room.player_names.get(agent, agent)
            sio.emit("player_left", {"agent": agent, "name": name}, to=room.room_id)
            room.finished = True


# ── game logic ────────────────────────────────────────────────────────────────

def _start_game(room: Room):
    bot_offset = room.num_players - len(room.bot_config)
    for i, bot_name in enumerate(room.bot_config):
        agent = bot_offset + i
        sentinel = f"__bot_{bot_name}_{i}"
        room.seat_map[agent] = sentinel
        room.player_names[agent] = f"{bot_name.capitalize()} Bot"
        room._bot_instances[agent] = BOT_REGISTRY[bot_name]()

    room.env = BSEnv.BSEnv()
    room.env.reset(seed=None)
    room.started = True

    sio.emit("game_started", {
        "room_id": room.room_id,
        "num_players": room.num_players,
        "players": _player_list(room),
    }, to=room.room_id)

    _advance_game(room)


def _advance_game(room: Room):
    env = room.env

    while True:
        if env.agent_selection is None:
            _end_game(room)
            return

        current = env.agent_selection
        _, reward, termination, truncation, info = env.last()
        obs = env.observe(current)

        if termination or truncation:
            env.step(None)
            if all(env.terminations.get(a, False) or env.truncations.get(a, False)
                   for a in env.agents):
                _end_game(room)
                return
            continue

        mask = info["action_mask"]
        seat = room.seat_map.get(current, "")

        if seat.startswith("__bot"):
            bot = room._bot_instances[current]
            action = bot.act(obs, mask)
            if mask[action] == 0:
                action = next(i for i, v in enumerate(mask) if v != 0)

            bot_type = seat.split("_")[2]
            sio.emit("bot_action", {
                "agent": current,
                "name": room.player_names.get(current, current),
                "bot_type": bot_type,
                "action": int(action),
                "action_short": VERBOSE_ACTIONS[action],
                "action_long": VERBOSE_ACTIONS_LONG[action],
            }, to=room.room_id)

            env.step(action)
            _broadcast_table(room)
            continue

        # human's turn
        state = build_game_state(current, obs, mask)
        target_sid = room.seat_map[current]
        sio.emit("your_turn", state, to=target_sid)
        sio.emit("waiting_for", {
            "agent": current,
            "name": room.player_names.get(current, current),
        }, to=room.room_id)
        _broadcast_table(room)
        return


def _end_game(room: Room):
    room.finished = True
    env = room.env
    results = {a: float(env.rewards.get(a, 0)) for a in env.possible_agents}
    winner = max(results, key=results.get)
    winner_name = room.player_names.get(winner, winner)
    sio.emit("game_over", {
        "winner": winner,
        "winner_name": winner_name,
        "rewards": results,
        "names": room.player_names,
    }, to=room.room_id)
    try:
        env.close()
    except Exception:
        pass


# ── entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    print(f"BS Game Server running at http://localhost:{args.port}")
    sio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)