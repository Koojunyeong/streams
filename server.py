"""
Streams PVE game server.

This version uses Supabase for persistent storage.
"""

import csv
import io
import json
import os
import random
from datetime import datetime, timedelta, timezone
from functools import cmp_to_key
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from flask import Flask, Response, jsonify, request, send_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None

SUPABASE_AVAILABLE = False
SUPABASE_CLIENT = None
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
if not SUPABASE_SERVICE_ROLE_KEY:
    SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_KEY", "")

if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        from supabase import create_client

        try:
            from supabase.client import ClientOptions

            SUPABASE_CLIENT = create_client(
                SUPABASE_URL,
                SUPABASE_SERVICE_ROLE_KEY,
                options=ClientOptions(auto_refresh_token=False, persist_session=False),
            )
        except Exception:
            SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

        SUPABASE_AVAILABLE = True
    except Exception as exc:
        print(f"[Supabase] init failed: {exc}")
else:
    print("[Supabase] missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

app = Flask(__name__)

if TORCH_AVAILABLE and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    DEVICE = torch.device("cuda")
else:
    DEVICE = "cpu"

print(f"[Server Device] {DEVICE}")

KST = ZoneInfo("Asia/Seoul")
PERIOD_LABELS = {"daily": "24 Hours", "weekly": "7 Days", "overall": "Overall"}


def sb():
    if not SUPABASE_AVAILABLE or SUPABASE_CLIENT is None:
        raise RuntimeError("Supabase client is not configured")
    return SUPABASE_CLIENT


def resp_data(response):
    return getattr(response, "data", []) or []


def resp_count(response):
    count = getattr(response, "count", None)
    return int(count) if count is not None else None


def normalize_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def sanitize_phone_number(value):
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def masked_phone_suffix(name, phone_number, duplicate_names):
    digits = sanitize_phone_number(phone_number)
    if not digits:
        return f"{name} (unknown)"
    suffix_len = 5 if duplicate_names.get(name, 0) > 1 else 4
    suffix = digits[-suffix_len:] if len(digits) >= suffix_len else digits
    return f"{name} ({suffix})"


def parse_played_at(value):
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def duration_sort_value(duration_ms):
    try:
        duration = int(duration_ms)
    except (TypeError, ValueError):
        return 10**15
    return duration if duration > 0 else 10**15


def format_duration_label(duration_ms):
    duration = max(0, int(duration_ms or 0))
    total_seconds = duration // 1000
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def compare_rank_entries(left, right):
    if left["player_score"] != right["player_score"]:
        return -1 if left["player_score"] > right["player_score"] else 1

    left_duration = duration_sort_value(left.get("duration_ms"))
    right_duration = duration_sort_value(right.get("duration_ms"))
    if left_duration != right_duration:
        return -1 if left_duration < right_duration else 1

    if left["played_at"] != right["played_at"]:
        return -1 if left["played_at"] > right["played_at"] else 1

    if left["identity_key"] == right["identity_key"]:
        return 0
    return -1 if left["identity_key"] < right["identity_key"] else 1


def build_leaderboards(player_rows, game_rows, target_player_id=None):
    players = {normalize_id(row["id"]): row for row in player_rows if row.get("id") is not None}
    target_phone = ""
    if target_player_id is not None:
        target_player = players.get(normalize_id(target_player_id))
        if target_player:
            target_phone = sanitize_phone_number(target_player.get("phone_number", ""))

    now_kst = datetime.now(KST)
    cutoffs = {
        "daily": now_kst - timedelta(days=1),
        "weekly": now_kst - timedelta(days=7),
        "overall": None,
    }
    best_by_period = {period: {} for period in cutoffs}

    for game in game_rows:
        pid = normalize_id(game.get("player_id"))
        player = players.get(pid)
        if not player:
            continue

        phone_number = sanitize_phone_number(player.get("phone_number", ""))
        if not phone_number:
            continue

        played_at = parse_played_at(game.get("played_at"))
        if not played_at:
            continue
        played_at_kst = played_at.astimezone(KST)

        candidate = {
            "player_id": pid,
            "player_name": player.get("player_name") or game.get("player_name") or "Anonymous",
            "phone_number": phone_number,
            "identity_key": phone_number,
            "player_score": int(game.get("player_score") or 0),
            "duration_ms": max(0, int(game.get("duration_ms") or 0)),
            "played_at": played_at.isoformat(),
            "played_at_label": played_at_kst.strftime("%m/%d %H:%M"),
        }

        for period, cutoff in cutoffs.items():
            if cutoff and played_at_kst < cutoff:
                continue
            current = best_by_period[period].get(phone_number)
            if current is None or compare_rank_entries(candidate, current) < 0:
                best_by_period[period][phone_number] = dict(candidate)

    periods = {}
    player_summary = {}

    for period, entries_by_phone in best_by_period.items():
        entries = list(entries_by_phone.values())
        name_counts = {}
        for entry in entries:
            name_counts[entry["player_name"]] = name_counts.get(entry["player_name"], 0) + 1
        for entry in entries:
            entry["display_name"] = masked_phone_suffix(entry["player_name"], entry["phone_number"], name_counts)

        ranked = sorted(entries, key=cmp_to_key(compare_rank_entries))
        rows = []
        for rank, entry in enumerate(ranked[:100], start=1):
            rows.append(
                {
                    "rank": rank,
                    "player_id": entry["player_id"],
                    "display_name": entry["display_name"],
                    "score": entry["player_score"],
                    "duration_ms": entry["duration_ms"],
                    "duration_label": format_duration_label(entry["duration_ms"]),
                    "played_at_label": entry["played_at_label"],
                }
            )

        periods[period] = {
            "label": PERIOD_LABELS[period],
            "rows": rows,
        }

        if target_phone:
            rank = next((idx for idx, entry in enumerate(ranked, start=1) if entry["identity_key"] == target_phone), None)
            current = entries_by_phone.get(target_phone)
            player_summary[period] = (
                {
                    "rank": rank,
                    "score": current["player_score"],
                    "duration_ms": current["duration_ms"],
                    "duration_label": format_duration_label(current["duration_ms"]),
                }
                if current
                else None
            )

    return {
        "periods": periods,
        "player": player_summary if target_phone else None,
    }


def fetch_all_rows(table, columns="*", chunk_size=1000, order_by=None, desc=False, filters=None):
    rows = []
    offset = 0
    while True:
        query = sb().table(table).select(columns)
        if filters:
            for method, args in filters:
                query = getattr(query, method)(*args)
        if order_by:
            query = query.order(order_by, desc=desc)
        page = query.range(offset, offset + chunk_size - 1).execute()
        page_rows = resp_data(page)
        rows.extend(page_rows)
        if len(page_rows) < chunk_size:
            break
        offset += chunk_size
    return rows


def init_schema_check():
    if not SUPABASE_AVAILABLE:
        return
    try:
        sb().table("players").select("id").limit(1).execute()
        print("[Supabase] schema check ok")
    except Exception as exc:
        print(f"[Supabase] schema check warning: {exc}")


init_schema_check()


if TORCH_AVAILABLE:

    class DuelingQNetwork(nn.Module):
        def __init__(self, state_size, action_size):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.feature = nn.Sequential(
                nn.Linear(64 * 20 + 51, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            )
            self.value_stream = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
            self.advantage_stream = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_size))

        def forward(self, x):
            board_norm = x[:, :20].unsqueeze(1)
            board_occ = x[:, 20:40].unsqueeze(1)
            conv_out = self.conv(torch.cat([board_norm, board_occ], dim=1)).flatten(1)
            features = self.feature(torch.cat([conv_out, x[:, 40:]], dim=1))
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))


NUMBER_POOL = []
for i in range(1, 11):
    NUMBER_POOL.append(i)
for i in range(11, 21):
    NUMBER_POOL.extend([i, i])
for i in range(21, 31):
    NUMBER_POOL.append(i)

TOTAL_COUNTS = {x: NUMBER_POOL.count(x) for x in range(1, 31)}


def get_prob_mask(board, num, deck, ci):
    rem = deck[ci + 1 :]
    mask = []
    for i in range(20):
        if board[i]:
            mask.append(0.0)
            continue

        lv, li = 0, -1
        for l in range(i - 1, -1, -1):
            if board[l]:
                lv, li = board[l], l
                break

        rv, ri = 31, 20
        for r in range(i + 1, 20):
            if board[r]:
                rv, ri = board[r], r
                break

        if not (lv < num < rv):
            mask.append(0.0)
            continue

        if i - li - 1 > 0 and sum(1 for c in rem if lv < c < num) < i - li - 1:
            mask.append(0.0)
            continue

        if ri - i - 1 > 0 and sum(1 for c in rem if num < c < rv) < ri - i - 1:
            mask.append(0.0)
            continue

        mask.append(1.0)

    return np.array(mask)


def build_state(board, num, deck, ci):
    bn = [x / 30.0 if x else -1.0 for x in board]
    bo = [1.0 if x else 0.0 for x in board]
    drawn = {}
    for c in deck[: ci + 1]:
        drawn[c] = drawn.get(c, 0) + 1
    rv = [max(0, TOTAL_COUNTS.get(n, 0) - drawn.get(n, 0)) / 2.0 for n in range(1, 31)]
    lm = get_prob_mask(board, num, deck, ci) if num > 0 else np.zeros(20)
    return np.concatenate([np.array(bn), np.array(bo), np.array([num / 30.0]), np.array(rv), lm])


model = None
loaded = False
if TORCH_AVAILABLE:
    model = DuelingQNetwork(91, 20).to(DEVICE)
    for name in ["best_model.pth", "final_model.pth"]:
        model_path = os.path.join(BASE_DIR, name)
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                loaded = True
                print(f"[Model] {name}")
                break
            except Exception as exc:
                print(f"[WARNING] Failed to load {name}: {exc}")
    if not loaded:
        print("[WARNING] No model file")
else:
    print("[WARNING] PyTorch not installed; using fallback AI")


def require_supabase():
    if not SUPABASE_AVAILABLE:
        return jsonify({"status": "error", "message": "Supabase is not configured"}), 500
    return None


@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "game.html"))


@app.route("/admin")
def admin():
    return send_file(os.path.join(BASE_DIR, "admin.html"))


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": loaded,
            "ai_ready": True,
            "ai_mode": "torch" if loaded else "fallback",
            "device": str(DEVICE),
            "supabase_ready": SUPABASE_AVAILABLE,
        }
    )


@app.route("/api/ai_move", methods=["POST"])
def ai_move():
    d = request.json
    board, num, deck, ci = d["board"], d["current_num"], d["deck"], d["current_index"]
    pm = get_prob_mask(board, num, deck, ci)
    st = build_state(board, num, deck, ci)

    if TORCH_AVAILABLE and loaded:
        fb = np.where(np.array(board) == 0)[0]
        if len(fb) == 0:
            return jsonify({"action": 0, "q_values": [0] * 20, "prob_mask": pm.tolist()})

        safe = np.where((np.array(board) == 0) & (pm == 1.0))[0]
        valid_actions = safe if len(safe) else fb

        with torch.no_grad():
            st_t = torch.FloatTensor(st).unsqueeze(0).to(DEVICE, non_blocking=True)
            qv = model(st_t)
            mask = torch.full(qv.shape, -1e9, device=DEVICE)
            mask[0, valid_actions] = 0
            action = (qv + mask).max(1)[1].item()
            qv_list = qv.squeeze(0).cpu().numpy().tolist()
    else:
        empty = [i for i in range(20) if board[i] == 0]
        safe = [i for i in empty if pm[i] == 1.0]
        action = random.choice(safe if safe else empty) if (safe or empty) else 0
        qv_list = [0] * 20

    return jsonify({"action": int(action), "q_values": qv_list, "prob_mask": pm.tolist()})


@app.route("/api/save_survey", methods=["POST"])
def save_survey():
    err = require_supabase()
    if err:
        return err

    d = request.json
    payload = {
        "player_name": (d.get("player_name", "Anonymous") or "Anonymous").strip(),
        "phone_number": sanitize_phone_number(d.get("phone_number", "")),
        "age": d.get("age", ""),
        "gender": d.get("gender", ""),
        "mbti": d.get("mbti", ""),
        "playtime": d.get("playtime", ""),
        "genres": d.get("genres", []),
    }
    try:
        resp = sb().table("players").insert(payload).execute()
        data = resp_data(resp)
        pid = data[0]["id"] if data else None
        return jsonify({"status": "saved", "player_id": pid})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/save_game", methods=["POST"])
def save_game():
    err = require_supabase()
    if err:
        return err

    d = request.json
    try:
        game_payload = {
            "player_id": d.get("player_id"),
            "player_name": d.get("player_name", "Anonymous"),
            "deck": d["deck"],
            "player_board": d["player_board"],
            "ai_board": d["ai_board"],
            "player_score": d["player_score"],
            "ai_score": d["ai_score"],
            "duration_ms": max(0, int(d.get("duration_ms") or 0)),
            "result": d["result"],
            "turn_sequence": [t.get("card") for t in d.get("turns", [])],
        }
        try:
            game_resp = sb().table("games").insert(game_payload).execute()
        except Exception:
            legacy_game_payload = {k: v for k, v in game_payload.items() if k not in ("turn_sequence", "duration_ms")}
            game_resp = sb().table("games").insert(legacy_game_payload).execute()
        game_rows = resp_data(game_resp)
        if not game_rows:
            return jsonify({"status": "error", "message": "Failed to create game"}), 500

        gid = game_rows[0]["id"]
        turn_rows = [
            {
                "game_id": gid,
                "turn_number": t["turn"],
                "card_value": t["card"],
                "card_order": t.get("card_order", t["turn"]),
                "deck_index": t.get("deck_index", t["turn"] - 1),
                "player_slot": t["player_slot"],
                "ai_slot": t["ai_slot"],
                "player_score_after": t.get("player_score_after", 0),
                "ai_score_after": t.get("ai_score_after", 0),
            }
            for t in d.get("turns", [])
        ]
        if turn_rows:
            try:
                sb().table("turns").insert(turn_rows).execute()
            except Exception:
                legacy_turn_rows = [
                    {k: v for k, v in row.items() if k not in ("card_order", "deck_index")}
                    for row in turn_rows
                ]
                sb().table("turns").insert(legacy_turn_rows).execute()

        return jsonify({"status": "saved", "game_id": gid})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/records")
def get_records():
    err = require_supabase()
    if err:
        return err

    try:
        page = int(request.args.get("page", 1))
        pp = int(request.args.get("per_page", 50))
        offset = (page - 1) * pp

        count_resp = sb().table("games").select("id", count="exact").range(0, 0).execute()
        total = resp_count(count_resp) or 0

        games_resp = sb().table("games").select("*").order("played_at", desc=True).range(offset, offset + pp - 1).execute()
        games = resp_data(games_resp)

        player_ids = sorted({normalize_id(g["player_id"]) for g in games if g.get("player_id") is not None})
        players_map = {}
        if player_ids:
            players_resp = sb().table("players").select("id, phone_number, age, gender, mbti, playtime, genres").in_("id", player_ids).execute()
            players_map = {normalize_id(row["id"]): row for row in resp_data(players_resp)}

        merged_games = []
        for g in games:
            p = players_map.get(normalize_id(g.get("player_id")), {})
            merged = dict(g)
            merged["age"] = p.get("age", "")
            merged["gender"] = p.get("gender", "")
            merged["mbti"] = p.get("mbti", "")
            merged["playtime"] = p.get("playtime", "")
            merged["phone_number"] = p.get("phone_number", "")
            merged["survey_genres"] = json.dumps(p.get("genres", []))
            merged["deck"] = json.dumps(merged.get("deck", []))
            merged["turn_sequence"] = json.dumps(merged.get("turn_sequence", []))
            merged["player_board"] = json.dumps(merged.get("player_board", []))
            merged["ai_board"] = json.dumps(merged.get("ai_board", []))
            merged_games.append(merged)

        all_games = fetch_all_rows("games", columns="result,player_score,ai_score", order_by="played_at", desc=True)
        stats = {}
        total_player_score = 0.0
        total_ai_score = 0.0
        for row in all_games:
            result = row.get("result", "")
            stats[result] = stats.get(result, 0) + 1
            total_player_score += float(row.get("player_score") or 0)
            total_ai_score += float(row.get("ai_score") or 0)

        avg_p = total_player_score / len(all_games) if all_games else None
        avg_a = total_ai_score / len(all_games) if all_games else None

        return jsonify(
            {
                "games": merged_games,
                "total": total,
                "page": page,
                "stats": stats,
                "averages": {"avg_p": avg_p, "avg_a": avg_a},
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/records/<int:gid>/turns")
def get_turns(gid):
    err = require_supabase()
    if err:
        return err

    try:
        turns_resp = sb().table("turns").select("*").eq("game_id", gid).order("turn_number").execute()
        turns = resp_data(turns_resp)
        return jsonify({"turns": turns})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/records/download")
def download_csv():
    err = require_supabase()
    if err:
        return err

    try:
        games = fetch_all_rows("games", columns="*")
        player_ids = sorted({normalize_id(g["player_id"]) for g in games if g.get("player_id") is not None})
        players_map = {}
        if player_ids:
            players_resp = sb().table("players").select("id, phone_number, age, gender, mbti, playtime, genres").in_("id", player_ids).execute()
            players_map = {normalize_id(row["id"]): row for row in resp_data(players_resp)}

        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(
            [
                "ID",
                "Player",
                "Phone",
                "Age",
                "Gender",
                "MBTI",
                "Playtime",
                "Genres",
                "Played At",
                "Duration (ms)",
                "Deck",
                "Turn Sequence",
                "Player Board",
                "AI Board",
                "Player Score",
                "AI Score",
                "Result",
            ]
        )
        for g in games:
            p = players_map.get(normalize_id(g.get("player_id")), {})
            w.writerow(
                [
                    g.get("id"),
                    g.get("player_name", "Anonymous"),
                    p.get("phone_number", ""),
                    p.get("age", ""),
                    p.get("gender", ""),
                    p.get("mbti", ""),
                    p.get("playtime", ""),
                    json.dumps(p.get("genres", [])),
                    g.get("played_at"),
                    g.get("duration_ms", 0),
                    json.dumps(g.get("deck", [])),
                    json.dumps(g.get("turn_sequence", [])),
                    json.dumps(g.get("player_board", [])),
                    json.dumps(g.get("ai_board", [])),
                    g.get("player_score"),
                    g.get("ai_score"),
                    g.get("result"),
                ]
            )

        out.seek(0)
        csv_text = "\ufeff" + out.getvalue()
        return Response(
            csv_text,
            mimetype="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename=streams_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            },
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/survey_stats")
def survey_stats():
    err = require_supabase()
    if err:
        return err

    try:
        games = fetch_all_rows("games", columns="player_id,player_score,ai_score,result")
        player_ids = sorted({normalize_id(g["player_id"]) for g in games if g.get("player_id") is not None})
        players_map = {}
        if player_ids:
            players_resp = sb().table("players").select("id, age, gender, mbti, playtime").in_("id", player_ids).execute()
            players_map = {normalize_id(row["id"]): row for row in resp_data(players_resp)}

        def bucket_stats(key_name):
            grouped = {}
            for g in games:
                pid = normalize_id(g.get("player_id"))
                p = players_map.get(pid)
                if not p:
                    continue
                key = p.get(key_name, "")
                if not key:
                    continue
                item = grouped.setdefault(key, {"sum": 0.0, "cnt": 0, "wins": 0})
                item["sum"] += float(g.get("player_score") or 0)
                item["cnt"] += 1
                if g.get("result") == "win":
                    item["wins"] += 1
            rows = []
            for key, item in grouped.items():
                rows.append(
                    {
                        key_name: key,
                        "avg_score": item["sum"] / item["cnt"] if item["cnt"] else 0,
                        "cnt": item["cnt"],
                        "wins": item["wins"],
                    }
                )
            rows.sort(key=lambda x: x["avg_score"], reverse=True)
            return rows

        return jsonify(
            {
                "mbti": bucket_stats("mbti"),
                "age": bucket_stats("age"),
                "playtime": bucket_stats("playtime"),
                "gender": bucket_stats("gender"),
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/leaderboard")
def leaderboard():
    err = require_supabase()
    if err:
        return err

    try:
        target_player_id = request.args.get("player_id")
        player_rows = fetch_all_rows("players", columns="id,player_name,phone_number")
        game_rows = fetch_all_rows("games", columns="player_id,player_name,player_score,duration_ms,played_at")
        return jsonify(build_leaderboards(player_rows, game_rows, target_player_id=target_player_id))
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Streams PVE v3 - Survey + Game Recording")
    print("  http://localhost:5000       Game")
    print("  http://localhost:5000/admin Admin")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
