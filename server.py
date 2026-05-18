"""
Streams PVE game server.

This version uses Supabase for persistent storage.
"""

import csv
import io
import json
import os
import random
import base64
import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta, timezone
from collections import Counter
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
app.json.sort_keys = False

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
TOKEN_MAX_AGE_SECONDS = 6 * 60 * 60
LEGACY_AI_MODEL_LABEL = "legacy-dqn"
CURRENT_AI_MODEL_LABEL = "v16-double-dueling-dqn"
FALLBACK_AI_MODEL_LABEL = "fallback-rules"
STREAMS_TOKEN_SECRET = (
    os.environ.get("STREAMS_TOKEN_SECRET")
    or os.environ.get("VERCEL_OIDC_TOKEN")
    or SUPABASE_SERVICE_ROLE_KEY
    or "streams-dev-secret"
).encode("utf-8")


def sb():
    if not SUPABASE_AVAILABLE or SUPABASE_CLIENT is None:
        raise RuntimeError("Supabase client is not configured")
    return SUPABASE_CLIENT


def _urlsafe_b64encode(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _urlsafe_b64decode(text):
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def issue_signed_token(payload):
    body = _urlsafe_b64encode(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(STREAMS_TOKEN_SECRET, body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_urlsafe_b64encode(signature)}"


def verify_signed_token(token, purpose, max_age_seconds=TOKEN_MAX_AGE_SECONDS):
    if not token or "." not in str(token):
        return None
    body, signature = str(token).split(".", 1)
    expected = _urlsafe_b64encode(hmac.new(STREAMS_TOKEN_SECRET, body.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(_urlsafe_b64decode(body).decode("utf-8"))
    except Exception:
        return None
    if payload.get("purpose") != purpose:
        return None
    issued_at = int(payload.get("iat") or 0)
    if issued_at <= 0 or (int(time.time()) - issued_at) > max_age_seconds:
        return None
    return payload


def issue_player_token(player_id, player_name):
    return issue_signed_token(
        {
            "purpose": "player",
            "player_id": normalize_id(player_id),
            "player_name": str(player_name or "Anonymous"),
            "iat": int(time.time()),
        }
    )


def issue_game_token(player_id, deck):
    return issue_signed_token(
        {
            "purpose": "game",
            "player_id": normalize_id(player_id),
            "deck": deck,
            "nonce": secrets.token_hex(8),
            "iat": int(time.time()),
        }
    )


def build_number_pool():
    pool = list(range(1, 11))
    for value in range(11, 21):
        pool.extend([value, value])
    pool.extend(range(21, 31))
    return pool


def draw_server_deck():
    return random.sample(build_number_pool(), 20)


def validate_int_list(values, expected_length):
    if not isinstance(values, list) or len(values) != expected_length:
        return None
    normalized = []
    for value in values:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError):
            return None
    return normalized


def validate_deck(deck):
    normalized = validate_int_list(deck, 20)
    if normalized is None:
        return None
    allowed = Counter(build_number_pool())
    counts = Counter(normalized)
    for value, count in counts.items():
        if value < 1 or value > 30 or count > allowed.get(value, 0):
            return None
    return normalized


def calc_streams_score(board):
    values = [value for value in board if value]
    if not values:
        return 0
    streak_points = {
        1: 0,
        2: 1,
        3: 3,
        4: 5,
        5: 7,
        6: 9,
        7: 11,
        8: 15,
        9: 20,
        10: 25,
        11: 30,
        12: 35,
        13: 40,
        14: 50,
        15: 60,
        16: 70,
        17: 85,
        18: 100,
        19: 150,
        20: 300,
    }
    score = 0
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[end] <= values[end + 1]:
            end += 1
        score += streak_points.get(end - index + 1, 0)
        index = end + 1
    return score


def validate_turn_rows(turns, deck):
    if not isinstance(turns, list) or len(turns) != len(deck):
        return None, "Turn log is incomplete"
    player_slots_seen = set()
    ai_slots_seen = set()
    player_board = [0] * len(deck)
    ai_board = [0] * len(deck)

    for expected_index, turn in enumerate(turns):
        try:
            turn_number = int(turn.get("turn"))
            card_value = int(turn.get("card"))
            card_order = int(turn.get("card_order", turn_number))
            deck_index = int(turn.get("deck_index", turn_number - 1))
            player_slot = int(turn.get("player_slot"))
            ai_slot = int(turn.get("ai_slot"))
        except (TypeError, ValueError, AttributeError):
            return None, "Turn log contains invalid values"

        if turn_number != expected_index + 1 or card_order != expected_index + 1 or deck_index != expected_index:
            return None, "Turn log order is invalid"
        if card_value != deck[expected_index]:
            return None, "Turn log does not match the issued deck"
        if not 0 <= player_slot < len(deck) or not 0 <= ai_slot < len(deck):
            return None, "Turn slots are out of range"
        if player_slot in player_slots_seen or ai_slot in ai_slots_seen:
            return None, "Turn slots are duplicated"

        player_slots_seen.add(player_slot)
        ai_slots_seen.add(ai_slot)
        player_board[player_slot] = card_value
        ai_board[ai_slot] = card_value

    return {"player_board": player_board, "ai_board": ai_board}, None


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


def leaderboard_identity(player_row):
    pid = normalize_id(player_row.get("id"))
    phone_number = sanitize_phone_number(player_row.get("phone_number", ""))
    if phone_number:
        return {
            "identity_key": f"phone:{phone_number}",
            "phone_number": phone_number,
            "identity_suffix": phone_number,
            "identity_kind": "phone",
        }
    if pid is None:
        return None
    player_suffix = str(pid).zfill(4)[-4:]
    return {
        "identity_key": f"player:{pid}",
        "phone_number": "",
        "identity_suffix": player_suffix,
        "identity_kind": "player",
    }


def leaderboard_display_name(name, duplicate_names, duplicate_index, identity):
    if identity["identity_kind"] == "phone":
        return masked_phone_suffix(name, identity["phone_number"], duplicate_names)

    if duplicate_names.get(name, 0) > 1:
        prefix, dash, suffix = name.rpartition("-")
        if dash and suffix.isdigit():
            return f"{prefix}-{duplicate_index}"
        return f"{name}-{duplicate_index}"
    return name


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
    target_identity_key = ""
    if target_player_id is not None:
        target_player = players.get(normalize_id(target_player_id))
        if target_player:
            target_identity = leaderboard_identity(target_player)
            if target_identity:
                target_identity_key = target_identity["identity_key"]

    now_kst = datetime.now(KST)
    start_of_today_kst = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_week_kst = start_of_today_kst - timedelta(days=start_of_today_kst.weekday())
    cutoffs = {
        "daily": start_of_today_kst,
        "weekly": start_of_week_kst,
        "overall": None,
    }
    best_by_period = {period: {} for period in cutoffs}

    for game in game_rows:
        pid = normalize_id(game.get("player_id"))
        player = players.get(pid)
        if not player:
            continue

        identity = leaderboard_identity(player)
        if not identity:
            continue

        played_at = parse_played_at(game.get("played_at"))
        if not played_at:
            continue
        played_at_kst = played_at.astimezone(KST)

        candidate = {
            "player_id": pid,
            "player_name": player.get("player_name") or game.get("player_name") or "Anonymous",
            "phone_number": identity["phone_number"],
            "identity_key": identity["identity_key"],
            "identity_suffix": identity["identity_suffix"],
            "identity_kind": identity["identity_kind"],
            "player_score": int(game.get("player_score") or 0),
            "duration_ms": max(0, int(game.get("duration_ms") or 0)),
            "played_at": played_at.isoformat(),
            "played_at_label": played_at_kst.strftime("%m/%d %H:%M"),
            "played_weekday": played_at_kst.weekday(),
            "played_day_label": played_at_kst.strftime("%m/%d"),
        }

        for period, cutoff in cutoffs.items():
            if cutoff and played_at_kst < cutoff:
                continue
            current = best_by_period[period].get(identity["identity_key"])
            if current is None or compare_rank_entries(candidate, current) < 0:
                best_by_period[period][identity["identity_key"]] = dict(candidate)

    periods = {}
    player_summary = {}

    for period, entries_by_identity in best_by_period.items():
        entries = list(entries_by_identity.values())
        name_counts = {}
        for entry in entries:
            name_counts[entry["player_name"]] = name_counts.get(entry["player_name"], 0) + 1

        ranked = sorted(entries, key=cmp_to_key(compare_rank_entries))
        name_seen = {}
        for entry in ranked:
            name = entry["player_name"]
            name_seen[name] = name_seen.get(name, 0) + 1
            entry["display_name"] = leaderboard_display_name(name, name_counts, name_seen[name], entry)
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
                    "played_weekday": entry["played_weekday"],
                    "played_day_label": entry["played_day_label"],
                }
            )

        periods[period] = {
            "label": PERIOD_LABELS[period],
            "rows": rows,
        }

        if target_identity_key:
            rank = next((idx for idx, entry in enumerate(ranked, start=1) if entry["identity_key"] == target_identity_key), None)
            current = entries_by_identity.get(target_identity_key)
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
        "player": player_summary if target_identity_key else None,
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


def coerce_game_mode(value):
    return "tutorial" if str(value or "").strip().lower() == "tutorial" else "main"


def normalize_game_mode_filter(value):
    text = str(value or "").strip().lower()
    return text if text in {"main", "tutorial"} else ""


def fetch_players_map(player_ids):
    player_ids = sorted({normalize_id(pid) for pid in player_ids if pid is not None})
    if not player_ids:
        return {}
    players_resp = (
        sb()
        .table("players")
        .select("id, phone_number, age, gender, mbti, playtime, genres")
        .in_("id", player_ids)
        .execute()
    )
    return {normalize_id(row["id"]): row for row in resp_data(players_resp)}


def fetch_turns_map(game_ids):
    game_ids = sorted({normalize_id(gid) for gid in game_ids if gid is not None})
    if not game_ids:
        return {}
    turns_resp = sb().table("turns").select("*").in_("game_id", game_ids).execute()
    grouped = {}
    ordered_turns = sorted(
        resp_data(turns_resp),
        key=lambda row: (
            normalize_id(row.get("game_id")) or 0,
            int(row.get("turn_number") or 0),
            int(row.get("deck_index") or 0),
        ),
    )
    for row in ordered_turns:
        grouped.setdefault(normalize_id(row.get("game_id")), []).append(row)
    return grouped


def turn_rows_to_export_payload(turn_rows):
    ordered_rows = sorted(
        turn_rows,
        key=lambda row: (int(row.get("turn_number") or 0), int(row.get("deck_index") or 0)),
    )
    detail_rows = [
        {
            "turn": int(row.get("turn_number") or 0),
            "card": int(row.get("card_value") or 0),
            "card_order": int(row.get("card_order") or row.get("turn_number") or 0),
            "deck_index": int(row.get("deck_index") or 0),
            "player_slot": int(row.get("player_slot") or 0),
            "ai_slot": int(row.get("ai_slot") or 0),
            "player_score_after": int(row.get("player_score_after") or 0),
            "ai_score_after": int(row.get("ai_score_after") or 0),
        }
        for row in ordered_rows
    ]
    return {
        "turn_count": len(detail_rows),
        "turn_cards": [row["card"] for row in detail_rows],
        "turn_player_slots": [row["player_slot"] for row in detail_rows],
        "turn_ai_slots": [row["ai_slot"] for row in detail_rows],
        "turn_player_scores": [row["player_score_after"] for row in detail_rows],
        "turn_ai_scores": [row["ai_score_after"] for row in detail_rows],
        "turn_detail_rows": detail_rows,
        "turn_detail": json.dumps(detail_rows, ensure_ascii=False),
    }


def enrich_game_row(game_row, player_row=None, turn_rows=None):
    merged = dict(game_row)
    player_row = player_row or {}
    merged["game_mode"] = coerce_game_mode(game_row.get("game_mode"))
    merged["ai_model_label"] = str(game_row.get("ai_model_label") or LEGACY_AI_MODEL_LABEL)
    merged["age"] = player_row.get("age", "")
    merged["gender"] = player_row.get("gender", "")
    merged["mbti"] = player_row.get("mbti", "")
    merged["playtime"] = player_row.get("playtime", "")
    merged["phone_number"] = player_row.get("phone_number", "")
    genres = player_row.get("genres", [])
    if not isinstance(genres, list):
        genres = []
    merged["survey_genres_list"] = genres
    merged["survey_genres"] = json.dumps(genres, ensure_ascii=False)
    merged["deck"] = json.dumps(game_row.get("deck", []), ensure_ascii=False)
    merged["turn_sequence"] = json.dumps(game_row.get("turn_sequence", []), ensure_ascii=False)
    merged["player_board"] = json.dumps(game_row.get("player_board", []), ensure_ascii=False)
    merged["ai_board"] = json.dumps(game_row.get("ai_board", []), ensure_ascii=False)
    turn_payload = turn_rows_to_export_payload(turn_rows or [])
    merged.update(turn_payload)
    return merged


def build_record_search_text(record):
    searchable = [
        record.get("id"),
        record.get("player_name"),
        record.get("phone_number"),
        record.get("age"),
        record.get("gender"),
        record.get("mbti"),
        record.get("playtime"),
        record.get("result"),
        record.get("game_mode"),
        record.get("ai_model_label"),
        " ".join(record.get("survey_genres_list", [])),
    ]
    return " ".join(str(item or "") for item in searchable).lower()


def filter_record_rows(records, search_text="", result_filter="", game_mode_filter=""):
    search_text = str(search_text or "").strip().lower()
    result_filter = str(result_filter or "").strip().lower()
    game_mode_filter = normalize_game_mode_filter(game_mode_filter)
    filtered = []
    for record in records:
        if result_filter and str(record.get("result") or "").lower() != result_filter:
            continue
        if game_mode_filter and coerce_game_mode(record.get("game_mode")) != game_mode_filter:
            continue
        if search_text and search_text not in build_record_search_text(record):
            continue
        filtered.append(record)
    return filtered


def summarize_record_rows(records):
    stats = {}
    total_player_score = 0.0
    total_ai_score = 0.0
    for row in records:
        result = row.get("result", "")
        stats[result] = stats.get(result, 0) + 1
        total_player_score += float(row.get("player_score") or 0)
        total_ai_score += float(row.get("ai_score") or 0)
    count = len(records)
    return {
        "stats": stats,
        "averages": {
            "avg_p": total_player_score / count if count else None,
            "avg_a": total_ai_score / count if count else None,
        },
    }


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
            tail_feature_size = max(0, int(state_size) - 40)
            self.feature = nn.Sequential(
                nn.Linear(64 * 20 + tail_feature_size, 512),
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
MODEL_STATE_SIZE = 96
RESERVED_STATE_FEATURES = 5


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
    reserved = np.zeros(RESERVED_STATE_FEATURES, dtype=np.float32)
    return np.concatenate(
        [
            np.array(bn),
            np.array(bo),
            np.array([num / 30.0]),
            np.array(rv),
            lm,
            reserved,
        ]
    )


model = None
loaded = False
active_ai_model_label = FALLBACK_AI_MODEL_LABEL
if TORCH_AVAILABLE:
    model = DuelingQNetwork(MODEL_STATE_SIZE, 20).to(DEVICE)
    for name in ["best_model.pth", "final_model.pth"]:
        candidate_paths = [
            os.path.join(BASE_DIR, name),
            os.path.join(BASE_DIR, "api", name),
        ]
        for model_path in candidate_paths:
            if not os.path.exists(model_path):
                continue
            try:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()
                loaded = True
                active_ai_model_label = CURRENT_AI_MODEL_LABEL
                print(f"[Model] {name} @ {model_path}")
                break
            except Exception as exc:
                print(f"[WARNING] Failed to load {model_path}: {exc}")
        if loaded:
            break
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
            "ai_model_label": active_ai_model_label,
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
        return jsonify({"status": "saved", "player_id": pid, "player_token": issue_player_token(pid, payload["player_name"])})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/start_game", methods=["POST"])
def start_game_session():
    d = request.json or {}
    player_id = normalize_id(d.get("player_id"))
    player_token = verify_signed_token(d.get("player_token"), "player")
    if not player_token or player_token.get("player_id") != player_id:
        return jsonify({"status": "error", "message": "Invalid player session"}), 403

    deck = draw_server_deck()
    return jsonify({"status": "ok", "deck": deck, "game_token": issue_game_token(player_id, deck)})


@app.route("/api/save_game", methods=["POST"])
def save_game():
    err = require_supabase()
    if err:
        return err

    d = request.json or {}
    try:
        player_id = normalize_id(d.get("player_id"))
        player_token = verify_signed_token(d.get("player_token"), "player")
        game_token = verify_signed_token(d.get("game_token"), "game")
        if not player_token or player_token.get("player_id") != player_id:
            return jsonify({"status": "error", "message": "Invalid player token"}), 403
        if not game_token or game_token.get("player_id") != player_id:
            return jsonify({"status": "error", "message": "Invalid game token"}), 403

        deck = validate_deck(d.get("deck"))
        if deck is None:
            return jsonify({"status": "error", "message": "Invalid deck"}), 400
        token_deck = validate_deck(game_token.get("deck"))
        if token_deck is None or deck != token_deck:
            return jsonify({"status": "error", "message": "Deck mismatch"}), 400

        validated_turns, turn_error = validate_turn_rows(d.get("turns", []), deck)
        if turn_error:
            return jsonify({"status": "error", "message": turn_error}), 400

        player_board = validate_int_list(d.get("player_board"), 20)
        ai_board = validate_int_list(d.get("ai_board"), 20)
        if player_board is None or ai_board is None:
            return jsonify({"status": "error", "message": "Invalid board"}), 400
        if player_board != validated_turns["player_board"] or ai_board != validated_turns["ai_board"]:
            return jsonify({"status": "error", "message": "Submitted board does not match turn log"}), 400

        if Counter(player_board) != Counter(deck) or Counter(ai_board) != Counter(deck):
            return jsonify({"status": "error", "message": "Board contents do not match deck"}), 400

        computed_player_score = calc_streams_score(player_board)
        computed_ai_score = calc_streams_score(ai_board)
        computed_result = "draw"
        if computed_player_score > computed_ai_score:
            computed_result = "win"
        elif computed_player_score < computed_ai_score:
            computed_result = "lose"

        game_payload = {
            "player_id": player_id,
            "player_name": d.get("player_name", "Anonymous"),
            "game_mode": coerce_game_mode(d.get("game_mode")),
            "ai_model_label": active_ai_model_label,
            "deck": deck,
            "player_board": player_board,
            "ai_board": ai_board,
            "player_score": computed_player_score,
            "ai_score": computed_ai_score,
            "duration_ms": max(0, int(d.get("duration_ms") or 0)),
            "result": computed_result,
            "turn_sequence": deck,
        }
        try:
            game_resp = sb().table("games").insert(game_payload).execute()
        except Exception:
            legacy_game_payload = {
                k: v
                for k, v in game_payload.items()
                if k not in ("turn_sequence", "duration_ms", "game_mode", "ai_model_label")
            }
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
        page = max(1, int(request.args.get("page", 1)))
        pp = max(1, min(200, int(request.args.get("per_page", 50))))
        offset = (page - 1) * pp
        search_text = request.args.get("q", "")
        result_filter = request.args.get("result", "")
        game_mode_filter = request.args.get("game_mode", "")

        games = fetch_all_rows("games", columns="*", order_by="played_at", desc=True)
        players_map = fetch_players_map([g.get("player_id") for g in games])
        merged_games = [enrich_game_row(g, players_map.get(normalize_id(g.get("player_id")), {})) for g in games]
        filtered_games = filter_record_rows(
            merged_games,
            search_text=search_text,
            result_filter=result_filter,
            game_mode_filter=game_mode_filter,
        )
        total = len(filtered_games)
        summary = summarize_record_rows(filtered_games)
        page_games = filtered_games[offset : offset + pp]

        return jsonify(
            {
                "games": page_games,
                "total": total,
                "page": page,
                "per_page": pp,
                "stats": summary["stats"],
                "averages": summary["averages"],
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/records/<int:gid>")
def get_record(gid):
    err = require_supabase()
    if err:
        return err

    try:
        game_resp = sb().table("games").select("*").eq("id", gid).limit(1).execute()
        game_rows = resp_data(game_resp)
        if not game_rows:
            return jsonify({"status": "error", "message": "Record not found"}), 404

        game = game_rows[0]
        player_id = normalize_id(game.get("player_id"))
        players_map = fetch_players_map([player_id])
        return jsonify({"game": enrich_game_row(game, players_map.get(player_id, {}))})
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
        games = fetch_all_rows("games", columns="*", order_by="played_at", desc=True)
        players_map = fetch_players_map([g.get("player_id") for g in games])
        turns_map = fetch_turns_map([g.get("id") for g in games])

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
                "Game Mode",
                "AI Model Label",
                "Duration (ms)",
                "Deck",
                "Turn Sequence",
                "Turn Count",
                "Turn Cards",
                "Turn Player Slots",
                "Turn AI Slots",
                "Turn Player Scores",
                "Turn AI Scores",
                "Turn Detail",
                "Player Board",
                "AI Board",
                "Player Score",
                "AI Score",
                "Result",
            ]
        )
        for g in games:
            merged = enrich_game_row(
                g,
                players_map.get(normalize_id(g.get("player_id")), {}),
                turns_map.get(normalize_id(g.get("id")), []),
            )
            w.writerow(
                [
                    merged.get("id"),
                    merged.get("player_name", "Anonymous"),
                    merged.get("phone_number", ""),
                    merged.get("age", ""),
                    merged.get("gender", ""),
                    merged.get("mbti", ""),
                    merged.get("playtime", ""),
                    merged.get("survey_genres", "[]"),
                    merged.get("played_at"),
                    merged.get("game_mode", "main"),
                    merged.get("ai_model_label", LEGACY_AI_MODEL_LABEL),
                    merged.get("duration_ms", 0),
                    merged.get("deck", "[]"),
                    merged.get("turn_sequence", "[]"),
                    merged.get("turn_count", 0),
                    json.dumps(merged.get("turn_cards", []), ensure_ascii=False),
                    json.dumps(merged.get("turn_player_slots", []), ensure_ascii=False),
                    json.dumps(merged.get("turn_ai_slots", []), ensure_ascii=False),
                    json.dumps(merged.get("turn_player_scores", []), ensure_ascii=False),
                    json.dumps(merged.get("turn_ai_scores", []), ensure_ascii=False),
                    merged.get("turn_detail", "[]"),
                    merged.get("player_board", "[]"),
                    merged.get("ai_board", "[]"),
                    merged.get("player_score"),
                    merged.get("ai_score"),
                    merged.get("result"),
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
        games = [
            row
            for row in fetch_all_rows("games", columns="*", order_by="played_at", desc=True)
            if coerce_game_mode(row.get("game_mode")) == "main"
        ]
        players_map = fetch_players_map([g.get("player_id") for g in games])

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
        game_rows = [
            row
            for row in fetch_all_rows("games", columns="*", order_by="played_at", desc=True)
            if coerce_game_mode(row.get("game_mode")) == "main"
        ]
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
