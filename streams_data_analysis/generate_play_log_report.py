from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from dotenv import load_dotenv
from supabase import create_client


SCORE_TABLE = {
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

CENTER_LOW = 0.35
CENTER_HIGH = 0.65


@dataclass
class ValidationSummary:
    total_games: int = 0
    analyzed_games: int = 0
    excluded_games: int = 0
    missing_turn_games: int = 0
    invalid_turn_count_games: int = 0
    invalid_turn_order_games: int = 0
    board_mismatch_games: int = 0
    player_missing_games: int = 0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_environment(root: Path) -> None:
    load_dotenv(root / ".env.vercel", override=False)
    load_dotenv(root / ".env", override=False)


def env_value(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return ""


def is_placeholder_value(value: str) -> bool:
    lowered = value.lower()
    return "your-project" in lowered or "your_" in lowered or "example" in lowered


def get_supabase_client():
    candidate_urls = [
        os.getenv("game_SUPABASE_URL", ""),
        os.getenv("SUPABASE_URL", ""),
    ]
    candidate_keys = [
        os.getenv("game_SUPABASE_SERVICE_ROLE_KEY", ""),
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    ]
    url = next((value for value in candidate_urls if value and not is_placeholder_value(value)), "") or env_value(
        "game_SUPABASE_URL", "SUPABASE_URL"
    )
    service_key = next((value for value in candidate_keys if value and not is_placeholder_value(value)), "") or env_value(
        "game_SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SERVICE_ROLE_KEY"
    )
    if not url or not service_key:
        raise RuntimeError("Supabase credentials were not found in the environment.")
    return create_client(url, service_key)


def fetch_all_rows(client, table: str, columns: str = "*", order_by: str = "id", page_size: int = 1000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        query = client.table(table).select(columns).order(order_by).range(start, start + page_size - 1)
        response = query.execute()
        data = response.data or []
        rows.extend(data)
        if len(data) < page_size:
            break
        start += page_size
    return rows


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_json_field(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return fallback
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return fallback
    return fallback


def parse_played_at(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_text(value: Any, default: str = "미기재") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def mean_or_zero(values: list[float | int]) -> float:
    return round(mean(values), 3) if values else 0.0


def round_or_zero(value: float) -> float:
    return round(value, 3) if value else 0.0


def calc_streams_score(board: list[int]) -> int:
    filtered = [value for value in board if value]
    if not filtered:
        return 0
    score = 0
    start = 0
    while start < len(filtered):
        end = start
        while end + 1 < len(filtered) and filtered[end] <= filtered[end + 1]:
            end += 1
        score += SCORE_TABLE.get(end - start + 1, 0)
        start = end + 1
    return score


def monotonic_segments(board: list[int]) -> list[dict[str, Any]]:
    filtered = [value for value in board if value]
    if not filtered:
        return []
    segments: list[dict[str, Any]] = []
    start = 0
    while start < len(filtered):
        end = start
        while end + 1 < len(filtered) and filtered[end] <= filtered[end + 1]:
            end += 1
        values = filtered[start : end + 1]
        segments.append(
            {
                "start": start,
                "end": end,
                "length": len(values),
                "values": values,
                "score": SCORE_TABLE.get(len(values), 0),
            }
        )
        start = end + 1
    return segments


def longest_segment_slots(board: list[int]) -> set[int]:
    segments: list[tuple[int, int]] = []
    start = 0
    while start < len(board):
        end = start
        while end + 1 < len(board) and board[end] <= board[end + 1]:
            end += 1
        segments.append((start, end))
        start = end + 1
    if not segments:
        return set()
    longest = max(end - start + 1 for start, end in segments)
    slots: set[int] = set()
    for start, end in segments:
        if end - start + 1 == longest:
            slots.update(range(start, end + 1))
    return slots


def find_left_anchor(board_state: list[int], slot: int) -> int | None:
    for idx in range(slot - 1, -1, -1):
        if board_state[idx]:
            return board_state[idx]
    return None


def find_right_anchor(board_state: list[int], slot: int) -> int | None:
    for idx in range(slot + 1, len(board_state)):
        if board_state[idx]:
            return board_state[idx]
    return None


def phase_for_turn(turn_number: int) -> str:
    if turn_number <= 6:
        return "early"
    if turn_number <= 14:
        return "mid"
    return "late"


def card_band(card_value: int) -> str:
    if card_value <= 10:
        return "low"
    if card_value <= 20:
        return "mid"
    return "high"


def placement_type_for_turn(card_value: int, left_anchor: int | None, right_anchor: int | None) -> str:
    if left_anchor is None and right_anchor is None:
        return "middle_insert"
    if left_anchor is None and right_anchor is not None:
        return "left_end_extension" if card_value <= right_anchor else "risky_insert"
    if right_anchor is None and left_anchor is not None:
        return "right_end_extension" if left_anchor <= card_value else "risky_insert"
    if left_anchor is not None and right_anchor is not None and left_anchor <= card_value <= right_anchor:
        return "middle_insert"
    return "risky_insert"


def adjacent_gap_profile(deck: list[int]) -> dict[str, Any]:
    if len(deck) < 2:
        return {"mean_abs_gap": 0.0, "median_abs_gap": 0.0, "max_abs_gap": 0, "gt10_rate": 0.0}
    gaps = [abs(deck[idx + 1] - deck[idx]) for idx in range(len(deck) - 1)]
    return {
        "mean_abs_gap": round(mean(gaps), 3),
        "median_abs_gap": round(median(gaps), 3),
        "max_abs_gap": max(gaps),
        "gt10_rate": round(sum(1 for gap in gaps if gap > 10) / len(gaps), 3),
    }


def normalize_board(board: Any) -> list[int]:
    raw = parse_json_field(board, [])
    return [safe_int(value, 0) for value in raw]


def validate_turn_sequence(turns: list[dict[str, Any]]) -> set[str]:
    errors: set[str] = set()
    if len(turns) != 20:
        errors.add("turn_count_mismatch")
    expected_turns = list(range(1, len(turns) + 1))
    actual_turns = [safe_int(turn.get("turn_number")) for turn in turns]
    if actual_turns != expected_turns:
        errors.add("turn_order_invalid")
    actual_deck_indexes = [safe_int(turn.get("deck_index")) for turn in turns]
    if actual_deck_indexes != list(range(len(turns))):
        errors.add("deck_index_invalid")
    return errors


def derive_turn_metrics(game: dict[str, Any], player: dict[str, Any], turns: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    errors = validate_turn_sequence(turns)
    deck = normalize_board(game.get("deck"))
    final_player_board = normalize_board(game.get("player_board"))
    final_ai_board = normalize_board(game.get("ai_board"))

    current_player_board = [0] * len(final_player_board)
    current_ai_board = [0] * len(final_ai_board)
    seen_cards: Counter[int] = Counter()
    final_longest_slots = longest_segment_slots(final_player_board)
    derived: list[dict[str, Any]] = []

    previous_player_score = 0
    previous_ai_score = 0

    for turn in turns:
        turn_number = safe_int(turn.get("turn_number"))
        deck_index = safe_int(turn.get("deck_index"))
        card_value = safe_int(turn.get("card_value"))
        player_slot = safe_int(turn.get("player_slot"))
        ai_slot = safe_int(turn.get("ai_slot"))
        player_score_after = safe_int(turn.get("player_score_after"))
        ai_score_after = safe_int(turn.get("ai_score_after"))

        if deck_index < 0 or deck_index >= len(deck) or deck[deck_index] != card_value:
            errors.add("deck_card_mismatch")

        left_anchor = find_left_anchor(current_player_board, player_slot)
        right_anchor = find_right_anchor(current_player_board, player_slot)
        empty_slots_before_turn = sum(1 for value in current_player_board if value == 0)
        local_gap_width = (right_anchor - left_anchor) if left_anchor is not None and right_anchor is not None else None
        placement_type = placement_type_for_turn(card_value, left_anchor, right_anchor)
        duplicate_card_turn = seen_cards[card_value] > 0

        current_player_board[player_slot] = card_value
        current_ai_board[ai_slot] = card_value
        seen_cards[card_value] += 1

        player_score_delta = player_score_after - previous_player_score
        ai_score_delta = ai_score_after - previous_ai_score
        previous_player_score = player_score_after
        previous_ai_score = ai_score_after

        derived.append(
            {
                "game_id": safe_int(game.get("id")),
                "player_id": safe_int(game.get("player_id")),
                "player_name": clean_text(game.get("player_name")),
                "age": clean_text(player.get("age")),
                "playtime": clean_text(player.get("playtime")),
                "turn_number": turn_number,
                "card_value": card_value,
                "card_order": safe_int(turn.get("card_order"), turn_number),
                "deck_index": deck_index,
                "player_slot": player_slot,
                "ai_slot": ai_slot,
                "player_score_after": player_score_after,
                "ai_score_after": ai_score_after,
                "player_score_delta": player_score_delta,
                "ai_score_delta": ai_score_delta,
                "score_advantage_delta": player_score_delta - ai_score_delta,
                "slot_position_norm": round(player_slot / 19, 3) if len(final_player_board) > 1 else 0.0,
                "phase": phase_for_turn(turn_number),
                "empty_slots_before_turn": empty_slots_before_turn,
                "left_anchor": left_anchor,
                "right_anchor": right_anchor,
                "local_gap_width": local_gap_width if local_gap_width is not None else "",
                "placement_type": placement_type,
                "duplicate_card_turn": int(duplicate_card_turn),
                "future_segment_gain_proxy": int(player_slot in final_longest_slots),
                "slot_distance_to_ai": abs(player_slot - ai_slot),
                "center_slot_flag": int(CENTER_LOW <= (player_slot / 19) <= CENTER_HIGH),
                "card_band": card_band(card_value),
            }
        )

    if current_player_board != final_player_board:
        errors.add("board_mismatch")
    if current_ai_board != final_ai_board:
        errors.add("ai_board_mismatch")
    if previous_player_score != safe_int(game.get("player_score")):
        errors.add("player_score_mismatch")
    if previous_ai_score != safe_int(game.get("ai_score")):
        errors.add("ai_score_mismatch")

    return derived, errors


def derive_game_metrics(game: dict[str, Any], player: dict[str, Any], turn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    deck = normalize_board(game.get("deck"))
    player_board = normalize_board(game.get("player_board"))
    ai_board = normalize_board(game.get("ai_board"))
    segments = monotonic_segments(player_board)
    gap_profile = adjacent_gap_profile(deck)
    duplicate_count = sum(count - 1 for count in Counter(deck).values() if count > 1)
    late_game_efficiency = sum(row["player_score_delta"] for row in turn_rows if row["phase"] == "late")

    return {
        "game_id": safe_int(game.get("id")),
        "player_id": safe_int(game.get("player_id")),
        "player_name": clean_text(game.get("player_name")),
        "age": clean_text(player.get("age")),
        "playtime": clean_text(player.get("playtime")),
        "played_at": game.get("played_at"),
        "result": clean_text(game.get("result"), "unknown"),
        "player_score": safe_int(game.get("player_score")),
        "ai_score": safe_int(game.get("ai_score")),
        "score_gap": safe_int(game.get("player_score")) - safe_int(game.get("ai_score")),
        "duration_ms": safe_int(game.get("duration_ms")),
        "duration_sec": round(safe_int(game.get("duration_ms")) / 1000, 3),
        "win_flag": int(clean_text(game.get("result"), "unknown") == "win"),
        "draw_flag": int(clean_text(game.get("result"), "unknown") == "draw"),
        "deck": json.dumps(deck, ensure_ascii=False),
        "player_board": json.dumps(player_board, ensure_ascii=False),
        "ai_board": json.dumps(ai_board, ensure_ascii=False),
        "board_monotonic_segments": len(segments),
        "board_longest_segment": max((segment["length"] for segment in segments), default=0),
        "board_avg_segment_length": round(mean([segment["length"] for segment in segments]), 3) if segments else 0.0,
        "late_game_efficiency": late_game_efficiency,
        "duplicate_count": duplicate_count,
        "spread": max(deck) - min(deck) if deck else 0,
        "deck_variance": round(mean([(value - mean(deck)) ** 2 for value in deck]), 3) if deck else 0.0,
        "mean_abs_gap": gap_profile["mean_abs_gap"],
        "median_abs_gap": gap_profile["median_abs_gap"],
        "max_abs_gap": gap_profile["max_abs_gap"],
        "gt10_rate": gap_profile["gt10_rate"],
        "avg_slot_distance_to_ai": mean_or_zero([row["slot_distance_to_ai"] for row in turn_rows]),
        "max_slot_distance_to_ai": max((row["slot_distance_to_ai"] for row in turn_rows), default=0),
        "avg_turn_score_delta": mean_or_zero([row["player_score_delta"] for row in turn_rows]),
        "avg_turn_advantage_delta": mean_or_zero([row["score_advantage_delta"] for row in turn_rows]),
    }


def percentile_group_ids(game_rows: list[dict[str, Any]]) -> tuple[set[int], set[int], str]:
    if not game_rows:
        return set(), set(), "데이터 없음"
    ordered = sorted(game_rows, key=lambda row: row["player_score"], reverse=True)
    if len(ordered) >= 40:
        count = max(1, math.ceil(len(ordered) * 0.25))
        label = "상위 25% vs 하위 25%"
    else:
        count = max(1, min(10, len(ordered) // 2 or 1))
        label = f"상위 {count}게임 vs 하위 {count}게임"
    high_ids = {row["game_id"] for row in ordered[:count]}
    low_ids = {row["game_id"] for row in ordered[-count:]}
    return high_ids, low_ids, label


def summarize_game_group(game_rows: list[dict[str, Any]], turn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    game_ids = {row["game_id"] for row in game_rows}
    relevant_turns = [row for row in turn_rows if row["game_id"] in game_ids]
    early_turns = [row for row in relevant_turns if row["phase"] == "early"]
    return {
        "games": len(game_rows),
        "avg_score": mean_or_zero([row["player_score"] for row in game_rows]),
        "avg_duration_sec": mean_or_zero([row["duration_sec"] for row in game_rows]),
        "early_center_rate": mean_or_zero([row["center_slot_flag"] for row in early_turns]),
        "middle_insert_rate": mean_or_zero([1 if row["placement_type"] == "middle_insert" else 0 for row in relevant_turns]),
        "risky_insert_rate": mean_or_zero([1 if row["placement_type"] == "risky_insert" else 0 for row in relevant_turns]),
        "avg_late_game_efficiency": mean_or_zero([row["late_game_efficiency"] for row in game_rows]),
        "avg_longest_segment": mean_or_zero([row["board_longest_segment"] for row in game_rows]),
        "avg_score_gap": mean_or_zero([row["score_gap"] for row in game_rows]),
    }


def placement_phase_summary(turn_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for phase in ["early", "mid", "late"]:
        rows = [row for row in turn_rows if row["phase"] == phase]
        output.append(
            {
                "phase": phase,
                "turns": len(rows),
                "left_end_extension_rate": mean_or_zero([1 if row["placement_type"] == "left_end_extension" else 0 for row in rows]),
                "right_end_extension_rate": mean_or_zero([1 if row["placement_type"] == "right_end_extension" else 0 for row in rows]),
                "middle_insert_rate": mean_or_zero([1 if row["placement_type"] == "middle_insert" else 0 for row in rows]),
                "risky_insert_rate": mean_or_zero([1 if row["placement_type"] == "risky_insert" else 0 for row in rows]),
            }
        )
    return output


def card_band_summary(turn_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for band in ["low", "mid", "high"]:
        rows = [row for row in turn_rows if row["card_band"] == band]
        output.append(
            {
                "card_band": band,
                "turns": len(rows),
                "avg_score_delta": mean_or_zero([row["player_score_delta"] for row in rows]),
                "middle_insert_rate": mean_or_zero([1 if row["placement_type"] == "middle_insert" else 0 for row in rows]),
                "risky_insert_rate": mean_or_zero([1 if row["placement_type"] == "risky_insert" else 0 for row in rows]),
                "avg_gap_width": mean_or_zero([safe_float(row["local_gap_width"], 0.0) for row in rows if row["local_gap_width"] != ""]),
            }
        )
    return output


def player_vs_ai_summary(game_rows: list[dict[str, Any]], turn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "games": len(game_rows),
        "human_win_rate": mean_or_zero([row["win_flag"] for row in game_rows]),
        "avg_score_gap": mean_or_zero([row["score_gap"] for row in game_rows]),
        "avg_slot_distance": mean_or_zero([row["slot_distance_to_ai"] for row in turn_rows]),
        "avg_final_longest_segment": mean_or_zero([row["board_longest_segment"] for row in game_rows]),
    }


def case_studies(game_rows: list[dict[str, Any]], turn_rows: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    wins = [row for row in game_rows if row["result"] == "win"]
    wins = sorted(wins, key=lambda row: (row["score_gap"], row["player_score"]), reverse=True)[:limit]
    studies: list[dict[str, Any]] = []
    for game in wins:
        relevant_turns = [row for row in turn_rows if row["game_id"] == game["game_id"]]
        decisive = sorted(relevant_turns, key=lambda row: (row["score_advantage_delta"], row["future_segment_gain_proxy"]), reverse=True)[:3]
        studies.append(
            {
                "game_id": game["game_id"],
                "player_score": game["player_score"],
                "ai_score": game["ai_score"],
                "score_gap": game["score_gap"],
                "duration_sec": game["duration_sec"],
                "longest_segment": game["board_longest_segment"],
                "decisive_turns": [
                    {
                        "turn": row["turn_number"],
                        "card": row["card_value"],
                        "slot": row["player_slot"] + 1,
                        "placement_type": row["placement_type"],
                        "delta_advantage": row["score_advantage_delta"],
                    }
                    for row in decisive
                ],
            }
        )
    return studies


def group_summary_rows(game_rows: list[dict[str, Any]], turn_rows: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    grouped_games: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_turns: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    game_group_map: dict[int, str] = {}

    for row in game_rows:
        key = clean_text(row.get(key_name))
        grouped_games[key].append(row)
        game_group_map[row["game_id"]] = key

    for row in turn_rows:
        key = game_group_map.get(row["game_id"])
        if key:
            grouped_turns[key].append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(grouped_games):
        g_rows = grouped_games[key]
        t_rows = grouped_turns.get(key, [])
        output.append(
            {
                "group_type": key_name,
                "group_value": key,
                "games": len(g_rows),
                "players": len({row["player_id"] for row in g_rows if row["player_id"]}),
                "avg_score": mean_or_zero([row["player_score"] for row in g_rows]),
                "avg_duration_sec": mean_or_zero([row["duration_sec"] for row in g_rows]),
                "risky_insert_rate": mean_or_zero([1 if row["placement_type"] == "risky_insert" else 0 for row in t_rows]),
                "middle_insert_rate": mean_or_zero([1 if row["placement_type"] == "middle_insert" else 0 for row in t_rows]),
            }
        )
    return output


def validation_report_text(summary: ValidationSummary) -> list[str]:
    return [
        f"- 전체 게임 수: {summary.total_games}",
        f"- 분석 포함 게임 수: {summary.analyzed_games}",
        f"- 제외 게임 수: {summary.excluded_games}",
        f"- 턴 로그 누락 게임 수: {summary.missing_turn_games}",
        f"- 턴 수 불일치 게임 수: {summary.invalid_turn_count_games}",
        f"- 턴 순서 이상 게임 수: {summary.invalid_turn_order_games}",
        f"- 복원 보드 불일치 게임 수: {summary.board_mismatch_games}",
        f"- 플레이어 설문 연결 누락 게임 수: {summary.player_missing_games}",
    ]


def render_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_데이터 없음_"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, divider, *body])


def histogram(values: list[float], bins: int, width: int = 24) -> str:
    if not values:
        return "_데이터 없음_"
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return f"{low:.1f}: {'#' * width} ({len(values)})"
    step = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(int((value - low) / step), bins - 1)
        counts[index] += 1
    peak = max(counts) or 1
    lines = []
    for idx, count in enumerate(counts):
        bin_start = low + idx * step
        bin_end = bin_start + step
        bar = "#" * max(1, round((count / peak) * width)) if count else ""
        lines.append(f"{bin_start:6.1f} ~ {bin_end:6.1f} | {bar} ({count})")
    return "\n".join(lines)


def top_differences(high: dict[str, Any], low: dict[str, Any]) -> list[str]:
    comparisons = [
        ("초반 중앙 배치 비율", high["early_center_rate"] - low["early_center_rate"], "높습니다"),
        ("중간 삽입 비율", high["middle_insert_rate"] - low["middle_insert_rate"], "높습니다"),
        ("위험 삽입 비율", low["risky_insert_rate"] - high["risky_insert_rate"], "낮습니다"),
        ("후반 점수 효율", high["avg_late_game_efficiency"] - low["avg_late_game_efficiency"], "높습니다"),
        ("최종 긴 구간 길이", high["avg_longest_segment"] - low["avg_longest_segment"], "깁니다"),
        ("평균 점수 격차", high["avg_score_gap"] - low["avg_score_gap"], "좋습니다"),
    ]
    ranked = sorted(comparisons, key=lambda item: abs(item[1]), reverse=True)
    return [
        f"- 고득점 그룹은 `{label}`이(가) 저득점 그룹보다 `{abs(diff):.3f}` 만큼 {phrase}."
        for label, diff, phrase in ranked[:4]
    ]


def build_report(
    summary: ValidationSummary,
    game_rows: list[dict[str, Any]],
    turn_rows: list[dict[str, Any]],
    high_summary: dict[str, Any],
    low_summary: dict[str, Any],
    group_label: str,
    phase_rows: list[dict[str, Any]],
    card_rows: list[dict[str, Any]],
    player_ai: dict[str, Any],
    cases: list[dict[str, Any]],
    grouped_rows: list[dict[str, Any]],
) -> str:
    total_players = len({row["player_id"] for row in game_rows if row["player_id"]})
    scores = [row["player_score"] for row in game_rows]
    durations = [row["duration_sec"] for row in game_rows]
    played_times = [parse_played_at(row["played_at"]) for row in game_rows]
    played_times = [value for value in played_times if value is not None]
    period = f"{min(played_times).date()} ~ {max(played_times).date()}" if played_times else "N/A"

    outcome_rows = [
        {"항목": "평균 플레이어 점수", "값": round(mean(scores), 3) if scores else 0},
        {"항목": "중앙값 플레이어 점수", "값": round(median(scores), 3) if scores else 0},
        {"항목": "평균 플레이 시간(초)", "값": round(mean(durations), 3) if durations else 0},
        {"항목": "플레이어 승률", "값": round(mean([row["win_flag"] for row in game_rows]), 3) if game_rows else 0},
        {"항목": "평균 AI 대비 점수 차", "값": round(mean([row["score_gap"] for row in game_rows]), 3) if game_rows else 0},
    ]

    high_low_rows = [
        {"지표": "게임 수", "고득점 그룹": high_summary["games"], "저득점 그룹": low_summary["games"]},
        {"지표": "평균 점수", "고득점 그룹": high_summary["avg_score"], "저득점 그룹": low_summary["avg_score"]},
        {"지표": "평균 플레이 시간(초)", "고득점 그룹": high_summary["avg_duration_sec"], "저득점 그룹": low_summary["avg_duration_sec"]},
        {"지표": "초반 중앙 배치 비율", "고득점 그룹": high_summary["early_center_rate"], "저득점 그룹": low_summary["early_center_rate"]},
        {"지표": "중간 삽입 비율", "고득점 그룹": high_summary["middle_insert_rate"], "저득점 그룹": low_summary["middle_insert_rate"]},
        {"지표": "위험 삽입 비율", "고득점 그룹": high_summary["risky_insert_rate"], "저득점 그룹": low_summary["risky_insert_rate"]},
        {"지표": "후반 점수 효율", "고득점 그룹": high_summary["avg_late_game_efficiency"], "저득점 그룹": low_summary["avg_late_game_efficiency"]},
        {"지표": "최종 긴 구간 길이", "고득점 그룹": high_summary["avg_longest_segment"], "저득점 그룹": low_summary["avg_longest_segment"]},
    ]

    grouped_age_rows = [row for row in grouped_rows if row["group_type"] == "age"]
    grouped_playtime_rows = [row for row in grouped_rows if row["group_type"] == "playtime"]

    report_lines = [
        "# 플레이 로그 행동 패턴 분석 보고서",
        "",
        "## 1. 데이터 개요",
        f"- 분석 기간: {period}",
        f"- 전체 플레이어 수: {total_players}",
        *validation_report_text(summary),
        "",
        "## 2. 전체 성과 분포",
        render_markdown_table(outcome_rows, ["항목", "값"]),
        "",
        "### 점수 분포",
        "```text",
        histogram(scores, bins=8),
        "```",
        "",
        "### 플레이 시간 분포(초)",
        "```text",
        histogram(durations, bins=8),
        "```",
        "",
        "## 3. 고득점 플레이의 공통 특징",
        f"- 비교 기준: {group_label}",
        render_markdown_table(high_low_rows, ["지표", "고득점 그룹", "저득점 그룹"]),
        "",
        "### 핵심 해석",
        *top_differences(high_summary, low_summary),
        "",
        "## 4. 턴 단위 의사결정 패턴",
        "### 구간별 배치 유형",
        render_markdown_table(
            phase_rows,
            ["phase", "turns", "left_end_extension_rate", "right_end_extension_rate", "middle_insert_rate", "risky_insert_rate"],
        ),
        "",
        "### 카드 범위별 선택 특성",
        render_markdown_table(
            card_rows,
            ["card_band", "turns", "avg_score_delta", "middle_insert_rate", "risky_insert_rate", "avg_gap_width"],
        ),
        "",
        "## 5. 사람 vs AI 비교",
        render_markdown_table(
            [
                {"항목": "게임 수", "값": player_ai["games"]},
                {"항목": "사람 승률", "값": player_ai["human_win_rate"]},
                {"항목": "평균 점수 차", "값": player_ai["avg_score_gap"]},
                {"항목": "평균 슬롯 거리", "값": player_ai["avg_slot_distance"]},
                {"항목": "평균 최종 긴 구간 길이", "값": player_ai["avg_final_longest_segment"]},
            ],
            ["항목", "값"],
        ),
        "",
        "### AI를 이긴 대표 사례",
    ]

    if cases:
        for case in cases:
            decisive_turn_text = ", ".join(
                f"T{item['turn']} 카드 {item['card']} -> {item['slot']}칸 ({item['placement_type']}, Δ{item['delta_advantage']})"
                for item in case["decisive_turns"]
            )
            report_lines.extend(
                [
                    f"- 게임 `{case['game_id']}`: 플레이어 {case['player_score']}점 / AI {case['ai_score']}점 / 점수 차 {case['score_gap']} / 긴 구간 {case['longest_segment']}",
                    f"  - 결정적 턴: {decisive_turn_text}",
                ]
            )
    else:
        report_lines.append("- 분석 가능한 승리 사례가 없습니다.")

    report_lines.extend(
        [
            "",
            "## 6. 기본 집단 비교",
            "### 연령대 비교",
            render_markdown_table(
                grouped_age_rows,
                ["group_value", "games", "players", "avg_score", "avg_duration_sec", "risky_insert_rate", "middle_insert_rate"],
            ),
            "",
            "### 주간 플레이타임 비교",
            render_markdown_table(
                grouped_playtime_rows,
                ["group_value", "games", "players", "avg_score", "avg_duration_sec", "risky_insert_rate", "middle_insert_rate"],
            ),
            "",
            "## 7. 한계와 다음 단계",
            "- 현재 로그만으로는 클릭 대기 시간, 취소 행동, 보드 재탐색 같은 미세 행동 신호를 직접 관찰할 수 없습니다.",
            "- 설문 집단 비교는 표본 수가 작을 때 과해석 위험이 있으므로, 이번 보고서에서는 연령대와 주간 플레이타임만 기초 비교로 사용했습니다.",
            "- 다음 단계에서는 턴별 반응 시간, 추천 배치 노출 여부, 세션 종료 사유를 추가하면 모방학습용 분석 정확도를 더 높일 수 있습니다.",
        ]
    )
    return "\n".join(report_lines) + "\n"


def markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    html_parts = [
        "<!doctype html>",
        "<html lang='ko'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>플레이 로그 행동 패턴 분석 보고서</title>",
        "<style>",
        "body{font-family:Segoe UI,Apple SD Gothic Neo,sans-serif;background:#0b1020;color:#e8eefc;line-height:1.65;margin:0;padding:32px;}",
        "main{max-width:1100px;margin:0 auto;}",
        "h1,h2,h3{color:#ffffff;line-height:1.3;}",
        "h1{font-size:2.2rem;margin-bottom:0.75rem;}",
        "h2{font-size:1.5rem;margin-top:2.2rem;padding-bottom:0.4rem;border-bottom:1px solid #2a3558;}",
        "h3{font-size:1.1rem;margin-top:1.6rem;}",
        "p,li{color:#d7def5;}",
        "code{background:#16203a;padding:0.15rem 0.35rem;border-radius:6px;}",
        "pre{background:#0f172a;color:#dbeafe;padding:16px;border-radius:14px;overflow:auto;border:1px solid #223055;}",
        "table{width:100%;border-collapse:collapse;margin:16px 0 24px;background:#10182d;border:1px solid #243455;}",
        "th,td{border:1px solid #243455;padding:10px 12px;text-align:left;vertical-align:top;}",
        "th{background:#17233f;color:#ffffff;}",
        "tr:nth-child(even) td{background:#0f172a;}",
        ".muted{color:#93a3c7;}",
        "</style>",
        "</head>",
        "<body>",
        "<main>",
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("### "):
            html_parts.append(f"<h3>{html.escape(line[4:])}</h3>")
            i += 1
            continue
        if line.startswith("## "):
            html_parts.append(f"<h2>{html.escape(line[3:])}</h2>")
            i += 1
            continue
        if line.startswith("# "):
            html_parts.append(f"<h1>{html.escape(line[2:])}</h1>")
            i += 1
            continue
        if line.startswith("```"):
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            html_parts.append(f"<pre>{html.escape(chr(10).join(code_lines))}</pre>")
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            html_parts.append(markdown_table_to_html(table_lines))
            continue
        if line.startswith("- "):
            list_items = []
            while i < len(lines) and lines[i].startswith("- "):
                list_items.append(lines[i][2:])
                i += 1
            html_parts.append("<ul>")
            for item in list_items:
                html_parts.append(f"<li>{inline_markdown_to_html(item)}</li>")
            html_parts.append("</ul>")
            continue
        if line.startswith("_") and line.endswith("_"):
            html_parts.append(f"<p class='muted'>{html.escape(line.strip('_'))}</p>")
            i += 1
            continue
        html_parts.append(f"<p>{inline_markdown_to_html(line)}</p>")
        i += 1

    html_parts.extend(["</main>", "</body>", "</html>"])
    return "\n".join(html_parts)


def markdown_table_to_html(lines: list[str]) -> str:
    rows = []
    for line in lines:
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        rows.append(parts)
    if len(rows) < 2:
        return ""
    header = rows[0]
    body = rows[2:]
    out = ["<table>", "<thead><tr>"]
    out.extend(f"<th>{inline_markdown_to_html(cell)}</th>" for cell in header)
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>")
        out.extend(f"<td>{inline_markdown_to_html(cell)}</td>" for cell in row)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def inline_markdown_to_html(text: str) -> str:
    escaped = html.escape(text)
    parts = escaped.split("`")
    if len(parts) == 1:
        return escaped
    rebuilt = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            rebuilt.append(f"<code>{part}</code>")
        else:
            rebuilt.append(part)
    return "".join(rebuilt)


def build_analysis() -> tuple[ValidationSummary, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = repo_root()
    load_environment(root)
    client = get_supabase_client()

    players = fetch_all_rows(client, "players")
    games = fetch_all_rows(client, "games")
    turns = fetch_all_rows(client, "turns", order_by="turn_number")

    players_by_id = {safe_int(player.get("id")): player for player in players}
    turns_by_game: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        turns_by_game[safe_int(turn.get("game_id"))].append(turn)

    summary = ValidationSummary(total_games=len(games))
    game_metrics: list[dict[str, Any]] = []
    turn_metrics: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for game in games:
        game_id = safe_int(game.get("id"))
        player_id = safe_int(game.get("player_id"))
        player = players_by_id.get(player_id)
        if not player:
            summary.player_missing_games += 1
            summary.excluded_games += 1
            continue

        game_turns = turns_by_game.get(game_id, [])
        if not game_turns:
            summary.missing_turn_games += 1
            summary.excluded_games += 1
            continue

        derived_turns, errors = derive_turn_metrics(game, player, game_turns)
        if errors:
            if "turn_count_mismatch" in errors:
                summary.invalid_turn_count_games += 1
            if "turn_order_invalid" in errors or "deck_index_invalid" in errors:
                summary.invalid_turn_order_games += 1
            if "board_mismatch" in errors:
                summary.board_mismatch_games += 1
            summary.excluded_games += 1
            continue

        summary.analyzed_games += 1
        turn_metrics.extend(derived_turns)
        game_metric = derive_game_metrics(game, player, derived_turns)
        game_metrics.append(game_metric)
        group_rows.append(
            {
                "game_id": game_metric["game_id"],
                "player_id": game_metric["player_id"],
                "age": game_metric["age"],
                "playtime": game_metric["playtime"],
                "player_score": game_metric["player_score"],
                "duration_sec": game_metric["duration_sec"],
            }
        )

    return summary, game_metrics, turn_metrics, group_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis artifacts from Streams play logs.")
    parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "streams_data_analysis" / "output"),
        help="Directory where analysis outputs will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    summary, game_metrics, turn_metrics, _ = build_analysis()
    high_ids, low_ids, group_label = percentile_group_ids(game_metrics)
    high_rows = [row for row in game_metrics if row["game_id"] in high_ids]
    low_rows = [row for row in game_metrics if row["game_id"] in low_ids]

    high_summary = summarize_game_group(high_rows, turn_metrics)
    low_summary = summarize_game_group(low_rows, turn_metrics)
    phase_rows = placement_phase_summary(turn_metrics)
    card_rows = card_band_summary(turn_metrics)
    player_ai = player_vs_ai_summary(game_metrics, turn_metrics)
    grouped_rows = group_summary_rows(game_metrics, turn_metrics, "age") + group_summary_rows(game_metrics, turn_metrics, "playtime")
    cases = case_studies(game_metrics, turn_metrics)

    markdown_report = build_report(
        summary=summary,
        game_rows=game_metrics,
        turn_rows=turn_metrics,
        high_summary=high_summary,
        low_summary=low_summary,
        group_label=group_label,
        phase_rows=phase_rows,
        card_rows=card_rows,
        player_ai=player_ai,
        cases=cases,
        grouped_rows=grouped_rows,
    )
    html_report = markdown_to_html(markdown_report)

    write_csv(output_dir / "game_level_metrics.csv", game_metrics)
    write_csv(output_dir / "turn_level_metrics.csv", turn_metrics)
    write_csv(output_dir / "group_summary.csv", grouped_rows)
    (output_dir / "play_log_analysis_report.md").write_text(markdown_report, encoding="utf-8")
    (output_dir / "play_log_analysis_report.html").write_text(html_report, encoding="utf-8")

    print(f"[Analysis] game rows: {len(game_metrics)}")
    print(f"[Analysis] turn rows: {len(turn_metrics)}")
    print(f"[Analysis] outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
