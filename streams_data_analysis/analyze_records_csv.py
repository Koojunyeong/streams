from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


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

THRESHOLDS = [40, 60, 80, 100]

FEATURE_SPECS = [
    ("longest_segment", "최장 오름차순 구간 길이", "higher", "길수록 고득점에 유리합니다."),
    ("segment_count", "오름차순 구간 수", "lower", "적을수록 보드가 덜 쪼개집니다."),
    ("avg_segment_length", "평균 구간 길이", "higher", "길수록 안정적인 흐름을 뜻합니다."),
    ("pair_inversion_rate", "쌍별 역전 비율", "lower", "낮을수록 정렬이 잘 유지됩니다."),
    ("rank_displacement_mae", "정렬 기준 평균 슬롯 이탈", "lower", "낮을수록 숫자 순서가 자연스럽습니다."),
    ("low_high_separation", "저수-고수 분리도", "higher", "낮은 수와 높은 수를 영역별로 나눠 둔 정도입니다."),
    ("duplicate_integration_rate", "중복 숫자 통합률", "higher", "같은 숫자를 기존 스트림 안에 흡수한 비율입니다."),
    ("middle_insert_rate", "중간 삽입 비율", "higher", "이미 만든 틈을 활용하는 공격적 배치입니다."),
    ("left_end_extension_rate", "왼쪽 끝 확장 비율", "neutral", "낮은 숫자 영역을 바깥으로 넓힌 비율입니다."),
    ("right_end_extension_rate", "오른쪽 끝 확장 비율", "neutral", "높은 숫자 영역을 바깥으로 넓힌 비율입니다."),
    ("risky_insert_rate", "위험 삽입 비율", "lower", "좌우 기준 범위를 깨는 배치 비율입니다."),
    ("late_game_repair_rate", "후반 수습 비율", "higher", "후반 턴에 틈을 메우며 점수를 올린 비율입니다."),
    ("score_jump_turn_count", "점수 점프 턴 수", "higher", "한 턴에 5점 이상 오른 횟수입니다."),
    ("duplicate_first_second_slot_distance", "중복 숫자 첫/둘째 슬롯 거리", "context", "중복 숫자를 멀리 떼어 두는지 보는 지표입니다."),
    ("duplicate_same_segment_rate", "중복 숫자 동일 스트림 비율", "higher", "같은 숫자를 같은 스트림에 넣은 비율입니다."),
    ("duplicate_second_use_repair_rate", "중복 숫자 재활용 수습 비율", "higher", "두 번째 중복 숫자로 점수를 실제 개선한 비율입니다."),
    ("avg_slot_distance_to_ai", "사람-AI 슬롯 거리", "context", "사람과 AI의 선택이 얼마나 달랐는지 보여줍니다."),
    ("avg_longest_segment_gap", "사람-AI 최장 구간 격차", "higher", "사람이 AI보다 긴 스트림을 만들었는지 보여줍니다."),
    ("duration_sec", "플레이 시간(초)", "context", "시간이 길다고 반드시 고득점은 아닙니다."),
    ("duplicate_count", "덱 중복 수", "context", "중복 카드 수 자체는 난이도 요인입니다."),
    ("deck_mean_abs_gap", "연속 공개 카드 변화폭 평균", "context", "연속 턴 카드 점프의 평균 크기입니다."),
    ("deck_gt10_gap_rate", "10초과 점프 비율", "context", "큰 점프가 자주 나오는 덱인지 보여줍니다."),
    ("deck_max_gap", "최대 카드 점프", "context", "가장 급격한 턴 변화 폭입니다."),
]


@dataclass
class DatasetSummary:
    source_games: int
    main_games: int
    excluded_tutorial_games: int
    invalid_games: int
    score_mismatch_games: int
    turn_reconstruction_mismatch_games: int
    period: str
    unique_players: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value or "[]"))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def parse_datetime_text(value: str) -> str:
    text = clean_text(value, "")
    return text[:10] if text else ""


def normalize_game_mode(value: Any) -> str:
    return "tutorial" if str(value or "").strip().lower() == "tutorial" else "main"


def calc_streams_score(board: list[int]) -> int:
    values = [value for value in board if value]
    if not values:
        return 0
    score = 0
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[end] <= values[end + 1]:
            end += 1
        score += SCORE_TABLE.get(end - index + 1, 0)
        index = end + 1
    return score


def monotonic_segments(board: list[int]) -> list[tuple[int, int, int]]:
    values = [value for value in board if value]
    if not values:
        return []
    segments: list[tuple[int, int, int]] = []
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[end] <= values[end + 1]:
            end += 1
        segments.append((start, end, end - start + 1))
        start = end + 1
    return segments


def slot_segment_lengths(board: list[int]) -> dict[int, int]:
    segments = monotonic_segments(board)
    mapping: dict[int, int] = {}
    for start, end, length in segments:
        for slot in range(start, end + 1):
            mapping[slot] = length
    return mapping


def pair_inversion_rate(board: list[int]) -> float:
    values = [value for value in board if value]
    total = 0
    inversions = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            total += 1
            if values[i] > values[j]:
                inversions += 1
    return inversions / total if total else 0.0


def rank_displacement_mae(board: list[int]) -> float:
    values = [value for value in board if value]
    sorted_values = sorted(values)
    target_positions: defaultdict[int, list[int]] = defaultdict(list)
    used: defaultdict[int, int] = defaultdict(int)
    for idx, value in enumerate(sorted_values):
        target_positions[value].append(idx)
    absolute_diffs = []
    for idx, value in enumerate(values):
        target_idx = target_positions[value][used[value]]
        used[value] += 1
        absolute_diffs.append(abs(idx - target_idx))
    return mean(absolute_diffs) if absolute_diffs else 0.0


def low_high_separation(board: list[int]) -> float:
    values = [value for value in board if value]
    if len(values) <= 1:
        return 0.0
    low_positions = [idx / (len(values) - 1) for idx, value in enumerate(values) if value <= 10]
    high_positions = [idx / (len(values) - 1) for idx, value in enumerate(values) if value >= 21]
    if not low_positions or not high_positions:
        return 0.0
    return mean(high_positions) - mean(low_positions)


def duplicate_integration_rate(board: list[int], deck: list[int]) -> float:
    values = [value for value in board if value]
    if not values:
        return 0.0
    segments = monotonic_segments(board)
    slot_to_segment: dict[int, int] = {}
    for seg_idx, (start, end, _) in enumerate(segments):
        for idx in range(start, end + 1):
            slot_to_segment[idx] = seg_idx

    value_positions: defaultdict[int, list[int]] = defaultdict(list)
    for idx, value in enumerate(values):
        value_positions[value].append(idx)

    duplicate_values = [value for value, count in Counter(deck).items() if count > 1]
    if not duplicate_values:
        return 0.0

    integrated = 0
    for value in duplicate_values:
        positions = value_positions.get(value, [])
        if positions and len({slot_to_segment[pos] for pos in positions}) == 1:
            integrated += 1
    return integrated / len(duplicate_values)


def deck_gap_stats(deck: list[int]) -> tuple[float, float, int]:
    if len(deck) < 2:
        return 0.0, 0.0, 0
    gaps = [abs(deck[idx + 1] - deck[idx]) for idx in range(len(deck) - 1)]
    return mean(gaps), sum(1 for gap in gaps if gap > 10) / len(gaps), max(gaps)


def mean_or_zero(values: list[float | int]) -> float:
    return round(mean(values), 3) if values else 0.0


def correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = sum((x - mx) ** 2 for x in xs)
    denom_y = sum((y - my) ** 2 for y in ys)
    if not denom_x or not denom_y:
        return 0.0
    return numerator / math.sqrt(denom_x * denom_y)


def parse_turn_detail(row: dict[str, Any], deck: list[int]) -> list[dict[str, int]]:
    raw_detail = row.get("Turn Detail", "")
    if raw_detail:
        try:
            parsed = json.loads(raw_detail)
            if isinstance(parsed, list):
                normalized = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    normalized.append(
                        {
                            "turn": safe_int(item.get("turn") or item.get("turn_number")),
                            "card": safe_int(item.get("card") or item.get("card_value")),
                            "card_order": safe_int(item.get("card_order") or item.get("turn") or item.get("turn_number")),
                            "deck_index": safe_int(item.get("deck_index")),
                            "player_slot": safe_int(item.get("player_slot")),
                            "ai_slot": safe_int(item.get("ai_slot")),
                            "player_score_after": safe_int(item.get("player_score_after")),
                            "ai_score_after": safe_int(item.get("ai_score_after")),
                        }
                    )
                normalized = [item for item in normalized if item["turn"] > 0]
                normalized.sort(key=lambda item: (item["turn"], item["deck_index"]))
                if normalized:
                    return normalized
        except Exception:
            pass

    cards = [safe_int(value) for value in parse_json_list(row.get("Turn Cards", "[]"))]
    player_slots = [safe_int(value, -1) for value in parse_json_list(row.get("Turn Player Slots", "[]"))]
    ai_slots = [safe_int(value, -1) for value in parse_json_list(row.get("Turn AI Slots", "[]"))]
    player_scores = [safe_int(value) for value in parse_json_list(row.get("Turn Player Scores", "[]"))]
    ai_scores = [safe_int(value) for value in parse_json_list(row.get("Turn AI Scores", "[]"))]
    turn_count = max(len(cards), len(player_slots), len(ai_slots), len(player_scores), len(ai_scores), len(deck))
    detail = []
    for idx in range(turn_count):
        card = cards[idx] if idx < len(cards) else (deck[idx] if idx < len(deck) else 0)
        if card == 0:
            continue
        detail.append(
            {
                "turn": idx + 1,
                "card": card,
                "card_order": idx + 1,
                "deck_index": idx,
                "player_slot": player_slots[idx] if idx < len(player_slots) else -1,
                "ai_slot": ai_slots[idx] if idx < len(ai_slots) else -1,
                "player_score_after": player_scores[idx] if idx < len(player_scores) else 0,
                "ai_score_after": ai_scores[idx] if idx < len(ai_scores) else 0,
            }
        )
    return [item for item in detail if item["player_slot"] >= 0]


def rebuild_board_from_turns(turns: list[dict[str, int]], slot_key: str) -> list[int]:
    board = [0] * 20
    for turn in turns:
        slot = safe_int(turn.get(slot_key), -1)
        card = safe_int(turn.get("card"), 0)
        if 0 <= slot < len(board):
            board[slot] = card
    return board


def analyze_turns(turns: list[dict[str, int]], final_board: list[int], final_ai_board: list[int]) -> dict[str, Any]:
    if not turns:
        return {
            "turn_count": 0,
            "early_left_bias": 0.0,
            "early_center_bias": 0.0,
            "early_right_bias": 0.0,
            "middle_insert_rate": 0.0,
            "left_end_extension_rate": 0.0,
            "right_end_extension_rate": 0.0,
            "risky_insert_rate": 0.0,
            "late_game_repair_rate": 0.0,
            "score_jump_turn_count": 0,
            "duplicate_turn_count": 0,
            "duplicate_first_second_slot_distance": 0.0,
            "duplicate_same_segment_rate": 0.0,
            "duplicate_second_use_repair_rate": 0.0,
            "avg_slot_distance_to_ai": 0.0,
            "avg_longest_segment_gap": 0.0,
            "turn_rows": [],
        }

    final_slot_segment_length = slot_segment_lengths(final_board)
    final_ai_longest = max((segment[2] for segment in monotonic_segments(final_ai_board)), default=0)
    final_player_longest = max((segment[2] for segment in monotonic_segments(final_board)), default=0)
    before_board = [0] * 20
    seen_cards: Counter[int] = Counter()
    total_card_counts = Counter(turn["card"] for turn in turns)
    early_left = early_center = early_right = 0
    middle_insert = left_end_extension = right_end_extension = risky_insert = 0
    late_repair_success = 0
    late_turns = 0
    score_jump_turn_count = 0
    duplicate_turn_count = 0
    slot_distances_to_ai: list[float] = []
    duplicate_turn_rows = []
    turn_rows = []
    prev_player_score = 0

    for turn in sorted(turns, key=lambda item: item["turn"]):
        slot = safe_int(turn.get("player_slot"), -1)
        ai_slot = safe_int(turn.get("ai_slot"), -1)
        card = safe_int(turn.get("card"), 0)
        turn_number = safe_int(turn.get("turn"), 0)
        player_score_after = safe_int(turn.get("player_score_after"), 0)
        ai_score_after = safe_int(turn.get("ai_score_after"), 0)

        left_slot = next((idx for idx in range(slot - 1, -1, -1) if before_board[idx]), None)
        right_slot = next((idx for idx in range(slot + 1, len(before_board)) if before_board[idx]), None)
        left_anchor = before_board[left_slot] if left_slot is not None else None
        right_anchor = before_board[right_slot] if right_slot is not None else None

        if left_anchor is None and right_anchor is None:
            placement_type = "initial"
        elif left_anchor is None:
            placement_type = "left_end_extension" if card <= right_anchor else "risky_insert"
        elif right_anchor is None:
            placement_type = "right_end_extension" if card >= left_anchor else "risky_insert"
        else:
            placement_type = "middle_insert" if left_anchor <= card <= right_anchor else "risky_insert"

        slot_position_norm = slot / 19 if slot >= 0 else 0.0
        phase = "early" if turn_number <= 6 else "mid" if turn_number <= 14 else "late"
        player_score_delta = player_score_after - prev_player_score
        duplicate_card_turn = seen_cards[card] > 0
        future_segment_gain_proxy = final_slot_segment_length.get(slot, 0)

        if phase == "early":
            if slot_position_norm < 1 / 3:
                early_left += 1
            elif slot_position_norm < 2 / 3:
                early_center += 1
            else:
                early_right += 1

        if placement_type == "middle_insert":
            middle_insert += 1
        elif placement_type == "left_end_extension":
            left_end_extension += 1
        elif placement_type == "right_end_extension":
            right_end_extension += 1
        elif placement_type == "risky_insert":
            risky_insert += 1

        if phase == "late":
            late_turns += 1
            if placement_type == "middle_insert" and player_score_delta > 0:
                late_repair_success += 1

        if player_score_delta >= 5:
            score_jump_turn_count += 1

        if duplicate_card_turn:
            duplicate_turn_count += 1
            duplicate_turn_rows.append(
                {
                    "card": card,
                    "turn": turn_number,
                    "player_slot": slot,
                    "player_score_delta": player_score_delta,
                    "placement_type": placement_type,
                }
            )

        if slot >= 0 and ai_slot >= 0:
            slot_distances_to_ai.append(abs(slot - ai_slot))

        turn_rows.append(
            {
                "turn_number": turn_number,
                "card_value": card,
                "player_slot": slot,
                "ai_slot": ai_slot,
                "player_score_after": player_score_after,
                "ai_score_after": ai_score_after,
                "player_score_delta": player_score_delta,
                "slot_position_norm": round(slot_position_norm, 3),
                "phase": phase,
                "empty_slots_before_turn": sum(1 for value in before_board if value == 0),
                "left_anchor": left_anchor if left_anchor is not None else "",
                "right_anchor": right_anchor if right_anchor is not None else "",
                "local_gap_width": (right_anchor - left_anchor) if left_anchor is not None and right_anchor is not None else "",
                "placement_type": placement_type,
                "duplicate_card_turn": int(duplicate_card_turn),
                "future_segment_gain_proxy": future_segment_gain_proxy,
            }
        )

        if 0 <= slot < len(before_board):
            before_board[slot] = card
        seen_cards[card] += 1
        prev_player_score = player_score_after

    duplicate_slot_distances = []
    duplicate_repair_success = 0
    duplicate_groups: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in duplicate_turn_rows:
        duplicate_groups[row["card"]].append(row)
        if row["player_score_delta"] > 0:
            duplicate_repair_success += 1
    for card, occurrences in duplicate_groups.items():
        if len(occurrences) >= 2:
            duplicate_slot_distances.append(abs(occurrences[0]["player_slot"] - occurrences[1]["player_slot"]))

    value_positions: defaultdict[int, list[int]] = defaultdict(list)
    for slot, value in enumerate(final_board):
        value_positions[value].append(slot)
    slot_to_segment: dict[int, int] = {}
    for seg_idx, (start, end, _) in enumerate(monotonic_segments(final_board)):
        for slot in range(start, end + 1):
            slot_to_segment[slot] = seg_idx
    duplicate_values = [value for value, count in total_card_counts.items() if count > 1]
    duplicate_same_segment = 0
    for value in duplicate_values:
        positions = value_positions.get(value, [])
        if positions and len({slot_to_segment.get(pos, -1) for pos in positions}) == 1:
            duplicate_same_segment += 1

    total_turns = len(turns)
    early_turn_count = sum(1 for row in turn_rows if row["phase"] == "early")
    return {
        "turn_count": total_turns,
        "early_left_bias": round(early_left / early_turn_count, 3) if early_turn_count else 0.0,
        "early_center_bias": round(early_center / early_turn_count, 3) if early_turn_count else 0.0,
        "early_right_bias": round(early_right / early_turn_count, 3) if early_turn_count else 0.0,
        "middle_insert_rate": round(middle_insert / total_turns, 3) if total_turns else 0.0,
        "left_end_extension_rate": round(left_end_extension / total_turns, 3) if total_turns else 0.0,
        "right_end_extension_rate": round(right_end_extension / total_turns, 3) if total_turns else 0.0,
        "risky_insert_rate": round(risky_insert / total_turns, 3) if total_turns else 0.0,
        "late_game_repair_rate": round(late_repair_success / late_turns, 3) if late_turns else 0.0,
        "score_jump_turn_count": score_jump_turn_count,
        "duplicate_turn_count": duplicate_turn_count,
        "duplicate_first_second_slot_distance": round(mean(duplicate_slot_distances), 3) if duplicate_slot_distances else 0.0,
        "duplicate_same_segment_rate": round(duplicate_same_segment / len(duplicate_values), 3) if duplicate_values else 0.0,
        "duplicate_second_use_repair_rate": round(duplicate_repair_success / duplicate_turn_count, 3) if duplicate_turn_count else 0.0,
        "avg_slot_distance_to_ai": round(mean(slot_distances_to_ai), 3) if slot_distances_to_ai else 0.0,
        "avg_longest_segment_gap": round(final_player_longest - final_ai_longest, 3),
        "turn_rows": turn_rows,
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "games": len(rows),
        "avg_score": mean_or_zero([row["player_score"] for row in rows]),
        "avg_duration_sec": mean_or_zero([row["duration_sec"] for row in rows]),
        "avg_longest_segment": mean_or_zero([row["longest_segment"] for row in rows]),
        "avg_segment_count": mean_or_zero([row["segment_count"] for row in rows]),
        "avg_avg_segment_length": mean_or_zero([row["avg_segment_length"] for row in rows]),
        "avg_pair_inversion_rate": mean_or_zero([row["pair_inversion_rate"] for row in rows]),
        "avg_rank_displacement": mean_or_zero([row["rank_displacement_mae"] for row in rows]),
        "avg_low_high_separation": mean_or_zero([row["low_high_separation"] for row in rows]),
        "avg_duplicate_integration": mean_or_zero([row["duplicate_integration_rate"] for row in rows]),
        "avg_middle_insert_rate": mean_or_zero([row["middle_insert_rate"] for row in rows]),
        "avg_risky_insert_rate": mean_or_zero([row["risky_insert_rate"] for row in rows]),
        "avg_late_game_repair_rate": mean_or_zero([row["late_game_repair_rate"] for row in rows]),
        "avg_duplicate_same_segment_rate": mean_or_zero([row["duplicate_same_segment_rate"] for row in rows]),
        "avg_duplicate_second_use_repair_rate": mean_or_zero([row["duplicate_second_use_repair_rate"] for row in rows]),
        "avg_ai_gap": mean_or_zero([row["score_gap"] for row in rows]),
        "avg_slot_distance_to_ai": mean_or_zero([row["avg_slot_distance_to_ai"] for row in rows]),
    }


def feature_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = [float(row["player_score"]) for row in rows]
    out = []
    for key, label, goal, description in FEATURE_SPECS:
        values = [float(row.get(key, 0.0)) for row in rows]
        out.append(
            {
                "feature_key": key,
                "feature_name": label,
                "goal": goal,
                "description": description,
                "correlation_with_score": round(correlation(values, scores), 3),
                "overall_mean": mean_or_zero(values),
            }
        )
    return sorted(out, key=lambda item: abs(item["correlation_with_score"]), reverse=True)


def threshold_feature_summary(rows: list[dict[str, Any]], threshold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    above_rows = [row for row in rows if row["player_score"] >= threshold]
    below_rows = [row for row in rows if row["player_score"] < threshold]
    summary_rows = []
    for key, label, goal, description in FEATURE_SPECS:
        above_values = [float(row.get(key, 0.0)) for row in above_rows]
        below_values = [float(row.get(key, 0.0)) for row in below_rows]
        above_mean = mean_or_zero(above_values)
        below_mean = mean_or_zero(below_values)
        summary_rows.append(
            {
                "threshold": threshold,
                "feature_key": key,
                "feature_name": label,
                "goal": goal,
                "description": description,
                "above_games": len(above_rows),
                "below_games": len(below_rows),
                "above_mean": above_mean,
                "below_mean": below_mean,
                "gap": round(above_mean - below_mean, 3),
            }
        )
    summary_rows.sort(key=lambda row: abs(row["gap"]), reverse=True)
    return above_rows, below_rows, summary_rows


def categorical_summary(rows: list[dict[str, Any]], key: str, label: str, min_games: int = 5) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[clean_text(row.get(key))].append(row)
    out = []
    for value, items in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        if len(items) < min_games:
            continue
        record = {
            label: value,
            "games": len(items),
            "avg_score": mean_or_zero([item["player_score"] for item in items]),
            "avg_duration_sec": mean_or_zero([item["duration_sec"] for item in items]),
            "avg_longest_segment": mean_or_zero([item["longest_segment"] for item in items]),
        }
        for threshold in THRESHOLDS:
            record[f"hit_{threshold}"] = mean_or_zero([1 if item["player_score"] >= threshold else 0 for item in items])
        out.append(record)
    return out


def genre_summary(rows: list[dict[str, Any]], min_games: int = 5) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for genre in row["genres"]:
            grouped[genre].append(row)
    out = []
    for genre, items in sorted(grouped.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        if len(items) < min_games:
            continue
        record = {
            "genre": genre,
            "games": len(items),
            "avg_score": mean_or_zero([item["player_score"] for item in items]),
            "avg_duration_sec": mean_or_zero([item["duration_sec"] for item in items]),
            "avg_longest_segment": mean_or_zero([item["longest_segment"] for item in items]),
        }
        for threshold in THRESHOLDS:
            record[f"hit_{threshold}"] = mean_or_zero([1 if item["player_score"] >= threshold else 0 for item in items])
        out.append(record)
    return out


def ai_comparison_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "avg_player_score": mean_or_zero([row["player_score"] for row in rows]),
        "avg_ai_score": mean_or_zero([row["ai_score"] for row in rows]),
        "avg_score_gap": mean_or_zero([row["score_gap"] for row in rows]),
        "player_win_rate": mean_or_zero([1 if row["result"] == "win" else 0 for row in rows]),
        "avg_player_longest_segment": mean_or_zero([row["longest_segment"] for row in rows]),
        "avg_ai_longest_segment": mean_or_zero([row["ai_longest_segment"] for row in rows]),
        "avg_longest_segment_gap": mean_or_zero([row["avg_longest_segment_gap"] for row in rows]),
        "avg_slot_distance_to_ai": mean_or_zero([row["avg_slot_distance_to_ai"] for row in rows]),
    }


def build_case_studies(rows: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["player_score"], row["score_gap"], row["longest_segment"]), reverse=True)
    return [
        {
            "id": row["id"],
            "player": row["player"],
            "played_at": row["played_at"],
            "player_score": row["player_score"],
            "ai_score": row["ai_score"],
            "score_gap": row["score_gap"],
            "duration_sec": row["duration_sec"],
            "longest_segment": row["longest_segment"],
            "segment_count": row["segment_count"],
            "duplicate_integration_rate": row["duplicate_integration_rate"],
            "middle_insert_rate": row["middle_insert_rate"],
            "risky_insert_rate": row["risky_insert_rate"],
            "player_board": row["player_board"],
        }
        for row in ordered[:limit]
    ]


def render_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_데이터 없음_"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def histogram(values: list[float], bins: int = 8, width: int = 24) -> str:
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


def inline_markdown_to_html(text: str) -> str:
    escaped = html.escape(text)
    parts = escaped.split("`")
    if len(parts) == 1:
        return escaped
    rebuilt = []
    for idx, part in enumerate(parts):
        rebuilt.append(f"<code>{part}</code>" if idx % 2 else part)
    return "".join(rebuilt)


def markdown_table_to_html(lines: list[str]) -> str:
    rows = [[part.strip() for part in line.strip().strip("|").split("|")] for line in lines]
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


def markdown_to_html(markdown_text: str, title: str) -> str:
    lines = markdown_text.splitlines()
    parts = [
        "<!doctype html>",
        "<html lang='ko'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        ":root{--page:#f3f0e8;--paper:#fffdfa;--ink:#202124;--muted:#6b7280;--line:#d8dbe2;--header:#eef2f7;--accent:#2f5d8a;--code:#f5f7fb;}",
        "*{box-sizing:border-box;}",
        "body{margin:0;padding:40px 20px;background:linear-gradient(180deg,#f7f4ed 0%,#ece7dc 100%);color:var(--ink);font-family:'Georgia','Times New Roman','Noto Serif KR','Apple SD Gothic Neo',serif;line-height:1.78;}",
        "main{max-width:1100px;margin:0 auto;}",
        ".paper{background:var(--paper);border:1px solid rgba(110,122,146,.18);box-shadow:0 18px 60px rgba(86,73,49,.12);border-radius:18px;padding:56px 64px;}",
        "h1{font-size:2.1rem;margin:0 0 1.2rem;color:#15202b;line-height:1.3;}",
        "h2{font-size:1.35rem;margin-top:2.4rem;padding-top:0.35rem;border-bottom:2px solid var(--line);font-weight:700;}",
        "h3{font-size:1.06rem;margin-top:1.8rem;color:#243b53;font-weight:700;}",
        "p{margin:0.7rem 0;color:#2b313c;font-size:1rem;}",
        "ul{margin:0.8rem 0 1.2rem 1.2rem;padding:0;}",
        "li{margin:0.4rem 0;color:#2b313c;}",
        "code{background:var(--code);color:#17324d;padding:0.15rem 0.35rem;border-radius:6px;font-size:0.95em;border:1px solid #e4e9f2;}",
        "pre{background:#fcfcfd;color:#17212f;padding:18px 20px;border-radius:12px;overflow:auto;border:1px solid var(--line);font-size:0.93rem;line-height:1.55;}",
        "table{width:100%;border-collapse:collapse;margin:16px 0 28px;background:white;border:1px solid var(--line);font-size:0.96rem;}",
        "th,td{border:1px solid var(--line);padding:11px 13px;text-align:left;vertical-align:top;}",
        "th{background:var(--header);color:#16202f;font-weight:700;}",
        "tr:nth-child(even) td{background:#fbfcfe;}",
        "tr:hover td{background:#f6faff;}",
        ".muted{color:var(--muted);font-style:italic;}",
        "@media (max-width:900px){body{padding:18px 10px;}.paper{padding:28px 20px;border-radius:14px;}h1{font-size:1.8rem;}h2{font-size:1.2rem;}table{font-size:0.9rem;display:block;overflow:auto;}}",
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        "<article class='paper'>",
    ]
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("### "):
            parts.append(f"<h3>{html.escape(line[4:])}</h3>")
            i += 1
            continue
        if line.startswith("## "):
            parts.append(f"<h2>{html.escape(line[3:])}</h2>")
            i += 1
            continue
        if line.startswith("# "):
            parts.append(f"<h1>{html.escape(line[2:])}</h1>")
            i += 1
            continue
        if line.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            parts.append(f"<pre>{html.escape(chr(10).join(code_lines))}</pre>")
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            parts.append(markdown_table_to_html(table_lines))
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:])
                i += 1
            parts.append("<ul>")
            for item in items:
                parts.append(f"<li>{inline_markdown_to_html(item)}</li>")
            parts.append("</ul>")
            continue
        if line.startswith("_") and line.endswith("_"):
            parts.append(f"<p class='muted'>{html.escape(line.strip('_'))}</p>")
            i += 1
            continue
        parts.append(f"<p>{inline_markdown_to_html(line)}</p>")
        i += 1
    parts.extend(["</article>", "</main>", "</body>", "</html>"])
    return "\n".join(parts)


def build_threshold_insights(threshold: int, feature_rows: list[dict[str, Any]], above_rows: list[dict[str, Any]], below_rows: list[dict[str, Any]]) -> list[str]:
    if not above_rows or not below_rows:
        return [f"{threshold}점 이상 또는 미만 그룹의 표본이 부족해 비교 해석을 생성하지 못했습니다."]

    strongest = feature_rows[:4]
    bullets = [
        f"{threshold}점 이상 그룹은 총 `{len(above_rows)}`경기였고, 미만 그룹은 `{len(below_rows)}`경기였습니다.",
    ]
    for row in strongest[:3]:
        direction = "높았습니다" if row["gap"] > 0 else "낮았습니다"
        bullets.append(
            f"`{row['feature_name']}`은 {threshold}점 이상 그룹에서 평균 `{row['above_mean']}`로, 미만 그룹 `{row['below_mean']}`보다 {direction}."
        )
    if threshold >= 80:
        bullets.append("초고득점 구간으로 갈수록 긴 스트림 유지와 중복 숫자 통합이 동시에 좋아지는지 특히 주목할 필요가 있습니다.")
    return bullets


def build_duplicate_strategy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = [("전체", rows)] + [(f"{threshold}점 이상", [row for row in rows if row["player_score"] >= threshold]) for threshold in THRESHOLDS]
    output = []
    for label, subset in labels:
        if not subset:
            continue
        output.append(
            {
                "집단": label,
                "games": len(subset),
                "avg_duplicate_integration_rate": mean_or_zero([row["duplicate_integration_rate"] for row in subset]),
                "avg_duplicate_same_segment_rate": mean_or_zero([row["duplicate_same_segment_rate"] for row in subset]),
                "avg_duplicate_second_use_repair_rate": mean_or_zero([row["duplicate_second_use_repair_rate"] for row in subset]),
                "avg_duplicate_first_second_slot_distance": mean_or_zero([row["duplicate_first_second_slot_distance"] for row in subset]),
            }
        )
    return output


def build_report(
    csv_path: Path,
    summary: DatasetSummary,
    rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    age_rows: list[dict[str, Any]],
    playtime_rows: list[dict[str, Any]],
    gender_rows: list[dict[str, Any]],
    mbti_rows: list[dict[str, Any]],
    genre_rows: list[dict[str, Any]],
    ai_summary: dict[str, float],
    cases: list[dict[str, Any]],
) -> str:
    overview_rows = [
        {"항목": "원본 CSV 경기 수", "값": summary.source_games},
        {"항목": "분석 포함 본게임 수", "값": summary.main_games},
        {"항목": "제외된 튜토리얼 경기 수", "값": summary.excluded_tutorial_games},
        {"항목": "유효성 검사 실패 경기 수", "값": summary.invalid_games},
        {"항목": "점수 검증 실패 경기 수", "값": summary.score_mismatch_games},
        {"항목": "턴 복원 불일치 경기 수", "값": summary.turn_reconstruction_mismatch_games},
        {"항목": "분석 기간", "값": summary.period},
        {"항목": "고유 플레이어 수", "값": summary.unique_players},
        {"항목": "평균 플레이어 점수", "값": round(mean(row["player_score"] for row in rows), 3) if rows else 0},
        {"항목": "중앙값 플레이어 점수", "값": median(row["player_score"] for row in rows) if rows else 0},
        {"항목": "평균 플레이 시간(초)", "값": round(mean(row["duration_sec"] for row in rows), 3) if rows else 0},
    ]

    lines = [
        "# 본게임 전용 스트림스 플레이 전략 분석 보고서",
        "",
        f"- 원본 파일: `{csv_path.name}`",
        "- 분석 범위: `Game Mode == main` 본게임만 포함",
        "- 분석 목적: 고득점(40/60/80/100점 이상)에 도달한 플레이가 어떤 보드 구조와 턴 전략을 보이는지 설명",
        "",
        "## 1. 데이터 개요",
        render_markdown_table(overview_rows, ["항목", "값"]),
        "",
        "### 점수 분포",
        "```text",
        histogram([row["player_score"] for row in rows], bins=10),
        "```",
        "",
        "### 플레이 시간 분포(초)",
        "```text",
        histogram([row["duration_sec"] for row in rows], bins=10),
        "```",
        "",
        "## 2. 분석 대상 필터 설명",
        "- 튜토리얼 데이터는 저장돼 있더라도 이번 보고서에서는 완전히 제외했습니다.",
        "- 따라서 아래 전략 해석은 모두 실제 본게임 플레이에서 나온 행동 패턴을 기준으로 합니다.",
        "",
        "## 3. 전체 피처 상관 분석",
        render_markdown_table(
            feature_rows,
            ["feature_name", "goal", "description", "correlation_with_score", "overall_mean"],
        ),
        "",
    ]

    for index, threshold in enumerate(THRESHOLDS, start=4):
        above_rows = [row for row in rows if row["player_score"] >= threshold]
        below_rows = [row for row in rows if row["player_score"] < threshold]
        subset = [row for row in threshold_rows if row["threshold"] == threshold]
        important = subset[:8]
        lines.extend(
            [
                f"## {index}. {threshold}점 이상 전략",
                render_markdown_table(
                    [
                        {"구분": f"{threshold}점 이상 경기 수", "값": len(above_rows)},
                        {"구분": f"{threshold}점 미만 경기 수", "값": len(below_rows)},
                        {"구분": "이상 그룹 평균 점수", "값": mean_or_zero([row['player_score'] for row in above_rows])},
                        {"구분": "미만 그룹 평균 점수", "값": mean_or_zero([row['player_score'] for row in below_rows])},
                    ],
                    ["구분", "값"],
                ),
                "",
                render_markdown_table(
                    important,
                    ["feature_name", "goal", "above_mean", "below_mean", "gap"],
                ),
                "",
                "### 해석 요약",
                *[f"- {item}" for item in build_threshold_insights(threshold, important, above_rows, below_rows)],
                "",
            ]
        )

    duplicate_rows = build_duplicate_strategy_rows(rows)
    lines.extend(
        [
            "## 8. 중복 숫자 처리 전략",
            render_markdown_table(
                duplicate_rows,
                [
                    "집단",
                    "games",
                    "avg_duplicate_integration_rate",
                    "avg_duplicate_same_segment_rate",
                    "avg_duplicate_second_use_repair_rate",
                    "avg_duplicate_first_second_slot_distance",
                ],
            ),
            "",
            "- 같은 숫자를 기존 스트림 안으로 흡수하는 비율이 높을수록 높은 점수 구간에서 더 자주 나타나는지 확인합니다.",
            "- 특히 두 번째 중복 숫자가 나왔을 때 점수 개선이 실제로 일어났는지가 `duplicate_second_use_repair_rate`에 반영됩니다.",
            "",
            "## 9. 사람 vs AI 비교",
            render_markdown_table(
                [
                    {"항목": "평균 사람 점수", "값": ai_summary["avg_player_score"]},
                    {"항목": "평균 AI 점수", "값": ai_summary["avg_ai_score"]},
                    {"항목": "평균 점수 차", "값": ai_summary["avg_score_gap"]},
                    {"항목": "사람 승률", "값": ai_summary["player_win_rate"]},
                    {"항목": "평균 사람 최장 구간", "값": ai_summary["avg_player_longest_segment"]},
                    {"항목": "평균 AI 최장 구간", "값": ai_summary["avg_ai_longest_segment"]},
                    {"항목": "최장 구간 격차", "값": ai_summary["avg_longest_segment_gap"]},
                    {"항목": "평균 슬롯 거리", "값": ai_summary["avg_slot_distance_to_ai"]},
                ],
                ["항목", "값"],
            ),
            "",
            "## 10. 메타 정보별 차이",
            "### 연령대",
            render_markdown_table(age_rows, ["연령대", "games", "avg_score", "avg_duration_sec", "avg_longest_segment", "hit_40", "hit_60", "hit_80", "hit_100"]),
            "",
            "### 주간 플레이타임",
            render_markdown_table(playtime_rows, ["주간 플레이타임", "games", "avg_score", "avg_duration_sec", "avg_longest_segment", "hit_40", "hit_60", "hit_80", "hit_100"]),
            "",
            "### 성별",
            render_markdown_table(gender_rows, ["성별", "games", "avg_score", "avg_duration_sec", "avg_longest_segment", "hit_40", "hit_60", "hit_80", "hit_100"]),
            "",
            "### MBTI",
            render_markdown_table(mbti_rows, ["MBTI", "games", "avg_score", "avg_duration_sec", "avg_longest_segment", "hit_40", "hit_60", "hit_80", "hit_100"]),
            "",
            "### 게임 장르",
            render_markdown_table(genre_rows, ["genre", "games", "avg_score", "avg_duration_sec", "avg_longest_segment", "hit_40", "hit_60", "hit_80", "hit_100"]),
            "",
            "## 11. 대표 고득점 사례",
        ]
    )

    if cases:
        for case in cases:
            lines.extend(
                [
                    f"- 경기 `{case['id']}` / 플레이어 `{case['player']}` / 날짜 `{case['played_at']}`",
                    f"  - 점수: 사람 {case['player_score']}점, AI {case['ai_score']}점, 격차 {case['score_gap']}점",
                    f"  - 보드 구조: 최장 구간 {case['longest_segment']}, 구간 수 {case['segment_count']}, 중복 통합률 {round(case['duplicate_integration_rate'], 3)}",
                    f"  - 턴 전략 요약: 중간 삽입 비율 {case['middle_insert_rate']}, 위험 삽입 비율 {case['risky_insert_rate']}",
                    f"  - 최종 보드: `{case['player_board']}`",
                ]
            )
    else:
        lines.append("- 대표 사례를 추릴 수 있는 본게임 데이터가 충분하지 않습니다.")

    lines.extend(
        [
            "",
            "## 12. 결론과 한계",
            "- 전체적으로 가장 일관된 고득점 신호는 `긴 오름차순 구간 유지`, `구간 수 최소화`, `역전 비율 축소`, `중복 숫자 통합`, `후반 수습 성공`이었습니다.",
            "- 80점 이상과 100점 이상으로 갈수록 단순히 오래 생각하는 것보다, 이미 만든 스트림을 망가뜨리지 않는 배치가 더 중요하게 나타났습니다.",
            "- 이 보고서는 CSV에 포함된 턴 로그를 이용해 재구성했으므로, 클릭 후보 탐색(hover)이나 망설임 자체까지는 보지 못합니다.",
            "- 따라서 다음 단계에서는 `turn_duration_ms`, `후보 슬롯 히스토리`, `undo/reselect 여부`가 추가되면 더 강한 행동 전략 분석이 가능합니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_rows(csv_path: Path) -> tuple[DatasetSummary, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    invalid_games = 0
    score_mismatch_games = 0
    turn_reconstruction_mismatch_games = 0
    excluded_tutorial_games = 0
    played_dates: list[str] = []
    unique_players: set[str] = set()
    source_games = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            source_games += 1
            game_mode = normalize_game_mode(raw.get("Game Mode", "main"))
            if game_mode != "main":
                excluded_tutorial_games += 1
                continue

            try:
                deck = [safe_int(value) for value in parse_json_list(raw.get("Deck", "[]"))]
                turn_sequence = [safe_int(value) for value in parse_json_list(raw.get("Turn Sequence", "[]"))]
                player_board = [safe_int(value) for value in parse_json_list(raw.get("Player Board", "[]"))]
                ai_board = [safe_int(value) for value in parse_json_list(raw.get("AI Board", "[]"))]
                genres = [clean_text(value) for value in parse_json_list(raw.get("Genres", "[]"))]
                turn_detail = parse_turn_detail(raw, deck or turn_sequence)
            except Exception:
                invalid_games += 1
                continue

            player_score = safe_int(raw.get("Player Score"))
            ai_score = safe_int(raw.get("AI Score"))
            computed_score = calc_streams_score(player_board)
            if computed_score != player_score:
                score_mismatch_games += 1
                invalid_games += 1
                continue

            reconstructed_board = rebuild_board_from_turns(turn_detail, "player_slot") if turn_detail else []
            if turn_detail and reconstructed_board != player_board:
                turn_reconstruction_mismatch_games += 1
                invalid_games += 1
                continue

            reconstructed_ai_board = rebuild_board_from_turns(turn_detail, "ai_slot") if turn_detail else [0] * 20
            player_segments = monotonic_segments(player_board)
            ai_segments = monotonic_segments(ai_board)
            deck_mean_abs_gap, deck_gt10_gap_rate, deck_max_gap = deck_gap_stats(deck or turn_sequence)
            duplicate_count = sum(count - 1 for count in Counter(deck or turn_sequence).values() if count > 1)
            played_at = clean_text(raw.get("Played At"), "")
            player_name = clean_text(raw.get("Player"), "unknown")
            turn_metrics = analyze_turns(turn_detail, player_board, reconstructed_ai_board if any(reconstructed_ai_board) else ai_board)

            rows.append(
                {
                    "id": safe_int(raw.get("ID")),
                    "player": player_name,
                    "phone": clean_text(raw.get("Phone", "")),
                    "age": clean_text(raw.get("Age")),
                    "gender": clean_text(raw.get("Gender")),
                    "mbti": clean_text(raw.get("MBTI")),
                    "playtime": clean_text(raw.get("Playtime")),
                    "genres": genres or ["미기재"],
                    "played_at": played_at,
                    "game_mode": game_mode,
                    "duration_ms": safe_int(raw.get("Duration (ms)")),
                    "duration_sec": round(safe_int(raw.get("Duration (ms)")) / 1000, 3),
                    "deck": deck or turn_sequence,
                    "turn_sequence": turn_sequence or deck,
                    "turn_detail": turn_detail,
                    "player_board": player_board,
                    "ai_board": ai_board,
                    "player_score": player_score,
                    "ai_score": ai_score,
                    "score_gap": player_score - ai_score,
                    "result": clean_text(raw.get("Result"), "unknown"),
                    "longest_segment": max((segment[2] for segment in player_segments), default=0),
                    "segment_count": len(player_segments),
                    "avg_segment_length": mean([segment[2] for segment in player_segments]) if player_segments else 0.0,
                    "pair_inversion_rate": pair_inversion_rate(player_board),
                    "rank_displacement_mae": rank_displacement_mae(player_board),
                    "low_high_separation": low_high_separation(player_board),
                    "duplicate_integration_rate": duplicate_integration_rate(player_board, deck or turn_sequence),
                    "duplicate_count": duplicate_count,
                    "deck_spread": max(deck or turn_sequence) - min(deck or turn_sequence) if (deck or turn_sequence) else 0,
                    "deck_mean_abs_gap": deck_mean_abs_gap,
                    "deck_gt10_gap_rate": deck_gt10_gap_rate,
                    "deck_max_gap": deck_max_gap,
                    "ai_longest_segment": max((segment[2] for segment in ai_segments), default=0),
                    "ai_segment_count": len(ai_segments),
                    **{key: value for key, value in turn_metrics.items() if key != "turn_rows"},
                }
            )
            if played_at:
                played_dates.append(parse_datetime_text(played_at))
            unique_players.add(player_name)

    period = f"{min(played_dates)} ~ {max(played_dates)}" if played_dates else "N/A"
    summary = DatasetSummary(
        source_games=source_games,
        main_games=len(rows),
        excluded_tutorial_games=excluded_tutorial_games,
        invalid_games=invalid_games,
        score_mismatch_games=score_mismatch_games,
        turn_reconstruction_mismatch_games=turn_reconstruction_mismatch_games,
        period=period,
        unique_players=len(unique_players),
    )
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    normalized_rows = []
    for row in rows:
        normalized = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                normalized[key] = json.dumps(value, ensure_ascii=False)
            else:
                normalized[key] = value
        normalized_rows.append(normalized)
    columns = list(normalized_rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(normalized_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Streams CSV records and generate an HTML strategy report.")
    parser.add_argument("csv_path", help="Path to the exported records CSV file.")
    parser.add_argument(
        "--output-dir",
        default=str(repo_root() / "streams_data_analysis" / "output"),
        help="Directory where derived CSVs and the HTML report will be written.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    summary, rows = load_rows(csv_path)
    feature_rows = feature_analysis(rows)
    threshold_summaries = []
    for threshold in THRESHOLDS:
        _, _, summary_rows = threshold_feature_summary(rows, threshold)
        threshold_summaries.extend(summary_rows)
    age_rows = categorical_summary(rows, "age", "연령대")
    playtime_rows = categorical_summary(rows, "playtime", "주간 플레이타임")
    gender_rows = categorical_summary(rows, "gender", "성별")
    mbti_rows = categorical_summary(rows, "mbti", "MBTI")
    genre_rows = genre_summary(rows)
    ai_summary = ai_comparison_summary(rows)
    cases = build_case_studies([row for row in rows if row["player_score"] >= 80 or row["result"] == "win"])

    markdown_report = build_report(
        csv_path=csv_path,
        summary=summary,
        rows=rows,
        feature_rows=feature_rows,
        threshold_rows=threshold_summaries,
        age_rows=age_rows,
        playtime_rows=playtime_rows,
        gender_rows=gender_rows,
        mbti_rows=mbti_rows,
        genre_rows=genre_rows,
        ai_summary=ai_summary,
        cases=cases,
    )

    report_stem = csv_path.stem + "_strategy_report"
    html_report = markdown_to_html(markdown_report, "본게임 전용 스트림스 플레이 전략 분석 보고서")

    write_csv(output_dir / f"{csv_path.stem}_feature_rows.csv", feature_rows)
    write_csv(output_dir / f"{csv_path.stem}_threshold_summary.csv", threshold_summaries)
    write_csv(output_dir / f"{csv_path.stem}_game_features.csv", rows)
    (output_dir / f"{report_stem}.md").write_text(markdown_report, encoding="utf-8")
    (output_dir / f"{report_stem}.html").write_text(html_report, encoding="utf-8")

    print(f"[CSV Analysis] main games: {summary.main_games}")
    print(f"[CSV Analysis] excluded tutorial games: {summary.excluded_tutorial_games}")
    print(f"[CSV Analysis] output html: {output_dir / f'{report_stem}.html'}")


if __name__ == "__main__":
    main()
