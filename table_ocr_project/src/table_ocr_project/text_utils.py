from __future__ import annotations

import re
from difflib import SequenceMatcher, get_close_matches
from typing import Dict, Iterable, List, Optional, Sequence


def normalize_text(text: str) -> str:
    if text is None:
        return ''
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = text.replace('：', ':')
    text = text.replace('|', 'I')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_time_string(text: str) -> str:
    text = normalize_text(text)
    text = text.replace('.', ':').replace('；', ':').replace(';', ':')
    text = re.sub(r'[^0-9:]', '', text)
    if re.fullmatch(r'\d{3,4}', text):
        if len(text) == 3:
            text = '0' + text
        return f'{text[:2]}:{text[2:]}'
    m = re.search(r'(\d{1,2}):(\d{1,2})', text)
    if m:
        return f'{int(m.group(1)):02d}:{int(m.group(2)):02d}'
    return text


def extract_first_date(text: str) -> str:
    text = normalize_text(text)
    m = re.search(r'(20\d{2})[./年-](\d{1,2})[./月-](\d{1,2})', text)
    if not m:
        return ''
    y, mo, d = m.groups()
    return f'{int(y):04d}/{int(mo):02d}/{int(d):02d}'


def best_fuzzy_match(text: str, candidates: Sequence[str], min_score: float = 0.55) -> str:
    text = normalize_text(text)
    if not text or not candidates:
        return text
    direct = get_close_matches(text, list(candidates), n=1, cutoff=min_score)
    if direct:
        return direct[0]
    best = text
    best_score = min_score
    for cand in candidates:
        score = SequenceMatcher(None, text, cand).ratio()
        if score > best_score:
            best = cand
            best_score = score
    return best


def correct_with_lexicon(text: str, lexicon: Dict[str, List[str]], field: str) -> str:
    if not text:
        return ''
    candidates = lexicon.get(field, [])
    return best_fuzzy_match(text, candidates) if candidates else text


def flatten_region_lines(items: Sequence[Dict]) -> List[str]:
    lines: List[str] = []
    for item in items:
        text = normalize_text(str(item.get('text', '')))
        if text:
            lines.append(text)
    return lines


def search_time_after_label(text: str, label: str) -> str:
    text = normalize_text(text)
    m = re.search(re.escape(label) + r'\s*[:：]?\s*([0-9OIl]{1,2}[:：.]?[0-9OIl]{2})', text)
    if m:
        cand = m.group(1).replace('O', '0').replace('I', '1').replace('l', '1')
        return normalize_time_string(cand)

    # Fuzzy fallback for OCR label drift such as "天风时刻" -> "天黑时刻".
    best_time = ''
    best_score = 0.0
    for raw_label, raw_time in re.findall(r'([^\s:]{2,8})\s*[:：]?\s*([0-9OIl]{1,2}[:：.]?[0-9OIl]{2})', text):
        score = SequenceMatcher(None, normalize_text(raw_label), label).ratio()
        if score >= 0.72 and score > best_score:
            best_score = score
            best_time = raw_time.replace('O', '0').replace('I', '1').replace('l', '1')

    return normalize_time_string(best_time) if best_time else ''


def extract_after_label(text: str, label: str) -> str:
    text = normalize_text(text)
    m = re.search(re.escape(label) + r'\s*[:：]?\s*(\S+)', text)
    return normalize_text(m.group(1)) if m else ''
