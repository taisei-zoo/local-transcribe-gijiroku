# prompts.py
# -*- coding: utf-8 -*-
"""V3用プロンプトテンプレート。

V4の考え方を取り込みつつ、V3では標準生成タイプを3つに整理する。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GenType(str, Enum):
    POINT_SUMMARY_LONG = "要点要約（長）"
    MINUTES = "議事録"
    GENERAL_SUMMARY = "汎用要約（自由指示）"

    # 互換用。GUIでは標準表示しないが、過去運用から呼ばれても落ちないよう残す。
    CORRECT_FULL = "誤認識補正（全文）"
    SUMMARY_SHORT = "要約（短）"
    SUMMARY_MID = "要約（中）"
    DECISIONS_TODOS = "決定事項/ToDo"


ALIASES = {
    "要約（長）": GenType.POINT_SUMMARY_LONG.value,
    "議事録（整形）": GenType.MINUTES.value,
    "汎用要約": GenType.GENERAL_SUMMARY.value,
    "自由要約": GenType.GENERAL_SUMMARY.value,
}


@dataclass(frozen=True)
class PromptPack:
    system: str
    user: str
    used_instruction: str


def normalize_gen_type_label(label: str | None) -> str:
    raw = (label or "").strip() or GenType.POINT_SUMMARY_LONG.value
    return ALIASES.get(raw, raw)


def resolve_gen_type(label: str | None) -> GenType:
    normalized = normalize_gen_type_label(label)
    try:
        return GenType(normalized)
    except Exception:
        return GenType.POINT_SUMMARY_LONG


_SYSTEM_MEETING = """あなたは自治体内部で利用されるローカルLLMです。
入力テキストは、音声文字起こし、既存テキスト、Word・Excel・PDFから抽出された資料テキストを結合したものです。

重要な制約:
・入力に書かれていない事実を推測して補わないでください。
・日付、場所、出席者、担当者、期限などが明示されていない場合は「記載なし」または「不明」としてください。
・決定事項、確認事項、保留事項、今後の対応を混同しないでください。
・資料テキストと発言内容が食い違う場合は、断定せず「資料上は〜、発言では〜」のように区別してください。
・音声認識の誤りと思われる箇所は、参考固有名詞や文脈に基づき補正してよいです。ただし、不確かな補正は断定しないでください。
・明らかな重複、言い淀み、つなぎ言葉、雑談は内容が変わらない範囲で整理してよいです。
・出力は日本語で、自治体内部文書・会議録として読みやすい形式にしてください。
・箇条書きは「・」を基本にしてください。
""".strip()

_SYSTEM_GENERAL = """あなたは自治体内部で利用されるローカルLLMです。
入力テキストを、ユーザーの指示に従って整理してください。
入力に書かれていない事実は推測せず、不明な事項は「不明」または「記載なし」としてください。
""".strip()

_SYSTEM_CORRECT_FULL = """あなたは音声文字起こしの誤認識補正の専門家です。
要約・削除・再構成をせず、原文の内容を保ったまま、音声起因の誤認識のみ補正してください。
入力にない事実は補わないでください。
""".strip()


def _reference_block(terms: list[str] | None) -> str:
    items = [x.strip() for x in (terms or []) if x and x.strip()]
    if not items:
        return ""
    items = items[:200]
    lines = [
        "【参考固有名詞】",
        "以下は会議・資料で出やすい固有名詞です。音声認識の誤りや表記ゆれがある場合、参考にしてください。",
    ]
    lines.extend([f"・{x}" for x in items])
    return "\n".join(lines).strip() + "\n\n"


def default_instruction(gen_type: GenType | str, meeting_title: str | None = None) -> str:
    if not isinstance(gen_type, GenType):
        gen_type = resolve_gen_type(str(gen_type))
    title = (meeting_title or "").strip()
    mt = f"（会議名：{title}）" if title else ""

    if gen_type == GenType.POINT_SUMMARY_LONG:
        return """【タスク】要点要約（長）
以下の入力テキストをもとに、会議で話された内容、および添付資料から読み取れる重要事項を、論点ごとに詳しく整理してください。

目的:
・会議内容・資料内容の要点を、できるだけ取りこぼさず把握できる詳しい要約にする。
・短くまとめることを優先せず、重要な説明、意見、質疑、決定事項、保留事項、今後の対応を落とさない。
・逐語録ではなく、読みやすい要点整理にする。

出力形式:
要点要約（長）

１　全体概要
・会議または資料全体の趣旨を簡潔に整理する。

２　主な論点
【論点1】〇〇について
・説明、意見、質疑、確認事項などを整理する。

３　決定事項・確認事項
・決定または確認された内容を記載する。
・明確な決定がない場合は「明確な決定事項は確認できない」と記載する。

４　保留事項・今後の対応
・継続確認が必要な事項、今後の作業、担当、期限を整理する。
・担当や期限が不明な場合は「担当：記載なし」「期限：記載なし」とする。

５　資料から読み取れる補足
・添付資料から読み取れる重要事項がある場合に記載する。
・該当がない場合は「特記事項なし」とする。
""".strip()

    if gen_type == GenType.MINUTES:
        return f"""【タスク】議事録{mt}
以下の入力テキストをもとに、自治体内部会議の議事録の叩き台を作成してください。

目的:
・会議全体の流れ、議題、説明内容、主な意見、質疑、決定事項、保留事項、今後の対応を整理する。
・単なる短い要約ではなく、後から人間が編集して正式な議事録に近づけられる材料にする。
・発言の逐語録ではなく、議題ごとに読みやすく整理した議事録形式にする。

重要な制約:
・入力に書かれていない事実を推測して補わない。
・担当者、期限、出席者、日付、場所などが明示されていない場合は「記載なし」または「不明」とする。
・決定事項と未決事項を混同しない。
・資料テキストと発言内容が食い違う場合は、断定せず区別して記載する。

出力形式:
議事録

１　会議概要
・会議名：
・日時：
・場所：
・出席者：
・作成対象資料：

２　議題
【議題1】〇〇について
【議題2】〇〇について

３　議事内容
【議題1】〇〇について
（1）説明・報告内容
・説明された内容を整理する。

（2）主な意見・質疑
・主な意見、質問、回答を整理する。

（3）確認された内容
・会議内で確認された内容を整理する。

（4）決定事項・方向性
・決定した内容、今後の方向性を整理する。
・明確な決定がない場合は「明確な決定事項は確認できない」と記載する。

（5）保留事項・課題
・保留となった事項、今後確認が必要な事項を整理する。

４　決定事項
・決定事項を整理する。
・該当がない場合は「明確な決定事項は確認できない」と記載する。

５　今後の対応・ToDo
・対応内容：
　担当：記載なし
　期限：記載なし

６　保留事項・確認事項
・継続確認が必要な事項を整理する。

７　資料から読み取れる補足
・資料から読み取れる重要事項がある場合に記載する。
""".strip()

    if gen_type == GenType.GENERAL_SUMMARY:
        return """以下の入力テキストを、指定された目的に沿って整理してください。

目的:
・内容を読みやすく要約する。
・重要な論点、事実関係、決定事項、課題、今後の対応が分かるように整理する。
・入力に書かれていない事実は推測して補わない。
・不明な点は「不明」または「記載なし」とする。

出力形式:
要約

１　概要
・...

２　主な内容
・...

３　重要な論点
・...

４　今後確認すべき事項
・...
""".strip()

    if gen_type == GenType.CORRECT_FULL:
        return """【タスク】誤認識補正（全文）
要約・削除・再構成は禁止です。原文の内容を保ったまま、音声起因の誤認識のみ補正してください。
""".strip()

    if gen_type == GenType.SUMMARY_SHORT:
        return """入力テキストを5行以内で要約してください。結論・重要点を先に書いてください。""".strip()

    if gen_type == GenType.SUMMARY_MID:
        return """入力テキストを10〜15行程度で要約してください。論点、結論、理由、次アクションを整理してください。""".strip()

    if gen_type == GenType.DECISIONS_TODOS:
        return """入力テキストから、決定事項とToDoを分けて抽出してください。担当者・期限が明示されている場合のみ記載してください。""".strip()

    return "入力テキストを、重要点が分かるように要約してください。入力にない事実は推測しないでください。"


def build_prompt(
    gen_type: GenType | str,
    input_text: str,
    *,
    meeting_title: str | None = None,
    terms: list[str] | None = None,
    custom_instruction: str | None = None,
) -> PromptPack:
    if not isinstance(gen_type, GenType):
        gen_type = resolve_gen_type(str(gen_type))

    instruction = (custom_instruction or "").strip() or default_instruction(gen_type, meeting_title)
    ref = _reference_block(terms)
    system = _SYSTEM_GENERAL if gen_type == GenType.GENERAL_SUMMARY else _SYSTEM_MEETING
    if gen_type == GenType.CORRECT_FULL:
        system = _SYSTEM_CORRECT_FULL

    user = f"""{ref}【AIへの指示】
{instruction}

------------------
【入力テキスト】
{(input_text or '').strip()}
"""
    return PromptPack(system=system, user=user, used_instruction=instruction)


def template_for_label(label: str) -> str:
    return default_instruction(resolve_gen_type(label))
