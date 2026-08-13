"""Shared prompt, schema, and prediction parser for VARM-Bench."""

from __future__ import annotations

import re
from typing import Any, Mapping


FIELDS = (
    "text",
    "target",
    "target_type",
    "target_explicitness",
    "stance",
    "label",
    "category",
    "cot",
)
CATEGORIES = (
    "非攻击",
    "一般辱骂",
    "地域族群攻击",
    "性别攻击",
    "性少数攻击",
    "阶层职业攻击",
    "身心疾病攻击",
)
TARGET_TYPES = ("单一对象", "群体对象", "无明确对象")
TARGET_EXPLICITNESS = ("明示", "隐含", "无明确对象")
STANCES = ("攻击或认同", "反对攻击", "引用或转述", "中性提及")
LABELS = ("有害", "无害")
ANCHORS = ("T", "TY", "TT", "S", "L", "C")
ANCHOR_RE = re.compile(r"\[(TY|TT|T|S|L|C):([^\]\r\n]+)\]")

SYSTEM_PROMPT = "你是中文网络攻击性言论识别助手。"
USER_TEMPLATE = """作为中文网络攻击性言论检测专家，请分析以下文本。

类别包括：非攻击、一般辱骂、地域族群攻击、性别攻击、性少数攻击、阶层职业攻击、身心疾病攻击。

请用80-180个字符（含锚点）完成一段连贯推理，必须以“分析：”开头，不要换行、分点、JSON、Markdown或额外说明。先用完整句子说明文本指向谁或什么对象、对象如何出现，再判断作者立场和有害性，最后解释类别依据；不要先罗列锚点，不能只罗列锚点或类别。

分析中必须使用半角方括号“[]”和半角冒号“:”，恰好包含六个锚点，顺序固定为T、TY、TT、S、L、C，每个只出现一次。每个锚点都要自然放在对应判断句中；任何两个相邻锚点之间都必须有说明判断关系的中文词句，不能只用空格、标点或箭头连接，也不能在正文末尾集中追加六个锚点。

组织句式可参考：分析：原文指向[T:{{对象}}]，从范围看属于[TY:{{对象类型}}]，对象以[TT:{{显隐性}}]方式出现。结合上下文，作者是在[S:{{立场}}]，所以判断为[L:{{有害性}}]；攻击依据或非攻击理由表明类别是[C:{{类别}}]。只参考锚点嵌入方式，必须根据待分析文本重新表述并填写实际值。
- [T:...]填写原文主要描述、评价、反驳或攻击的最短可识别对象；对象可以是个人、群体、事物或议题。例如对象是“这部电影”时写[T:这部电影]。确实没有对象时写[T:无明确对象]。
- [TY:...]只能写[TY:单一对象]、[TY:群体对象]或[TY:无明确对象]。
- [TT:...]只能写[TT:明示]、[TT:隐含]或[TT:无明确对象]。
- [S:...]只能写[S:攻击或认同]、[S:反对攻击]、[S:引用或转述]或[S:中性提及]。
- [L:...]只能写[L:有害]或[L:无害]。
- [C:...]只能写[C:非攻击]、[C:一般辱骂]、[C:地域族群攻击]、[C:性别攻击]、[C:性少数攻击]、[C:阶层职业攻击]或[C:身心疾病攻击]。

逻辑必须一致：[L:无害]只能对应[C:非攻击]；六类攻击必须对应[L:有害]和[S:攻击或认同]。当[T:无明确对象]时，[TY:无明确对象]和[TT:无明确对象]必须同时成立。冒号后必须填写本条文本的实际值，不得照抄省略号，也不得把“主要对象、对象类型、对象显隐性、作者立场、是否有害、最终类别”写进锚点。

待分析文本：{text}"""

KNOWLEDGE = """判定与类别边界知识：
先判断作者是否实施或认同攻击；反对攻击、引用或转述、中性提及及针对行为、作品或议题的正常批评均不构成作者攻击，不能仅因出现身份词或攻击词就判为攻击。
判为有害后，以攻击成立所依据的属性分类，而不是按文本中出现的词分类；未以身份属性为攻击依据的人身辱骂、人格贬损或威胁归入一般辱骂。
非攻击：作者没有实施或认同辱骂、贬损、歧视、排斥或威胁。
一般辱骂：未以特定身份属性为攻击依据的一般人身攻击。
地域族群攻击：以地域、籍贯、民族、族裔、国籍或文化身份为依据的攻击。
性别攻击：以性别或性别角色为依据的贬损、排斥或刻板泛化。
性少数攻击：以性取向或性别认同为依据的贬损、污名化或排斥。
阶层职业攻击：以职业、收入、教育、阶层或社会角色为依据的攻击。
身心疾病攻击：以外貌、身体状况、疾病、心理状态或残障为依据的攻击。
涉及多个属性时，选择直接支撑主要贬损结论的核心属性。"""


def normalize_assistant_cot(value: Any) -> str:
    cot = str(value or "").strip()
    if not cot:
        raise ValueError("cot is empty")
    return cot if cot.startswith("分析：") else f"分析：{cot}"


def build_messages(
    row: Mapping[str, Any],
    *,
    include_answer: bool,
    prompt_variant: str = "base",
) -> list[dict[str, str]]:
    text = " ".join(str(row.get("text") or "").split()).strip()
    if not text:
        raise ValueError("text is empty")
    if prompt_variant not in {"base", "knowledge"}:
        raise ValueError(f"unknown prompt variant: {prompt_variant}")
    user_prompt = USER_TEMPLATE.format(text=text)
    if prompt_variant == "knowledge":
        user_prompt = f"{KNOWLEDGE}\n\n{user_prompt}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if include_answer:
        messages.append(
            {"role": "assistant", "content": normalize_assistant_cot(row.get("cot"))}
        )
    return messages


def parse_prediction(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    matches = list(ANCHOR_RE.finditer(raw))
    names = [match.group(1) for match in matches]
    values = {match.group(1): match.group(2).strip() for match in matches}
    anchor_gap_chinese_counts = [
        len(re.findall(r"[\u3400-\u9fff]", raw[left.end() : right.start()]))
        for left, right in zip(matches, matches[1:])
    ]
    anchor_schema_valid = names == list(ANCHORS)
    if anchor_schema_valid:
        anchor_schema_valid = (
            values.get("TY") in TARGET_TYPES
            and values.get("TT") in TARGET_EXPLICITNESS
            and values.get("S") in STANCES
            and values.get("L") in LABELS
            and values.get("C") in CATEGORIES
            and bool(values.get("T"))
        )

    format_issues: list[str] = []
    if not anchor_schema_valid:
        format_issues.append("anchor_sequence_or_value_invalid")
    elif any(count == 0 for count in anchor_gap_chinese_counts):
        format_issues.append("anchors_not_naturally_integrated")
    if "\n" in raw or "\r" in raw:
        format_issues.append("multiple_paragraphs")
    if not raw.startswith("分析："):
        format_issues.append("missing_analysis_prefix")
    if not 80 <= len(raw) <= 180:
        format_issues.append("length_out_of_range")
    reasoning_text = ANCHOR_RE.sub("", raw)
    if len(re.findall(r"[\u3400-\u9fff]", reasoning_text)) < 30:
        format_issues.append("insufficient_reasoning_text")
    if raw.startswith(("{", "[", "```", "#", "- ")):
        format_issues.append("forbidden_wrapper")

    placeholder_values = {
        "...",
        "…",
        "主要对象",
        "实际对象",
        "对象文本",
        "对象类型",
        "对象显隐性",
        "作者立场",
        "是否有害",
        "最终类别",
    }
    if any(value in placeholder_values for value in values.values()):
        format_issues.append("placeholder_value")

    if anchor_schema_valid:
        if (values["L"] == "无害") != (values["C"] == "非攻击"):
            format_issues.append("label_category_inconsistent")
        no_target = values["T"] == "无明确对象"
        if not (
            no_target
            == (values["TY"] == "无明确对象")
            == (values["TT"] == "无明确对象")
        ):
            format_issues.append("no_target_fields_inconsistent")
        if (values["L"] == "有害") != (values["S"] == "攻击或认同"):
            format_issues.append("stance_label_inconsistent")

    schema_valid = anchor_schema_valid and not format_issues

    category = values.get("C", "") if values.get("C") in CATEGORIES else ""
    label = values.get("L", "") if values.get("L") in LABELS else ""

    return {
        "schema_valid": schema_valid,
        "anchor_schema_valid": anchor_schema_valid,
        "format_issues": format_issues,
        "output_char_count": len(raw),
        "target": values.get("T", ""),
        "target_type": values.get("TY", ""),
        "target_explicitness": values.get("TT", ""),
        "stance": values.get("S", ""),
        "label": label,
        "category": category,
        "anchor_order": names,
        "anchor_gap_chinese_counts": anchor_gap_chinese_counts,
        "raw": raw,
    }


def validate_record(row: Mapping[str, Any]) -> None:
    missing = [field for field in FIELDS if field not in row]
    extra = [field for field in row if field not in FIELDS]
    if missing or extra:
        raise ValueError(f"schema mismatch: missing={missing}, extra={extra}")
    parsed = parse_prediction(normalize_assistant_cot(row["cot"]))
    if not parsed["schema_valid"]:
        raise ValueError("cot does not satisfy the six-anchor schema")
    for field in ("target", "target_type", "target_explicitness", "stance", "label", "category"):
        expected = row[field]
        if field == "target" and not str(expected).strip() and row["target_type"] == "无明确对象":
            expected = "无明确对象"
        if parsed[field] != expected:
            raise ValueError(f"cot/{field} mismatch: {parsed[field]!r} != {expected!r}")
