# V3_main.py
# -*- coding: utf-8 -*-
"""ローカル文字起こし・要約ツール V3（端末1台版）。

構成方針:
- V3はGUIでその場実行する単体版。
- V4共有フォルダー版のキュー処理、_UPLOAD_DONE.txt、server.lock、処理中/完了/失敗フォルダー管理は入れない。
- V4由来の改善は「文書前処理」「LLM上限・timeout」「プロンプト整理」だけ取り込む。
"""

from __future__ import annotations

import json
import os
import site
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover
    WhisperModel = None  # type: ignore

try:
    from pykakasi import kakasi
    _KAKASI = kakasi()
    _KAKASI.setMode("J", "H")
    _KAKASI.setMode("K", "H")
    _KAKASI.setMode("H", "H")
    _CONV = _KAKASI.getConverter()
except Exception:  # pragma: no cover
    _CONV = None

from document_preprocessor import (
    DOCUMENT_EXTENSIONS,
    TEXT_EXTENSIONS,
    preprocess_file,
    safe_filename,
    write_extracted_text,
)
from lmstudio_client import LMStudioClient
from prompts import GenType, build_prompt, resolve_gen_type, template_for_label


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".mp4", ".mov", ".mkv"}
MAX_TERMS = 500
MODEL_CACHE: dict[str, Any] = {}
IS_RUNNING = False

DEFAULT_CONFIG: dict[str, Any] = {
    "whisper": {
        "model_size": "medium",
        "device": "auto",
        "compute_type": "auto",
        "language": "ja",
        "beam_size": 5,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "timeout_sec": 3600,
        "temperature": 0.2,
        "model_label": "gpt-oss-20b",
        "model_id": "openai/gpt-oss-20b",
        "model_catalog": {
            "gpt-oss-20b": "openai/gpt-oss-20b",
            "gemma-4-e4b": "google/gemma-4-e4b",
            "gemma-4-12b": "google/gemma-4-12b",
            "gemma-3-12b": "google/gemma-3-12b",
        },
        "max_tokens_default": 90000,
        "max_tokens_map": {
            "要点要約（長）": 90000,
            "議事録": 90000,
            "汎用要約（自由指示）": 90000,
            "誤認識補正（全文）": 90000,
            "要約（短）": 800,
            "要約（中）": 1800,
            "決定事項/ToDo": 2500,
        },
        "context_tokens": 262144,
        "input_max_estimated_tokens": 150000,
        "prompt_buffer_tokens": 22000,
        "too_long_behavior": "send_anyway",
    },
    "document_extract": {
        "word_extract_tables_as_markdown": True,
        "excel_max_sheets_per_book": 30,
        "excel_max_rows_per_sheet": 5000,
        "excel_max_cols_per_sheet": 80,
        "excel_max_cells_per_sheet": 80000,
        "excel_max_total_cells_per_book": 200000,
        "excel_max_cell_chars": 2000,
        "excel_include_hidden_sheets": False,
        "pdf_min_extracted_chars": 100,
    },
    "output": {
        "write_plain_text": True,
        "write_timestamped_text": True,
        "write_extracted_text": True,
        "write_combined_input_text": False,
        "write_used_prompt": False,
        "write_segments_csv": False,
        "write_process_result_json": False,
    },
}


def add_nvidia_bins_to_path() -> int:
    """Windows + pip版CUDAライブラリ用のDLLパス補助。"""
    candidates = []
    try:
        candidates += site.getsitepackages()
    except Exception:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass

    subdirs = [
        ("nvidia", "cublas", "bin"),
        ("nvidia", "cudnn", "bin"),
        ("nvidia", "cuda_runtime", "bin"),
        ("nvidia", "cuda_nvrtc", "bin"),
        ("nvidia", "curand", "bin"),
        ("nvidia", "cufft", "bin"),
    ]

    added = 0
    for sp in candidates:
        if not sp:
            continue
        for parts in subdirs:
            p = os.path.join(sp, *parts)
            if os.path.isdir(p):
                current = os.environ.get("PATH", "")
                if p not in current:
                    os.environ["PATH"] = p + os.pathsep + current
                    added += 1
    return added


NVIDIA_PATH_ADDED = add_nvidia_bins_to_path()


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config() -> dict:
    path = Path(__file__).with_name("config.json")
    if not path.exists():
        return DEFAULT_CONFIG.copy()
    try:
        user_cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(user_cfg, dict):
            return deep_merge(DEFAULT_CONFIG, user_cfg)
    except Exception as e:
        print(f"[WARN] config.json の読み込みに失敗しました: {e}")
    return DEFAULT_CONFIG.copy()


CONFIG = load_config()


def guess_hiragana(text: str) -> str:
    text = text.strip()
    if not text or _CONV is None:
        return ""
    return _CONV.do(text)


def append_log(text_widget: tk.Text, msg: str) -> None:
    text_widget.insert(tk.END, msg + "\n")
    text_widget.see(tk.END)


def collect_list(listbox: tk.Listbox) -> list[str]:
    return [listbox.get(i) for i in range(listbox.size())]


def build_initial_prompt(terms: list[str], persons: list[str]) -> str | None:
    parts: list[str] = []
    if terms:
        parts.append("用語: " + "、".join(terms[:200]))
    if persons:
        parts.append("人名: " + "、".join(persons[:200]))
    return " / ".join(parts) if parts else None


def update_status(
    status_label: tk.Label,
    model_var: tk.StringVar,
    terms_listbox: tk.Listbox,
    persons_listbox: tk.Listbox,
    device_pref_var: tk.StringVar,
    suffix: str = "",
) -> None:
    base = (
        f"Whisper: {model_var.get()} ｜ 実行: {device_pref_var.get()} ｜ "
        f"用語: {terms_listbox.size()}/{MAX_TERMS} ｜ 人名: {persons_listbox.size()}/{MAX_TERMS}"
    )
    status_label.config(text=base + (f"（{suffix}）" if suffix else ""))


def get_model(model_size: str, device_pref: str = "auto"):
    if WhisperModel is None:
        raise RuntimeError("faster-whisper が見つかりません。requirements.txt を参照してインストールしてください。")

    model_size = (model_size or "medium").strip()
    device_pref = (device_pref or "auto").strip().lower()

    if device_pref == "cpu":
        candidates = [("cpu", "int8")]
    elif device_pref == "gpu":
        candidates = [("cuda", "float16")]
    else:
        candidates = [("cuda", "float16"), ("cpu", "int8")]

    last_err: Exception | None = None
    for device, compute_type in candidates:
        key = f"{model_size}|{device}|{compute_type}"
        if key in MODEL_CACHE:
            return MODEL_CACHE[key]
        try:
            print(f"[INFO] Whisper model loading: {model_size} device={device} compute_type={compute_type}")
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            setattr(model, "_loaded_device", device)
            setattr(model, "_loaded_compute_type", compute_type)
            MODEL_CACHE[key] = model
            return model
        except Exception as e:
            last_err = e
            print(f"[WARN] Whisper model load failed: device={device} compute_type={compute_type} err={e}")
            if device_pref == "gpu":
                raise RuntimeError("GPU優先（固定）が選択されていますが、CUDAが利用できませんでした。自動またはCPU優先にしてください。") from e
    raise RuntimeError(f"Whisperモデルのロードに失敗しました: {last_err}") from last_err


def transcribe_audio_file(model: Any, audio_path: Path, initial_prompt: str | None) -> tuple[str, str, list[dict], dict]:
    t0 = time.perf_counter()
    whisper_cfg = CONFIG.get("whisper", {})
    segments, info = model.transcribe(
        str(audio_path),
        language=str(whisper_cfg.get("language", "ja")),
        beam_size=int(whisper_cfg.get("beam_size", 5)),
        initial_prompt=initial_prompt,
    )

    timestamped_lines: list[str] = []
    plain_lines: list[str] = []
    segment_rows: list[dict] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        timestamped_lines.append(f"[{seg.start:.1f} - {seg.end:.1f}] {text}")
        plain_lines.append(text)
        segment_rows.append({"start": float(seg.start), "end": float(seg.end), "text": text})

    elapsed = time.perf_counter() - t0
    meta = {
        "elapsed_sec": elapsed,
        "duration_sec": getattr(info, "duration", None),
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
    }
    return "\n".join(timestamped_lines), "\n".join(plain_lines), segment_rows, meta


def estimate_tokens_japanese(text: str) -> int:
    # PoC用のざっくり推定。日本語中心なので 1文字≒1トークン として安全側に見る。
    return len(text or "")


def resolve_lm_model(model_label: str) -> tuple[str, str]:
    lm_cfg = CONFIG.get("lmstudio", {})
    catalog = dict(lm_cfg.get("model_catalog", {}) or {})
    label = (model_label or lm_cfg.get("model_label") or "").strip()
    if not label:
        label = "gpt-oss-20b"
    explicit = str(lm_cfg.get("model_id", "")).strip()
    if explicit and label == str(lm_cfg.get("model_label", "")).strip():
        return label, explicit
    return label, catalog.get(label, label)


def max_tokens_for(gen_type_label: str) -> int:
    lm_cfg = CONFIG.get("lmstudio", {})
    mp = dict(lm_cfg.get("max_tokens_map", {}) or {})
    default = int(lm_cfg.get("max_tokens_default", 90000))
    return int(mp.get(gen_type_label, default))


def gen_output_suffix(gen_type_label: str) -> tuple[str, str]:
    label = resolve_gen_type(gen_type_label).value
    mapping = {
        GenType.POINT_SUMMARY_LONG.value: ("要点要約（長）", ".txt"),
        GenType.MINUTES.value: ("議事録", ".txt"),
        GenType.GENERAL_SUMMARY.value: ("汎用要約", ".txt"),
        GenType.CORRECT_FULL.value: ("文字起こし_補正済全文", ".md"),
        GenType.SUMMARY_SHORT.value: ("要約（短）", ".txt"),
        GenType.SUMMARY_MID.value: ("要約（中）", ".txt"),
        GenType.DECISIONS_TODOS.value: ("決定事項ToDo", ".md"),
    }
    return mapping.get(label, (safe_filename(label), ".txt"))


def build_combined_input(transcribed_parts: list[tuple[str, str]], document_parts: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    if transcribed_parts:
        parts.append("=" * 80)
        parts.append("音声文字起こし")
        parts.append("=" * 80)
        for name, text in transcribed_parts:
            parts.append(f"\n【音声ファイル】{name}")
            parts.append(text.strip())
    if document_parts:
        parts.append("\n" + "=" * 80)
        parts.append("資料・既存テキスト")
        parts.append("=" * 80)
        for name, text in document_parts:
            parts.append(f"\n【資料ファイル】{name}")
            parts.append(text.strip())
    return "\n".join(parts).strip()


def run_lmstudio(
    *,
    input_text: str,
    gen_type_label: str,
    meeting_title: str,
    model_label: str,
    terms: list[str],
    custom_instruction: str,
) -> tuple[bool, str, str, list[str], str]:
    gen_type = resolve_gen_type(gen_type_label)
    pack = build_prompt(
        gen_type,
        input_text,
        meeting_title=meeting_title,
        terms=terms,
        custom_instruction=custom_instruction.strip() or None,
    )

    lm_cfg = CONFIG.get("lmstudio", {})
    base_url = str(lm_cfg.get("base_url", "http://localhost:1234/v1")).strip()
    timeout_sec = float(lm_cfg.get("timeout_sec", 3600))
    temperature = float(lm_cfg.get("temperature", 0.2))
    max_tokens = max_tokens_for(gen_type.value)
    _label, model_id = resolve_lm_model(model_label)

    estimated = estimate_tokens_japanese(input_text)
    input_limit = int(lm_cfg.get("input_max_estimated_tokens", 150000))
    context_tokens = int(lm_cfg.get("context_tokens", 262144))
    prompt_buffer = int(lm_cfg.get("prompt_buffer_tokens", 22000))
    too_long_behavior = str(lm_cfg.get("too_long_behavior", "send_anyway"))
    warnings: list[str] = []

    if estimated > input_limit:
        msg = f"推定入力トークン数が推奨上限を超えています。推定={estimated} / 推奨上限={input_limit} / 挙動={too_long_behavior}"
        warnings.append(msg)
        if too_long_behavior == "skip_postprocess":
            return True, "", "", warnings, pack.used_instruction
        if too_long_behavior == "fail_job":
            return False, "", msg, warnings, pack.used_instruction

    if estimated + max_tokens + prompt_buffer > context_tokens:
        warnings.append(
            "推定上、入力+出力上限+プロンプト余白がコンテキスト長を超える可能性があります。"
            f" context={context_tokens}, input={estimated}, output={max_tokens}, buffer={prompt_buffer}"
        )

    client = LMStudioClient(
        base_url=base_url,
        api_key=str(lm_cfg.get("api_key", "lm-studio")),
        timeout_sec=timeout_sec,
    )
    res = client.generate(
        model=model_id,
        system_prompt=pack.system,
        user_prompt=pack.user,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
    )
    if not res.ok:
        return False, "", res.error, warnings, pack.used_instruction
    return True, res.text, "", warnings, pack.used_instruction


def categorize_files(files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    audio_files: list[Path] = []
    text_files: list[Path] = []
    doc_files: list[Path] = []
    for p in files:
        ext = p.suffix.lower()
        if ext in AUDIO_EXTENSIONS:
            audio_files.append(p)
        elif ext in TEXT_EXTENSIONS:
            text_files.append(p)
        elif ext in DOCUMENT_EXTENSIONS:
            doc_files.append(p)
    return audio_files, text_files, doc_files


def run_job(
    *,
    files: list[Path],
    out_parent: Path,
    text_widget: tk.Text,
    status_label: tk.Label,
    model_var: tk.StringVar,
    device_pref_var: tk.StringVar,
    terms_listbox: tk.Listbox,
    persons_listbox: tk.Listbox,
    root: tk.Tk,
    use_ai: bool,
    gen_type_label: str,
    lm_model_label: str,
    meeting_title: str,
    custom_instruction: str,
) -> None:
    global IS_RUNNING
    try:
        title = (meeting_title or "").strip()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_stem = safe_filename(title or "V3処理")
        out_dir = out_parent / f"{job_stem}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        transcribe_dir = out_dir / "文字起こし"
        extract_dir = out_dir / "抽出テキスト"

        terms = collect_list(terms_listbox)
        persons = collect_list(persons_listbox)
        all_terms = terms + persons
        initial_prompt = build_initial_prompt(terms, persons)

        audio_files, text_files, doc_files = categorize_files(files)
        unsupported = [p for p in files if p not in audio_files and p not in text_files and p not in doc_files]

        root.after(0, lambda: text_widget.delete("1.0", tk.END))
        root.after(0, lambda: append_log(text_widget, f"出力先: {out_dir}"))
        root.after(0, lambda: append_log(text_widget, f"対象: 音声/動画 {len(audio_files)} 件、txt/md {len(text_files)} 件、Word/Excel/PDF {len(doc_files)} 件"))
        if unsupported:
            root.after(0, lambda: append_log(text_widget, f"⚠ 非対応ファイル {len(unsupported)} 件はスキップします。"))
            for p in unsupported:
                root.after(0, lambda name=p.name: append_log(text_widget, f"  - {name}"))

        root.after(0, lambda: update_status(status_label, model_var, terms_listbox, persons_listbox, device_pref_var, "実行中…"))

        transcribed_parts: list[tuple[str, str]] = []
        document_parts: list[tuple[str, str]] = []
        warnings: list[str] = []

        # 1. 音声文字起こし
        if audio_files:
            model_size = model_var.get()
            device_pref = device_pref_var.get()
            root.after(0, lambda: append_log(text_widget, f"Whisperモデル読み込み: {model_size} / {device_pref}"))
            model = get_model(model_size, device_pref=device_pref)
            loaded_device = getattr(model, "_loaded_device", "N/A")
            loaded_compute = getattr(model, "_loaded_compute_type", "N/A")
            root.after(0, lambda d=loaded_device, c=loaded_compute: append_log(text_widget, f"Whisper runtime: device={d} compute_type={c}"))

            transcribe_dir.mkdir(exist_ok=True)
            for idx, audio_path in enumerate(audio_files, start=1):
                root.after(0, lambda p=audio_path, i=idx, n=len(audio_files): append_log(text_widget, f"[{i}/{n}] 文字起こし: {p.name}"))
                try:
                    timestamped, plain, _segments, meta = transcribe_audio_file(model, audio_path, initial_prompt)
                    base = safe_filename(audio_path.stem)
                    if CONFIG.get("output", {}).get("write_timestamped_text", True):
                        (transcribe_dir / f"{base}_文字起こし_秒数付き.txt").write_text(timestamped, encoding="utf-8", newline="\n")
                    if CONFIG.get("output", {}).get("write_plain_text", True):
                        (transcribe_dir / f"{base}_文字起こし.txt").write_text(plain, encoding="utf-8", newline="\n")
                    transcribed_parts.append((audio_path.name, plain))
                    dur = meta.get("duration_sec")
                    elp = meta.get("elapsed_sec")
                    dur_s = f"{dur:.1f}s" if isinstance(dur, (int, float)) else "N/A"
                    elp_s = f"{elp:.1f}s" if isinstance(elp, (int, float)) else "N/A"
                    root.after(0, lambda ds=dur_s, es=elp_s: append_log(text_widget, f"  ✅ 完了 音声長={ds} / 所要={es}"))
                except Exception as e:
                    warnings.append(f"{audio_path.name}: 文字起こし失敗: {type(e).__name__}: {e}")
                    root.after(0, lambda err=e: append_log(text_widget, f"  ❌ 文字起こし失敗: {err}"))

        # 2. txt/md/docx/xlsx/pdf前処理
        preprocess_targets = text_files + doc_files
        doc_cfg = CONFIG.get("document_extract", {})
        for idx, path in enumerate(preprocess_targets, start=1):
            root.after(0, lambda p=path, i=idx, n=len(preprocess_targets): append_log(text_widget, f"[{i}/{n}] 前処理: {p.name}"))
            try:
                result = preprocess_file(path, doc_cfg)
                warnings.extend(result.warnings)
                if result.text.strip():
                    if CONFIG.get("output", {}).get("write_extracted_text", True):
                        out_path = write_extracted_text(result, extract_dir)
                        if out_path:
                            root.after(0, lambda f=out_path: append_log(text_widget, f"  ✅ 抽出保存: {f.relative_to(out_dir)}"))
                    document_parts.append((path.name, result.text))
                else:
                    root.after(0, lambda: append_log(text_widget, "  ⚠ 抽出テキストなし"))
            except Exception as e:
                msg = f"{path.name}: 前処理失敗: {type(e).__name__}: {e}"
                warnings.append(msg)
                root.after(0, lambda m=msg: append_log(text_widget, f"  ❌ {m}"))

        combined_input = build_combined_input(transcribed_parts, document_parts)
        if CONFIG.get("output", {}).get("write_combined_input_text", False) and combined_input:
            (out_dir / "AI入力_統合テキスト.txt").write_text(combined_input, encoding="utf-8", newline="\n")

        if warnings:
            root.after(0, lambda: append_log(text_widget, "----- 警告 -----"))
            for w in warnings:
                root.after(0, lambda msg=w: append_log(text_widget, f"⚠ {msg}"))

        # 3. AI後処理
        if bool(use_ai):
            if not combined_input.strip():
                root.after(0, lambda: append_log(text_widget, "⚠ AIに渡すテキストがありません。要約はスキップしました。"))
            else:
                gen_label = gen_type_label
                suffix, ext = gen_output_suffix(gen_label)
                out_path = out_dir / f"{safe_filename(title or 'V3処理')}_{suffix}{ext}"
                root.after(0, lambda: append_log(text_widget, f"LM Studio実行: {gen_label} / model={lm_model_label}"))
                ok, result_text, err, lm_warnings, used_instruction = run_lmstudio(
                    input_text=combined_input,
                    gen_type_label=gen_label,
                    meeting_title=title,
                    model_label=lm_model_label,
                    terms=all_terms,
                    custom_instruction=custom_instruction,
                )
                for w in lm_warnings:
                    root.after(0, lambda msg=w: append_log(text_widget, f"⚠ {msg}"))
                if ok and result_text.strip():
                    out_path.write_text(result_text, encoding="utf-8", newline="\n")
                    root.after(0, lambda f=out_path: append_log(text_widget, f"✅ AI結果保存: {f.name}"))
                    if CONFIG.get("output", {}).get("write_used_prompt", False):
                        (out_dir / "使用したAIへの指示.txt").write_text(used_instruction, encoding="utf-8", newline="\n")
                elif ok:
                    root.after(0, lambda: append_log(text_widget, "⚠ AI処理はスキップまたは空応答でした。"))
                else:
                    err_path = out_dir / f"{safe_filename(title or 'V3処理')}_{suffix}_ERROR.txt"
                    err_path.write_text(err, encoding="utf-8", newline="\n")
                    root.after(0, lambda m=err: append_log(text_widget, f"❌ LM Studioエラー: {m}"))
        else:
            root.after(0, lambda: append_log(text_widget, "LM Studio: OFF（文字起こし・前処理のみ）"))

        root.after(0, lambda: append_log(text_widget, "----- 完了 -----"))
        root.after(0, lambda: update_status(status_label, model_var, terms_listbox, persons_listbox, device_pref_var, "完了"))
    finally:
        IS_RUNNING = False


def select_files_and_run(
    *,
    text_widget: tk.Text,
    terms_listbox: tk.Listbox,
    persons_listbox: tk.Listbox,
    status_label: tk.Label,
    model_var: tk.StringVar,
    root: tk.Tk,
    device_pref_var: tk.StringVar,
    use_ai_var: tk.BooleanVar,
    gen_type_var: tk.StringVar,
    lm_model_var: tk.StringVar,
    meeting_title_var: tk.StringVar,
    custom_instruction_widget: tk.Text,
) -> None:
    global IS_RUNNING
    if IS_RUNNING:
        messagebox.showinfo("情報", "現在処理中です。完了をお待ちください。")
        return

    file_paths = filedialog.askopenfilenames(
        title="処理するファイルを選択（音声/txt/md/Word/Excel/PDF 複数可）",
        filetypes=[
            ("対応ファイル", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.mp4 *.mov *.mkv *.txt *.md *.docx *.xlsx *.pdf"),
            ("音声/動画", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.mp4 *.mov *.mkv"),
            ("文書", "*.txt *.md *.docx *.xlsx *.pdf"),
            ("すべてのファイル", "*.*"),
        ],
    )
    if not file_paths:
        return

    out_dir = filedialog.askdirectory(title="出力先フォルダーを選択")
    if not out_dir:
        return

    files = [Path(p) for p in file_paths]
    out_parent = Path(out_dir)

    IS_RUNNING = True
    threading.Thread(
        target=lambda: run_job(
            files=files,
            out_parent=out_parent,
            text_widget=text_widget,
            status_label=status_label,
            model_var=model_var,
            device_pref_var=device_pref_var,
            terms_listbox=terms_listbox,
            persons_listbox=persons_listbox,
            root=root,
            use_ai=bool(use_ai_var.get()),
            gen_type_label=gen_type_var.get(),
            lm_model_label=lm_model_var.get(),
            meeting_title=(meeting_title_var.get() or ""),
            custom_instruction=custom_instruction_widget.get("1.0", tk.END).strip(),
        ),
        daemon=True,
    ).start()


# ---------- リスト操作 ----------
def add_item(entry_widget: tk.Entry, listbox: tk.Listbox, status_label: tk.Label, model_var: tk.StringVar, terms_listbox: tk.Listbox, persons_listbox: tk.Listbox, device_pref_var: tk.StringVar, label_name: str) -> None:
    val = entry_widget.get().strip()
    if not val:
        return
    if listbox.size() >= MAX_TERMS:
        messagebox.showwarning("上限", f"{label_name}は最大 {MAX_TERMS} 個までです。")
        return
    existing = [listbox.get(i) for i in range(listbox.size())]
    if val in existing:
        messagebox.showinfo("情報", "既に登録されています。")
        return
    listbox.insert(tk.END, val)
    entry_widget.delete(0, tk.END)
    update_status(status_label, model_var, terms_listbox, persons_listbox, device_pref_var)


def delete_selected(listbox: tk.Listbox, status_label: tk.Label, model_var: tk.StringVar, terms_listbox: tk.Listbox, persons_listbox: tk.Listbox, device_pref_var: tk.StringVar) -> None:
    for index in reversed(listbox.curselection()):
        listbox.delete(index)
    update_status(status_label, model_var, terms_listbox, persons_listbox, device_pref_var)


def export_list(listbox: tk.Listbox, title: str, default_name: str) -> None:
    items = [listbox.get(i) for i in range(listbox.size())]
    if not items:
        messagebox.showwarning("警告", "エクスポートする内容がありません。")
        return
    save_path = filedialog.asksaveasfilename(title=title, defaultextension=".txt", initialfile=default_name, filetypes=[("テキストファイル", "*.txt")])
    if save_path:
        Path(save_path).write_text("\n".join(items) + "\n", encoding="utf-8")
        messagebox.showinfo("保存完了", f"{save_path} に保存しました。")


def import_list(listbox: tk.Listbox, title: str, status_label: tk.Label, model_var: tk.StringVar, terms_listbox: tk.Listbox, persons_listbox: tk.Listbox, device_pref_var: tk.StringVar) -> None:
    file_path = filedialog.askopenfilename(title=title, filetypes=[("テキストファイル", "*.txt"), ("すべてのファイル", "*.*")])
    if not file_path:
        return
    try:
        lines = [line.strip() for line in Path(file_path).read_text(encoding="utf-8-sig", errors="replace").splitlines() if line.strip()]
    except Exception as e:
        messagebox.showerror("エラー", f"辞書ファイルの読み込みに失敗しました:\n{e}")
        return

    existing = set(listbox.get(i) for i in range(listbox.size()))
    added = 0
    skipped_dup = 0
    skipped_limit = 0
    for term in lines:
        if term in existing:
            skipped_dup += 1
            continue
        if listbox.size() >= MAX_TERMS:
            skipped_limit += 1
            continue
        listbox.insert(tk.END, term)
        existing.add(term)
        added += 1
    update_status(status_label, model_var, terms_listbox, persons_listbox, device_pref_var)

    msg = f"読み込み完了：{added} 件追加"
    if skipped_dup:
        msg += f"\n（重複 {skipped_dup} 件はスキップ）"
    if skipped_limit:
        msg += f"\n（上限超過のため {skipped_limit} 件は読み込みできず）"
    messagebox.showinfo("結果", msg)


def attach_auto_kana(name_entry: tk.Entry, kana_entry: tk.Entry) -> None:
    kana_user_edited = {"flag": False}

    def on_kana_edited(event=None):
        kana_user_edited["flag"] = True

    def on_name_focus_out(event=None):
        name = name_entry.get().strip()
        if name and not kana_user_edited["flag"]:
            kana_entry.delete(0, tk.END)
            kana_entry.insert(0, guess_hiragana(name))

    def on_name_changed(event=None):
        kana_user_edited["flag"] = False

    kana_entry.bind("<KeyRelease>", on_kana_edited)
    name_entry.bind("<FocusOut>", on_name_focus_out)
    name_entry.bind("<KeyRelease>", on_name_changed)


def add_person_expanded(name_entry: tk.Entry, kana_entry: tk.Entry, persons_listbox: tk.Listbox, status_label: tk.Label, model_var: tk.StringVar, terms_listbox: tk.Listbox, device_pref_var: tk.StringVar) -> None:
    name = name_entry.get().strip()
    kana = kana_entry.get().strip()
    if not name:
        return
    candidates = [name, f"{name}さん"]
    if kana:
        candidates.extend([kana, f"{kana}さん"])
    existing = set(persons_listbox.get(i) for i in range(persons_listbox.size()))
    added = 0
    for term in candidates:
        if persons_listbox.size() >= MAX_TERMS:
            messagebox.showwarning("上限", f"人名は最大 {MAX_TERMS} 個までです。")
            break
        if term in existing:
            continue
        persons_listbox.insert(tk.END, term)
        existing.add(term)
        added += 1
    name_entry.delete(0, tk.END)
    kana_entry.delete(0, tk.END)
    update_status(status_label, model_var, terms_listbox, persons_listbox, device_pref_var)
    if added > 0:
        messagebox.showinfo("追加", f"{added} 件追加しました。")


def main() -> None:
    root = tk.Tk()
    root.title("ローカル文字起こし・要約ツール V3（端末1台版）")
    root.geometry("1180x920")

    whisper_cfg = CONFIG.get("whisper", {})
    lm_cfg = CONFIG.get("lmstudio", {})
    catalog = dict(lm_cfg.get("model_catalog", {}) or {})

    model_var = tk.StringVar(value=str(whisper_cfg.get("model_size", "medium")))
    device_pref_var = tk.StringVar(value=str(whisper_cfg.get("device", "auto")))
    use_ai_var = tk.BooleanVar(value=True)
    gen_type_var = tk.StringVar(value=GenType.POINT_SUMMARY_LONG.value)
    lm_model_var = tk.StringVar(value=str(lm_cfg.get("model_label", "gpt-oss-20b")))
    meeting_title_var = tk.StringVar(value="")

    # 上段: 基本設定
    setting_frame = tk.LabelFrame(root, text="基本設定", padx=8, pady=8)
    setting_frame.pack(pady=6, fill="x", padx=10)

    tk.Label(setting_frame, text="Whisperモデル:").grid(row=0, column=0, sticky="w")
    tk.OptionMenu(setting_frame, model_var, "small", "medium", "large-v2", "large-v3").grid(row=0, column=1, sticky="w")

    tk.Label(setting_frame, text="実行モード:").grid(row=0, column=2, sticky="w", padx=(20, 0))
    tk.Radiobutton(setting_frame, text="自動", variable=device_pref_var, value="auto").grid(row=0, column=3, sticky="w")
    tk.Radiobutton(setting_frame, text="CPU", variable=device_pref_var, value="cpu").grid(row=0, column=4, sticky="w")
    tk.Radiobutton(setting_frame, text="GPU固定", variable=device_pref_var, value="gpu").grid(row=0, column=5, sticky="w")

    tk.Checkbutton(setting_frame, text="生成AIを使用する", variable=use_ai_var).grid(row=1, column=0, sticky="w", pady=(6, 0))

    tk.Label(setting_frame, text="生成タイプ:").grid(row=1, column=1, sticky="w", pady=(6, 0))
    gen_menu = tk.OptionMenu(setting_frame, gen_type_var, GenType.POINT_SUMMARY_LONG.value, GenType.MINUTES.value, GenType.GENERAL_SUMMARY.value)
    gen_menu.grid(row=1, column=2, sticky="w", pady=(6, 0))

    tk.Label(setting_frame, text="LMモデル:").grid(row=1, column=3, sticky="w", padx=(20, 0), pady=(6, 0))
    model_labels = list(catalog.keys()) or [lm_model_var.get()]
    tk.OptionMenu(setting_frame, lm_model_var, *model_labels).grid(row=1, column=4, sticky="w", pady=(6, 0))

    tk.Label(setting_frame, text="会議名/案件名:").grid(row=2, column=0, sticky="w", pady=(6, 0))
    tk.Entry(setting_frame, textvariable=meeting_title_var, width=60).grid(row=2, column=1, columnspan=4, sticky="we", pady=(6, 0))

    # AI指示
    prompt_frame = tk.LabelFrame(root, text="AIへの指示（必要に応じて編集可）", padx=8, pady=8)
    prompt_frame.pack(pady=4, fill="x", padx=10)
    custom_instruction_text = tk.Text(prompt_frame, height=7, wrap="word", font=("Meiryo", 9))
    custom_instruction_text.pack(fill="x")

    def set_prompt_template(label: str) -> None:
        """生成タイプに対応する標準プロンプトをGUIへ反映する。"""
        custom_instruction_text.delete("1.0", tk.END)
        custom_instruction_text.insert("1.0", template_for_label(label))

    set_prompt_template(gen_type_var.get())

    def on_gen_type_changed(*_args) -> None:
        # 生成タイプを変更したら、表示中のAI指示もそのタイプの標準テンプレートへ切り替える。
        set_prompt_template(gen_type_var.get())

    gen_type_var.trace_add("write", on_gen_type_changed)

    def refresh_prompt_template() -> None:
        if messagebox.askyesno("確認", "現在のAI指示を標準テンプレートで上書きしますか？"):
            set_prompt_template(gen_type_var.get())

    tk.Button(prompt_frame, text="選択中の生成タイプの標準指示に戻す", command=refresh_prompt_template).pack(anchor="e", pady=(4, 0))

    # 固有名詞
    top_outer = tk.Frame(root)
    top_outer.pack(pady=5, fill="x", padx=10)

    terms_frame = tk.LabelFrame(top_outer, text="固有名詞（用語）", padx=8, pady=8)
    terms_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    top_outer.grid_columnconfigure(0, weight=1)

    persons_frame = tk.LabelFrame(top_outer, text="登場人物名（人名）", padx=8, pady=8)
    persons_frame.grid(row=0, column=1, sticky="nsew")
    top_outer.grid_columnconfigure(1, weight=1)

    tk.Label(terms_frame, text="用語:").grid(row=0, column=0, sticky="w")
    term_entry = tk.Entry(terms_frame, width=34)
    term_entry.grid(row=0, column=1, padx=5, sticky="w")
    terms_listbox = tk.Listbox(terms_frame, height=6, width=48)
    terms_listbox.grid(row=1, column=0, columnspan=2, sticky="we", pady=5)

    tk.Label(persons_frame, text="人名:").grid(row=0, column=0, sticky="w")
    person_name_entry = tk.Entry(persons_frame, width=24)
    person_name_entry.grid(row=0, column=1, padx=5, sticky="w")
    tk.Label(persons_frame, text="読み:").grid(row=1, column=0, sticky="w")
    person_kana_entry = tk.Entry(persons_frame, width=24)
    person_kana_entry.grid(row=1, column=1, padx=5, sticky="w")
    persons_listbox = tk.Listbox(persons_frame, height=6, width=48)
    persons_listbox.grid(row=2, column=0, columnspan=2, sticky="we", pady=5)
    attach_auto_kana(person_name_entry, person_kana_entry)

    status_label = tk.Label(root, text="")
    status_label.pack(pady=2, anchor="w", padx=12)
    update_status(status_label, model_var, terms_listbox, persons_listbox, device_pref_var)

    tk.Button(terms_frame, text="追加", command=lambda: add_item(term_entry, terms_listbox, status_label, model_var, terms_listbox, persons_listbox, device_pref_var, "用語")).grid(row=0, column=2, padx=5)
    tk.Button(terms_frame, text="選択削除", command=lambda: delete_selected(terms_listbox, status_label, model_var, terms_listbox, persons_listbox, device_pref_var)).grid(row=1, column=2, padx=5, sticky="n")
    tk.Button(terms_frame, text="辞書読み込み", command=lambda: import_list(terms_listbox, "用語辞書を読み込み", status_label, model_var, terms_listbox, persons_listbox, device_pref_var)).grid(row=2, column=1, sticky="w")
    tk.Button(terms_frame, text="辞書保存", command=lambda: export_list(terms_listbox, "用語辞書を保存", "terms.txt")).grid(row=2, column=2, sticky="w")

    tk.Button(persons_frame, text="追加", command=lambda: add_item(person_name_entry, persons_listbox, status_label, model_var, terms_listbox, persons_listbox, device_pref_var, "人名")).grid(row=0, column=2, padx=5)
    tk.Button(persons_frame, text="敬称・読み付きで追加", command=lambda: add_person_expanded(person_name_entry, person_kana_entry, persons_listbox, status_label, model_var, terms_listbox, device_pref_var)).grid(row=1, column=2, padx=5, sticky="w")
    tk.Button(persons_frame, text="選択削除", command=lambda: delete_selected(persons_listbox, status_label, model_var, terms_listbox, persons_listbox, device_pref_var)).grid(row=2, column=2, padx=5, sticky="n")
    tk.Button(persons_frame, text="辞書読み込み", command=lambda: import_list(persons_listbox, "人名辞書を読み込み", status_label, model_var, terms_listbox, persons_listbox, device_pref_var)).grid(row=3, column=1, sticky="w")
    tk.Button(persons_frame, text="辞書保存", command=lambda: export_list(persons_listbox, "人名辞書を保存", "persons.txt")).grid(row=3, column=2, sticky="w")

    # 実行ボタンとログ
    middle_frame = tk.Frame(root)
    middle_frame.pack(pady=6)
    text_box = tk.Text(root, wrap="word", font=("Meiryo", 10))

    tk.Button(
        middle_frame,
        text="ファイルを選択して処理（音声/txt/md/Word/Excel/PDF → 必要ならAI処理）",
        command=lambda: select_files_and_run(
            text_widget=text_box,
            terms_listbox=terms_listbox,
            persons_listbox=persons_listbox,
            status_label=status_label,
            model_var=model_var,
            root=root,
            device_pref_var=device_pref_var,
            use_ai_var=use_ai_var,
            gen_type_var=gen_type_var,
            lm_model_var=lm_model_var,
            meeting_title_var=meeting_title_var,
            custom_instruction_widget=custom_instruction_text,
        ),
        width=76,
    ).pack()

    text_box.pack(expand=True, fill="both", padx=10, pady=10)
    append_log(text_box, f"起動完了。NVIDIA DLL PATH追加: {NVIDIA_PATH_ADDED} dirs")
    append_log(text_box, "V3単体版: 共有フォルダーキュー処理は含みません。")
    append_log(text_box, "対応: 音声/動画、txt/md、Word(docx)、Excel(xlsx)、PDF(テキストPDF)")

    if _CONV is None:
        append_log(text_box, "注意: pykakasiが見つかりません。人名の読み自動入力は無効です。")

    root.mainloop()


if __name__ == "__main__":
    main()
