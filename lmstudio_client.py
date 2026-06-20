# lmstudio_client.py
# -*- coding: utf-8 -*-
"""LM Studio（OpenAI互換API）呼び出し。

V3ではGUIで実行状況を確認できるため、サーバー運用用の詳細成果物は作らず、
このモジュールは「投げる・受け取る」だけに絞る。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from openai import OpenAI
    import httpx
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    httpx = None  # type: ignore


@dataclass
class LMStudioResult:
    ok: bool
    text: str
    error: str = ""


class LMStudioClient:
    """OpenAI Python SDK v1系で LM Studio のOpenAI互換APIを叩くクライアント。"""

    def __init__(self, base_url: str, api_key: str = "lm-studio", timeout_sec: float = 3600.0):
        if OpenAI is None or httpx is None:
            raise RuntimeError("openai / httpx が見つかりません。requirements.txt を参照してインストールしてください。")

        self.base_url = base_url.rstrip("/")
        http_client = httpx.Client(
            timeout=httpx.Timeout(
                timeout=timeout_sec,
                connect=10.0,
                read=timeout_sec,
                write=60.0,
                pool=10.0,
            )
        )
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            http_client=http_client,
            timeout=timeout_sec,
            max_retries=0,
        )

    def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 90000,
        timeout_sec: Optional[float] = None,
    ) -> LMStudioResult:
        try:
            client = self.client.with_options(timeout=timeout_sec) if timeout_sec is not None else self.client
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content if resp.choices else ""
            return LMStudioResult(ok=True, text=(content or "").strip())
        except Exception as e:
            return LMStudioResult(ok=False, text="", error=f"{type(e).__name__}: {e}")
