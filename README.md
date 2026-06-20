# ローカル文字起こし・議事録ツール V3（端末1台版）

V3は、1台のWindows端末上で、Whisper / faster-whisper による音声文字起こし、PythonによるWord・Excel・PDF等のテキスト抽出、LM Studio上のローカルLLMによる要約・議事録化をGUIから実行する単体利用版です。
共有フォルダーを巡回して複数人の依頼を処理するV4とは別物です。

## 技術構成

本ツールは、端末1台で完結するローカル処理を前提としています。

* 音声文字起こし：Whisper / faster-whisper
* 文書テキスト抽出：Python

  * Word（docx）：python-docx
  * Excel（xlsx）：openpyxl
  * PDF：pypdf
  * txt / md：Python標準処理
* ローカルLLM連携：LM Studio の OpenAI互換API
* GUI：Tkinter

ローカルLLMは、LM Studio上で動作するモデルを想定しています。
サンプル設定では、以下のようなモデル候補を想定しています。

* GPT-OSS 20B
* Gemma 4 E4B
* Gemma 4 12B
* Gemma 3 12B

実際に使用するモデルIDやLM StudioのURLは、`config.sample.json` を `config.json` にコピーしたうえで、利用環境に合わせて変更してください。

## 構成

```
V3_端末1台版/
├─ V3_main.py
├─ lmstudio_client.py
├─ prompts.py
├─ document_preprocessor.py
├─ config.sample.json
├─ requirements.txt
├─ run_v3_app.bat
└─ README.md
```

## 対応入力

- 音声/動画: wav, mp3, m4a, aac, flac, ogg, mp4, mov, mkv
- テキスト: txt, md
- Word: docx
- Excel: xlsx
- PDF: テキストPDF

PDFはOCRを行いません。スキャンPDFや画像PDFは、十分な文字を抽出できない場合があります。

## 出力

実行時に選んだ出力先に、案件名と日時のフォルダーを作成します。

```text
出力先/
└─ 案件名_YYYYMMDD_HHMMSS/
   ├─ 文字起こし/
   │  ├─ xxx_文字起こし.txt
   │  └─ xxx_文字起こし_秒数付き.txt
   ├─ 抽出テキスト/
   │  ├─ xxx_docx_抽出.txt
   │  ├─ xxx_xlsx_抽出.txt
   │  └─ xxx_pdf_抽出.txt
   └─ 案件名_要点要約（長）.txt など
```

V3はGUIで警告や処理状況を確認できるため、V4で生成する複数ユーザー向けの確認用ファイルは標準出力しません。

- 文字抽出できなかったファイル.json
- 使用したAIへの指示.txt
- segments.csv
- 処理結果.json

必要な場合のみ、`config.json` で詳細出力を有効化してください。

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

GPUで faster-whisper を使う場合は、CUDA/CUDNNまわりの環境も別途整えてください。GPUが使えない場合は、GUIの実行モードを「自動」または「CPU」にしてください。

## config.json

`config.sample.json` を `config.json` にコピーして、必要に応じて編集します。

```powershell
copy config.sample.json config.json
```

LM StudioのURLやモデルIDは、実際の環境に合わせてください。

## 実行

```powershell
python V3_main.py
```

または `run_v3_app.bat` を実行します。

## 使い方

1. Whisperモデル、実行モード、生成タイプ、LMモデルを選びます。
2. 必要に応じて会議名/案件名を入力します。
3. AIへの指示を確認し、必要なら編集します。
4. 固有名詞や人名を登録します。
5. 「ファイルを選択して処理」を押し、音声・txt・md・docx・xlsx・pdfを選びます。
6. 出力先を選ぶと処理が始まります。

音声と資料を同時に選ぶと、音声文字起こしと資料抽出結果をまとめてAIに渡します。

## 注意

- 実際の音声、議事録、個人情報、庁内パスをGitHubに含めないでください。
- `config.json` は公開せず、`config.sample.json` のみ公開してください。
- ローカルLLMを利用する場合も、モデルやツールの利用条件を確認してください。
- 長文入力はモデルのコンテキスト上限に注意してください。現時点では本格的な「分割要約→統合要約」は未実装です。
