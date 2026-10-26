"""Download the CTranslate2 Whisper turbo model from ModelScope (no audio upload)."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
from urllib.request import urlopen

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models/faster-whisper-large-v3-turbo"
REPO = "https://modelscope.cn/api/v1/models/pengzhendong/faster-whisper-large-v3-turbo/repo"
REQUIRED = {"config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"}


def ensure_model(directory: Path = DEFAULT_MODEL_DIR) -> Path:
    directory = directory.resolve()
    if all((directory / name).is_file() for name in REQUIRED):
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    with urlopen(REPO + "/files?Revision=master&Recursive=true", timeout=60) as response:
        manifest = json.load(response)
    if manifest.get("Code") != 200:
        raise RuntimeError(f"无法读取模型列表: {manifest}")
    files = {item["Path"]: item for item in manifest["Data"]["Files"]}
    for name in sorted(REQUIRED):
        item = files[name]
        destination = directory / name
        if destination.is_file() and destination.stat().st_size == item["Size"]:
            with destination.open("rb") as existing:
                matches = hashlib.file_digest(existing, "sha256").hexdigest() == item["Sha256"]
            if matches:
                continue
        url = REPO + "?" + urlencode({"Revision": "master", "FilePath": name})
        temporary = directory / (name + ".part")
        for attempt in range(3):
            try:
                digest = hashlib.sha256()
                count = 0
                last_report = 0.0
                with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                    while block := response.read(4 * 1024 * 1024):
                        output.write(block)
                        digest.update(block)
                        count += len(block)
                        if time.monotonic() - last_report >= 10:
                            print(f"下载 {name}: {count / item['Size']:.1%}", file=sys.stderr, flush=True)
                            last_report = time.monotonic()
                if count != item["Size"] or digest.hexdigest() != item["Sha256"]:
                    raise RuntimeError(f"模型文件校验失败: {name}")
                temporary.replace(destination)
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载并校验 Whisper large-v3-turbo 模型，约 1.6 GB。")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    try:
        print(ensure_model(args.output))
    except (OSError, RuntimeError, KeyError, ValueError) as error:
        parser.exit(1, f"下载失败: {error}\n")
