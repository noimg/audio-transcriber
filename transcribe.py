#!/usr/bin/env python3
"""Local, bounded-memory audio transcription with VAD and GPU batching."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from download_model import DEFAULT_MODEL_DIR, ensure_model

SAMPLE_RATE = 16000


@dataclass
class Caption:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


def timestamp(seconds: float, subtitle: bool = False) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{',' if subtitle else '.'}{milliseconds:03}"


def select_caption(segment, offset: float, owned_start: float, owned_end: float,
                   duration: float, previous_end: float) -> Caption | None:
    """Assign overlap words to exactly one window using their midpoint."""
    if segment.no_speech_prob > 0.6 and segment.avg_logprob < -1.0:
        return None
    if segment.compression_ratio > 2.4 and segment.avg_logprob < -1.0:
        return None
    if segment.words:
        words = [word for word in segment.words
                 if owned_start <= offset + (word.start + word.end) / 2 < owned_end]
        if not words:
            return None
        text = "".join(word.word for word in words).strip()
        start, end = offset + words[0].start, offset + words[-1].end
    else:
        if not owned_start <= offset + (segment.start + segment.end) / 2 < owned_end:
            return None
        text = segment.text.strip()
        start, end = offset + segment.start, offset + segment.end
    start = max(0.0, previous_end, start)
    end = min(duration, end)
    if not text or end <= start:
        return None
    return Caption(round(start, 3), round(end, 3), text,
                   round(segment.avg_logprob, 4), round(segment.no_speech_prob, 4))


def prepare_audio(source: Path, destination: Path) -> float:
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(source), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE)]
    command += ["-c:a", "pcm_s16le", "-f", "s16le", str(destination)]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError("音频解码失败: " + result.stderr[-2000:])
    duration = destination.stat().st_size / (SAMPLE_RATE * 2)
    if duration <= 0:
        raise ValueError("输入中没有可解码的音频。")
    return duration


def denoise_chunk(pcm: bytes, sample_count: int):
    """Denoise only after VAD; preserve the original sample clock for subtitles."""
    import numpy as np

    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
         "-af", "highpass=f=70,afftdn=nf=-25", "-f", "s16le", "pipe:1"],
        input=pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError("降噪失败: " + result.stderr.decode(errors="replace")[-2000:])
    audio = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0
    if len(audio) < sample_count:
        audio = np.pad(audio, (0, sample_count - len(audio)))
    return audio[:sample_count]


def detect_speech(audio, threshold: float) -> list[dict]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    spans = get_speech_timestamps(audio, VadOptions(
        threshold=threshold, min_speech_duration_ms=200,
        min_silence_duration_ms=500, speech_pad_ms=200,
        max_speech_duration_s=29,
    ))
    # Batch nearby utterances while keeping every model input below its 30s limit.
    merged = []
    for span in spans:
        if (merged and span["start"] - merged[-1]["end"] < SAMPLE_RATE
                and span["end"] - merged[-1]["start"] <= 29.5 * SAMPLE_RATE):
            merged[-1]["end"] = span["end"]
        else:
            merged.append(dict(span))
    return [{key: value / SAMPLE_RATE for key, value in span.items()} for span in merged]


def save_outputs(prefix: Path, captions: list[Caption], metadata: dict) -> None:
    # Replace complete files atomically so readers never see half-written JSON/SRT.
    contents = {
        ".txt": "\n".join(item.text for item in captions) + ("\n" if captions else ""),
        ".srt": "".join(f"{i}\n{timestamp(item.start, True)} --> {timestamp(item.end, True)}\n{item.text}\n\n"
                         for i, item in enumerate(captions, 1)),
        ".json": json.dumps({**metadata, "segments": [asdict(item) for item in captions]},
                            ensure_ascii=False, indent=2) + "\n",
    }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, content in contents.items():
        target = Path(str(prefix) + suffix)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix="." + target.name, delete=False) as output:
            temporary = Path(output.name)
            try:
                output.write(content)
                output.close()
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def run(args) -> dict:
    import ctranslate2
    import numpy as np
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    started = time.perf_counter()
    source = args.audio.resolve()
    if not source.is_file():
        raise ValueError(f"音频文件不存在: {source}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("找不到 ffmpeg，请先安装: sudo apt install ffmpeg")
    prefix = args.output_dir.resolve() / source.stem
    if source in [Path(str(prefix) + suffix) for suffix in (".txt", ".srt", ".json")]:
        raise ValueError("输出路径与输入文件相同，请更换 --output-dir。")
    device = args.device
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    compute_type = args.compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    model_path = str(ensure_model()) if args.model is None else args.model
    print(f"加载模型: {model_path}\n设备: {device} / {compute_type}", file=sys.stderr, flush=True)
    model = WhisperModel(model_path, device=device, compute_type=compute_type,
                         cpu_threads=min(8, os.cpu_count() or 1))
    pipeline = BatchedInferencePipeline(model)
    batch_size = args.batch_size or (16 if device == "cuda" else 4)
    captions = []
    language = None if args.language == "auto" else args.language
    language_probability = None
    filtered_segments = 0
    with tempfile.TemporaryDirectory(prefix="audio-transcribe-", dir=args.temp_dir) as temporary:
        print("解码、重采样；先检测人声，再降噪识别…", file=sys.stderr, flush=True)
        pcm = Path(temporary) / "audio.pcm"
        duration = prepare_audio(source, pcm)
        total_samples = pcm.stat().st_size // 2
        window_samples = round(args.chunk_seconds * SAMPLE_RATE)
        overlap_samples = round(args.overlap_seconds * SAMPLE_RATE)
        num_windows = math.ceil(total_samples / window_samples)
        print(f"音频时长: {timestamp(duration)}，共 {num_windows} 块", file=sys.stderr, flush=True)
        inference_started = time.perf_counter()
        with pcm.open("rb") as audio_file:
            for index, owned_start_sample in enumerate(range(0, total_samples, window_samples), 1):
                owned_end_sample = min(owned_start_sample + window_samples, total_samples)
                begin = max(0, owned_start_sample - overlap_samples)
                end = min(total_samples, owned_end_sample + overlap_samples)
                audio_file.seek(begin * 2)
                pcm_bytes = audio_file.read((end - begin) * 2)
                audio = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
                audio /= 32768.0
                clips = detect_speech(audio, args.vad_threshold)
                if clips and args.denoise == "light":
                    audio = denoise_chunk(pcm_bytes, end - begin)
                segments, info = [], None
                while clips:
                    try:
                        pipeline.last_speech_timestamp = 0.0
                        segments, info = pipeline.transcribe(
                            audio, language=language, batch_size=batch_size,
                            beam_size=args.beam_size, word_timestamps=True,
                            clip_timestamps=clips, vad_filter=False,
                            condition_on_previous_text=False,
                            initial_prompt=args.prompt, temperature=0.0,
                        )
                        segments = list(segments)
                        break
                    except RuntimeError as error:
                        if "out of memory" not in str(error).lower() or batch_size <= 1:
                            raise
                        batch_size = max(1, batch_size // 2)
                        print(f"显存不足，降低 batch_size 至 {batch_size} 后重试。", file=sys.stderr, flush=True)
                if segments and language is None:
                    language_probability = info.language_probability
                    if info.language_probability >= 0.5:
                        language = info.language
                    print(f"检测语言: {info.language} ({info.language_probability:.1%})", file=sys.stderr, flush=True)
                for segment in segments:
                    caption = select_caption(segment, begin / SAMPLE_RATE,
                                             owned_start_sample / SAMPLE_RATE,
                                             owned_end_sample / SAMPLE_RATE, duration,
                                             captions[-1].end if captions else 0.0)
                    if caption:
                        captions.append(caption)
                    else:
                        filtered_segments += 1
                completed = owned_end_sample / SAMPLE_RATE
                elapsed = time.perf_counter() - inference_started
                print(f"[{index}/{num_windows}] {completed / duration:.0%} | "
                      f"{timestamp(completed)} | {completed / max(elapsed, 0.001):.1f} 倍速 | "
                      f"{len(captions)} 段文字", file=sys.stderr, flush=True)
        inference_seconds = time.perf_counter() - inference_started
    elapsed = time.perf_counter() - started
    metadata = {
        "source": str(source), "duration_seconds": round(duration, 3),
        "language": language, "language_probability": language_probability,
        "model": model_path, "device": device, "compute_type": compute_type,
        "batch_size": batch_size, "beam_size": args.beam_size, "denoise": args.denoise,
        "vad_threshold": args.vad_threshold, "chunk_seconds": args.chunk_seconds,
        "overlap_seconds": args.overlap_seconds, "filtered_segments": filtered_segments,
        "elapsed_seconds": round(elapsed, 3),
        "inference_seconds": round(inference_seconds, 3),
        "speedup": round(duration / max(elapsed, 0.001), 2),
    }
    save_outputs(prefix, captions, metadata)
    print(f"完成: {len(captions)} 段，耗时 {elapsed:.1f} 秒，{metadata['speedup']} 倍速。", file=sys.stderr)
    for suffix in (".txt", ".srt", ".json"):
        print(str(prefix) + suffix)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="本地长音频转文字：降噪、人声检测、GPU 批量识别；输出 TXT/SRT/JSON。")
    parser.add_argument("audio", type=Path, help="输入音频，如 audios/ROF.m4a")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="输出目录（默认 outputs）")
    parser.add_argument("--model", help="本地 CTranslate2 模型目录或 faster-whisper 模型名；默认自动下载 large-v3-turbo")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--compute-type", choices=["auto", "float16", "float32", "int8", "int8_float16"], default="auto")
    parser.add_argument("--language", default="auto", help="auto 或语言代码，如 zh/en")
    parser.add_argument("--batch-size", type=int, help="默认 GPU 16、CPU 4；显存不足时自动减半")
    parser.add_argument("--beam-size", type=int, default=3, help="搜索宽度（默认 3；1 更快，5 更精细）")
    parser.add_argument("--denoise", choices=["off", "light"], default="light", help="轻度降噪；人声失真时可设 off")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="人声检测阈值 0~1（默认 0.5）")
    parser.add_argument("--chunk-seconds", type=float, default=300, help="内存中每块长度（默认 300 秒）")
    parser.add_argument("--overlap-seconds", type=float, default=2, help="每侧重叠长度（默认 2 秒）")
    parser.add_argument("--prompt", help="词汇提示，如人名、专业术语；不要填写指令")
    parser.add_argument("--temp-dir", type=Path, help="临时 PCM 目录，约占 115 MB/小时，完成后自动删除")
    args = parser.parse_args(argv)
    if not math.isfinite(args.chunk_seconds) or args.chunk_seconds < 30:
        parser.error("--chunk-seconds 必须为大于等于 30 的有限数")
    if not math.isfinite(args.overlap_seconds) or not 0 <= args.overlap_seconds < args.chunk_seconds / 2:
        parser.error("--overlap-seconds 必须大于等于 0 且小于块长的一半")
    if not 0 < args.vad_threshold < 1:
        parser.error("--vad-threshold 必须在 0 与 1 之间")
    if args.beam_size < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("--beam-size 和 --batch-size 必须为正整数")
    return args


def main():
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n已取消，临时音频已清理。", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
