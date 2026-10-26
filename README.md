# Audio Transcriber

将长录音转换为文字和字幕的本地命令行工具。基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 和 Whisper large-v3-turbo，结合人声检测、轻度降噪和分块处理，减少静音及背景杂音造成的误识别。

## 功能

- 支持 M4A、MP3、WAV 等 FFmpeg 可解码的音频格式。
- 自动检测语言，也可指定中文、英文等语言。
- 自动选择 NVIDIA GPU 或 CPU，支持批量推理和显存不足时降低批量大小。
- 按块读取长录音，使用重叠上下文和词时间戳减少边界处的漏字、重复。
- 同时输出 TXT、SRT 和 JSON，保留原始录音的时间戳。
- 在本机处理音频；模型下载完成后，使用本地模型无需联网。

## 安装

需要 Python 3.11 或更新版本，以及 FFmpeg。以下命令适用于 Ubuntu/Debian，请先进入项目目录：

```bash
sudo apt install ffmpeg
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU 推理需要兼容 CTranslate2 的 CUDA 和 cuDNN 运行库。具体要求参见 [faster-whisper 安装说明](https://github.com/SYSTRAN/faster-whisper#gpu)。没有兼容 GPU 时可以使用 CPU。

## 快速开始

将音频放入 `audios/`，或直接传入其他位置的音频路径：

```bash
python transcribe.py audios/input.m4a
```

首次运行会从 ModelScope 下载约 1.6 GB 的默认模型，校验下载文件的 SHA-256，并缓存到 `models/faster-whisper-large-v3-turbo/`。也可以提前下载：

```bash
python download_model.py
```

转写完成后，默认在 `outputs/` 生成以下文件：

| 文件 | 内容 |
| --- | --- |
| `input.txt` | UTF-8 文字，按识别段落换行 |
| `input.srt` | 带原始录音时间戳的字幕 |
| `input.json` | 文字段落、时间戳、模型信息和处理耗时 |

同一输出目录中，同名音频的结果会被覆盖。处理来自不同目录的同名文件时，请通过 `--output-dir` 分别指定输出目录。

## 使用示例

```bash
# 指定中文，并提供人名、术语等词汇提示
python transcribe.py audios/input.m4a --language zh --prompt "傅里叶变换，信噪比"

# 指定输出目录
python transcribe.py audios/input.m4a --output-dir results

# 减小搜索宽度，提高转写速度
python transcribe.py audios/input.m4a --beam-size 1

# 使用 CPU
python transcribe.py audios/input.m4a --device cpu --compute-type int8

# 使用已有的本地模型
python transcribe.py audios/input.m4a --model ./models/faster-whisper-large-v3-turbo

# 降低人声检测阈值，尝试保留较轻的讲话声
python transcribe.py audios/input.m4a --vad-threshold 0.35

# 人声因降噪而失真时，关闭降噪
python transcribe.py audios/input.m4a --denoise off
```

`--model` 也接受 `small`、`large-v3` 等 faster-whisper 模型名。使用模型名时，由 faster-whisper 从 Hugging Face 下载；默认模型通过 ModelScope 下载。

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--output-dir` | `outputs` | 输出目录 |
| `--model` | large-v3-turbo 本地缓存 | 本地 CTranslate2 模型目录或模型名 |
| `--language` | `auto` | 语言代码，例如 `zh`、`en` |
| `--device` | `auto` | `auto`、`cuda` 或 `cpu` |
| `--compute-type` | `auto` | GPU 默认 `float16`，CPU 默认 `int8` |
| `--batch-size` | GPU 16、CPU 4 | 批量大小，显存不足时自动减半重试 |
| `--beam-size` | `3` | 搜索宽度；较小通常更快 |
| `--denoise` | `light` | 轻度降噪；`off` 为关闭 |
| `--vad-threshold` | `0.5` | 人声检测阈值，介于 0 和 1；降低可能引入更多杂音 |
| `--chunk-seconds` | `300` | 每次读入的音频长度，单位为秒 |
| `--overlap-seconds` | `2` | 每个音频块两侧的重叠长度，单位为秒 |
| `--prompt` | 无 | 词汇提示，例如人名和专业术语 |
| `--temp-dir` | 系统临时目录 | 临时 PCM 文件的存放目录 |

完整帮助：

```bash
python transcribe.py --help
```

## 处理方式

音频先解码为 16 kHz 单声道 PCM，再按块读取。工具先使用 Silero VAD 在原始音频上检测人声，再对检测通过的片段进行高通和轻度频谱降噪，避免降噪产生的伪影触发人声检测。人声片段随后以不超过 30 秒的小块批量识别。

分块两侧保留重叠上下文，并按词时间戳划分重叠区域的归属。临时 PCM 每小时音频约占 115 MB 磁盘，正常结束、报错或通过 Ctrl+C 取消后会自动清理。

背景谈话、强音乐、多人重叠讲话和较重口音仍可能产生识别错误。人声检测阈值过高可能漏掉轻声讲话，过低则可能保留更多杂音。工具不区分说话人，字幕按识别段落输出，部分段落可能较长。

## 性能参考

以下为模型已缓存时的一次完整转写结果，实际速度取决于硬件、模型和音频内容：

| 项目 | 结果 |
| --- | --- |
| GPU | NVIDIA RTX 4090 |
| 模型 | Whisper large-v3-turbo，FP16 |
| 音频 | 英语录音，37 分 26 秒 |
| 总耗时 | 76.4 秒 |
| 处理速度 | 约 29.4 倍实时速度 |
| 输出段落 | 118 段 |

JSON 中的 `elapsed_seconds` 包含模型加载、解码、人声检测、降噪和识别；`inference_seconds` 记录分块阶段的总耗时。首次下载模型的耗时也会计入 `elapsed_seconds`。

该录音没有人工标注的参考文本，因此以上结果不代表字词准确率评估。

## 测试

运行参数校验、分块边界、字幕时间格式和 UTF-8 输出测试：

```bash
python -m unittest discover -s tests -v
```

设置本地模型路径后，还会使用真实模型检查静音、白噪声和 60 Hz 嗡声样本是否生成文字：

```bash
ASR_TEST_MODEL=./models/faster-whisper-large-v3-turbo python -m unittest discover -s tests -v
```

## 项目结构

```text
audio-transcriber/
├── transcribe.py          # 转写入口
├── download_model.py      # 模型下载与校验
├── requirements.txt       # Python 依赖
├── tests/                 # 单元测试和可选模型测试
├── audios/                # 本地输入音频
├── models/                # 下载的模型缓存
└── outputs/               # 生成的转写结果
```

输入音频、模型缓存、转写结果和 Python 缓存已加入 Git 忽略规则，仓库不包含这些文件。
