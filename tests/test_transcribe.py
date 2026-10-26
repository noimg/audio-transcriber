import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from transcribe import Caption, parse_args, save_outputs, select_caption


def segment(words, *, logprob=-0.2, no_speech=0.01):
    return SimpleNamespace(words=[SimpleNamespace(start=a, end=b, word=text)
                                  for a, b, text in words],
                           text="".join(text for _, _, text in words),
                           start=words[0][0], end=words[-1][1],
                           avg_logprob=logprob, no_speech_prob=no_speech,
                           compression_ratio=1.1)


class TranscriptionTests(unittest.TestCase):
    def test_overlap_preserves_sentence_once(self):
        original = segment([(298, 299, " a"), (299, 301, " boundary"), (301, 302, " word")])
        left = select_caption(original, 0, 0, 300, 600, 0)
        # Same utterance decoded again in the following window, whose audio starts at 298s.
        shifted = segment([(0, 1, " a"), (1, 3, " boundary"), (3, 4, " word")])
        right = select_caption(shifted, 298, 300, 600, 600, left.end)
        self.assertEqual(left.text + " " + right.text, "a boundary word")
        self.assertLessEqual(left.end, right.start)

    def test_noise_filter_preserves_confident_speech(self):
        words = [(0, 1, "Hello")]
        self.assertIsNone(select_caption(segment(words, logprob=-1.5, no_speech=0.9), 0, 0, 5, 5, 0))
        self.assertEqual(select_caption(segment(words, logprob=-0.2, no_speech=0.9), 0, 0, 5, 5, 0).text,
                         "Hello")

    def test_unicode_subtitles_and_empty_result(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "录音"
            captions = [Caption(3599.9996, 3601, "测试音频 Hello", -0.2, 0.01)]
            save_outputs(prefix, captions, {"language": "zh"})
            self.assertEqual(prefix.with_suffix(".txt").read_text(), "测试音频 Hello\n")
            self.assertIn("01:00:00,000 --> 01:00:01,000", prefix.with_suffix(".srt").read_text())
            self.assertEqual(json.loads(prefix.with_suffix(".json").read_text())["segments"][0]["text"],
                             "测试音频 Hello")
            save_outputs(prefix, [], {"language": None})
            self.assertEqual(prefix.with_suffix(".txt").read_text(), "")
            self.assertEqual(prefix.with_suffix(".srt").read_text(), "")
            self.assertEqual(json.loads(prefix.with_suffix(".json").read_text())["segments"], [])

    def test_invalid_options_fail_before_model_download(self):
        for flags in [["--chunk-seconds", "nan"], ["--overlap-seconds", "-1"],
                      ["--vad-threshold", "1"], ["--batch-size", "0"]]:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(["input.wav", *flags])
                self.assertEqual(error.exception.code, 2)


@unittest.skipUnless(os.environ.get("ASR_TEST_MODEL"), "设置 ASR_TEST_MODEL 运行真实模型静音/杂音测试")
class NoiseIntegrationTests(unittest.TestCase):
    def test_silence_and_non_speech_produce_no_text(self):
        import numpy as np
        from transcribe import run
        import wave

        rng = np.random.default_rng(42)
        samples = 16000 * 8
        signals = {"silence": np.zeros(samples),
                   "white_noise": rng.normal(0, 0.08, samples),
                   "hum": 0.15 * np.sin(2 * np.pi * 60 * np.arange(samples) / 16000)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, signal in signals.items():
                with self.subTest(name=name):
                    source = root / f"{name}.wav"
                    with wave.open(str(source), "wb") as output:
                        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                        output.writeframes((signal * 32767).astype("<i2").tobytes())
                    args = parse_args([str(source), "--model", os.environ["ASR_TEST_MODEL"],
                                       "--output-dir", str(root / "out")])
                    run(args)
                    result = json.loads((root / "out" / f"{name}.json").read_text())
                    self.assertEqual(result["segments"], [])
                    self.assertEqual((root / "out" / f"{name}.txt").read_text(), "")


if __name__ == "__main__":
    unittest.main()
