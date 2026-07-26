from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_rag.schemas import Keyframe, TimedText


@dataclass(frozen=True, slots=True)
class VideoInfo:
    duration: float
    fps: float
    frame_count: int
    width: int
    height: int


def probe_video(path: str | Path) -> VideoInfo:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video probing requires: pip install -e '.[video]'") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Video has invalid FPS or frame count: {path}")
    return VideoInfo(
        duration=frame_count / fps,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


class WhisperTranscriber:
    """Lazy Transformers Whisper adapter that preserves segment timestamps."""

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        *,
        device: str = "cuda",
        chunk_length_seconds: int = 30,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.chunk_length_seconds = chunk_length_seconds
        self._pipeline: Any = None

    def _load(self):
        if self._pipeline is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("Whisper requires torch and transformers") from exc
            device_index = 0 if self.device.startswith("cuda") and torch.cuda.is_available() else -1
            dtype = torch.float16 if device_index >= 0 else torch.float32
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                torch_dtype=dtype,
                device=device_index,
                chunk_length_s=self.chunk_length_seconds,
            )
        return self._pipeline

    def transcribe(self, media_path: str | Path, language: str | None = None) -> list[TimedText]:
        generate_kwargs = {"language": language} if language else {}
        result = self._load()(
            str(media_path),
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )
        timed: list[TimedText] = []
        for chunk in result.get("chunks", []):
            timestamp = chunk.get("timestamp") or (None, None)
            start, end = timestamp
            text = str(chunk.get("text", "")).strip()
            if start is None or end is None or end <= start or not text:
                continue
            timed.append(TimedText(float(start), float(end), text))
        return timed

    def unload(self) -> None:
        self._pipeline = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class SceneKeyframeExtractor:
    """Samples frames, detects visual changes, deduplicates nearby selections."""

    def __init__(
        self,
        *,
        sample_interval_seconds: float = 1.0,
        scene_threshold: float = 0.22,
        minimum_gap_seconds: float = 2.0,
        max_frames_per_minute: int = 12,
    ) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self.scene_threshold = scene_threshold
        self.minimum_gap_seconds = minimum_gap_seconds
        self.max_frames_per_minute = max_frames_per_minute

    def extract(self, video_path: str | Path, output_dir: str | Path) -> list[Keyframe]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Keyframe extraction requires OpenCV and NumPy") from exc

        info = probe_video(video_path)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        candidates: list[tuple[float, float, Any]] = []
        previous_histogram = None
        timestamp = 0.0
        try:
            while timestamp < info.duration:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = capture.read()
                if not ok:
                    timestamp += self.sample_interval_seconds
                    continue
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                histogram = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
                cv2.normalize(histogram, histogram)
                change = (
                    1.0
                    if previous_histogram is None
                    else float(cv2.compareHist(previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA))
                )
                if previous_histogram is None or change >= self.scene_threshold:
                    sharpness = float(
                        cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    )
                    candidates.append((timestamp, change + np.log1p(sharpness) / 20.0, frame.copy()))
                previous_histogram = histogram
                timestamp += self.sample_interval_seconds
        finally:
            capture.release()

        limit = max(1, round(info.duration / 60 * self.max_frames_per_minute))
        selected: list[tuple[float, float, Any]] = []
        for candidate in sorted(candidates, key=lambda item: item[1], reverse=True):
            if all(abs(candidate[0] - existing[0]) >= self.minimum_gap_seconds for existing in selected):
                selected.append(candidate)
            if len(selected) >= limit:
                break
        selected.sort(key=lambda item: item[0])

        frames: list[Keyframe] = []
        stem = Path(video_path).stem
        for frame_number, (frame_time, _, frame) in enumerate(selected):
            frame_path = destination / f"{stem}_{frame_number:04d}_{frame_time:.3f}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise IOError(f"Failed to save frame: {frame_path}")
            frames.append(Keyframe(timestamp=frame_time, path=str(frame_path)))
        return frames
