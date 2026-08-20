from __future__ import annotations

from pathlib import Path
from typing import Any

from video_rag.schemas import Keyframe, VideoSegment


class Qwen3Reranker:
    """Qwen3-Reranker adapter that scores query against multimodal segment evidence."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        *,
        device_map: str = "auto",
        max_length: int = 8192,
        unload_after_score: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.max_length = max_length
        self.unload_after_score = unload_after_score
        self._tokenizer: Any = None
        self._model: Any = None
        self._prefix_tokens: list[int] = []
        self._suffix_tokens: list[int] = []
        self._true_token_id = -1
        self._false_token_id = -1

    def _load(self):
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Qwen reranker requires torch and transformers") from exc

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
            )
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device_map,
            ).eval()
            prefix = (
                "<|im_start|>system\nJudge whether the Document meets the requirements "
                "based on the Query. The answer can only be \"yes\" or \"no\"."
                "<|im_end|>\n<|im_start|>user\n"
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self._prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
            self._suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)
            self._true_token_id = self._tokenizer.convert_tokens_to_ids("yes")
            self._false_token_id = self._tokenizer.convert_tokens_to_ids("no")
        return self._model, self._tokenizer

    @staticmethod
    def _document(segment: VideoSegment) -> str:
        return (
            f"Segment ID: {segment.segment_id}\n"
            f"Time: {segment.start_time:.3f}-{segment.end_time:.3f} seconds\n"
            f"{segment.searchable_text}"
        )

    def score(self, query: str, segments: list[VideoSegment]) -> list[float]:
        import torch

        model, tokenizer = self._load()
        pairs = [
            f"<Query>: {query}\n<Document>: {self._document(segment)}"
            for segment in segments
        ]
        available_length = self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        inputs = tokenizer(
            pairs,
            padding=False,
            truncation=True,
            max_length=available_length,
            return_attention_mask=False,
        )
        for index, tokens in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = self._prefix_tokens + tokens + self._suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        inputs = {name: value.to(model.device) for name, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :]
            yes = logits[:, self._true_token_id]
            no = logits[:, self._false_token_id]
            probabilities = torch.softmax(torch.stack([no, yes], dim=1), dim=1)[:, 1]
        result = probabilities.float().cpu().tolist()
        if self.unload_after_score:
            self.unload()
        return result

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._prefix_tokens = []
        self._suffix_tokens = []
        try:
            import gc

            gc.collect()
            if "torch" in globals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except ImportError:
            pass


class QwenVLService:
    """Shared lazy Qwen2.5-VL model for captioning and grounded answer generation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        *,
        device_map: str = "auto",
        max_pixels: int = 420 * 360,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self._processor: Any = None
        self._model: Any = None

    def _load(self):
        if self._model is None:
            try:
                import torch
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            except ImportError as exc:
                raise RuntimeError("Qwen-VL requires torch and transformers") from exc
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device_map,
            ).eval()
        return self._model, self._processor

    def infer(self, content: list[dict[str, Any]], *, max_new_tokens: int | None = None) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError("Qwen-VL requires qwen-vl-utils") from exc
        import torch

        model, processor = self._load()
        messages = [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            output[len(source) :]
            for source, output in zip(inputs.input_ids, generated, strict=True)
        ]
        return processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class QwenVLCaptioner:
    def __init__(self, service: QwenVLService) -> None:
        self.service = service

    def caption(self, frame: Keyframe) -> str:
        if not Path(frame.path).exists():
            raise FileNotFoundError(frame.path)
        return self.service.infer(
            [
                {
                    "type": "image",
                    "image": frame.path,
                    "max_pixels": self.service.max_pixels,
                },
                {
                    "type": "text",
                    "text": (
                        "Objectively describe the visible people, objects, actions, "
                        "on-screen text, and scene. Do not infer information that is not visible."
                    ),
                },
            ],
            max_new_tokens=128,
        )


class QwenVLEvidenceGenerator:
    def __init__(
        self,
        service: QwenVLService,
        *,
        max_images: int = 6,
        unload_after_generate: bool = False,
    ) -> None:
        self.service = service
        self.max_images = max_images
        self.unload_after_generate = unload_after_generate

    def generate(self, query: str, segments: list[VideoSegment]) -> str:
        content: list[dict[str, Any]] = []
        evidence_text: list[str] = []
        image_count = 0
        for segment in segments:
            evidence_text.append(
                f"[{segment.segment_id}] Time {segment.start_time:.1f}-{segment.end_time:.1f} seconds\n"
                f"{segment.searchable_text}"
            )
            for frame in segment.keyframes:
                if image_count >= self.max_images or not Path(frame.path).exists():
                    continue
                content.append(
                    {
                        "type": "image",
                        "image": frame.path,
                        "max_pixels": self.service.max_pixels,
                    }
                )
                image_count += 1
        instruction = (
            "Answer the question using only the candidate video evidence below. "
            "Do not introduce external facts. If the evidence is insufficient, answer exactly: "
            "The current video evidence is insufficient to determine the answer. "
            "Give a concise answer and cite the supporting segment_id.\n\n"
            f"Question: {query}\n\nCandidate evidence:\n" + "\n\n".join(evidence_text)
        )
        content.append({"type": "text", "text": instruction})
        try:
            return self.service.infer(content)
        finally:
            if self.unload_after_generate:
                self.service.unload()
