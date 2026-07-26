from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_rag.evaluation import evaluate_answers


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated answers.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    predictions = read_jsonl(args.predictions)
    references = {item["question_id"]: item["answer"] for item in questions}
    predicted_answers = {item["question_id"]: item["answer"] for item in predictions}
    print(json.dumps(evaluate_answers(predicted_answers, references), indent=2))


if __name__ == "__main__":
    main()

