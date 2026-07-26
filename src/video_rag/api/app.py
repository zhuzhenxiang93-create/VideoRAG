from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from video_rag.pipeline import VideoRAGPipeline


def create_app(pipeline: VideoRAGPipeline) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/ask")
    def ask():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            return jsonify({"error": "question must be a non-empty string"}), 400
        try:
            result = pipeline.ask(question)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        response = result.to_dict()
        for evidence in response["evidence"]:
            evidence["video_url"] = f"/api/videos/{evidence['video_id']}"
        return jsonify(response)

    @app.get("/api/videos/<video_id>")
    def video(video_id: str):
        source = pipeline.video_path(video_id)
        if source is None:
            return jsonify({"error": "video not found"}), 404
        resolved = Path(source).resolve()
        if not resolved.is_file():
            return jsonify({"error": "video file is unavailable"}), 404
        return send_file(resolved, conditional=True)

    return app
