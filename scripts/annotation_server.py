from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from threading import Lock
import uuid

from flask import Flask, jsonify, request, send_file

from video_rag.evaluation.dataset_validation import read_jsonl, validate_questions
from video_rag.evaluation.review_events import load_review_events


HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>VideoRAG人工复核</title>
<style>body{font-family:sans-serif;background:#0e1322;color:#eef2ff;margin:0;padding:24px}main{max-width:1400px;margin:auto}.grid{display:grid;grid-template-columns:1.2fr 1fr;gap:20px}.card{background:#171e31;border:1px solid #33405f;border-radius:12px;padding:16px}video{width:100%;max-height:480px;background:#000}textarea{width:100%;height:430px;background:#0d1220;color:#dce6ff;border:1px solid #46577e;padding:10px;font:14px monospace}button{padding:10px 18px;margin:8px 8px 0 0;border:0;border-radius:8px;cursor:pointer}.accept{background:#2da66f;color:white}.reject{background:#bd4b55;color:white}.skip{background:#62739d;color:white}.evidence{white-space:pre-wrap;max-height:300px;overflow:auto}.frames img{max-width:220px;max-height:140px;margin:4px}input{padding:9px;width:55%;background:#0d1220;color:white;border:1px solid #46577e}</style></head>
<body><main><h1>VideoRAG 人工复核</h1><p><label>题型筛选：<select id="typeFilter" onchange="loadNext()"><option value="">全部</option><option value="audio">字幕事实</option><option value="visual">视觉</option><option value="ocr">OCR</option><option value="multimodal">跨模态</option><option value="unknown_route">未知路由</option><option value="unanswerable">无答案</option></select></label></p><p id="progress"></p><div class="grid"><section class="card"><video id="video" controls></video><div class="frames" id="frames"></div><h3>ASR / 视觉描述</h3><div class="evidence" id="evidence"></div></section><section class="card"><textarea id="editor"></textarea><div><button class="accept" onclick="review('accept')">接受并标为verified</button><button class="reject" onclick="review('reject')">拒绝</button><button class="skip" onclick="loadNext(current?.question.question_id)">暂跳过</button><button class="skip" id="reopenButton" onclick="reopenLast()" disabled>撤销上一次处理并重审</button></div><p><input id="reason" placeholder="拒绝原因、复核备注或撤销原因"></p><p id="message"></p></section></div></main>
<script>let current=null,lastReviewed=null,openedAt=0,played=false,minTime=null,maxTime=null;const video=document.getElementById('video');video.addEventListener('play',()=>played=true);video.addEventListener('timeupdate',()=>{minTime=minTime===null?video.currentTime:Math.min(minTime,video.currentTime);maxTime=maxTime===null?video.currentTime:Math.max(maxTime,video.currentTime)});async function loadNext(after=''){let kind=document.getElementById('typeFilter').value;let qs=new URLSearchParams({after:after||'',type:kind});let r=await fetch('/api/next?'+qs);let d=await r.json();document.getElementById('progress').textContent=`总计 ${d.progress.total}，已接受 ${d.progress.accepted}，已拒绝 ${d.progress.rejected}，待复核 ${d.progress.pending}`;if(!d.item){current=null;document.getElementById('message').textContent='当前筛选下没有待处理候选。';return}current=d.item;openedAt=Date.now();played=false;minTime=null;maxTime=null;document.getElementById('editor').value=JSON.stringify(d.item.question,null,2);video.src=d.item.video_url;video.onloadedmetadata=()=>{video.currentTime=d.item.question.evidence_start||0};document.getElementById('evidence').textContent=d.item.evidence_text;document.getElementById('frames').innerHTML=d.item.frame_urls.map(x=>`<img src="${x}">`).join('');document.getElementById('message').textContent='接受前请实际播放证据时段，并确认 evidence_by_modality 与画面/字幕一致。';}async function review(action){if(!current)return;let edited;try{edited=JSON.parse(document.getElementById('editor').value)}catch(e){alert('JSON格式错误');return}let reason=document.getElementById('reason').value.trim();if(action==='reject'&&!reason){alert('拒绝时必须填写原因');return}let behavior={video_played:played,playback_ranges:minTime===null?[]:[[minTime,maxTime]],frames_present:current.frame_urls.length>0,asr_and_caption_view_present:true,review_duration_seconds:(Date.now()-openedAt)/1000};let questionId=current.question.question_id;let r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question_id:questionId,base_event_id:current.revision,action,reason,edited,review_behavior:behavior})});let d=await r.json();if(!r.ok){document.getElementById('message').textContent=JSON.stringify(d);return}lastReviewed={question_id:questionId,event_id:d.event_id};document.getElementById('reopenButton').disabled=false;document.getElementById('reason').value='';await loadNext();}async function reopenLast(){if(!lastReviewed)return;let reason=document.getElementById('reason').value.trim();if(!reason){alert('撤销时必须填写原因');return}let r=await fetch('/api/reopen',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question_id:lastReviewed.question_id,base_event_id:lastReviewed.event_id,reason})});let d=await r.json();if(!r.ok){document.getElementById('message').textContent=JSON.stringify(d);return}lastReviewed=null;document.getElementById('reopenButton').disabled=true;document.getElementById('reason').value='';document.getElementById('typeFilter').value='';await loadNext();}loadNext();</script></body></html>"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_app(
    candidates_path: Path, segments_path: Path, events_path: Path, reviewer_id: str,
    *, allowed_video_root: Path | None = None, allowed_frame_root: Path | None = None,
) -> Flask:
    candidates = read_jsonl(candidates_path)
    for item in candidates:
        item.pop("_line_number", None)
    segments = read_jsonl(segments_path)
    for item in segments:
        item.pop("_line_number", None)
    segment_by_id = {item["segment_id"]: item for item in segments}
    segments_by_video: dict[str, list[dict]] = {}
    for item in segments:
        segments_by_video.setdefault(item["video_id"], []).append(item)
    candidate_by_id = {item["question_id"]: item for item in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    candidate_sha = sha256(candidates_path)
    video_root = (allowed_video_root or Path("data/raw")).resolve()
    frame_root = (allowed_frame_root or Path("artifacts")).resolve()
    event_lock = Lock()
    app = Flask(__name__)

    def event_state() -> tuple[list[dict], dict[str, dict]]:
        return load_review_events(events_path, candidate_sha256=candidate_sha, known_question_ids=set(candidate_by_id))

    def backup_events(sequence: int) -> None:
        backup_dir = events_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(events_path, backup_dir / f"review_events.sequence_{sequence:06d}.jsonl")

    def progress(latest: dict[str, dict]) -> dict:
        accepted = sum(item["action"] == "accept" for item in latest.values())
        rejected = sum(item["action"] == "reject" for item in latest.values())
        return {"total": len(candidates), "accepted": accepted, "rejected": rejected, "pending": len(candidates) - accepted - rejected}

    @app.get("/")
    def index():
        return HTML

    @app.get("/api/next")
    def next_candidate():
        _, latest = event_state()
        requested_type = request.args.get("type", "").strip()
        after = request.args.get("after", "").strip()
        eligible = [value for value in candidates if latest.get(value["question_id"], {}).get("action") not in {"accept", "reject"}]
        if requested_type == "unanswerable":
            eligible = [value for value in eligible if value.get("answerable") is False]
        elif requested_type:
            eligible = [value for value in eligible if value.get("question_type") == requested_type]
        if after and eligible:
            ids = [value["question_id"] for value in eligible]
            if after in ids:
                position = ids.index(after) + 1
                eligible = eligible[position:] + eligible[:position]
        item = eligible[0] if eligible else None
        if item is None:
            return jsonify({"item": None, "progress": progress(latest)})
        relevant = [segment_by_id[value] for value in item.get("relevant_segment_ids", []) if value in segment_by_id]
        if not relevant:
            relevant = sorted(segments_by_video.get(item["video_id"], []), key=lambda value: value["start_time"])[:2]
        evidence = "\n\n".join(
            f"[{value['segment_id']}] {value['start_time']:.1f}-{value['end_time']:.1f}s\n"
            f"ASR: {value.get('transcript','')}\n视觉描述: {value.get('visual_caption','')}"
            for value in relevant
        )
        frames = []
        for value in relevant:
            for frame in value.get("keyframes", []):
                token = hashlib.sha256(frame["path"].encode()).hexdigest()[:24]
                frames.append((token, frame["path"]))
        app.config.setdefault("KNOWN_FRAMES", {}).update(dict(frames))
        return jsonify({
            "progress": progress(latest),
            "item": {
                "question": item,
                "revision": latest.get(item["question_id"], {}).get("event_id"),
                "video_url": f"/media/{item['video_id']}",
                "frame_urls": [f"/frame/{token}" for token, _ in frames[:8]],
                "evidence_text": evidence,
            },
        })

    @app.get("/media/<video_id>")
    def media(video_id: str):
        values = segments_by_video.get(video_id)
        if not values:
            return jsonify({"error": "unknown video"}), 404
        path = Path(values[0]["source_path"])
        resolved = path.resolve()
        if not resolved.is_relative_to(video_root) or not resolved.is_file():
            return jsonify({"error": "video file missing"}), 404
        return send_file(resolved, conditional=True)

    @app.get("/frame/<token>")
    def frame(token: str):
        value = app.config.get("KNOWN_FRAMES", {}).get(token)
        resolved = Path(value).resolve() if value else None
        if not resolved or not resolved.is_relative_to(frame_root) or not resolved.is_file():
            return jsonify({"error": "unknown frame"}), 404
        return send_file(resolved, conditional=True)

    @app.post("/api/review")
    def review():
        payload = request.get_json(force=True)
        question_id = payload.get("question_id")
        action = payload.get("action")
        reason = str(payload.get("reason", "")).strip()
        edited = payload.get("edited")
        if question_id not in candidate_by_id or action not in {"accept", "reject"} or not isinstance(edited, dict):
            return jsonify({"error": "invalid review payload"}), 400
        if action == "reject" and not reason:
            return jsonify({"error": "reject requires reason"}), 400
        if edited.get("question_id") != question_id or edited.get("video_id") != candidate_by_id[question_id]["video_id"]:
            return jsonify({"error": "question_id and video_id are immutable"}), 400
        behavior = payload.get("review_behavior")
        if action == "accept" and (
            not isinstance(behavior, dict) or behavior.get("video_played") is not True
            or not behavior.get("playback_ranges")
        ):
            return jsonify({"error": "accept requires recorded video playback behavior"}), 400
        with event_lock:
            all_events, latest = event_state()
            previous = latest.get(question_id)
            previous_id = previous["event_id"] if previous else None
            if payload.get("base_event_id") != previous_id:
                return jsonify({"error": "stale review revision", "current_event_id": previous_id}), 409
            event_id = str(uuid.uuid4())
            reviewed_at = datetime.now(timezone.utc).isoformat()
            if action == "accept":
                edited.update({
                    "verification_status": "verified", "annotation_source": "human_review",
                    "reviewer_id": reviewer_id, "reviewed_at": reviewed_at, "review_event_id": event_id,
                })
                report = validate_questions([edited], segments)
                if not report.valid:
                    return jsonify({"error": "accepted annotation is invalid", "details": list(report.errors)}), 400
            event = {
                "event_id": event_id, "sequence": len(all_events) + 1, "previous_event_id": previous_id,
                "question_id": question_id, "action": action, "reason": reason,
                "reviewer_id": reviewer_id, "reviewed_at": reviewed_at,
                "candidate_file": str(candidates_path), "candidate_file_sha256": candidate_sha,
                "candidate_before": candidate_by_id[question_id], "annotation_after": edited if action == "accept" else None,
                "review_behavior": behavior,
            }
            events_path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(event, ensure_ascii=False) + "\n"
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            backup_events(event["sequence"])
            return jsonify({"ok": True, "event_id": event_id})

    @app.post("/api/reopen")
    def reopen():
        payload = request.get_json(force=True)
        question_id = payload.get("question_id")
        reason = str(payload.get("reason", "")).strip()
        if question_id not in candidate_by_id or not reason:
            return jsonify({"error": "reopen requires known question_id and reason"}), 400
        with event_lock:
            all_events, latest = event_state()
            previous = latest.get(question_id)
            if previous is None or payload.get("base_event_id") != previous["event_id"]:
                return jsonify({"error": "stale or missing review revision"}), 409
            event = {
                "event_id": str(uuid.uuid4()), "sequence": len(all_events) + 1,
                "previous_event_id": previous["event_id"], "question_id": question_id,
                "action": "reopen", "reason": reason, "reviewer_id": reviewer_id,
                "reviewed_at": datetime.now(timezone.utc).isoformat(), "candidate_file": str(candidates_path),
                "candidate_file_sha256": candidate_sha, "candidate_before": candidate_by_id[question_id],
                "annotation_after": None, "review_behavior": None,
            }
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
            backup_events(event["sequence"])
        return jsonify({"ok": True, "event_id": event["event_id"]})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-only human review UI with append-only audit events.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.supplement_a.jsonl"))
    parser.add_argument("--events", type=Path, default=Path("artifacts/annotations/review_events.jsonl"))
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5100)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("annotation server must remain localhost-only; use an SSH tunnel")
    create_app(
        args.candidates, args.segments, args.events, args.reviewer_id,
        allowed_video_root=Path("data/raw"), allowed_frame_root=Path("artifacts"),
    ).run(
        host=args.host, port=args.port, debug=False, threaded=True
    )


if __name__ == "__main__":
    main()
