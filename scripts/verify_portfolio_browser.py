"""Browser acceptance: real CPU retrieval/media plus isolated UI response fixtures.

Fixtures are explicitly NOT model acceptance. Public screenshots show real preview only.
"""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

out = Path("docs/media")
out.mkdir(parents=True, exist_ok=True)
report = {"mode": "CPU BM25 preview; no model acceptance", "real": {}, "ui_fixtures": {}}
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    errors = []
    console_errors = []
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto("http://127.0.0.1:5001")
    page.wait_for_function("document.querySelector('#videos').options.length===76")
    page.locator("#runtimeNotice").wait_for()
    assert page.locator("#allVideos").is_checked()
    assert page.locator("#videos option").first.text_content() != "14643"
    page.screenshot(path=str(out / "workspace-desktop.png"), full_page=True)
    page.get_by_role("button", name="保存为 PDF", exact=True).click()
    with page.expect_response("**/api/search") as response:
        page.click("#search")
    result = response.value.json()
    page.locator("#state").filter(has_text="片段已找到").wait_for()
    assert result["evidence"]
    assert "video_ids" not in response.value.request.post_data_json
    report["real"]["global_search"] = {"count": len(result["evidence"]),
                                      "video_ids": [e["video_id"] for e in result["evidence"]]}
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
    page.screenshot(path=str(out / "workspace-mobile.png"), full_page=True)
    page.set_viewport_size({"width": 1440, "height": 1080})
    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
    # Real video request, decode and seek; no video pixels are published.
    item = result["evidence"][0]
    page.locator(".play").first.click()
    page.wait_for_function("(start)=>{const v=document.querySelector('video');return v.readyState>=2 && v.currentTime>=start && v.currentTime<start+4;}", arg=item["start_time"], timeout=30000)
    page.locator("video").evaluate("(v)=>v.pause()")
    report["real"]["first_playback"] = page.locator("video").evaluate("(v)=>({src:v.currentSrc,time:v.currentTime,readyState:v.readyState})")
    assert page.locator(".evidence.active").count() == 1
    page.locator(".scope summary").click()
    page.uncheck("#allVideos")
    page.select_option("#videos", ["14644"])
    page.fill("#question", "text")
    with page.expect_response("**/api/search") as response:
        page.click("#search")
    scoped = response.value.json()["evidence"]
    page.locator("#state").filter(has_text="片段已找到").wait_for()
    assert scoped and all(e["video_id"] == "14644" for e in scoped)
    page.locator(".play").first.click()
    page.wait_for_function("(start)=>{const v=document.querySelector('video');return v.readyState>=2 && v.currentTime>=start && v.currentTime<start+4;}", arg=scoped[0]["start_time"], timeout=30000)
    page.locator("video").evaluate("(v)=>v.pause()")
    report["real"]["scoped_playback"] = page.locator("video").evaluate("(v)=>({src:v.currentSrc,time:v.currentTime,readyState:v.readyState})")
    assert "/14644" in report["real"]["scoped_playback"]["src"]
    report["real"]["scoped_search"] = {"video_id": "14644", "count": len(scoped)}
    report["real"]["console_errors_before_expected_failures"] = list(console_errors)
    assert not console_errors
    page.click("#submit")
    page.locator("#state").filter(has_text="请求未完成").wait_for()
    assert "预览" in page.locator("#answer").inner_text()
    report["real"]["preview_ask_503"] = True
    # Exercise UI states only using labelled test fixtures; never save these as demo media.
    page.route("**/api/ask", lambda route: route.fulfill(json={"answer": "<img src=x onerror=alert(1)> "+item["segment_id"], "citations": [item["segment_id"]], "evidence": [item], "status": "answered", "latency_ms": {"total": 100}}))
    page.click("#submit")
    page.locator("#state").filter(has_text="回答已返回").wait_for()
    assert page.locator("#answer img").count() == 0
    page.locator("#answer .citation").click()
    page.wait_for_function("(start)=>{const v=document.querySelector('video');return v.readyState>=2&&v.currentTime>=start&&v.currentTime<start+4;}", arg=item["start_time"], timeout=30000)
    page.locator("video").evaluate("(v)=>v.pause()")
    assert page.locator(".evidence.active").count() == 1
    report["ui_fixtures"]["safe_text_and_clickable_citation"] = True
    page.unroute("**/api/ask")
    page.route("**/api/ask", lambda route: route.fulfill(json={"answer": "根据当前视频内容无法确定。", "abstained": True, "evidence": []}))
    page.click("#submit")
    page.locator("#state").filter(has_text="证据不足").wait_for()
    report["ui_fixtures"]["abstention"] = True
    page.route("**/api/search", lambda route: route.fulfill(json={"evidence": [], "status": "search_results"}))
    page.click("#search")
    page.locator("#state").filter(has_text="未找到片段").wait_for()
    report["ui_fixtures"]["empty_results"] = True
    page.unroute("**/api/search")
    held = []
    page.route("**/api/search", lambda route: held.append(route))
    page.click("#search")
    page.locator("#state").filter(has_text="正在找片段").wait_for()
    assert page.locator("#search").is_disabled()
    assert "等待" in page.locator("#answer").inner_text()
    held[0].fulfill(json={"evidence": [], "status": "search_results"})
    page.locator("#state").filter(has_text="未找到片段").wait_for()
    report["ui_fixtures"]["honest_loading_and_disabled_submit"] = True
    page.unroute("**/api/search")
    page.route("**/api/search", lambda route: route.abort())
    page.click("#search")
    page.locator("#state").filter(has_text="请求未完成").wait_for()
    assert page.locator("#search").is_enabled()
    report["ui_fixtures"]["network_error_and_retry"] = True
    # Verify keyboard example activation and honest loading text before response.
    page.locator("[data-question]").first.focus()
    page.keyboard.press("Enter")
    assert page.input_value("#question") == "How do I save my design as a PDF?"
    report["real"]["keyboard_example"] = True
    report["real"]["viewports"] = [
        {"width": 1440, "height": 1080, "horizontal_overflow": False},
        {"width": 390, "height": 844, "horizontal_overflow": False},
    ]
    report["javascript_errors"] = errors
    assert not errors
    browser.close()
Path("reports/portfolio").mkdir(parents=True, exist_ok=True)
Path("reports/portfolio/browser-acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False, indent=2))
