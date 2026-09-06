"""Exercise the real web UI and media seeking on the server."""

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

out = Path("artifacts/tutorialvqa/acceptance")
out.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:5000")
    page.locator("#videos option").nth(4).wait_for()
    assert page.locator("#videos option").count() == 5
    page.select_option("#videos", ["14643"])
    page.fill("#question", "how can i save my file as a pdf file?")
    with page.expect_response(lambda r: "/api/ask" in r.url, timeout=180000) as response:
        page.click("#submit")
    answer = response.value.json()
    assert not answer["abstained"] and answer["answer"]
    page.locator(".evidence").first.wait_for()
    expected = answer["evidence"][0]["start_time"]
    page.click(".evidence >> nth=0")
    page.wait_for_function(
        '(t)=>{const v=document.querySelector("video"); return v.readyState>=2 && v.currentTime>=t && v.currentTime<t+5;}',
        arg=expected,
        timeout=30000,
    )
    page.locator("video").evaluate("(v)=>v.pause()")
    playback = page.locator("video").evaluate(
        "(v)=>({time:v.currentTime,duration:v.duration,readyState:v.readyState,error:v.error?.message,src:v.currentSrc})"
    )
    page.screenshot(path=str(out / "browser-answer.png"), full_page=True)
    page.select_option("#videos", ["14644"])
    page.fill("#question", "text")
    with page.expect_response(lambda r: "/api/search" in r.url) as result:
        page.click("#search")
    evidence = result.value.json()["evidence"]
    assert evidence and all(e["video_id"] == "14644" for e in evidence)
    page.locator("#answer").filter(has_text="找到").wait_for()
    assert not errors
    (out / "browser-report.json").write_text(
        json.dumps(
            {
                "passed": True,
                "selected_video": "14643",
                "answer": answer["answer"],
                "expected_seek": expected,
                "playback": playback,
                "search_video": "14644",
                "search_results": len(evidence),
                "javascript_errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    browser.close()
print("BROWSER ACCEPTANCE passed", flush=True)
