"""中间帧漏检定位测试：可定位场景、歧义/证据不足/已有框/解码失败/超大图，
以及“只读”与版本校验行为。"""

import io
import json
import random
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.localize import (
    AGREEMENT_IOU, MAX_DIM, MIN_SIDE_SCORE, CannotLocate,
    load_rgb, locate_missed, rect_iou,
)
from app.main import app

client = TestClient(app)

W, H = 320, 240
ANIMAL_RGB = (220, 40, 40)


# ---------------------------------------------------------------- 合成画面


def make_frame(animals, seed=1, size=(W, H), jitter=0.0):
    """在噪点草地上画十字标记的“动物”；animals = [(cx, cy, rgb, orient)]。"""
    rng = random.Random(seed)
    img = Image.new("RGB", size, (34, 52, 34))
    draw = ImageDraw.Draw(img)
    for _ in range(500):
        x = rng.randrange(size[0])
        y = rng.randrange(size[1])
        g = rng.randrange(45, 90)
        draw.point((x, y), fill=(g // 2, g, g // 3))
    for cx, cy, rgb, orient in animals:
        d = int(8 + jitter)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], fill=rgb, outline=(15, 15, 15))
        if orient == "h":
            draw.line([cx - d, cy, cx + d, cy], fill=(245, 245, 245), width=3)
        elif orient == "v":
            draw.line([cx, cy - d, cx, cy + d], fill=(245, 245, 245), width=3)
        else:  # 斜线：更明显的不同外观
            draw.line([cx - d, cy - d, cx + d, cy + d], fill=(245, 245, 245), width=3)
            draw.line([cx + d, cy - d, cx - d, cy + d], fill=(245, 245, 245), width=3)
    return img


def png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def box(cx, cy, s=16):
    return [cx - s // 2, cy - s // 2, s, s]


def reliable_frames(cx0=100, cx1=140, cy=120):
    left = make_frame([(cx0, cy, ANIMAL_RGB, "h")], seed=1)
    mid = make_frame([((cx0 + cx1) // 2, cy, ANIMAL_RGB, "h")], seed=2)
    right = make_frame([(cx1, cy, ANIMAL_RGB, "h")], seed=3)
    return left, mid, right


# ---------------------------------------------------------------- 单元：定位器


def test_locates_reliable_candidate_between_two_sides():
    left, mid, right = reliable_frames()
    res = locate_missed(left, mid, right, box(100, 120), box(140, 120), 1, [],
                        left_frame=0, right_frame=2)
    assert res["locatable"] is True
    cand = res["candidate"]
    assert cand["frame"] == 1
    assert cand["rect"][0] == pytest.approx(112, abs=2)
    assert cand["rect"][1] == pytest.approx(112, abs=2)
    assert cand["rect"][2] == pytest.approx(16, abs=1)
    assert cand["rect"][3] == pytest.approx(16, abs=1)
    ev = res["evidence"]
    assert ev["left"]["score"] >= MIN_SIDE_SCORE
    assert ev["right"]["score"] >= MIN_SIDE_SCORE
    assert ev["agreement_iou"] >= AGREEMENT_IOU
    # 两侧证据独立给出，且都指向同一处
    assert rect_iou(ev["left"]["rect"], ev["right"]["rect"]) >= AGREEMENT_IOU
    assert ev["left"]["frame"] == 0 and ev["right"]["frame"] == 2


def test_interpolated_area_has_no_animal_weak_evidence():
    """中间帧根本没有动物时不得凭插值给框。"""
    left, _, right = reliable_frames()
    empty_mid = make_frame([], seed=2)
    with pytest.raises(CannotLocate) as ei:
        locate_missed(left, empty_mid, right, box(100, 120), box(140, 120), 1, [],
                      left_frame=0, right_frame=2)
    assert ei.value.code == "weak_evidence"


def test_similar_target_nearby_is_ambiguous():
    """中间帧有两个几乎相同的目标 → 不知是哪一个，拒绝定位。"""
    left, _, right = reliable_frames()
    mid = make_frame([(112, 120, ANIMAL_RGB, "h"), (128, 120, ANIMAL_RGB, "h")], seed=2)
    with pytest.raises(CannotLocate) as ei:
        locate_missed(left, mid, right, box(100, 120), box(140, 120), 1, [],
                      left_frame=0, right_frame=2)
    assert ei.value.code == "ambiguous_peak"


def test_two_sides_disagree_is_rejected():
    """前帧动物与后帧动物外观/位置不合，两侧各自找到不同处 → 拒绝定位。"""
    left = make_frame([(100, 120, ANIMAL_RGB, "h")], seed=1)
    right = make_frame([(140, 120, ANIMAL_RGB, "v")], seed=3)
    mid = make_frame([(112, 120, ANIMAL_RGB, "h"),
                      (128, 120, ANIMAL_RGB, "v")], seed=2)
    with pytest.raises(CannotLocate) as ei:
        locate_missed(left, mid, right, box(100, 120), box(140, 120), 1, [],
                      left_frame=0, right_frame=2)
    assert ei.value.code in {"disagree", "ambiguous_peak"}


def test_candidate_matching_existing_mid_box_is_rejected():
    left, mid, right = reliable_frames()
    with pytest.raises(CannotLocate) as ei:
        locate_missed(left, mid, right, box(100, 120), box(140, 120), 1,
                      [box(120, 120, 18)], left_frame=0, right_frame=2)
    assert ei.value.code == "existing_box"
    assert "IoU" in ei.value.message


def test_box_near_existing_mid_detection_is_rejected():
    left, mid, right = reliable_frames()
    with pytest.raises(CannotLocate) as ei:
        locate_missed(left, mid, right, box(100, 120), box(140, 120), 1,
                      [box(132, 120, 16)], left_frame=0, right_frame=2)
    assert ei.value.code == "nearby_target"


def test_flat_patch_has_insufficient_evidence():
    flat = Image.new("RGB", (W, H), (80, 80, 80))
    with pytest.raises(CannotLocate) as ei:
        locate_missed(flat, flat, flat, box(100, 120), box(140, 120), 1, [],
                      left_frame=0, right_frame=2)
    assert ei.value.code == "flat_patch"


def test_tiny_box_rejected():
    left, mid, right = reliable_frames()
    with pytest.raises(CannotLocate) as ei:
        locate_missed(left, mid, right, [98, 118, 4, 4], [138, 118, 4, 4], 1, [],
                      left_frame=0, right_frame=2)
    assert ei.value.code == "box_too_small"


def test_size_mismatch_frames_rejected():
    left = make_frame([(100, 120, ANIMAL_RGB, "h")], seed=1, size=(320, 240))
    mid = make_frame([(120, 120, ANIMAL_RGB, "h")], seed=2, size=(300, 240))
    right = make_frame([(140, 120, ANIMAL_RGB, "h")], seed=3, size=(320, 240))
    with pytest.raises(CannotLocate) as ei:
        locate_missed(left, mid, right, box(100, 120), box(140, 120), 1, [],
                      left_frame=0, right_frame=2)
    assert ei.value.code == "size_mismatch"


def test_load_rgb_decode_failure():
    with pytest.raises(CannotLocate) as ei:
        load_rgb(b"not an image", "中间帧")
    assert ei.value.code == "decode_failed"


def test_load_rgb_too_large():
    img = Image.new("RGB", (MAX_DIM + 1, 10), (0, 0, 0))
    with pytest.raises(CannotLocate) as ei:
        load_rgb(png(img), "中间帧")
    assert ei.value.code == "image_too_large"


# ---------------------------------------------------------------- API


def upload(frames, dets):
    files = [("images", (f"f{i}.png", png(im), "image/png")) for i, im in enumerate(frames)]
    files.append(("detections", ("d.json", json.dumps({"detections": dets}),
                                 "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def add_vector(dets):
    for d in dets:
        d["vector"] = [1.0, 0.0, 0.0]
    return dets


def localize_payload(p, left, right, version=None):
    return client.post(
        f"/api/sessions/{p['session_id']}/localize",
        json={"left_id": left, "right_id": right,
              "version": p["version"] if version is None else version},
    )


def three_frame_session():
    frames = reliable_frames()
    dets = add_vector([
        {"id": "L", "frame": 0, "rect": box(100, 120)},
        {"id": "R", "frame": 2, "rect": box(140, 120)},
    ])
    return frames, dets


def test_api_localize_ok_and_read_only():
    frames, dets = three_frame_session()
    p = upload(frames, dets)
    before = client.get(f"/api/sessions/{p['session_id']}").json()
    r = localize_payload(p, "L", "R")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["locatable"] is True
    assert body["mid_frame"] == 1
    assert body["version"] == before["version"]
    assert body["candidate"]["frame"] == 1
    assert body["candidate"]["rect"][:2] == pytest.approx([112, 112], abs=2)
    # 只读：会话检测、版本均未变化；下载的检测 JSON 也不含候选
    after = client.get(f"/api/sessions/{p['session_id']}").json()
    assert after["detections"] == before["detections"]
    assert after["version"] == before["version"]
    dl = client.get(f"/api/sessions/{p['session_id']}/detections.json").json()
    assert dl["detections"] == before["detections"]


def test_api_localize_existing_mid_box_refused():
    frames, dets = three_frame_session()
    dets.append({"id": "M", "frame": 1, "rect": box(120, 120), "vector": [1.0, 0.0, 0.0]})
    p = upload(frames, dets)
    body = localize_payload(p, "L", "R").json()
    assert body["locatable"] is False
    assert body["code"] == "existing_box"
    assert "不能可靠定位" in body["message"]


def test_api_localize_rejects_non_gap_2():
    frames, dets = three_frame_session()
    dets = add_vector([
        {"id": "L", "frame": 0, "rect": box(100, 120)},
        {"id": "R", "frame": 1, "rect": box(120, 120)},
    ])
    p = upload(frames[:2], dets)
    r = localize_payload(p, "L", "R")
    assert r.status_code == 400 and "帧号相差 2" in r.json()["detail"]


def test_api_localize_rejects_stale_version():
    frames, dets = three_frame_session()
    p = upload(frames, dets)
    r = localize_payload(p, "L", "R", version=p["version"] + 5)
    assert r.status_code == 409


def test_api_localize_rejects_unknown_ids():
    frames, dets = three_frame_session()
    p = upload(frames, dets)
    r = localize_payload(p, "L", "nope")
    assert r.status_code == 400


def test_api_localize_oversized_image_refused():
    frames = [make_frame([(100, 120, ANIMAL_RGB, "h")], seed=1, size=(1100, 300)),
              make_frame([(120, 120, ANIMAL_RGB, "h")], seed=2, size=(1100, 300)),
              make_frame([(140, 120, ANIMAL_RGB, "h")], seed=3, size=(1100, 300))]
    dets = add_vector([
        {"id": "L", "frame": 0, "rect": box(100, 120)},
        {"id": "R", "frame": 2, "rect": box(140, 120)},
    ])
    p = upload(frames, dets)
    body = localize_payload(p, "L", "R").json()
    assert body["locatable"] is False
    assert body["code"] == "image_too_large"
    assert "1024×1024" in body["message"]


def test_api_localize_decode_failure_refused():
    frames = reliable_frames()
    dets = add_vector([
        {"id": "L", "frame": 0, "rect": box(100, 120)},
        {"id": "R", "frame": 2, "rect": box(140, 120)},
    ])
    files = [("images", ("f0.png", png(frames[0]), "image/png")),
             ("images", ("f1.png", b"not a real image", "image/png")),
             ("images", ("f2.png", png(frames[2]), "image/png")),
             ("detections", ("d.json", json.dumps({"detections": dets}),
                             "application/json"))]
    p = client.post("/api/sessions", files=files).json()
    body = localize_payload(p, "L", "R").json()
    assert body["locatable"] is False
    assert body["code"] == "decode_failed"
    assert "解码失败" in body["message"]


def test_api_localize_accepts_reversed_pair_order():
    frames, dets = three_frame_session()
    p = upload(frames, dets)
    body = localize_payload(p, "R", "L").json()
    assert body["locatable"] is True
    assert body["left"]["id"] == "L" and body["right"]["id"] == "R"
    assert body["mid_frame"] == 1


def test_api_localize_keeps_judgments_and_params():
    frames, dets = three_frame_session()
    p = upload(frames, dets)
    client.put(f"/api/sessions/{p['session_id']}/judgments",
               json={"judgments": [{"a": "L", "b": "R", "type": "same"}]})
    client.put(f"/api/sessions/{p['session_id']}/params",
               json={"w_dist": 3, "w_app": 4, "penalty": 5})
    before = client.get(f"/api/sessions/{p['session_id']}").json()
    body = localize_payload(p, "L", "R", version=before["version"]).json()
    assert body["locatable"] is True
    after = client.get(f"/api/sessions/{p['session_id']}").json()
    assert after["judgments"] == before["judgments"]
    assert after["params"] == before["params"]
    assert after["version"] == before["version"]


def test_api_localize_weak_evidence_message():
    frames = reliable_frames()
    empty_mid = make_frame([], seed=2)
    dets = add_vector([
        {"id": "L", "frame": 0, "rect": box(100, 120)},
        {"id": "R", "frame": 2, "rect": box(140, 120)},
    ])
    p = upload([frames[0], empty_mid, frames[2]], dets)
    body = localize_payload(p, "L", "R").json()
    assert body["locatable"] is False
    assert body["code"] == "weak_evidence"
    assert "不能定位" in body["message"]


def test_api_localize_same_box_rejected():
    frames, dets = three_frame_session()
    p = upload(frames, dets)
    r = localize_payload(p, "L", "L")
    assert r.status_code == 400


def test_api_heavy_and_example_unaffected():
    """定位请求不影响既有求解与计数。"""
    p = client.post("/api/example").json()
    job = client.post(f"/api/sessions/{p['session_id']}/solve").json()
    import time
    deadline = time.time() + 10
    while time.time() < deadline:
        done = client.get(f"/api/jobs/{job['job_id']}").json()
        if done["status"] != "running":
            break
        time.sleep(0.05)
    assert done["result"]["animal_count"] == 2
