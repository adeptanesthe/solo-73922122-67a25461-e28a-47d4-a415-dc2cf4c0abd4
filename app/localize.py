"""中间帧漏检定位。

场景：复核一条已求得的可行最优轨迹时，某相邻两框的帧号相差 2
（例如帧 2 与帧 4），说明中间帧（帧 3）里这只动物没有被检测器记录。
本模块尝试在中间帧中找出“画面里可见、但检测器漏掉”的动物候选。

做法（外观 + 位置，两侧独立取证）
--------------------------------
1. 分别从前一帧、后一帧的检测框位置裁出图像模板；
2. 以两框中心的线性中点为位置先验，在中间帧的一定半径内做逐位置
   归一化互相关（NCC，三通道联合）搜索 —— 位置先验只用于限定搜索
   范围，最终候选必须由中间帧的真实像素证据支持；
3. 每一侧取相关峰：峰值过低 → 证据不足；除主峰外还存在接近的次峰
   → 附近有相似目标 / 证据歧义；
4. 两侧各自得到的候选框必须指向同一处（IoU ≥ 0.5），否则判为不一致；
5. 候选若与中间帧已有框 IoU > 0.5，或被中间帧已有检测框包围/邻近，
   则无法可靠判定为漏检；
6. 任何一侧证据不足、尺寸/位置不一致、图像解码失败或相关图片尺寸
   超过 1024×1024，都返回“不能定位”，绝不凭插值直接猜框。

本模块只产出候选框，供前端叠加显示；不修改检测 JSON、轨迹或计数。
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

MAX_DIM = 1024
MIN_BOX_SIDE = 5             # 小于该边长的框外观证据不足
MIN_PATCH_SIDE = 8           # 模板最小边长（像素）
MIN_SIDE_SCORE = 0.55        # 单侧 NCC 主峰下限
AMBIGUITY_MARGIN = 0.10      # 次峰与主峰相差小于该值 → 歧义
AGREEMENT_IOU = 0.5          # 两侧候选框至少达到该 IoU 才算“同一处”
MAX_SIZE_RATIO = 1.5         # 两侧框尺寸相差过大 → 证据不一致
NEAR_FACTOR = 1.0            # 中间帧已有框中心距 ≤ 该系数 × 候选尺寸 → 附近有目标
EXISTING_IOU = 0.5           # 与中间帧已有框 IoU 超过该值 → 已有检测


class CannotLocate(Exception):
    """无法可靠定位；``code`` 供前端/测试区分原因，``message`` 面向用户。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------- 几何工具


def rect_center(rect):
    x, y, w, h = (float(v) for v in rect)
    return (x + w / 2.0, y + h / 2.0)


def rect_iou(a, b):
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _round_rect(rect):
    return [round(float(v), 2) for v in rect]


# ---------------------------------------------------------------- 图像加载


def load_rgb(data: bytes, label: str) -> Image.Image:
    """解码并检查尺寸；失败时抛出 CannotLocate。

    先读文件头尺寸再真正解码，避免超大图片的完整解码开销 / 解压炸弹。
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            if w > MAX_DIM or h > MAX_DIM:
                raise CannotLocate(
                    "image_too_large",
                    f"{label}图像尺寸 {w}×{h} 超过 1024×1024，不能定位。",
                )
            im.load()
            rgb = im.convert("RGB")
    except CannotLocate:
        raise
    except Exception as exc:  # noqa: BLE001 - Pillow 的异常类型较杂
        raise CannotLocate(
            "decode_failed", f"{label}图像解码失败，不能定位。"
        ) from exc
    return rgb


def _as_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float64) / 255.0


# ---------------------------------------------------------------- 模板与互相关


def _extract_template(arr: np.ndarray, rect):
    """按检测框裁模板（贴边时裁剪），返回 (模板, x 偏移, y 偏移)。

    偏移 = 模板左上角 − 检测框左上角，用于把匹配位置换回检测框坐标。
    """
    h_img, w_img = arr.shape[:2]
    x, y, w, h = (float(v) for v in rect)
    if w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
        raise CannotLocate(
            "box_too_small",
            f"检测框边长 {w:.0f}×{h:.0f} 过小，外观证据不足，不能定位。",
        )

    px0 = max(0, int(math.floor(x)))
    py0 = max(0, int(math.floor(y)))
    px1 = min(w_img, int(math.ceil(x + w)))
    py1 = min(h_img, int(math.ceil(y + h)))
    if px1 - px0 < MIN_PATCH_SIDE or py1 - py0 < MIN_PATCH_SIDE:
        raise CannotLocate(
            "patch_too_small", "检测框超出画面或图像过小，外观证据不足，不能定位。"
        )
    template = arr[py0:py1, px0:px1, :]
    if float(template.std()) < 1e-6:
        raise CannotLocate(
            "flat_patch", "检测框内图像几乎无纹理，外观证据不足，不能定位。"
        )
    return template, float(px0) - x, float(py0) - y


def _ncc_map(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """三通道联合的归一化互相关图（valid 模式）。

    对每个通道将模板去均值，并对局部图像块减去其均值，
    分子为各通道乘积之和，分母为两侧联合范数之积。
    """
    h_t, w_t = template.shape[:2]
    n_pix = float(h_t * w_t)
    numer = None
    energy = None
    t_energy = 0.0
    ones = np.ones((h_t, w_t), dtype=np.float64)

    for c in range(3):
        t = template[:, :, c] - template[:, :, c].mean()
        t_energy += float(np.sum(t * t))
        conv = fftconvolve(image[:, :, c], t[::-1, ::-1], mode="valid")
        numer = conv if numer is None else numer + conv
        s1 = fftconvolve(image[:, :, c], ones, mode="valid")
        s2 = fftconvolve(image[:, :, c] ** 2, ones, mode="valid")
        local_var = np.maximum(s2 - s1 * s1 / n_pix, 0.0)
        energy = local_var if energy is None else energy + local_var

    if t_energy <= 1e-12:
        raise CannotLocate(
            "flat_patch", "模板纹理不足，外观证据不足以定位。"
        )
    denom = np.sqrt(energy) * math.sqrt(t_energy)
    ncc = np.zeros_like(numer)
    good = denom > 1e-9
    ncc[good] = numer[good] / denom[good]
    return np.clip(ncc, -1.0, 1.0)


def _search_radius(w, h, motion):
    """位置先验的搜索半径：至少覆盖一个目标尺寸，并随帧间位移放宽。"""
    return max(1.5 * max(w, h), 0.75 * motion + 0.5 * max(w, h))


def _pick_peak(ncc, allowed, w, h, label):
    """在允许区域内取主峰；证据不足或存在接近的次峰时抛出 CannotLocate。"""
    masked = np.where(allowed, ncc, -np.inf)
    if not np.isfinite(masked).any() or float(np.max(masked)) == -np.inf:
        raise CannotLocate(
            "no_search_area",
            "位置先验附近没有可搜索的中间帧区域，不能定位。",
        )
    best_idx = int(np.argmax(masked))
    by, bx = divmod(best_idx, masked.shape[1])
    best = float(masked[by, bx])
    if best < MIN_SIDE_SCORE:
        raise CannotLocate(
            "weak_evidence",
            f"{label}外观匹配峰值仅 {best:.2f}（低于 {MIN_SIDE_SCORE:.2f}），"
            "该侧证据不足，不能定位。",
        )

    # 主峰附近较小邻域（0.6 × 目标尺寸）不参与“次峰”判断，吸收同一目标
    # 相关峰的平台 / 尺度轻微变化；该邻域外仍有分数接近的峰 → 附近有相似目标。
    suppress = max(2, int(round(0.6 * max(w, h))))
    rest = masked.copy()
    y0 = max(0, by - suppress)
    y1 = min(masked.shape[0], by + suppress + 1)
    x0 = max(0, bx - suppress)
    x1 = min(masked.shape[1], bx + suppress + 1)
    rest[y0:y1, x0:x1] = -np.inf
    second = float(np.max(rest)) if np.isfinite(rest).any() else -1.0
    if np.isfinite(rest).any() and second >= best - AMBIGUITY_MARGIN:
        raise CannotLocate(
            "ambiguous_peak",
            f"{label}在附近找到相似匹配（主峰 {best:.2f}，次峰 {second:.2f}），"
            "存在相似目标或证据歧义，不能可靠定位。",
        )
    return bx, by, best, second


def _refine_peak(ncc, bx, by, radius=3):
    """在主峰附近用非负相关值做加权质心，得到亚像素位置。

    只用于收紧两侧候选的一致性判断；权重取 max(ncc, 0)，
    因此不会把峰拉向负相关区域。
    """
    y0 = max(0, by - radius)
    y1 = min(ncc.shape[0], by + radius + 1)
    x0 = max(0, bx - radius)
    x1 = min(ncc.shape[1], bx + radius + 1)
    patch = np.maximum(ncc[y0:y1, x0:x1], 0.0)
    total = float(patch.sum())
    if total <= 1e-9:
        return float(bx), float(by)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((patch * xx).sum() / total), float((patch * yy).sum() / total)


# ---------------------------------------------------------------- 主流程


def locate_missed(left_img, mid_img, right_img,
                  left_rect, right_rect, mid_frame, mid_det_rects,
                  left_frame=None, right_frame=None):
    """在中间帧定位漏检动物。

    参数
    ----
    left_img, mid_img, right_img : PIL.Image（RGB）
    left_rect, right_rect : 前后帧检测框 [x, y, w, h]
    mid_frame : 中间帧帧号（写入结果）
    mid_det_rects : 中间帧已有检测框列表 [[x, y, w, h], ...]
    left_frame, right_frame : 前后帧帧号（写入证据，可选）

    返回 dict；无法可靠定位时抛出 CannotLocate。
    """
    if left_img.size != mid_img.size or right_img.size != mid_img.size:
        raise CannotLocate(
            "size_mismatch",
            "相邻帧与中间帧图像尺寸不一致，不能在同一坐标系内定位。",
        )

    lw, lh = float(left_rect[2]), float(left_rect[3])
    rw, rh = float(right_rect[2]), float(right_rect[3])
    for w, h, side in ((lw, lh, "前一帧"), (rw, rh, "后一帧")):
        if w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
            raise CannotLocate(
                "box_too_small",
                f"{side}检测框边长 {w:.0f}×{h:.0f} 过小，外观证据不足，不能定位。",
            )
    if (max(lw, rw) / max(1e-9, min(lw, rw)) > MAX_SIZE_RATIO or
            max(lh, rh) / max(1e-9, min(lh, rh)) > MAX_SIZE_RATIO):
        raise CannotLocate(
            "size_mismatch",
            "前后两框尺寸差异过大，无法判断为同一目标，不能定位。",
        )

    lc = rect_center(left_rect)
    rc = rect_center(right_rect)
    expected = ((lc[0] + rc[0]) / 2.0, (lc[1] + rc[1]) / 2.0)
    motion = math.hypot(rc[0] - lc[0], rc[1] - lc[1])
    radius = _search_radius(max(lw, rw), max(lh, rh), motion)

    left_arr = _as_array(left_img)
    mid_arr = _as_array(mid_img)
    right_arr = _as_array(right_img)

    side_results = []
    for side_arr, side_frame, rect, label in (
            (left_arr, left_frame, left_rect, "前一帧"),
            (right_arr, right_frame, right_rect, "后一帧")):
        template, off_x, off_y = _extract_template(side_arr, rect)
        ncc = _ncc_map(mid_arr, template)
        w, h = float(rect[2]), float(rect[3])

        # NCC 图中每个位置对应的“检测框中心”（模板位置 + 模板内偏移）
        xs = np.arange(ncc.shape[1], dtype=float) + off_x + w / 2.0
        ys = np.arange(ncc.shape[0], dtype=float) + off_y + h / 2.0
        grid_x, grid_y = np.meshgrid(xs, ys)
        allowed = np.hypot(grid_x - expected[0], grid_y - expected[1]) <= radius

        bx, by, score, second = _pick_peak(ncc, allowed, w, h, label)
        # 亚像素精修：在峰附近做加权质心，避免整数像素抖动被误判为“不同位置”。
        bx, by = _refine_peak(ncc, bx, by)
        cand = [float(bx) + off_x, float(by) + off_y, w, h]
        side_results.append(
            {"label": label, "frame": side_frame,
             "rect": cand, "score": score, "second": second}
        )

    lres, rres = side_results
    agreement = rect_iou(lres["rect"], rres["rect"])
    if agreement < AGREEMENT_IOU:
        raise CannotLocate(
            "disagree",
            f"前后两帧的证据指向不同位置（两侧候选 IoU {agreement:.2f} < "
            f"{AGREEMENT_IOU:.2f}），不能可靠定位。",
        )

    # 两侧得分加权融合（尺寸已在上面限制过差异）
    ws = lres["score"] + rres["score"]
    fused = [(lres["rect"][k] * lres["score"] + rres["rect"][k] * rres["score"]) / ws
             for k in range(4)]
    fc = rect_center(fused)

    if math.hypot(fc[0] - expected[0], fc[1] - expected[1]) > radius:
        raise CannotLocate(
            "position_inconsistent",
            "候选位置偏离前后帧给出的位置先验过远，不能可靠定位。",
        )

    for det_rect in mid_det_rects:
        iou = rect_iou(fused, det_rect)
        if iou > EXISTING_IOU:
            raise CannotLocate(
                "existing_box",
                f"候选与中间帧已有框重叠（IoU {iou:.2f} > {EXISTING_IOU:.2f}），"
                "无法判定为漏检，不能可靠定位。",
            )

    near = NEAR_FACTOR * max(fused[2], fused[3])
    for det_rect in mid_det_rects:
        dc = rect_center(det_rect)
        if math.hypot(dc[0] - fc[0], dc[1] - fc[1]) <= near:
            raise CannotLocate(
                "nearby_target",
                "中间帧候选附近已有检测框，存在相似目标，不能可靠判定漏检。",
            )

    return {
        "locatable": True,
        "message": "已在中间帧找到两侧证据一致的漏检候选（未改动检测与轨迹）。",
        "candidate": {
            "frame": int(mid_frame),
            "rect": _round_rect(fused),
            "center": [round(fc[0], 2), round(fc[1], 2)],
            "score": round(min(lres["score"], rres["score"]), 4),
        },
        "evidence": {
            "left": {
                "frame": left_frame,
                "rect": _round_rect(lres["rect"]),
                "score": round(lres["score"], 4),
                "second_score": round(lres["second"], 4),
            },
            "right": {
                "frame": right_frame,
                "rect": _round_rect(rres["rect"]),
                "score": round(rres["score"], 4),
                "second_score": round(rres["second"], 4),
            },
            "agreement_iou": round(agreement, 4),
            "search_radius": round(radius, 2),
        },
    }
