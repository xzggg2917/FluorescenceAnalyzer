"""
三分类荧光颜色识别模型 v3 — ROI门控 + 高对比边界
流程: 1) 背景锁定(鼠毛/黑衬底排除) -> 荧光ROI
      2) ROI内分层: 黑骨优先 + KMeans亮/暗红
      3) 边界用非红色霓虹色绘制, 与红色荧光零混淆
边界配色(刻意避开红/暗红):
  健康组织 霓虹绿  (0,255,0)   BGR
  坏组织   品红    (255,0,255) BGR
  坏死骨   青色    (255,255,0) BGR
  ROI外框  白色    (255,255,255) 粗线
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans

LABELS = {0: "背景", 1: "健康组织(暗红)", 2: "坏组织(亮红)", 3: "坏死骨(黑)"}
# 半透明叠加仍用可读色; 边界用霓虹色
OVERLAY_BGR = {1: (60, 30, 200), 2: (0, 200, 255), 3: (0, 0, 0)}  # 灰度候选在下方用红黄热区单独渲染
EDGE_BGR = {1: (0, 255, 0), 2: (255, 0, 255), 3: (255, 255, 0)}
EDGE_NAMES = {1: "healthy-green", 2: "diseased-magenta", 3: "bone-cyan"}

def bgr_to_hsv(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

def border_background(img_bgr):
    """黑背景: 与图像边缘连通的暗区, 一律视为衬底, 永不判为荧光/坏死骨.
    用 V<40 从四边泛洪, 鼠身内部暗块不与边缘连通, 不受影响."""
    V = bgr_to_hsv(img_bgr)[:, :, 2]
    dark = ((V < 40)).astype(np.uint8) * 255
    h, w = dark.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = dark.copy()
    for sx, sy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flood[sy, sx] == 255:
            cv2.floodFill(flood, mask, (sx, sy), 128)
    return ((flood == 128)).astype(np.uint8)

def fur_mask(img_bgr):
    """白鼠毛: 高亮低饱和区(灰度片中包围病灶的白色绒毛)"""
    hsv = bgr_to_hsv(img_bgr)
    S, V = hsv[:, :, 1].astype(float), hsv[:, :, 2].astype(float)
    return (((V > 150) & (S < 60))).astype(np.uint8)

def despeckle(m, min_frac=0.001):
    n, lab, areas, _ = cv2.connectedComponentsWithStats((m * 255).astype(np.uint8), 8)
    if n <= 1:
        return m
    keep = np.zeros_like(m)
    for i in range(1, n):
        if areas[i, cv2.CC_STAT_AREA] >= min_frac * float(m.size):
            keep[lab == i] = 1
    return keep.astype(np.uint8)

def gray_dark_region(img_bgr):
    """局部暗斑候选；不把鼠体外轮廓或图像边界作为病灶。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    bg = border_background(img_bgr)
    fur = fur_mask(img_bgr)
    valid = (bg == 0)
    # 局部对比核: 必须大于病灶本身, 否则深灰三角内部响应为0。
    size = max(15, int(min(h, w) * 0.22) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    response = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)
    positive = response[response > 0]
    threshold = max(10.0, float(np.percentile(positive, 70)) * 0.4) if positive.size else 10.0
    candidate = (((response > threshold) & (smooth < 130) & valid).astype(np.uint8)) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, properties, _ = cv2.connectedComponentsWithStats(candidate, 8)
    result = np.zeros_like(gray)
    best_score = 0.0
    for index in range(1, n):
        x, y, width, height, area = properties[index]
        if area < max(12, h * w * 0.0005) or area > h * w * 0.12:
            continue
        if x <= 1 or y <= 1 or x + width >= w - 1 or y + height >= h - 1:
            continue
        region = (labels == index) & valid
        if not region.any():
            continue
        ring = ((cv2.dilate(region.astype(np.uint8), kernel) > 0) & ~region) & valid
        if not ring.any():
            continue
        contrast = float(np.median(smooth[ring])) - float(np.median(smooth[region]))
        # 坏死骨应深于周围鼠毛, 但内部深灰三角也应保留, 不只收纯黑。
        if contrast < 18 or max(width, height) > min(width, height) * 5:
            continue
        score = contrast * np.sqrt(area)
        if score > best_score:
            best_score = score
            result[:] = 0
            result[region] = 1
    return result


def fluorescence_roi(img_bgr):
    hsv = bgr_to_hsv(img_bgr)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    R = img_bgr[:, :, 2].astype(float)
    G = img_bgr[:, :, 1].astype(float)
    B = img_bgr[:, :, 0].astype(float)
    gray = (R + G + B) / 3.0
    redness = R - gray
    bg = border_background(img_bgr)
    fur = fur_mask(img_bgr)
    red_hue = ((H < 10) | (H > 155)) & (S > 40)
    strong = (redness > 10) & (R > 35)
    roi = (((red_hue | strong)).astype(np.uint8) & (bg == 0) & (fur == 0)).astype(np.uint8) * 255
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, areas, _ = cv2.connectedComponentsWithStats(roi, 8)
    if n > 1:
        big = 1 + int(np.argmax(areas[1:, cv2.CC_STAT_AREA]))
        roi = ((lab == big)).astype(np.uint8) * 255
    # 只膨胀2px吃回暗红边缘; 之前7px会把ROI撑大, 边界线因此不贴荧光
    roi = cv2.dilate(roi, np.ones((3, 3), np.uint8), iterations=1)
    h, w = roi.shape
    # 兜底仅在非大黑背景时启用, 且排除背景
    if float((roi > 0).sum()) < 0.02 * h * w and float(bg.sum()) < 0.5 * h * w:
        dark = (((V < 60)).astype(np.uint8) & (bg == 0)).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        n2, lab2, areas2, _ = cv2.connectedComponentsWithStats(dark, 8)
        if n2 > 1:
            big2 = 1 + int(np.argmax(areas2[1:, cv2.CC_STAT_AREA]))
            roi = np.maximum(roi, ((lab2 == big2)).astype(np.uint8) * 255)
    roi = (((roi > 0)).astype(np.uint8) & (bg == 0)).astype(np.uint8)
    # 灰度片 fallback: 先用Otsu把"鼠体"和"黑背景"分开, 黑块=鼠体内部的洞
    if float(roi.sum()) < 0.02 * h * w:
        valid = (bg == 0)
        t_mouse, _ = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mouse = (((V > t_mouse)).astype(np.uint8) & valid).astype(np.uint8) * 255
        mouse = cv2.morphologyEx(mouse, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        n3, lab3, areas3, _ = cv2.connectedComponentsWithStats(mouse, 8)
        if n3 > 1:
            big3 = 1 + int(np.argmax(areas3[1:, cv2.CC_STAT_AREA]))
            mouse_bin = (lab3 == big3)
            ff = mouse_bin.astype(np.uint8) * 255
            fmask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(ff, fmask, (0, 0), 128)
            holes = (ff == 0).astype(np.uint8) * 255
            holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            nh, labh, arh, _ = cv2.connectedComponentsWithStats(holes, 8)
            good = np.zeros((h, w), np.uint8)
            for i in range(1, nh):
                if arh[i, cv2.CC_STAT_AREA] >= 0.0001 * h * w:
                    good[labh == i] = 1
            if not good.any():
                vm = V.copy(); vm[~mouse_bin] = 255
                vals = vm[mouse_bin]
                lo2 = float(np.percentile(vals, 30)) if vals.size else 80.0
                cand = (((V < lo2)) & mouse_bin).astype(np.uint8) * 255
                cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                nc, labc, arc, _ = cv2.connectedComponentsWithStats(cand, 8)
                if nc > 1:
                    dist = cv2.distanceTransform(mouse_bin.astype(np.uint8) * 255, cv2.DIST_L2, 5)
                    best, bs = -1, -1.0
                    for i in range(1, nc):
                        if arc[i, cv2.CC_STAT_AREA] < 0.0001 * h * w:
                            continue
                        s = float(dist[labc == i].mean())
                        if s > bs:
                            bs, best = s, i
                    if best > 0:
                        good[labc == best] = 1
            if good.any():
                mouse_d = cv2.dilate(mouse_bin.astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1)
                roi = np.maximum(mouse_d // 255, good).astype(np.uint8)
                extra_core = good
            else:
                extra_core = None
        else:
            extra_core = None
    else:
        extra_core = None
    return roi, redness, bg, extra_core

def detect_masks(img_bgr):
    h, w = img_bgr.shape[:2]
    hsv = bgr_to_hsv(img_bgr)
    V = hsv[:, :, 2]
    R = img_bgr[:, :, 2].astype(float)
    fur = fur_mask(img_bgr)
    red_signal = (hsv[:, :, 1] > 60) & (R > img_bgr[:, :, 1].astype(float) + 20)
    if np.count_nonzero(red_signal) < max(10, h * w * 0.002):
        bone = gray_dark_region(img_bgr)
        zero = np.zeros((h, w), np.uint8)
        masks = {0: (bone == 0).astype(np.uint8), 1: zero.copy(),
                 2: zero.copy(), 3: bone}
        # 灰度图只能输出暗斑候选，不能推断亮红/暗红或被遮挡的红光。
        extra = {"roi": bone.copy(), "bone_under_red": zero.copy(),
                 "tissue": zero.copy(), "V": V, "R": R,
                 "mode": "gray_candidate", "overlap_supported": False,
                 "status": "局部暗斑候选，需人工确认" if bone.any() else "未检出可靠暗斑，请人工指定病灶区域"}
        return masks, extra
    roi, redness, bg, gray_core = fluorescence_roi(img_bgr)
    # 纯荧光图: 先定荧光边界, 再在荧光内部分类; 坏死骨只允许是荧光内部的小暗芯,
    # 绝不允许把整个ROI/背景判成骨(否则出现骨64.9%的错误)。
    v_otsu_val, _ = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    black_thr = min(45.0, float(v_otsu_val) * 0.4)
    bone = (((V < black_thr) & (R < 60)) & (roi > 0)).astype(np.uint8)
    bone = cv2.morphologyEx(bone, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bone = despeckle(bone, min_frac=0.002)
    # 荧光内部小暗芯约束: 骨面积不得超过ROI的25%, 否则视为误检直接清零(荧光图骨不可见)
    if (roi > 0).any() and float(bone.sum()) > 0.25 * float((roi > 0).sum()):
        bone[:] = 0
    roi_mean_v = float(V[roi > 0].mean()) if (roi > 0).any() else 255.0
    # 灰度片增强: ROI内最暗的内部连通域(被亮区包围的暗芯)直接判骨
    # 白鼠毛 V 高、黑骨 V 低, Otsu 在 ROI 内天然分成两类
    if (roi > 0).any() and bone.sum() < 0.005 * roi.sum():
        rv = V.copy(); rv[roi == 0] = 255
        t_in, _ = cv2.threshold(rv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        core = (((V < t_in * 0.9)) & (roi > 0) & (fur == 0)).astype(np.uint8) * 255
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n4, lab4, ar4, _ = cv2.connectedComponentsWithStats(core, 8)
        if n4 > 1:
            # 选远离ROI边缘(内部)的暗块, 而非边缘阴影
            dist = cv2.distanceTransform((roi * 255).astype(np.uint8), cv2.DIST_L2, 5)
            best, best_s = -1, -1.0
            for i in range(1, n4):
                if ar4[i, cv2.CC_STAT_AREA] < 0.0005 * roi.size:
                    continue
                ys4, xs4 = np.where(lab4 == i)
                s = float(dist[ys4, xs4].mean())
                if s > best_s:
                    best_s, best = s, i
            if best > 0:
                bone = ((lab4 == best)).astype(np.uint8)
    tissue = (roi > 0).astype(np.uint8)
    # 骨必须被红组织包围(荧光内部暗芯); 悬空的背景暗块不算骨
    if bone.any():
        dil = cv2.dilate(bone * 255, np.ones((15, 15), np.uint8)) > 0
        ring = (dil & (tissue > 0) & (bone == 0))
        if ring.sum() < 0.5 * bone.sum():
            bone[:] = 0
    diseased = np.zeros((h, w), np.uint8)
    healthy = np.zeros((h, w), np.uint8)
    ys, xs = np.where(tissue > 0)
    if len(ys) > 100:
        feat = np.stack([V[ys, xs].astype(float), R[ys, xs].astype(float)], axis=1)
        try:
            km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(feat)
            c = km.cluster_centers_
            bright_id = int(np.argmax(c[:, 0] + c[:, 1]))
            lab = (km.labels_ == bright_id)
            for i, (y, x) in enumerate(zip(ys, xs)):
                if lab[i]:
                    diseased[y, x] = 1
                else:
                    healthy[y, x] = 1
        except Exception:
            thr = float(np.median(V[ys, xs]))
            diseased = ((V > thr) & (tissue > 0)).astype(np.uint8)
            healthy = ((V <= thr) & (tissue > 0)).astype(np.uint8)
    strong_bright = ((R > 150) & (V > 110) & (tissue > 0))
    diseased[strong_bright] = 1
    healthy[strong_bright] = 0
    diseased = despeckle(diseased)
    healthy = despeckle(healthy)
    bone_under_red = ((bone > 0) & (tissue > 0))
    final = np.zeros((h, w), dtype=np.uint8)
    final[healthy > 0] = 1
    final[diseased > 0] = 2
    final[bone > 0] = 3
    masks = {0: (final == 0).astype(np.uint8), 1: (final == 1).astype(np.uint8),
             2: (final == 2).astype(np.uint8), 3: (final == 3).astype(np.uint8)}
    extra = {"bone_under_red": bone_under_red.astype(np.uint8), "V": V, "R": R,
             "tissue": tissue, "roi": roi, "redness": redness, "bg": bg,
             "roi_mean_v": roi_mean_v}
    return masks, extra

def smooth_poly(contour, strength=0.008):
    """多边形近似 + 轻微平滑: 只去毛刺, 不切掉真实边缘凹凸。"""
    peri = cv2.arcLength(contour, True)
    if peri < 1:
        return contour
    approx = cv2.approxPolyDP(contour, strength * peri, True).reshape(-1, 2).astype(float)
    if len(approx) < 3:
        return approx.reshape(-1, 1, 2).astype(np.int32)
    padded = np.vstack([approx[-1], approx, approx[:2]])
    out = []
    for i in range(1, len(padded) - 2):
        out.append(0.2 * padded[i - 1] + 0.6 * padded[i] + 0.2 * padded[i + 1])
    return np.array(out, np.int32).reshape(-1, 1, 2)

def overlay_and_contours(img_bgr, masks, extra=None):
    """中图: ROI内染色 + 两条霓虹主线; 右图: 同样的两条主线, ROI外保持原图."""
    ov = img_bgr.copy()
    gray_mode = extra is not None and extra.get("mode") == "gray_candidate"
    if gray_mode:
        region = masks[3] > 0
        if region.any():
            distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
            peak = max(1.0, float(distance.max()))
            center = np.clip(distance / peak, 0.0, 1.0)
            dark_red = np.array([25, 25, 190], dtype=float)
            orange = np.array([25, 165, 255], dtype=float)
            yellow = np.array([135, 255, 255], dtype=float)
            heat = np.zeros_like(ov, dtype=float)
            low = center < 0.5
            heat[low] = dark_red[None, :] * (1 - center[low, None]) + orange[None, :] * center[low, None]
            heat[~low] = orange[None, :] * (1 - (center[~low] - 0.5) * 2)[:, None] + yellow[None, :] * ((center[~low] - 0.5) * 2)[:, None]
            ov[region] = (0.28 * ov[region].astype(float) + 0.72 * heat[region]).astype(np.uint8)
        vis = img_bgr.copy()
        bone_map = (masks[3] * 255).astype(np.uint8)
        outer, _ = cv2.findContours(bone_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if outer:
            hull = cv2.convexHull(max(outer, key=cv2.contourArea))
            peri = cv2.arcLength(hull, True)
            hull = cv2.approxPolyDP(hull, 0.008 * peri, True)
            cv2.drawContours(vis, [hull], -1, EDGE_BGR[1], 2)
        distance = cv2.distanceTransform(bone_map, cv2.DIST_L2, 5)
        peak = float(distance.max())
        if peak > 3:
            inner = ((distance > peak * 0.48) * 255).astype(np.uint8)
            inner, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in inner:
                if cv2.contourArea(contour) < 0.0002 * bone_map.size:
                    continue
                peri = cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, 0.008 * peri, True)
                cv2.drawContours(vis, [contour], -1, EDGE_BGR[2], 2)
        return ov, vis
    # 彩色荧光图: 先算两条主线, 生成线掩膜用于"染色不盖线"。
    outer_line = inner_line = None
    src = extra["roi"] if (extra is not None and "roi" in extra) else ((masks[1] + masks[2]) > 0).astype(np.uint8)
    rm = cv2.medianBlur((src * 255).astype(np.uint8), 3)
    cnts, _ = cv2.findContours(rm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        outer_line = smooth_poly(max(cnts, key=cv2.contourArea), strength=0.006)
    inner_map = cv2.medianBlur((masks[2] * 255).astype(np.uint8), 3)
    cnts2, _ = cv2.findContours(inner_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts2:
        main2 = max(cnts2, key=cv2.contourArea)
        if cv2.contourArea(main2) >= 0.0005 * inner_map.size:
            inner_line = smooth_poly(main2, strength=0.010)
    line_mask = np.zeros(img_bgr.shape[:2], np.uint8)
    if outer_line is not None:
        cv2.drawContours(line_mask, [outer_line], -1, 1, 5)
    if inner_line is not None:
        cv2.drawContours(line_mask, [inner_line], -1, 1, 5)
    keep = line_mask > 0
    for l in (1, 2, 3):
        m = (masks[l] > 0) & (~keep)
        c = np.array(OVERLAY_BGR[l], dtype=float)
        ov[m] = (0.45 * ov[m].astype(float) + 0.55 * c).astype(np.uint8)
    # 细线: 1px深色衬底 + 2px霓虹色, 保证清晰但不粗糙。
    if outer_line is not None:
        cv2.drawContours(ov, [outer_line], -1, (0, 0, 0), 4)
        cv2.drawContours(ov, [outer_line], -1, EDGE_BGR[1], 2)
    if inner_line is not None:
        cv2.drawContours(ov, [inner_line], -1, (0, 0, 0), 4)
        cv2.drawContours(ov, [inner_line], -1, EDGE_BGR[2], 2)
    vis = img_bgr.copy()
    if extra is not None and "roi" in extra:
        roi_m = extra["roi"] > 0
        vis[roi_m] = ov[roi_m]
    else:
        roi_m = (masks[1] + masks[2] + masks[3]) > 0
        vis[roi_m] = ov[roi_m]
    if outer_line is not None:
        cv2.drawContours(vis, [outer_line], -1, (0, 0, 0), 4)
        cv2.drawContours(vis, [outer_line], -1, EDGE_BGR[1], 2)
    if inner_line is not None:
        cv2.drawContours(vis, [inner_line], -1, (0, 0, 0), 4)
        cv2.drawContours(vis, [inner_line], -1, EDGE_BGR[2], 2)
    return ov, vis

def stats(masks, extra):
    tot = float(max(extra["roi"].sum(), 1))
    out = {}
    for l in (1, 2, 3):
        out[LABELS[l]] = round(float(masks[l].sum()) / tot * 100, 2)
    out["坏死骨下仍有红色信号占比(%)"] = round(float(extra["bone_under_red"].sum()) / tot * 100, 2)
    out["ROI占全图(%)"] = round(float(extra["roi"].sum()) / masks[1].size * 100, 2)
    return out
