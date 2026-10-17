# -*- coding: utf-8 -*-
"""荧光三分类桌面端 v2 — 均匀四卡 + 自带示例 + 高级样式"""
import sys, os
import numpy as np
import cv2
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtCore import *
from color_model import detect_masks, overlay_and_contours, stats

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE, "assets", "samples")

def load_qss(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""

def ensure_samples():
    """若无示例图, 自动生成3张仿真荧光示例(亮红/暗红/黑骨+叠加), 保证'一键示例'永远可用"""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    def save(name, draw):
        p = os.path.join(SAMPLE_DIR, name)
        if not os.path.exists(p):
            img = np.zeros((300, 300, 3), dtype=np.uint8)
            draw(img)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            cv2.imwrite(p, img)
        return p
    def d1(m):  # 亮红坏组织 + 黄芯
        cv2.circle(m, (150, 150), 120, (10, 60, 230), -1)
        cv2.circle(m, (150, 160), 70, (20, 160, 255), -1)
        cv2.circle(m, (150, 150), 110, (200, 200, 200), 6)
    def d2(m):  # 暗红健康
        cv2.circle(m, (150, 150), 120, (20, 20, 120), -1)
        cv2.circle(m, (150, 150), 80, (30, 25, 100), -1)
        cv2.circle(m, (150, 150), 110, (180, 180, 180), 5)
    def d3(m):  # 黑骨 + 外圈亮红叠加
        cv2.circle(m, (150, 150), 120, (5, 40, 200), -1)
        cv2.circle(m, (150, 150), 85, (5, 5, 5), -1)
        cv2.circle(m, (150, 150), 110, (200, 200, 200), 6)
    return [save("sample_bright_red.png", d1), save("sample_dark_red.png", d2),
            save("sample_black_bone.png", d3)]

def cv_to_qpix(img_bgr, box=460):
    h, w = img_bgr.shape[:2]
    s = box / max(h, w)
    img = cv2.resize(img_bgr, (int(w * s), int(h * s)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hh, ww, _ = rgb.shape
    q = QImage(rgb.data, ww, hh, ww * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())

class Worker(QThread):
    done = Signal(dict)
    def __init__(self, img): super().__init__(); self.img = img
    def run(self):
        masks, extra = detect_masks(self.img)
        ov, vis = overlay_and_contours(self.img, masks, extra)
        self.done.emit({"masks": masks, "extra": extra, "ov": ov, "vis": vis, "stats": stats(masks, extra)})

def stat_card(title, tip, bar_id, dot):
    f = QFrame(); f.setObjectName("card")
    f.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    f.setMinimumHeight(200)
    fl = QVBoxLayout(f); fl.setContentsMargins(16, 14, 16, 14); fl.setSpacing(8)
    top = QHBoxLayout(); d = QLabel(dot); d.setStyleSheet("font-size:15px;background:transparent;")
    a = QLabel(title); a.setObjectName("cardTitle"); a.setToolTip(tip)
    top.addWidget(d); top.addWidget(a, 1); fl.addLayout(top)
    v = QLabel("-- %"); v.setObjectName("bigSmall"); v.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    v.setMinimumHeight(44); fl.addWidget(v)
    b = QProgressBar(); b.setObjectName(bar_id); b.setRange(0, 100); b.setValue(0)
    b.setFormat("%p%"); fl.addWidget(b)
    note = QLabel(tip); note.setObjectName("stat"); note.setWordWrap(True); fl.addWidget(note)
    fl.addStretch(1)
    return f, v, b

class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("荧光三分类智能识别平台 · 颌骨骨髓炎")
        for _icon in (os.path.join(BASE, "assets", "logo.ico"), os.path.join(BASE, "logo.ico")):
            if os.path.exists(_icon):
                self.setWindowIcon(QIcon(_icon))
                break
        self.resize(1360, 860)
        self.img = None; self.last = None
        self.samples = ensure_samples()
        w = QWidget(); self.setCentralWidget(w)
        root = QHBoxLayout(w); root.setContentsMargins(14, 14, 14, 14); root.setSpacing(14)
        # 左栏
        side = QFrame(); side.setObjectName("side"); side.setFixedWidth(330)
        sl = QVBoxLayout(side); sl.setContentsMargins(22, 24, 22, 24); sl.setSpacing(12)
        t = QLabel("🦷 荧光三分类\n智能识别平台"); t.setObjectName("title"); sl.addWidget(t)
        st = QLabel("坏死骨(黑) · 坏组织(亮红) · 健康组织(暗红)\n黑可在亮红下叠加 · 自动边界划分"); st.setObjectName("subtitle"); st.setWordWrap(True); sl.addWidget(st)
        for s in ["① 上传 / 拖拽 / 点示例 — 载入片子", "② 点击开始识别 — 分层检测", "③ 三视图 + 四指标 — 出结论", "④ 导出报告 — 一键存 PNG"]:
            l = QLabel(s); l.setObjectName("step"); sl.addWidget(l)
        self.btn_open = QPushButton("📂  选择荧光图片"); self.btn_open.setObjectName("primary"); self.btn_open.clicked.connect(self.open_file); sl.addWidget(self.btn_open)
        self.btn_run = QPushButton("🚀  开始识别"); self.btn_run.setObjectName("primary"); self.btn_run.clicked.connect(self.run); sl.addWidget(self.btn_run)
        self.btn_save = QPushButton("💾  导出结果报告"); self.btn_save.setObjectName("ghost"); self.btn_save.clicked.connect(self.save); sl.addWidget(self.btn_save)
        self.bar = QProgressBar(); self.bar.setRange(0, 0); self.bar.hide(); sl.addWidget(self.bar)
        sl.addWidget(QLabel("识别图例(颜色 = 类别):"))
        for dot, name, mean in [
            ("🔴", "亮红覆盖 — 坏组织", "R>150,V>110,需清创"),
            ("🟤", "暗红覆盖 — 健康组织", "暗红保留,保守处理"),
            ("⬜", "白色描边 — 坏死骨边界", "V极低,死骨游离区"),
            ("🟣", "半透明叠加 — 病灶范围", "原图+45%染色,看分布"),
            ("⭕", "边界勾勒 — 定量边界", "连通域轮廓,量面积")]:
            c = QLabel(f"{dot} {name}\n     {mean}"); c.setObjectName("step"); c.setWordWrap(True); sl.addWidget(c)
        sl.addStretch(1)
        root.addWidget(side)
        # 右侧
        right = QVBoxLayout(); right.setSpacing(12)
        banner = QFrame(); banner.setObjectName("card")
        bl = QHBoxLayout(banner); bl.setContentsMargins(18, 14, 18, 14)
        left = QVBoxLayout()
        self.big = QLabel("等待上传片子"); self.big.setObjectName("conclusion"); left.addWidget(self.big)
        self.sub = QLabel("方法: 黑骨优先分层 + KMeans(V,R)亮暗红自适应 · 形态学去噪 · 白色描边=坏死骨边界"); self.sub.setObjectName("stat"); left.addWidget(self.sub)
        self.view_tip = QLabel("📖 视图说明 · 半透明叠加: 原像素45%+类别色55%,看病灶分布 | 边界勾勒: 每类mask外轮廓描边,用于量面积与切除规划"); self.view_tip.setObjectName("stat"); self.view_tip.setWordWrap(True); left.addWidget(self.view_tip)
        bl.addLayout(left, 1)
        for txt in ["三视图同步对照", "黑色病灶清晰覆盖", "四指标定量统计"]:
            b = QLabel("✓ " + txt); b.setObjectName("badge"); bl.addWidget(b, alignment=Qt.AlignVCenter)
        right.addWidget(banner)
        imgs = QHBoxLayout(); imgs.setSpacing(12)
        self.l1, self.l2, self.l3 = QLabel("原图"), QLabel("叠加"), QLabel("边界")
        for i, l in enumerate([self.l1, self.l2, self.l3]):
            l.setObjectName("imgLabel"); l.setAlignment(Qt.AlignCenter)
            l.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            l.setMinimumSize(280, 320); l.setScaledContents(False)
            cap = QLabel(["① 原图 Original", "② 半透明叠加 Overlay", "③ 边界勾勒 Contour"][i]); cap.setObjectName("cap"); cap.setAlignment(Qt.AlignCenter)
            f = QFrame(); f.setObjectName("card"); f.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            fl = QVBoxLayout(f); fl.setContentsMargins(12, 10, 12, 12); fl.addWidget(cap); fl.addWidget(l, 1)
            imgs.addWidget(f, 1)  # 关键: 三图 stretch=1 均匀
        right.addLayout(imgs, 1)
        stat = QHBoxLayout(); stat.setSpacing(12)
        self.cards = {}
        defs = [("健康组织(暗红)", "R≈80-150,V偏低 · 保守保留", "amber", "🟤"),
                ("坏组织(亮红)", "R>150,V>110 · 高亮橙黄", "red", "🔴"),
                ("坏死骨(黑)", "V极低,R<80 · 可压盖显示", "slate", "⚫"),
                ("叠加指数", "骨区仍有红光占比 · 重叠证据", "teal", "🟣")]
        self.stat_views = []
        for title, tip, bid, dot in defs:
            f, v, b = stat_card(title, tip, bid, dot)
            stat.addWidget(f, 1)  # 关键: 四卡 stretch=1 均匀
            self.stat_views.append((title, v, b))
        right.addLayout(stat)
        root.addLayout(right, 1)
        self.setStyleSheet(load_qss(os.path.join(BASE, "style.qss")))
        self.statusBar().showMessage(f"就绪 | 内置示例 {len(self.samples)} 张 | 拖拽图片到窗口也可载入")
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        self.load(e.mimeData().urls()[0].toLocalFile())
    def open_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择荧光图", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if p: self.load(p)
    def load_demo(self):
        p = os.path.join(BASE, "jaw_osteomyelitis_concept.png")
        if os.path.exists(p): self.load(p)
        else: self.load(self.samples[0])
    def load(self, p):
        img = cv2.imdecode(np.fromfile(p, np.uint8), 1)
        if img is None: QMessageBox.warning(self, "失败", "图像读取失败"); return
        self.img = img; self.cur = p
        self.l1.setPixmap(cv_to_qpix(img)); self.l2.setText("待识别"); self.l3.setText("待识别")
        self.big.setText("已载入 · 点击开始识别")
        self.statusBar().showMessage(os.path.basename(p))
    def run(self):
        if self.img is None: QMessageBox.information(self, "提示", "请先选择图片或点示例"); return
        self.bar.show(); self.btn_run.setEnabled(False)
        self.w = Worker(self.img); self.w.done.connect(self.show_result); self.w.start()
    def show_result(self, r):
        self.bar.hide(); self.btn_run.setEnabled(True); self.last = r
        self.l2.setPixmap(cv_to_qpix(r["ov"])); self.l3.setPixmap(cv_to_qpix(r["vis"]))
        vals = {"健康组织(暗红)": r["stats"].get("健康组织(暗红)", 0),
                "坏组织(亮红)": r["stats"].get("坏组织(亮红)", 0),
                "坏死骨(黑)": r["stats"].get("坏死骨(黑)", 0),
                "叠加指数": r["stats"].get("坏死骨下仍有红色信号占比(%)", 0)}
        for title, v, b in self.stat_views:
            val = vals[title]; v.setText(f"{val:.1f} %"); b.setValue(int(round(val)))
        if r["extra"].get("mode") == "gray_candidate":
            bone = float(r["stats"].get("坏死骨(黑)", 0))
            names = {title: (value, progress) for title, value, progress in self.stat_views}
            names["坏死骨(黑)"][0].setText(f"{bone:.1f} % 候选")
            names["坏死骨(黑)"][1].setValue(int(round(bone)))
            for title in ["健康组织(暗红)", "坏组织(亮红)"]:
                names[title][0].setText("不可见")
                names[title][1].setValue(0)
            names["叠加指数"][0].setText("不支持")
            names["叠加指数"][1].setValue(0)
            self.big.setText(f"灰度候选：坏死骨 {bone:.1f}% · 需人工确认")
            self.statusBar().showMessage("灰度图按明暗找候选坏死骨；红色组织不可见，因此不计算红组织比例")
            return
        dom = max([(vals[k], k) for k in ["健康组织(暗红)", "坏组织(亮红)", "坏死骨(黑)"]])
        self.big.setText(f"结论: {dom[1]}为主 ({dom[0]:.1f}%) · 叠加指数 {vals['叠加指数']:.1f}%")
        self.statusBar().showMessage("识别完成 · 可导出报告")
    def save(self):
        if not self.last: QMessageBox.information(self, "提示", "先识别再导出"); return
        p, _ = QFileDialog.getSaveFileName(self, "导出", os.path.join(BASE, "report.png"), "PNG (*.png)")
        if not p: return
        canvas = np.hstack([cv2.resize(self.img, (400, 400)), cv2.resize(self.last["ov"], (400, 400)), cv2.resize(self.last["vis"], (400, 400))])
        cv2.imwrite(p, canvas)
        QMessageBox.information(self, "完成", f"已导出:\n{p}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    m = MainWin(); m.show(); sys.exit(app.exec())
