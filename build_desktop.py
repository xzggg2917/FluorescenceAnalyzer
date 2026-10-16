# -*- coding: utf-8 -*-
"""一键打包: 生成与图二界面一致的 Windows 桌面端 EXE(含 Logo 图标)。"""
import os, shutil, subprocess, sys

BASE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(BASE, "dist")
APP_NAME = "FluorescenceAnalyzer"

ICON_PNG = os.path.join(BASE, "assets", "logo.png")
ICON_ICO = os.path.join(BASE, "assets", "logo.ico")

def ensure_icon():
    if os.path.exists(ICON_ICO):
        return ICON_ICO
    if not os.path.exists(ICON_PNG):
        print(f"[跳过图标] 未找到 {ICON_PNG}，请把你发的 Logo 保存为该路径后再打包。")
        return None
    try:
        from PIL import Image
        img = Image.open(ICON_PNG).convert("RGBA")
        img.save(ICON_ICO, sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])
        print(f"[图标] 已生成 {ICON_ICO}")
        return ICON_ICO
    except Exception as e:
        print(f"[图标失败] {e}，将使用默认图标打包。")
        return None

def main():
    try:
        import PyInstaller  # noqa
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller>=6.0"])
    icon = ensure_icon()
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--name", APP_NAME,
        "--windowed", "--onedir",
        "--collect-all", "sklearn",
        "--collect-all", "scipy",
        "--add-data", f"style.qss{os.pathsep}.",
        "--add-data", f"assets{os.pathsep}assets",
        "desktop_app.py",
    ]
    if icon:
        cmd.insert(-1, "--icon")
        cmd.insert(-1, icon)
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=BASE)
    print(f"\n打包完成: {os.path.join(DIST, APP_NAME, APP_NAME + '.exe')}")
    print("把整个 dist/FluorescenceAnalyzer 文件夹压缩为 ZIP 后上传到 GitHub Release 即可。")

if __name__ == "__main__":
    main()
