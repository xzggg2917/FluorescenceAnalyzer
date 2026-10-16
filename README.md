# FluorescenceAnalyzer · 荧光三分类智能识别平台

![platform](https://img.shields.io/badge/platform-Windows%20x64-blue)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-research--use-lightgrey)

颌骨骨髓炎荧光三分类桌面端：**健康组织(暗红) / 坏组织(亮红) / 坏死骨(黑)**，
三视图 + 四指标 + 一键导出，解压即用，无需安装 Python。

- 下载：[Releases](../../releases) → `FluorescenceAnalyzer-win64.zip`
- 解压后双击 `FluorescenceAnalyzer/FluorescenceAnalyzer.exe`

## 界面

- ① 原图 Original / ② 半透明叠加 Overlay / ③ 边界勾勒 Contour
- 底部四卡：健康组织(暗红) / 坏组织(亮红) / 坏死骨(黑) / 叠加指数
- 左侧：上传 / 拖拽 / 开始识别 / 导出报告 + 颜色图例

## 方法

1. 背景锁定：与边缘连通的暗区判衬底，永不计入荧光/坏死骨。
2. 荧光 ROI：红 hue + 红度阈值，取最大连通域，轻膨胀吃回边缘。
3. ROI 内分层：黑骨优先（荧光内部小暗芯，面积熔断 + 包围约束），
   亮/暗红用 `KMeans(V,R)` 自适应二分，`R>150 且 V>110` 强制亮红。
4. 灰度图：只输出局部暗斑候选（红组织不可见，不推断叠加）。
5. 边界：外圈细绿线 = 荧光 ROI，内圈细紫红线 = 亮芯；染色跳过线像素。

## 本地运行

```powershell
pip install -r requirements_desktop.txt
python desktop_app.py
```

## 打包

```powershell
python build_desktop.py
```

产物：`dist/FluorescenceAnalyzer/FluorescenceAnalyzer.exe`。
打 tag 即由 GitHub Actions 自动构建并发布 Release：

```powershell
git tag v1.0.x
git push origin v1.0.x
```

## 仓库结构

```text
desktop_app.py      桌面端 UI(三图 + 四卡)
color_model.py      荧光 ROI + 三分类 + 边界绘制
style.qss           深色大字体主题
assets/logo.ico     软件图标
assets/samples/    自动生成的仿真示例
build_desktop.py    PyInstaller 一键打包
.github/workflows/release-desktop.yml  tag 自动发布
```

## 已知限制

- 灰度图无红色通道，只能给坏死骨候选比例，需人工确认。
- EXE 约数百 MB（含 Python + OpenCV + sklearn），属正常现象。
