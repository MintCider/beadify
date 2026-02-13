# Beadify — 拼豆图纸生成器

把像素画自动转换成拼豆图纸，标注色号、用量，直接照着拼。

支持 macOS、Windows、Linux。

## 功能

- 自动识别像素画网格
- 匹配最接近的拼豆色号（支持 Mard 色系）
- 智能合并相近颜色，减少用色数量
- 可选包边、连接断开的部件
- 导出带坐标和图例的高清图纸
- 提供命令行和 GUI 两种使用方式

## 快速开始

### 方式一：用 uv（推荐，无需手动装依赖）

[uv](https://docs.astral.sh/uv/) 是一个快速的 Python 包管理器。如果还没装：

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

装好后直接运行，uv 会自动处理 Python 版本和所有依赖：

```bash
# 命令行版
uv run beadify.py your_image.png

# GUI 版
uv run beadify_gui.py
```

### 方式二：用 pip

需要先安装 Python 3.10 或更高版本。

```bash
# 1. 克隆项目
git clone https://github.com/你的用户名/beadify.git
cd beadify

# 2. 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate   # Windows 用: .venv\Scripts\activate

# 3. 安装依赖
pip install numpy pillow          # 命令行版
pip install numpy pillow PySide6  # GUI 版（需要额外安装 PySide6）

# 4. 运行
python beadify.py your_image.png  # 命令行版
python beadify_gui.py             # GUI 版
```

## 命令行用法

```bash
python beadify.py <图片路径> [选项]
```

常用选项：

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-o` | 输出文件路径 | `<输入文件名>_beadify.png` |
| `-t` | 颜色合并容差（见下方说明） | 取决于算法 |
| `-e` | 启用扩展色系 (P/Q/R/T/Y/ZG) | 关闭 |
| `-c` | 图纸格子大小，单位像素，越大图纸越清晰 | 60 |
| `-b` | 手动指定输入图片的像素块大小（0=自动检测） | 0 |
| `--origin` | 坐标原点：bl/tl/br/tr | bl（左下） |
| `--metric` | 色差算法（见下方说明） | hybrid |

示例：

```bash
# 基本用法
python beadify.py fox.png

# 指定输出路径，增大容差减少颜色数
python beadify.py fox.png -o output.png -t 18

# 启用扩展色系，使用 CIEDE2000 算法
python beadify.py fox.png -e --metric ciede2000
```

## 颜色合并：容差与色差算法

### 这个功能在做什么？

像素画的颜色和拼豆的颜色不会完全一致。程序先把每个像素块匹配到最接近的拼豆色号，但这样往往会产生很多只用了一两颗的颜色——实际拼的时候更希望减少负担。

**颜色合并**就是把这些用量极少、且和其他颜色很接近的色号合并掉，减少需要使用的颜色种类。

1. **全局相似合并**：在所有使用中的颜色里，把非常接近的两个色号合为一个
2. **局部平滑**：如果一个格子的四周邻居大多是同一个颜色，且差距不大，就跟着邻居走

### 容差 `-t`

容差控制合并的激进程度：

- **容差越大**，合并越多，最终颜色越少，但可能丢失细节
- **容差越小**，保留的颜色越多，越忠实于原图，但需要使用更多颜色
- 设为 **0** 则完全不合并

默认值取决于色差算法：CIE76 默认 15，CIEDE2000 默认 10，hybrid 默认 12。

### 色差算法 `--metric`

三种算法衡量"两个颜色有多不一样"：

| 算法 | 特点 |
| ------ | ------ |
| `cie76` | 色相更精准，明度精度较低 |
| `ciede2000` | 明度更精准，色相精度较低 |
| `hybrid`（默认） | 取 CIE76 和 CIEDE2000 的几何平均值，兼顾色相和明度 |

一般用默认的 `hybrid` 就好。不过可以都切换试试，看看哪个观感更好。

## GUI 用法

启动后点击「打开图片」选择像素画，程序会自动处理并显示四个面板：

- **原图** — 你的输入图片
- **网格检测** — 红线标出识别到的像素块边界
- **拼豆色图** — 匹配后的实际拼豆颜色预览
- **拼豆图纸** — 带色号标注、坐标、图例的最终图纸

左侧面板可以实时调整所有参数（容差、算法、包边、连接等），右侧即时预览效果。满意后点击「导出图纸」保存。

### 容差对比

在左侧「视图」下拉框切换到「容差探索」模式，可以同时输入三个不同的容差值，点击「对比」后并排显示效果。开启「高亮融合区域」会用彩虹边框标出不同容差之间被合并的区域，帮你直观地选择最合适的容差。

## 输入要求

输入图片应该是**透明背景像素画**——每个逻辑像素由多个实际像素组成的网格图（比如从像素画编辑器导出的放大图，或者 AI 生成的像素画）。程序会通过梯度分析自动检测像素块大小。

## 许可证

AGPL-3.0 — 详见 [LICENSE](LICENSE)
