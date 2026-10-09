# 第 9 篇 · 数值计算与图像处理：NumPy、SciPy、Pillow

数据分析链路中，除表格处理外还涉及两类底层能力：数值计算（数组运算、线性代数、插值与优化、信号处理）与图像处理。本节以 NumPy、SciPy、Pillow 三个库为例，说明这些能力的调用方式及各自的适用条件。

前几节的数据处理以 pandas 为主，其底层依赖 NumPy 的数组结构；而线性代数、插值、优化与信号处理由 SciPy 提供，图像读写与变换由 Pillow 提供。三者的分工为：NumPy 提供数组与向量化运算，SciPy 在其上提供数值算法，Pillow 提供图像接口。本节不重复各库的基础教程，而是按"数组计算 → 数值算法 → 信号处理 → 图像处理"的顺序，说明常用接口的调用方式与适用条件，并与本系列的 NumPy、SciPy、Pillow 专题博客互为补充。实现基于 NumPy 1.26+、SciPy 1.11+ 与 Pillow 10.0+，运行环境为 Python 3.10+（OpenCV 4.8+ 为可选依赖）。

---



## 1 NumPy：数组计算

NumPy 的基本用法是以数组运算替代逐元素循环。借助广播机制，形状不同的数组在满足广播规则时可直接进行逐元素运算，无需显式复制：

```python
import numpy as np
rng = np.random.default_rng(42)
X = rng.random((1000, 50))

Xc = X - X.mean(axis=0)                  # 广播：(1000,50) - (50,) 每列减均值
out = np.empty_like(Xc)
np.multiply(Xc, 2, out=out)               # out= 复用缓冲，省一次内存分配

a = np.arange(6).reshape(2, 3)
print((a - a.mean(axis=0)).round(2))
```

```text
[[-1.5 -1.5 -1.5]
 [ 1.5  1.5  1.5]]
```

上述结果为按列去均值：`a.mean(axis=0)` 得到形状为 (3,) 的行均值，与形状 (2,3) 的数组相减时按列广播。向量化与广播的作用在于避免 Python 层循环，减少解释执行开销；其性能提升幅度取决于运算类型与数据规模，因此加速比应以实际测量为准。`out=` 参数用于复用已有数组的存储空间，减少内存分配。

---



## 2 SciPy：线性代数

线性代数中的常见操作是求解线性方程组 `Ax = b`。直接计算 `A` 的逆再与 `b` 相乘的做法在数值上不稳定：当 `A` 接近奇异时，其逆矩阵的元素会被放大，求解结果的误差随之增大。矩阵的性态可用**条件数**衡量，条件数越大表示矩阵越接近奇异：

```python
from scipy import linalg

def robust_solve(A, b):
    cond = np.linalg.cond(A)
    if cond > 1e10:                                  # 病态：改用最小二乘
        return linalg.lstsq(A, b)[0], {"cond": cond, "method": "lstsq"}
    return linalg.solve(A, b), {"cond": cond, "method": "solve"}
```

函数按条件数选择解法：条件数较小时使用 `solve`（基于 LU 分解），求解方阵方程组；条件数较大时改用 `lstsq`（最小二乘），在方程无精确解或矩阵病态时给出误差最小的解。条件数阈值为经验取值，实际使用时可按问题的精度要求调整。

---



## 3 SciPy：插值与优化

插值用于在已知数据点之间估计中间取值。`interp1d` 已不推荐使用，宜改用 `CubicSpline`：三次样条在相邻节点处保证函数值与一阶、二阶导数连续，插值曲线较为平滑。需注意插值仅适用于数据范围之内，超出范围的外推结果不再受数据约束：

```python
from scipy import interpolate, optimize

cs = interpolate.CubicSpline(x, y)          # 推荐；interp1d 已弃用

res = optimize.minimize(fun, x0, jac=grad, method="L-BFGS-B", bounds=bounds)
if not res.success:
    raise RuntimeError(res.message)          # 必须检查收敛，不能假定成功
```

优化部分使用 `minimize` 求解无约束或有界约束问题，`method="L-BFGS-B"` 适用于带边界的可微目标函数，`jac` 提供梯度可加快收敛。优化结果需检查 `res.success`：数值优化可能在迭代次数或精度限制下提前终止，未检查即使用结果会引入错误。

---



## 4 信号处理：滤波与频谱

滤波是对序列进行频域或时域的局部处理。高阶滤波器若直接以传递函数系数（`b, a`）实现，其极点分布会导致数值不稳定，因此改用二阶节（SOS）形式：

```python
from scipy import signal, fft

sos = signal.butter(6, 20, btype="low", fs=fs, output="sos")   # SOS：高阶滤波更稳定
y = signal.sosfiltfilt(sos, x)                                  # 零相位滤波，避免端点失真
amp = np.abs(fft.rfft(y)) / len(y) * 2                          # 实信号用 rfft + 归一化
```

`sosfiltfilt` 对序列作前向与反向两次滤波，从而抵消滤波引入的相位偏移（零相位滤波）；代价是无法用于在线处理，仅适用于已获得完整序列的场景。频谱部分使用 `rfft`：实信号的频谱具有共轭对称性，只需计算一半频点；除以序列长度并乘以 2，可将幅值归一化到与时域幅值一致的量级。

---



## 5 Pillow：图像处理与批量规格统一

单张图像的读取与变换较为直接，实际工程中的难点在于批量处理时的规格统一。以下函数将输入图像统一为固定尺寸的方形画布，并输出处理结果路径：

```python
from pathlib import Path
from PIL import Image, ImageOps

def normalize_image(src: Path, dst: Path, size=512):
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)          # 按 EXIF 纠方向（否则可能躺倒）
        im = im.convert("RGB")                    # 去 alpha，避免透明变黑
        im.thumbnail((size, size), Image.LANCZOS) # 等比缩放 + 抗锯齿
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        dst.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dst, format="PNG")
    return str(dst)
```

该函数依次完成四项处理：按 EXIF 信息纠正方向（部分图像的像素数据与显示方向不一致）；转换为 RGB 以去除透明通道（避免透明区域在后续合成中变为黑色）；等比缩放至画布范围内（`thumbnail` 保持宽高比，`LANCZOS` 为重采样算法）；在以黑色铺底的方形画布上居中粘贴，使输出尺寸统一。

批量处理时，可用进程池并行执行上述函数，并对结果作质量校验（检查文件可正常打开、尺寸符合规格）：

```python
from concurrent.futures import ProcessPoolExecutor
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
```

对 JPEG 大图，可在解码阶段即降采样，以减少内存占用：`im.draft("RGB", (512, 512))` 会在读取时按比例缩小，而非读取完整分辨率后再缩放。

---



## 6 小结

NumPy 提供数组与向量化运算，广播机制使不同形状的数组可直接参与逐元素计算，`out=` 参数用于复用存储空间。SciPy 在此基础上提供数值算法：线性代数部分按条件数在 `solve` 与 `lstsq` 之间选择解法，避免直接求逆；插值使用 `CubicSpline`，且仅适用于数据范围之内；优化需检查收敛标志后再使用结果。信号处理使用二阶节形式的高阶滤波与零相位滤波，频谱分析对实信号使用 `rfft` 并作幅值归一化。Pillow 用于图像读写与变换，批量处理时需统一尺寸、方向与颜色模式，并对结果作质量校验。

至此，本系列以"实习僧 · 互联网 IT 类实习岗位"数据为线索，走完了从环境搭建、数据获取与加工，到分析与建模、数值与图像处理的完整链路。各篇共享若干共同约定：处理前先确定统计范围与口径，结论以此为前提；预处理参数仅在训练集估计，特征文本去除目标信息；原始数据与原文保持只读，口径调整另行留档；方法前提无法满足时，以修订并记录的方式处理；随机种子、依赖版本、阶段编号与回归锚点统一固定，以保证结果可复现。采集环节需遵守目标网站的使用协议并控制访问频率；文中不公开原始数据，以及任何包含个人或公司敏感信息的明细。

**（全系列完）**

---



## 7 延伸参考

- 相关文章：[0. NumPy 系列教程](https://blog.csdn.net/MoRanzhi1203/article/details/152146993)、[NumPy 数据分析与图像处理入门](https://blog.csdn.net/MoRanzhi1203/article/details/152096309)、[SciPy 信号处理解析](https://blog.csdn.net/MoRanzhi1203/article/details/153080126)、[SciPy 插值方法详解](https://blog.csdn.net/MoRanzhi1203/article/details/152191746)、[Pillow 基础图像操作与数据预处理](https://blog.csdn.net/MoRanzhi1203/article/details/153296506)、[Pillow 图像分割、切片与拼接处理](https://blog.csdn.net/MoRanzhi1203/article/details/159001210)；
- NumPy / SciPy / Pillow 官方文档；


上一篇：[08 机器学习建模与实验评估](./第 8 篇 · 机器学习建模与实验评估：建模流程与对照实验.md)








