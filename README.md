# Qwen-IR

Qwen-IR 是一个面向 All-in-One 图像修复的研究项目：使用 Qwen3-VL 将低质量图像解析为结构化退化先验，再以全局语义条件和 layout 驱动的空间专家共同控制 Conditional UNet 恢复主干。

项目当前已经验证了退化类型概率与严重度先验（A3）的有效性；  
残差预测（R 系列）、全局 layout FiLM（A5.0）和多尺度空间专家（S 系列）正在按消融计划逐步推进。

> 研究目标不是让 Qwen 直接生成修复图像，而是让大模型回答“发生了什么退化”，再由专门的图像恢复网络回答“如何修复”。

## 方法总览

<img width="2333" height="796" alt="image" src="https://github.com/user-attachments/assets/71e5691f-b076-4def-9019-b72a8449cd86" />


图中白底块表示当前已经实现或验证的主干，灰色块表示最终设计中的 layout 空间路由模块。

## 核心思路

整个方法分为四个阶段：

1. **图像评估**：Qwen3-VL 对 LQ 图像进行视觉语言推理，输出结构化退化描述。
2. **全局退化理解**：校准后的退化概率与五类严重度编码为 `deg_context`，告诉 UNet“修什么”。
3. **Confidence修正**：根据模型对退化判断的Confidence，对把握不高的图像进行修正。
4. **Layout 空间路由**：layout 属性决定是否调用 depth、segmentation、DoG 等解析专家，告诉 UNet“在哪里修”。

### 图像评估

| 字段 | 维度 | 含义 |
|---|---:|---|
| `severity_5` | 5 | noise、blur、haze、rain、low-light 的严重度 |
| `main_logits_5` | 5 | Qwen 对五种主退化候选的原始平均 log-probability |
| `main_probs_5` | 5 | temperature calibration 后的五类概率 |
| `layout_10` | 10 | 全局、局部、物体、方向、深度、阴影、纹理等属性 |
| `raw_margin_confidence` | 1 | 原始 top-1 与 top-2 概率差 |
| `calibrated_confidence` | 1 | `max(main_probs_5)`，用于可靠性分析与可选专家路由 |




### 全局退化理解
图像五种退化严重程度由Qwen直接输出，限定为如下几种描述词，并映射到0-1之间：
```text
none = 0
mild = 1/3
moderate = 2/3
severe / serious = 1
```
calibrated probabilities通过固定main degradation输出token，计算不同退化输出的概率得到。Qwen 原始 logits 使用训练集拟合的 temperature 进行校准。
当前退化理解流程如下：
```text
[severity_5, calibrated probabilities_5]
              ↓
LayerNorm → Linear → GELU → Linear
              ↓
deg_context [B, 512]
```
将deg_context在当前 Conditional UNet 中转换为 degradation prompt，并加到 timestep embedding：
```math
t' = t + \mathrm{Prompt}(z_{deg})
```
每个 ResBlock 根据 `t'` 产生通道级 scale/shift，从而形成全局退化条件。

### Confidence 修正

当前Confidence的计算方法为top前2的calibrated probabilities的差值。

目前已经完成的 A4-clean 和 A4-corrupt 使用如下修正方式：

```math
z = c z_{Qwen} + (1-c)z_{unknown}
```
最终模型在错误例子上有一定提高，但在比例更高的正确例子上降低更明显，导致最终损害了模型性能。  
后续如何利用仍待探索。

### Layout 与空间专家

`layout_10` 包含以下属性：

```text
global
local_region
object_specific
continuous
discrete
directional
depth_dependent
shadow_dependent
texture_dependent
uncertain
```

最终空间路由计划为：

| 专家 | Layout gate | 输出 |
|---|---|---|
| Depth Anything | `depth_dependent` | 深度相关空间图 |
| SegFormer | `max(local_region, object_specific)` | 区域/物体 mask |
| DoG | `max(texture_dependent, directional, discrete)` | 纹理与方向响应图 |

各个专家输出将会通过confidence与uncertain调制：
```math
F_{out} = F_{base} + c \cdot u \cdot g_{layout} \cdot F_{expert}
```

专家输出计划生成多尺度 `[B,C,H,W]` gamma/beta map，通过 SFT 调制 UNet feature：

```math
F' = F \odot (1 + \gamma) + \beta
```

### Image 与 Residual 输出
目前正在尝试对比直接输出修复后图像，与输出修复前后的Residual


## 实验系列

| 系列 | 用途 | 配置目录 |
|---|---|---|
| G | 单任务与数值链路 sanity check | `config/train/G/`、`config/test/G/` |
| A | Direct-image prior 消融 | `config/train/A/`、`config/test/A/` |
| R | A0–A3 的 residual 对照 | `config/train/R/`、`config/test/R/` |
| A5.0 | A3 + global layout FiLM | 待实现 |
| S | 控制集与多尺度空间专家 | `config/train/S/`、`config/test/S/` |

R 系列与 A 系列对应关系：

```text
R0 ↔ A0
R1 ↔ A1
R2 ↔ A2
R3 ↔ A3
```


## 项目结构

```text
Qwen_IR/
├── config/
│   ├── train/                 # G/A/R/S 训练配置
│   ├── test/                  # G/A/R/S 测试配置
│   └── qwen_temperature_0611_sample.json
├── docs/assets/               # README 与论文草图资产
├── module/
│   ├── backbone/              # Conditional UNet wrapper
│   ├── degradation_prompt/    # severity/probability encoder
│   ├── layout_prompt/         # StructuredPriorV2 与 layout adapter
│   ├── pipeline/              # 数据、训练、验证、测试流程
│   ├── qwen/                  # Qwen structured-prior 导出
│   ├── runtime/               # 在线推理适配器
│   └── vendor/tpgdiff/        # 内置 Conditional UNet/SDE 兼容运行源码
├── script/
│   ├── train_assess_tpgd.py
│   ├── test_assess_tpgd.py
│   ├── run_train_sequence.bash
│   └── calibrate_qwen_prior.py
├── tests/                     # schema、prior、confidence、residual 测试
├── log/                       # 指标、恢复图与队列日志
└── 实施计划.md
```



## 当前状态与路线图

- [x] A0: Plain UNet Direct-GT baseline
- [x] Temperature calibration 与置信度统计
- [x] A1-A3: Oracle / Qwen probabilities / severity 消融
- [x] A4, A4cor: Clean 与 corruption confidence gate 负结果分析
- [x] R: Image / residual prediction target
- [ ] R0–R3 完整 residual 对照
- [ ] A5.0：A3 + global layout FiLM
- [ ] 局部/复合退化控制集
- [ ] Depth/SegFormer/DoG 多尺度空间专家



