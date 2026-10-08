# scripts/_rework_history — 返工脚本归档

本目录存放**已不再参与现行流程**的返工 / 被取代脚本，仅作追溯之用。

归档判据（两条同时满足）：

1. **零引用**：不被任何其他脚本（`importlib` / `subprocess` / 字符串路径）或文档引用；
2. **已失效**：或已被后续脚本取代，或其产出目标目录（`outputs/paper`、`docs/paper/stage23`、
   `docs/prompts`、`docs/records` 等）已不存在。

> 现行流程（Stage 00~11 流水线 + Stage 12~17 专项）**不依赖**本目录任何文件，
> 删除本目录不会影响 `run_data_pipeline.py` 与正式产物。

## 归档清单

| 脚本 | 阶段 | 归档原因 |
| --- | --- | --- |
| `18d_stage23_1_caption_fix.py` | Stage23.1 | 图题修补一次性稿；零引用；目标 `docs/paper/stage23` 已不存在 |
| `26_stage26_temporal.py` | Stage26.0 | 初版时序脚本，被 `26b`~`26j` 系列取代；零引用 |
| `26h_stage26_4_manuscript_math_audit.py` | Stage26.4 | 中途数学/正文审计稿；零引用 |
| `26i_stage26_5_compute.py` | Stage26.5 | 中途计算稿；零引用 |
| `26j_stage26_6_compute.py` | Stage26.6 | 中途计算稿；零引用 |
| `33b_stage26_7_text_gates.py` | Stage26.7 | 文本门禁校验稿；零引用 |
| `34_stage26_8_word_verify.py` | Stage26.8 | Word 核验稿；零引用 |
| `35_stage26_9a_text_final_verify.py` | Stage26.9A | 文本最终校验稿；零引用 |
| `37_stage27_0_paper_build.py` | Stage27.0 | 论文装配，被 `41_stage27_0a_paper_build.py` 取代；仅被同样归档的 38 引用 |
| `38_stage27_0_verify.py` | Stage27.0 | 装配核验，被 `42_stage27_0a_verify.py` 取代；零引用 |
| `39_stage27_0_image_privacy_scan.ps1` | Stage27.0 | 旧证据图隐私扫描；零引用 |
| `63_fix_figure_7_7_word.py` | 收尾 | 对 Word 的一次性补丁；零引用 |
| `64_replace_stale_heatmap_in_word.py` | 收尾 | 替换 Word 内旧热力图的一次性补丁；零引用 |

## 恢复方式

如需重新启用某个脚本，将其移回上一级目录即可：

```powershell
git mv scripts/_rework_history/26_stage26_temporal.py scripts/26_stage26_temporal.py
```

注意：多数脚本引用的 `outputs/paper`、`docs/paper/stage23` 等目录当前不存在，直接运行会失败，
需先恢复对应目录或调整路径。
