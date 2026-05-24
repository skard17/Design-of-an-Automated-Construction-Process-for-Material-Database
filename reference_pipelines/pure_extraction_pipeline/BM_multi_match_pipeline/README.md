# 多物文献抽取流程说明

这个目录是独立于单物流程的新支线，专门用于处理“一篇文献包含多个材料目标”的抽取问题。

它的核心目标不是把单物流程硬套到多物论文上，而是把任务拆成：

1. 先识别论文里有哪些材料目标
2. 再从整篇论文里抽候选事实
3. 然后按目标材料做归属判断
4. 最后汇总成单材料 JSON

这样做的主要目的是减少串料，也就是把 A 材料的信息错误归到 B 材料下面。

## 一、当前主流程

当前真正会参与运行的主流程脚本是：

- `run_manifest.py`
- `run_fact_candidates.py`
- `run_matcher.py`
- `run_aggregate.py`
- `run_pipeline.py`

它们分别对应：

1. `manifest`
   - 先识别这篇论文里有哪些材料目标
2. `fact_candidates`
   - 面向整篇论文抽候选事实池
3. `matcher`
   - 针对单个目标材料，判断哪些候选事实属于它
4. `aggregate`
   - 把已接受的事实汇总成最终单材料 JSON
5. `pipeline`
   - 串联整条流程

## 二、当前目录结构

### 1. 必须保留的代码和配置

这些文件/目录是当前流程真正要用的，不建议删除：

- `run_manifest.py`
- `run_fact_candidates.py`
- `run_matcher.py`
- `run_aggregate.py`
- `run_pipeline.py`
- `utils/`
- `prompts/`
- `schemas/`
- `config.yaml`
- `requirements.txt`

### 2. 建议保留的辅助文件

这些文件不是主流程本体，但现在对测试、回归、维护有用，建议保留：

- `prepare_benchmark_papers.py`
  - 把 benchmark 样例整理到 `papers/`
- `cleanup_stale_outputs.py`
  - 清掉 manifest 缩减后残留的旧结果
- `build_regression_baseline.py`
  - 生成当前回归基线
- `diff_regression_baseline.py`
  - 对比两份基线，查看是否退化
- `benchmark_mapping.json`
- `benchmark_audit.md`
- `regression_baseline.json`
- `regression_baseline_prev.json`

### 3. 输入与输出目录

- `papers/`
  - 存放待处理的 markdown 文献
- `outputs/manifests/`
  - 论文级材料清单
- `outputs/fact_candidates/`
  - 整篇论文的候选事实池
- `outputs/fact_candidates_chunks/`
  - 长文分块候选结果
- `outputs/matched/`
  - 单材料归属结果
- `outputs/final_targets/`
  - 最终单材料 JSON

### 4. 可以删除的目录或文件

这些不是核心代码，删掉不会破坏流程本身，后面需要时可以重新生成：

- `__pycache__/`
- `utils/__pycache__/`
- `logs/`
- `outputs/`
  - 如果你不需要保留已有测试结果，可以整体删除后重跑
- `regression_baseline.json`
- `regression_baseline_prev.json`
  - 如果你不需要保留当前回归快照，也可以删

### 5. 暂时可留可不留的目录

- `examples/`
  - 目前不是主流程必须项
  - 只是放一个最小示例，影响不大

## 三、为什么不建议现在删“旧代码”

从当前目录看，没有发现那种“已经完全废弃、确定不会再被调用的整套旧主流程脚本”。

现在大多数文件都分成三类：

1. 主流程代码
2. 回归/清理/准备工具
3. 测试输出与日志

真正适合删的是第 3 类，不是第 1 类和第 2 类。

也就是说：

- 现在不建议删主脚本
- 可以清缓存、日志、输出
- 如果后面你想进一步精简目录，建议做成“开发版”和“交付版”两套目录结构，而不是直接删脚本

## 四、流程设计思路

这个多物流程和旧的“先按材料切，再直接抽最终结果”不同。

当前方案是：

1. 论文级 `manifest`
2. 论文级 `fact_candidates`
3. 单目标 `matcher`
4. 单目标 `aggregate`

核心原则是：

- 宁可漏，不要串
- 宁可保留歧义，也不要强行归属
- 候选事实和归属判断分开做

## 五、运行方式

### 1. 跑整篇论文

```bash
python run_pipeline.py --paper_id K2Cr3As3_Rb2Cr3As3
```

### 2. 分阶段运行

```bash
python run_manifest.py --paper_id K2Cr3As3_Rb2Cr3As3
python run_fact_candidates.py --paper_id K2Cr3As3_Rb2Cr3As3
python run_matcher.py --paper_id K2Cr3As3_Rb2Cr3As3 --target_id K233
python run_aggregate.py --paper_id K2Cr3As3_Rb2Cr3As3 --target_id K233
```

### 3. 长文断点恢复

`run_fact_candidates.py` 支持长文分块和断点恢复。

如果之前已经跑出部分 chunk，可以只重建汇总结果，不再新增模型请求：

```bash
python run_fact_candidates.py --paper_id Nb085X015_Ti_Zr_Hf --max_new_chunks 0
```

这个命令会：

- 复用已有 chunk
- 重建 `outputs/fact_candidates/<paper_id>.json`
- 不再新增 chunk 请求

## 六、回归测试

### 1. 生成当前基线

```bash
python build_regression_baseline.py
```

会生成：

- `regression_baseline.json`

基线里会记录每个 target 的：

- 是否有 final JSON
- `section0/section1` 的事实数量
- accepted / ambiguous candidate 数量
- `ambiguity_flags`
- `omission_reasons`

### 2. 对比两份基线

```bash
python diff_regression_baseline.py --old C:\path\to\old.json --new C:\path\to\new.json
```

这个脚本会帮助判断：

- 哪些论文/材料结果变好了
- 哪些结果变干净了
- 哪些地方退化了

## 七、和旧 experimental / single 的衔接

如果你的策略是：

- 先走旧的实验文筛选
- 再走旧的单物判定
- 只有“不是单物”的论文才进入多物流程

那么现在这个目录已经可以直接承接这种用法。

### 1. 旧规则快照

为了避免来回切目录，我已经把旧入口使用的 prompt 快照复制到这里：

- `legacy_gate_prompts/experimental_L1.md`
- `legacy_gate_prompts/experimental_L2L3.md`
- `legacy_gate_prompts/single_system.md`

这三份文件是当前多物入口所参考的旧规则版本说明，不会影响主流程运行，但便于你手动核对边界。

### 2. 从旧 single 失败集导入多物

新增桥接脚本：

- `run_from_legacy_gate.py`

它的用途是：

1. 读取旧流程产生的 `experimental.txt`
2. 读取旧 `single_out/*.json`
3. 找出其中 `single_system=0` 的论文
4. 把这些论文复制到当前 `papers/`
5. 可选地直接启动当前多物流程

### 3. 典型用法

只导入旧 single 判定失败的论文，不立即运行：

```bash
python run_from_legacy_gate.py ^
  --source_papers_dir C:\path\to\papers_md ^
  --experimental_file C:\path\to\classification_out\experimental.txt ^
  --single_outputs_dir C:\path\to\single_out
```

导入后立刻跑多物流程：

```bash
python run_from_legacy_gate.py ^
  --source_papers_dir C:\path\to\papers_md ^
  --experimental_file C:\path\to\classification_out\experimental.txt ^
  --single_outputs_dir C:\path\to\single_out ^
  --run_pipeline
```

如果你希望把“旧 single 缺结果/解析失败”的论文也一并纳入多物候选池：

```bash
python run_from_legacy_gate.py ^
  --source_papers_dir C:\path\to\papers_md ^
  --experimental_file C:\path\to\classification_out\experimental.txt ^
  --single_outputs_dir C:\path\to\single_out ^
  --include_uncertain
```

脚本会额外生成：

- `legacy_not_single_ids.txt`
- `legacy_uncertain_ids.txt`

方便你人工检查。

## 八、当前已验证状态

当前这条多物支线已经能跑通从原文到 final JSON 的整条链路，并且已经做过 benchmark 回归。

目前可以认为：

- 规整数值型多物论文效果最好
- `respectively` 型多材料论文效果较稳
- family-style / 比较型论文目前仍偏保守，主要问题是召回不足，不再是严重串料

## 九、和单物流程的关系

这个目录是完全独立的。

请不要把结果写回单物流程的目录，尤其不要写入这些旧目录：

- `single_out`
- `results_part`
- `post_processed`

多物流程所有输入输出都应留在：

- `BM_multi_match_pipeline/`

## 十、如果你现在想精简目录

推荐的安全清理顺序是：

1. 删除 `__pycache__/`
2. 删除 `logs/`
3. 如无保留需要，再删除 `outputs/`
4. 如无保留需要，再删除 `regression_baseline*.json`

不建议删除：

- `run_*.py`
- `utils/`
- `prompts/`
- `schemas/`
- 回归工具脚本

如果你后面希望，我可以继续帮你做一步：

1. 直接清掉这个目录下所有缓存和日志
2. 或者进一步把这个目录整理成“最小交付版”
