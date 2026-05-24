# 单物 / 多物字段对照表

这份表的目的不是追求“历史上所有字段一字不漏”，而是帮助快速回答这几个问题：

1. 旧单物的字段骨架是什么
2. 当前多物实际落地了哪些字段
3. 哪些字段是沿用的
4. 哪些字段是多物新增的
5. 哪些旧字段目前没有完整接入多物

## 一、总体结论

### 1. 骨架层

当前多物流程仍然沿用了旧单物的两层骨架：

- `section0`
- `section1`

所以从结果组织方式上看，多物不是完全另起炉灶，而是在单物骨架上增加“多目标归属层”。

### 2. 当前多物的策略

多物不是把旧单物字段完整照搬，而是：

- 保留核心超导参数字段
- 压缩容易串料的宽字段
- 新增 target 归属、证据来源和质控字段

所以现在的多物结果更偏：

- 字段更少
- 归属更重
- 质控更强

## 二、字段分层对照

| 层级 | 旧单物 | 当前多物 | 说明 |
|---|---|---|---|
| 结果骨架 | `section0` / `section1` | `section0` / `section1` | 这一层是沿用的 |
| 材料标识 | 通常以单个主体系为中心 | `paper_id / target_id / canonical_name / aliases_used / excluded_siblings` | 多物新增，服务于多目标拆分 |
| 证据追踪 | 以抽取结果为主 | `provenance`、`supporting_evidence` | 多物新增 |
| 质控信息 | 相对弱 | `quality_control / ambiguity_flags / omission_reasons` | 多物新增 |

## 三、Section0 对照

### 1. 当前多物实际支持的 Section0 类型

当前多物代码里真正会聚合到 `section0` 的字段只有这 4 类：

- `tuning`
- `carrier_concentration`
- `secondary_phase`
- `stack_descriptor`

这一点直接写在 [run_aggregate.py](/C:/Users/Administrator/Desktop/fsdownload/sc-ie/BM_multi_match_pipeline/run_aggregate.py) 里。

### 2. 对照表

| Section0 字段 | 旧单物 | 当前多物 | 状态 |
|---|---|---|---|
| `tuning` | 有 | 有 | 保留 |
| `carrier_concentration` | 有 | 有 | 保留，但更保守 |
| `secondary_phase` | 有 | 有 | 保留，但目前出现较少 |
| `stack_descriptor` | 有 | 有 | 保留，但目前收得很紧 |

### 3. 当前多物的收缩点

虽然表面上 4 个字段都还在，但多物里的实际判定比旧单物严格很多：

- 测量设置类信息不再轻易当 `tuning`
- 一般结构标签不再轻易当 `stack_descriptor`
- 没有明确数值时，不轻易给 `carrier_concentration`

所以可以理解为：

- `Section0` 字段名沿用了
- 但字段边界明显收紧了

## 四、Section1 对照

### 1. 当前多物实际支持的 Section1 类型

当前多物代码里真正会聚合到 `section1` 的字段是：

- `Tc`
- `Jc`
- `Hc1`
- `Hc2`
- `Hc`
- `P_sc`
- `P_nsc`
- `lambda`
- `xi`

### 2. 对照表

| Section1 字段 | 旧单物 | 当前多物 | 状态 |
|---|---|---|---|
| `Tc` | 有 | 有 | 保留 |
| `Jc` | 有 | 有 | 保留 |
| `Hc1` | 有 | 有 | 保留 |
| `Hc2` | 有 | 有 | 保留 |
| `Hc` | 有 | 有 | 保留 |
| `P_sc` | 有 | 有 | 保留 |
| `P_nsc` | 有 | 有 | 保留 |
| `lambda` | 有 | 有 | 保留 |
| `xi` | 有 | 有 | 保留 |

### 3. 当前多物的特点

`section1` 是当前多物继承旧单物最完整的一块。

也就是说：

- 核心超导参数字段基本都沿用了
- 多物和单物最大的差别，不在字段名，而在“这些字段归谁”

## 五、多物新增字段

这些字段不是旧单物结果骨架里的核心字段，而是多物为了防串料、做归属和做审计新增的。

| 字段 | 当前多物 | 作用 |
|---|---|---|
| `paper_id` | 是 | 论文标识 |
| `target_id` | 是 | 单个目标材料标识 |
| `canonical_name` | 是 | 当前 target 的标准显示名 |
| `aliases_used` | 是 | 当前 target 实际识别时用到的别名 |
| `excluded_siblings` | 是 | 当前 target 的兄弟材料 |
| `provenance.accepted_candidate_ids` | 是 | 最终接受了哪些候选事实 |
| `provenance.ambiguous_candidate_ids` | 是 | 哪些候选事实仍然存在歧义 |
| `quality_control.has_target_specific_superconducting_evidence` | 是 | 是否有明确 target 级超导证据 |
| `quality_control.ambiguity_flags` | 是 | 当前 target 的歧义提示 |
| `quality_control.omission_reasons` | 是 | 为什么某些字段为空 |
| `supporting_evidence` | 是 | 同值重复证据的压缩保留 |

## 六、旧单物目前没有完整迁入多物的部分

从当前多物代码和输出结果来看，下面这些不是“完全没有”，但至少还没有像单物那样完整、稳定地迁入：

### 1. 更宽的 Section0 内容

旧单物里有些偏 family-level、结构描述型、材料比较型的信息，以前更容易进入结果；
现在多物里为了防串料，很多被压掉了。

所以目前多物对 `section0` 的支持是：

- 仍保留字段名
- 但实际收录范围更窄

### 2. family-level 信息

旧单物因为本质是“单体系”，很多体系级描述不会显得冲突；
现在多物里这类信息往往会：

- 留在 `fact_candidates`
- 留在 `matched` 的 ambiguity
- 或者在 final 阶段被过滤

也就是说：

- 这部分不是字段名没了
- 而是当前多物没有正式的 `system-level` 输出容器

### 3. 冲突值并存

单物流程里很多时候只需要给出一个主值；
多物里同一 target 可能有：

- 不同来源
- 不同定义
- 不同 criterion

当前多物已经有 `supporting_evidence` 和 `ambiguity_flags`，但对“冲突值并存”的显式建模仍然比较初级。

## 七、当前最准确的理解

如果用一句话总结当前字段关系，可以这样说：

- 多物沿用了单物的 `section0 / section1` 主骨架和核心超导参数字段
- 但没有完整继承旧单物的宽字段覆盖
- 同时新增了一整层 target 归属、证据追踪和质控字段

所以现在更像是：

- “单物字段骨架”
- 加上“多物归属与审计外壳”

而不是：

- “把旧单物字段完整平移到多物”

## 八、建议理解方式

目前最稳的理解是：

1. 如果你关心的是结果结构
   - 多物与单物是兼容的，因为都还是 `section0 / section1`

2. 如果你关心的是字段覆盖范围
   - 多物目前比单物更保守，尤其在 `section0`

3. 如果你关心的是可追溯性
   - 多物明显比单物更强，因为增加了：
   - `target_id`
   - `aliases_used`
   - `excluded_siblings`
   - `provenance`
   - `quality_control`

## 九、后续如果要继续对齐

如果后面要把多物再往单物字段体系上靠，可以优先做这三件事：

1. 明确列出旧单物所有最终字段清单
2. 对每个字段标记：
   - 多物已支持
   - 多物部分支持
   - 多物未支持
3. 再决定哪些字段值得迁入多物，哪些应该继续保持保守

这样会比直接“全搬过来”更稳。
