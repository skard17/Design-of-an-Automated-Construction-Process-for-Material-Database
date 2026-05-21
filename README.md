# Step 8 Section Design Agent 使用说明

本项目用于根据数据库目标、学科、查询需求、字段说明文件和参考论文，自动生成材料科学数据库的章节设计方案和 schema 定义。当前目录中的示例输出是针对“斯格明子材料磁性数据库”的一次带参考论文、带人工审核节点的完整运行结果。

## 目录文件说明

### 输入文件

| 文件 | 作用 |
| --- | --- |
| `key_description.yaml` | 上游字段/键描述文件，是 agent 设计数据库 schema 时的字段参考来源。 |
| `1-s2.0-S0304885322007909-main.md` | 参考论文 Markdown，运行时通过 `--reference-papers` 输入，用于补充斯格明子物理、材料体系、图表证据和应用场景。 |
| `050901_1_5.0042917.md` | 参考论文 Markdown，运行时通过 `--reference-papers` 输入，用于补充斯格明子器件、存储和逻辑应用相关内容。 |
| `142404_1_online.md` | 参考论文 Markdown，运行时通过 `--reference-papers` 输入，用于补充具体实验、材料和图表证据。 |

### 代码文件

| 文件 | 作用 |
| --- | --- |
| `code/section_design_agent.py` | 基础版 Step 8 Section Design Agent。按模块顺序调用 LLM，生成最终 `section_design` 和 `schema_definition` JSON。 |
| `code/section_design_agent_prompt.py` | 所有模块的 prompt 模板和最终输出 schema 描述。修改 agent 行为时主要改这里。 |
| `code/section_design_langgraph_human_gate.py` | LangGraph 版本。会在 `field_planning_module` 之后、`supervisor_module` 之前暂停，等待人工建议后再继续。适合需要人工审核图表归属风险的流程。 |
| `code/render_step8_flowchart.py` | 可选的流程图渲染脚本，会读取 `step8_flowchart.md` 并生成 `step8_flowchart_main.svg` 和 `step8_flowchart.html`。当前目录没有对应的 `step8_flowchart.md` 时不要运行。 |

### 当前生成文件

| 文件 | 作用 |
| --- | --- |
| `skyrmion_with_refs_human_gate_fullrerun.json` | LangGraph human gate 完整运行输出。包含输入参数、参考论文上下文、agent 流程记录、各模块输出、最终结果、校验错误和状态。 |
| `skyrmion_with_refs_human_gate_fullrerun.json.state.json` | 与上面输出对应的 LangGraph 状态快照。用于断点续跑、排查中间模块输出、恢复到某个节点继续执行。 |
| `skyrmion_with_refs_human_gate_fullrerun.json.human_gate_context.json` | 第一次运行暂停在人工审核节点时生成的上下文文件。里面包含需要人工检查的章节划分、字段规划、参考论文预览和建议填写要求。 |
| `skyrmion_with_refs_human_gate_fullrerun_postfixed.json` | 在完整运行结果基础上经过后处理/修正后的最终输出版本。建议优先查看这个文件作为最终 schema 设计结果。 |
| `skyrmion_with_refs_human_gate_fullrerun_postfixed.json.state.json` | 与 postfixed 输出对应的状态快照。用于记录修正后流程状态和复现/续跑。 |

最终结果 JSON 中最重要的字段通常是：

| 字段 | 说明 |
| --- | --- |
| `inputs` | 本次运行的数据库目标、学科、查询需求、字段说明路径、参考论文路径和人工建议状态。 |
| `reference_paper_context_used` | 从参考论文中截取并提供给模型的上下文。 |
| `agent_flow` | 面向阅读的 agent 流程追踪，方便理解每个模块如何影响最终设计。 |
| `module_outputs` | 每个中间模块的原始结构化输出。用于调试、审核和定位问题。 |
| `result.database_positioning` | 数据库定位，包括检索单元、设计目标和建模理由。 |
| `result.section_design` | 核心章节和非核心章节划分。 |
| `result.schema_definition` | 顶层 key 和字段注册表，是最接近最终数据库 schema 的部分。 |
| `validation_errors` | 自动校验发现的问题。为空时 `status` 通常为 `success`。 |
| `status` | 运行状态，常见值为 `success`、`needs_review`、`waiting_for_human_advice`。 |

## 环境准备
 

如果使用 SiliconFlow、DeepSeek 或其他兼容 OpenAI SDK 的服务，设置环境变量：

```powershell
$env:SECTION_AGENT_BASE_URL="https://api.siliconflow.cn/v1"
$env:SECTION_AGENT_API_KEY="你的 API Key"
$env:SECTION_AGENT_MODEL="deepseek-ai/DeepSeek-V3.2"
```

也可以不设置环境变量，直接在命令行传 `--base-url`、`--api-key` 和 `--model`。

## 运行方式一：基础版一次性运行

基础版不会强制中途等待人工建议，适合快速生成 schema：

```powershell
python code\section_design_agent.py `
  --database-goal "斯格明子材料磁性数据库，支持按材料体系、斯格明子相类型、拓扑霍尔效应、螺旋周期、斯格明子尺寸、临界磁场和图表证据检索" `
  --discipline "材料科学" `
  --query-requirements "按材料体系检索||按斯格明子相类型筛选||按拓扑霍尔信号和螺旋周期比较||按斯格明子尺寸和临界磁场区间筛选||需要保留图表中的关键磁拓扑结果" `
  --key-description-path ".\key_description.yaml" `
  --reference-papers ".\1-s2.0-S0304885322007909-main.md" ".\050901_1_5.0042917.md" ".\142404_1_online.md" `
  --output ".\section_design_output.json"
```

输出文件：

```text
section_design_output.json
```

## 运行方式二：带人工审核节点的 LangGraph 流程

这个流程会先运行到 `human_advice_gate`，在 `supervisor_module` 之前暂停，生成人工审核上下文。

### 第一步：启动并暂停到人工审核节点

```powershell
python code\section_design_langgraph_human_gate.py `
  --thread-id "skyrmion-human-gate-fullrerun" `
  --database-goal "斯格明子材料磁性数据库，支持按材料体系、斯格明子相类型、拓扑霍尔效应、螺旋周期、斯格明子尺寸、临界磁场和图表证据检索" `
  --discipline "材料科学" `
  --query-requirements "按材料体系检索||按斯格明子相类型筛选||按拓扑霍尔信号和螺旋周期比较||按斯格明子尺寸和临界磁场区间筛选||需要保留图表中的关键磁拓扑结果" `
  --key-description-path ".\key_description.yaml" `
  --reference-papers ".\1-s2.0-S0304885322007909-main.md" ".\050901_1_5.0042917.md" ".\142404_1_online.md" `
  --output ".\skyrmion_with_refs_human_gate_fullrerun.json"
```

暂停后会生成：

```text
skyrmion_with_refs_human_gate_fullrerun.json.state.json
skyrmion_with_refs_human_gate_fullrerun.json.human_gate_context.json
```

此时先打开 `*.human_gate_context.json`，重点检查：

```text
section_partition_module
field_planning_module
required_advice
reference_paper_context_preview
```

### 第二步：带人工建议恢复运行

```powershell
python code\section_design_langgraph_human_gate.py `
  --resume-from-state ".\skyrmion_with_refs_human_gate_fullrerun.json.state.json" `
  --thread-id "skyrmion-human-gate-fullrerun-resume" `
  --human-advice "主管应重点检查 section_magnetic_topology、section_physical_properties 和 section_characterization 之间的图表归属。LTEM/MFM/SANS/REXS 图应归属于 section_characterization，只作为 phase_identity、skyrmion_size、helical_period 的证据链接。Hall 和 M-H 曲线应归属于 section_physical_properties，不能单独用于高置信度 phase_type 判定。topological Hall only 的相识别必须标记为低置信度，并设置 hall_only_risk_flag=true。"
```

恢复后会继续执行：

```text
supervisor_module
figure_classification_module
schema_design_module
specialization_critic_module
aggregation
write_output
```

最终输出：

```text
skyrmion_with_refs_human_gate_fullrerun.json
skyrmion_with_refs_human_gate_fullrerun.json.state.json
```

## 运行方式三：交互式人工审核

如果希望程序暂停后直接在终端输入人工建议，可以加 `--interactive-human-gate`：

```powershell
python code\section_design_langgraph_human_gate.py `
  --interactive-human-gate `
  --thread-id "skyrmion-human-gate-interactive" `
  --database-goal "斯格明子材料磁性数据库，支持按材料体系、斯格明子相类型、拓扑霍尔效应、螺旋周期、斯格明子尺寸、临界磁场和图表证据检索" `
  --discipline "材料科学" `
  --query-requirements "按材料体系检索||按斯格明子相类型筛选||按拓扑霍尔信号和螺旋周期比较||按斯格明子尺寸和临界磁场区间筛选||需要保留图表中的关键磁拓扑结果" `
  --key-description-path ".\key_description.yaml" `
  --reference-papers ".\1-s2.0-S0304885322007909-main.md" ".\050901_1_5.0042917.md" ".\142404_1_online.md" `
  --output ".\skyrmion_interactive_output.json"
```

## 可选：生成流程图

只有当根目录存在 `step8_flowchart.md` 时再运行：

```powershell
python code\render_step8_flowchart.py
```

预期输出：

```text
step8_flowchart_main.svg
step8_flowchart.html
```

## 查看结果建议

优先查看：

```text
skyrmion_with_refs_human_gate_fullrerun_postfixed.json
```

重点看：

```text
result.section_design
result.schema_definition.top_level_keys
result.schema_definition.field_registry
validation_errors
status
```

如果 `status` 是 `needs_review`，先看 `validation_errors`，再回到 `module_outputs` 中定位是哪个模块输出不符合要求。

 
## 后续 Agent 流程说明

前面的 Step 8 负责从论文和需求中设计数据库 section 与 schema；Step 9 负责根据 section/schema 生成下游抽取 prompt。Step 9 之后，目前新增了一个独立的 Step 10，用来判断 prompt 是否可用，并把判断结果交给后续代码生成 Agent。

### Step 9：Prompt Generation Agent

输入文件通常是：

```text
skyrmion_with_refs_human_gate_fullrerun_postfixed.json
```

输出文件通常是：

```text
skyrmion_prompt_generation_output.json
```

这个 Agent 的作用是把 Step 8 生成的 section/schema 转换成可执行 prompt 包，主要包括：

- 分类 prompt：判断论文片段应该进入哪个抽取模块。
- section 抽取 prompt：按 section 抽取结构化字段。
- figure/table/chart 抽取 prompt：处理图、表、曲线、caption、OCR 等证据。
- postprocess repair prompt：修复 JSON 格式、缺失字段、证据链接和 section ownership 冲突。
- prompt package aggregation：把所有 prompt 模块整合成一个下游可复用的 prompt 包。

Step 9 的核心输出位置：

```text
result.prompt_modules
module_outputs
validation_errors
status
```

其中 `status=success` 且 `validation_errors=[]` 才适合进入后续 Agent。

### Step 10：Prompt Quality Judgement Agent

对应代码：

```text
code/prompt_quality_code_agent.py
```

输入：

```text
skyrmion_prompt_generation_output.json
```

输出：

```text
prompt_quality_judgement.json
```

这个 Agent 分为三层判断。

第一层是本地格式检查 `format_checks`，只判断机器可验证的问题：

- Step 9 的 `status` 是否为 `success`。
- `validation_errors` 是否为空。
- 必须的 prompt module 是否齐全。
- schema 中的 field 是否全部被 prompt 覆盖。
- `execution_order` 是否正确。
- 聚合后的 `result.prompt_modules` 是否完整。

第二层是本地质量检查 `quality_checks`，判断 prompt 是否包含基本质量约束：

- 是否要求输出合法 JSON。
- 是否要求 evidence/provenance/source/span。
- 是否有 anti-hallucination 约束。
- 是否限制 `field_path`。
- 是否保护 section ownership。
- 是否处理 figure constraint。
- repair prompt 是否覆盖 JSON、missing field、ownership、figure、provenance。
- prompt 是否过长、过短或疑似编码损坏。

第三层是 LLM 质量评审智能体，需要加 `--run-quality-review` 才会调用 API。它会主观评估 prompt 质量，并输出：

- `overall_score`
- `dimension_scores`
- `quality_findings`
- `must_fix_before_code_generation`
- `nice_to_have_improvements`
- `code_generation_guidance`

这些内容会被合并到 `prompt_quality_judgement.json`，作为后续代码生成 Agent 的依据。

只做本地判断：

```powershell
python code\prompt_quality_code_agent.py `
  --prompt-output skyrmion_prompt_generation_output.json `
  --judgement-output prompt_quality_judgement.json `
  --judge-only
```

调用 LLM 质量评审：

```powershell
$env:CODE_AGENT_API_KEY="your-key"
python code\prompt_quality_code_agent.py `
  --prompt-output skyrmion_prompt_generation_output.json `
  --judgement-output prompt_quality_judgement.json `
  --judge-only `
  --run-quality-review `
  --base-url https://chat.iphy.ac.cn/litellm/v1 `
  --model kimi-k2.6
```

目前 API 测试结果：

- `/models` 可用。
- `kimi-k2.6` 的 `/chat/completions` 可用。
- `deepseek-v4` 会映射到 `deepseek-v4-pro`，但当前服务端返回 500，原因是后端模型服务不可达。

### Step 11：Code Generation Agent

Step 11 仍然由 `code/prompt_quality_code_agent.py` 触发，但只有在不使用 `--judge-only` 时才会进入。

输入包括：

```text
skyrmion_prompt_generation_output.json
prompt_quality_judgement.json
```

输出：

```text
code_generation_agent_output.json
```

它的作用不是直接抽取论文数据，而是根据 Step 10 的判断结果，规划或生成后续代码，例如：

- 下游 extraction runner。
- prompt package preflight validator。
- JSON 输出校验器。
- evidence/provenance 检查器。
- repair agent runner。
- 回归测试或 dry-run 检查脚本。

默认情况下，代码生成 Agent 只把生成结果写入 JSON，不会直接覆盖项目原有代码。只有显式添加下面参数时，才会把 LLM 返回的 `generated_files` 写入新目录：

```powershell
--write-generated-files --generated-output-dir generated_code
```

完整运行示例：

```powershell
$env:CODE_AGENT_API_KEY="your-key"
python code\prompt_quality_code_agent.py `
  --prompt-output skyrmion_prompt_generation_output.json `
  --judgement-output prompt_quality_judgement.json `
  --code-agent-output code_generation_agent_output.json `
  --run-quality-review `
  --base-url https://chat.iphy.ac.cn/litellm/v1 `
  --model kimi-k2.6
```

如果需要把生成文件落盘：

```powershell
$env:CODE_AGENT_API_KEY="your-key"
python code\prompt_quality_code_agent.py `
  --prompt-output skyrmion_prompt_generation_output.json `
  --judgement-output prompt_quality_judgement.json `
  --code-agent-output code_generation_agent_output.json `
  --run-quality-review `
  --base-url https://chat.iphy.ac.cn/litellm/v1 `
  --model kimi-k2.6 `
  --write-generated-files `
  --generated-output-dir generated_code
```

### Step 10 流程图

相关文件：

```text
step10_prompt_quality_code_agent_flow.md
step10_prompt_quality_code_agent_flow.svg
step10_prompt_quality_code_agent_flow.html
code/render_step10_flowchart.py
```

重新生成流程图：

```powershell
python code\render_step10_flowchart.py
```

建议查看：

```text
step10_prompt_quality_code_agent_flow.html
```

这个 HTML 文件包含更直观的泳道流程图，展示从 Step 9 prompt 输出，到本地判断、LLM 质量评审、代码生成 Agent 的完整路径。
