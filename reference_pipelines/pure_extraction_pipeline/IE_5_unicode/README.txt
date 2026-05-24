IE_5_unicode 说明

一、脚本目的
- 对 inputs 下各分组 JSON 做“按 schema 指定字段的文本显示标准化处理，为后续直接放到网页做准备”，输出到 outputs。
- 处理方式：递归遍历 JSON，仅转换 schema 指定字段；未指定字段保持原样。


二、执行
- 进入环境sc_classify
- 在当前目录运行：
  python preprocess.py

三、模式说明（重点）
说明：代码共支持 4 个模式（0/1/2/3）。常用的“3个转换模式”通常是 1/2/3。

- mode 0（保留）
  不做转换，原样输出。

- mode 1（下划线转空格）
  把字符串中的 "_" 替换为 " "。
  示例："anion_substitution" -> "anion substitution"

- mode 2（内置 Unicode/LaTeX 归一化）
  主要做：
  - LaTeX 命令替换为 Unicode（如 \alpha -> α）
  - 上下标转换（如 MgB2-xCx -> MgB₂₋ₓCₓ）
  - unicode_escape 解码（遇到转义序列时）

- mode 3（调用LLM来处理）
  用 prompts/mode3_transform.md 作为提示词模板，调用 LLM 做更灵活的文本标准化。
  带有并发控制、重试、缓存（同文本会命中缓存减少请求）。
  prompt可以在/prompts里面对应的.md文件里面修改

四、文件与目录
- inputs/
  原始输入数据。第一层子目录名就是“part”，例如 metadata、resources、s0~s5、single。

- schema_unicode_transfer/
  每个 part 对应一个 schema 文件：schema_<part>.json。
  例如 inputs/s0/* 会使用 schema_unicode_transfer/schema_s0.json。

- outputs/
  输出目录，结构与 inputs 对应。
  示例：inputs/s2/A.json -> outputs/s2/A.json

- logs/
  每次运行生成 preprocess_时间戳.log，记录处理过程与统计。

- preprocess_list.txt
  本次运行每个文件的状态清单（success / skipped_empty_input / skipped_no_schema / transform_failed / invalid_json）。

- config.yaml
  LLM 相关参数（模型、base_url、并发、超时、重试、prompt 路径等）。

- prompts/mode3_transform.md
  mode 3 的提示词模板，定义输入/输出格式与规范化要求。

五、注意事项
1) inputs 第一层子目录名很重要
- 程序会把它当作 part，并去找 schema_<part>.json。
- 如果你改了目录名但没有对应 schema，文件会被标记 skipped_no_schema。

2) schema 才是“改哪里、怎么改”的唯一依据
- 只有 schema 声明到的字段才会被处理。
- 叶子节点必须是 0/1/2/3；不合法会直接报错。

3) 空数据会被跳过转换
- 若 JSON“实质为空”（全空字符串/空结构等），会直接镜像输出并标记 skipped_empty_input。

4) mode=3 需要联网和可用 key
- 没有 key 或网络异常会导致 transform_failed。

六、后续最常改的地方（改动指南）
A. 新增一个 part（例如 s6）
1) 新建 inputs/s6/
2) 新建 schema_unicode_transfer/schema_s6.json
3) 按 0/1/2/3 配置需要处理的字段

B. 调整某字段转换策略
- 去对应 schema_<part>.json 把该字段 mode 改成 0/1/2/3。
- 一般建议：
  - 结构标识类字段用 0
  - 仅格式空格修正用 1
  - 化学式/LaTeX 常用 2
  - 语义归一化/自由文本用 3

C. 调整 LLM 行为
- 改 config.yaml（IE_5_unicode/config.yaml 的 llm: 段）：
  - model / base_url / timeout_sec / max_retries
  - 并发相关：max_workers（处理文件的线程数上限）/ max_llm_inflight（同时“发出中的”LLM 请求上限）
  - API Key：优先写在 llm.api_keys（列表）；也支持 llm.api_key（单个）
- 也可以用环境变量配置 key（比写进文件更安全）：
  - LLM_API_KEY：单个 key
  - LLM_API_KEYS：多个 key，用英文逗号分隔
- 并发到底受哪些参数限制（mode=3 时生效，讲清楚）：
  - 这段代码的“并发”分两层：
    1) 文件并行处理（线程池）：由 max_workers 控制；但实际线程数会被 key 数量进一步上限
    2) LLM 请求并发（网络请求）：由 max_llm_inflight 控制；并且每个请求必须占用 1 个 key
    - 实际文件线程数：workers = min(max_workers, 待处理文件数, key数量)
    - 同时在飞的 LLM 请求数：inflight <= min(max_llm_inflight, workers, key数量)
  - 直观例子：
    - key=10，max_workers=6，max_llm_inflight=1 → workers=6，同时最多 1 个 LLM 请求
    - key=10，max_workers=6，max_llm_inflight=4 → workers=6，同时最多 4 个 LLM 请求
    - key=2，max_workers=8，max_llm_inflight=4 → workers=2，同时最多 2 个 LLM 请求
  - inflight=1 仍可能 429：因为限流常见是“每分钟请求数/每分钟 token 数”等速率限制；而且一个文件里可能有很多字符串叶子字段需要多次 LLM 调用（会很快累积请求量，短时间内高频次发出请求）
  - 调参建议：
    - 稳定优先：max_llm_inflight=1，max_workers=min(4, key数量)
    - 速度优先：先把 max_workers 提到接近 key 数量，再逐步增加 max_llm_inflight（不要超过 key 数量）
    - 如果遇到 429/限流：优先调小 max_llm_inflight，其次调小 max_workers；必要时缩小 schema 中 mode=3 字段范围
- 改 prompts/mode3_transform.md：规范化规则与输出约束

七、常见问题排查
- 报错：missing schema_xxx.json
  原因：inputs 子目录名与 schema 文件名不匹配。

- 报错：mode3 requires openai package
  处理：pip install -r requirements.txt

- 报错：schema contains mode=3 fields, but no API key found
  处理：设置 LLM_API_KEY 或 LLM_API_KEYS，或在 config.yaml 配置 key。

- 大量 skipped_empty_input
  说明输入文件内容本身是空值结构，不是程序异常。

