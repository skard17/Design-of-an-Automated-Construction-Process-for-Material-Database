IE_3_check 脚本说明

一、脚本目的
用于对 inputs 目录下的 JSON 文件做“格式层”检查与修复，确保文件最终可被标准 JSON 解析器读取。
注意：此处不做语义校验（例如字段是否符合学术抽取规范、值是否合理），语义校验可以后续扩充框架的时候再加。

二、处理流程
对每个 JSON 文件按以下顺序处理：
1) 直接解析：尝试 json.loads
2) 整体结构修复：调用 json_repair（没有在运行环境里面安装的时候会跳过这一步，代码不会报错）
3) 转义修复：修复字符串中的非法反斜杠转义（如 \(）
4) 再次解析：若成功则写入 outputs 对应路径；若失败则在 check.txt 记录失败原因

三、目录与文件说明
1) inputs/
   - 原始待检查 JSON 数据（按 metadata/resources/s0/s1/... 分子目录组织）

2) outputs/
   - 通过检查（或修复后通过）的 JSON 输出目录
   - 保持与 inputs 相同的相对路径结构
   - 写出为标准化 JSON（UTF-8，ensure_ascii=False，缩进 2 空格）

3) logs/
   - 每次运行生成一个日志文件：check_YYYYMMDD_HHMMSS.log
   - 记录运行状态、修复动作（structure_fixed/escape_fixed）、统计信息等

4) check.txt
   - 本次运行汇总表（覆盖写入）
   - 表头：path status steps detail

四、check.txt 字段含义
1) path
   - 文件相对路径（相对于 inputs）

2) status
   - ok：原始文件可解析，或流程最终可解析且无需标注为 fixed
   - fixed：经过修复后可解析
   - failed：最终仍不可解析

3) steps
   - -：未做修复
   - structure：做了结构修复（json_repair）
   - escape：做了非法转义修复
   - structure+escape：两种修复都做了

4) detail
   - 成功（ok/fixed）时当前实现写 "-"
   - failed 时写详细错误链路，例如：
     - initial_parse_error=...
     - after_structure_parse_error=...
     - after_escape_parse_error=...
    - 示例（失败）：
       - initial_parse_error=Invalid \\escape: line 12 column 35 (char 280) |
          after_structure_parse_error=Expecting ',' delimiter: line 13 column 5 (char 301) |
          after_escape_parse_error=Unterminated string starting at: line 13 column 18 (char 314)

五、可能需要修改的地方
1) 输入/输出根目录
   - 文件：check.py
   - 位置：第 28-32 行（BASE_DIR / INPUT_ROOT / OUTPUT_ROOT / LOG_ROOT / CHECK_PATH）
   - 当前行为：BASE_DIR 固定为脚本所在目录

2) “修复策略”
   - 结构修复逻辑：第 101-121 行（apply_structure_repair）
   - 转义修复逻辑：第 123-183 行（fix_invalid_escapes / is_valid_unicode_escape）
   - 单文件处理主流程：第 199-242 行（process_file）

3) 结果统计口径
   - steps -> 统计映射：第 35-40 行（STEP_TO_COUNTER）
   - 默认统计项：第 41-48 行（DEFAULT_STATS）
   - 统计累加与日志输出：第 245-272 行（main）

4) check.txt 输出格式
   - 行格式：第 58-59 行（FileOutcome.to_check_line）
   - 表头与写文件：第 187-191 行（write_check）
   - 成功记录 detail 默认值：第 194-196 行（build_success_outcome）

六、运行方式
先cd到 IE_3_check 目录
conda activate sc_classify进入环境
python check.py

七、补充说明
1) fixed 记录与 detail 字段
   - 当前实现仅在 failed 时写错误详情；fixed 只记录 steps，detail 为 "-"。
   - 若希望 fixed 也写修复摘要，可在 process_file()/build_success_outcome() 中调整。

2) inputs 与 outputs 文件数量关系
   - 正常情况下，每个可解析输入都会在 outputs 生成对应文件。
   - 若存在 failed，outputs 中对应文件会缺失，并在 check.txt 中体现。
