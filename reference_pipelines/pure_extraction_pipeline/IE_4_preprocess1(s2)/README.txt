IE_4_preprocess1(s2) 使用说明

一、脚本用途
- 脚本 `merge_s2.py` 用于将两路输入合并为统一输出 `section2`：
  - `inputs/s2_1/*.json`：来源字段 `method/description/geometry/conditions(键名列表)`
  - `inputs/s2_2/*.json`：来源字段 `conditions(对象)`
- 输出目录：`outputs/s2/*.json`
- 合并主键：同名文件 + `entry_id`

二、输出规则（当前代码真实行为）
- 顶层固定：`{"section2": [...]}`
- 每个 entry：
  - 从 s2_1 保留：`method`、`description`、`geometry`（存在才写）
  - 从 s2_2 注入：`conditions`（仅当其为对象）

三、检查顺序（显式固定）
- 顺序：`check-0 -> check-1 -> check-json -> check-2 -> check-3 -> check-4`
- `check-0` 是预扫描修复阶段；顺序由 `FULL_CHECK_SEQUENCE` 控制。

四、各 check 的含义与是否阻塞
0) `check_0.csv`（预扫描修复）
- 扫描：
  - `inputs/s2_1` 的文件应有顶层键 `section2_1`
  - `inputs/s2_2` 的文件应有顶层键 `section2_2`
- 自动修复（可恢复场景）：
  - 顶层键缺失（尝试从 `section2`/另一侧键/唯一列表键重命名）
  - 顶层值不是 list（必要时转成 list 或空 list）
  - 顶层不是对象（重建为 `{expected_key: []}`）
- 记录：被修复文件与修复动作会写入 `check_0.csv`。
- 若 JSON 无法解析，会在 `check_0.csv` 和 `check_json.csv` 记录，并跳过该文件。

1) `check_1.csv`（文件名配对检查）
- 记录：哪边缺同名文件。
- 影响：缺失文件不会进入合并；其余同名文件照常处理。

2) `check_json.csv`（JSON 可解析性）
- 记录：解析失败或顶层不是对象。
- 影响：该文件跳过，不输出。

3) `check_2.csv`（结构关键错误）
- 记录：section 键缺失/类型不对、entry_id 非 int/重复、entry_id 在 s2_2 不存在、s2_2.conditions 类型错误等。
- 影响：
  - 若是文件级结构错误（例如缺 section2_1），该文件跳过；
  - 若是单条 entry 错误，文件仍输出，其它可合并 entry 继续写。

4) `check_3.csv`（字段缺失提醒）
- 记录：`method/description/geometry/conditions` 缺失。
- 影响：仅记录，不阻塞。

5) `check_4.csv`（conditions 键集合对齐提醒）
- 记录：s2_1.conditions 列表 与 s2_2.conditions 对象键集合不一致，或类型不符。
- 影响：仅记录，不阻塞。

备注：某个 check 无问题时，对应 CSV 会写入 `ALL FILES PASSED`。

五、目录说明
- `merge_s2.py`：主程序
- `inputs/s2_1/`、`inputs/s2_2/`：输入
- `outputs/s2/`：合并结果
- `check_*.csv`：检查报告
- `logs/merge_s2_*.log`：运行日志

六、维护与修改建议
1) 想改检查顺序
- 修改 `CHECK_SEQUENCE` / `FULL_CHECK_SEQUENCE`。

2) 想改检查 CSV 列结构/路径
- 修改 `CHECK_CONFIG`。

3) 想改输出字段（目前是 method/description/geometry）
- 修改 `MERGE_FIELDS` 与 `merge_one_file()` 中的组装逻辑。

4) 想改输入/输出目录
- 修改顶部常量：`S2_1_DIR`、`S2_2_DIR`、`OUT_DIR`、`LOG_DIR`。

七、运行方式
- 在当前目录执行：
  python merge_s2.py

