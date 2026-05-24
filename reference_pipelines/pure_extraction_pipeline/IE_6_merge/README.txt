IE_6_merge 使用说明

一、脚本用途
- 作用：把 `inputs/` 下各分区 JSON 按 `schema/schema0_store.json` 的结构合并成单文件输出。
- 入口：`merge.py`
- 输出：`outputs/<filename>.json`
- 质量检查：`check.txt`
- 运行日志：`logs/merge_时间戳.log`

二、核心运行逻辑（按代码执行顺序）
2.1 读取模板 schema
- 文件：`schema/schema0_store.json`
- 这个 schema 同时承担两件事：
  - 定义最终输出 JSON 的骨架结构
  - 决定“从哪个输入目录取哪个键，放到输出哪里”

2.2 根据 schema 自动生成合并 routes
- `merge.py` 中 `build_routes_from_schema()` 会生成映射规则：
  - `material_info.sectionN` -> 从 `inputs/sN/<name>.json` 读取键 `sectionN`，写入 `material_info.sectionN`
  - `paper_info.metadata/resources` -> 从同名目录读取同名键，写入 `paper_info.<child_key>`
  - `primary_signature` -> 从 `inputs/single/<name>.json` 读取 `primary_signature`
  - 其他普通顶层键 -> 默认从同名目录读取同名键

2.3 汇总候选文件名
- 对所有参与目录收集 `*.json` 文件名，做并集（union）作为候选集合。

2.4 对每个文件名进行完整性检查
- 若某些目录缺这个文件：跳过该文件，并在 `check.txt` 记录 `missing_parts=...`
- 若某目录 JSON 解析失败：记录 `parse_error_<dir>=...`
- 若某目录缺少预期键：记录 `invalid_key_<dir>=expected:<key>`
- 上述任一情况都会跳过，不会输出该文件。

2.5 合并并写出
- 所有分区都通过校验时，深拷贝 schema，再把各路由值写入目标路径。
- 写到 `outputs/<name>.json`

2.6 收尾
- 统计 merged/skipped 数量
- 把所有跳过原因写入 `check.txt`

三、`check.txt` 是什么（重点）
- `check.txt` 是“本次未成功合并文件”的问题清单。
- 每行格式：
  - `<文件名>\t<问题1>; <问题2>; ...`
- 常见问题类型：
  - `missing_parts=...`：某些输入目录缺文件
  - `parse_error_<dir>=...`：JSON 读/解析失败
  - `invalid_key_<dir>=expected:<key>`：文件里缺预期根键
  - `merge_error=...`：合并阶段异常
- 若 `check.txt` 为空，通常表示本轮没有异常跳过项（或没有候选文件）。

四、目录与文件作用
- `inputs/`
  - 按分区存放输入文件（`metadata/resources/s0.../single`）
  - 同名文件（例如 `9.json`）会跨目录汇总后合并

- `schema/schema0_store.json`
  - 输出结构模板 + 路由来源规则

- `outputs/`
  - 合并结果目录（每个文件名对应一份最终结果）

- `logs/`
  - 运行日志（包含合并计数与路径）

- `check.txt`
  - 异常/缺失检查报告（用于快速定位为什么某文件没合并成功）

五、`inputs` 子文件名能不能改
- 可以改，但要把“同一个样本”在所有参与目录里的文件名一起改。
- 这个脚本是按“同名文件”做跨目录合并的，文件名本身就是对齐主键。
- 例如把 `inputs/metadata/9.json` 改成 `inputs/metadata/sample_9.json` 后，至少还要同步改：
  - `inputs/resources/9.json` -> `inputs/resources/sample_9.json`
  - `inputs/s0/9.json` -> `inputs/s0/sample_9.json`
  - `inputs/s1/9.json` -> `inputs/s1/sample_9.json`
  - ...直到所有参与合并的目录都改成同名
- 如果只改了部分目录，会在 `check.txt` 出现：`missing_parts=...`
- 只改文件名时，一般不需要改 schema；schema 约束的是“键名与结构”，不是文件名。
- 只有当你同时改变了目录命名规则或键名规则，才需要改 `merge.py` 的 `build_routes_from_schema()`。

六、复用与修改
6.1 想新增/调整输出结构
- 先改：`schema/schema0_store.json`
- 注意：改 schema 会直接影响输出骨架和自动路由。

6.2 想新增一个 section（例如 section6）
- 需要同时满足：
  1) schema 中出现对应位置（例如 `material_info.section6` 或顶层 `section6`）
  2) `inputs/s6/` 中存在同名文件
  3) 每个 `inputs/s6/<name>.json` 内有键 `section6`

6.3 想改“输入目录名”或“源键名”规则
- 当前规则写在 `merge.py` 的 `build_routes_from_schema()`。
- 若你的目录命名与当前约定不同，需要改这里。

6.4 想放宽/改变校验策略（例如允许部分缺失也输出）
- 改 `main()` 里对 `missing/invalid/parse_error` 的处理逻辑。
- 当前是严格模式：只要某一分区有问题，该文件整体跳过。

七、运行方式
- 在 `IE_6_merge` 目录执行：
  python merge.py

八、移交前后注意事项
1) 本脚本路径是相对项目目录构建的（无代码内绝对路径硬编码）。
2) 旧日志可能显示历史绝对路径，这是日志文本，不影响新环境运行。
3) 脚本默认不会自动清空旧 `outputs/`，建议每次跑前手动清理：
   - `outputs/*`
   - `logs/*`
   - `check.txt`
4) 输入文件需要“跨目录同名”才能成功合并（例如每个目录都有 `9.json`）。
5) `check.txt` 是首要排错入口；先看它再看日志。

九、已知实现特性（不一定是 bug）
- 严格整文件跳过：任何一个分区异常就不产出该文件。
- 候选文件名来自各目录并集，因此单目录多出来的文件也会进入检查并可能写入 `check.txt`。
