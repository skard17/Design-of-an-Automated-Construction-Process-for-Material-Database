preprocess2(s2)————把merge后的section2 json中的"method"从缩写映射为全称呼

这个脚本做什么
- 递归读取 inputs/ 下所有 JSON 文件。
- 按 method_mapping.yaml，把每个对象里的 method 从缩写改成全称。
- 输出到 outputs/，子目录结构和文件名保持不变。
- 只有出现 FAIL/WARN/SKIP 时，才会生成 check_list.txt（位于项目根目录，与 method_mapping.py 同级）。
- 每次运行都会在 logs/ 生成一份运行摘要日志。

如何运行
- 直接运行：
  python method_mapping.py
- 脚本会自动使用当前目录下的 inputs/、outputs/、method_mapping.yaml、logs/。

如何扩展或修改映射（重点）
1) 在 method_mapping.yaml 里增删改映射项。
   - 左侧：JSON 里的原始缩写（例如 SSR）
   - 右侧：要替换成的全称（例如 Solid-State Reaction）
2) 如果以后 JSON 字段名不叫 method，请修改 method_mapping.py 里的 process_json_file()。
   - 当前字段名是 "method"

check_list.txt 记录规则
- 只有有问题时才生成。
- 会记录：
  - FAIL：JSON 不合法 / 结构不合法 / method 不在映射表 / method 不是字符串
  - SKIP：顶层键对应列表为空（会记录，但仍写入 outputs）
  - WARN：列表对象缺少 method（该对象保持不变，文件仍写入 outputs）
- 处理成功的条目不会写进 check_list.txt。

依赖
- 必需：Python 3.10+
- 不强制第三方包。
- 可选：PyYAML（YAML 解析更稳健）。
  - 没装 PyYAML 也能运行：脚本会用内置的简易解析逻辑读取简单映射。
- 可以直接进入"sc_classify"这个环境来运行

补充说明
- 当前逻辑要求：顶层键的值是列表，且列表元素是对象。
- 对于 SKIP（顶层列表为空）和 WARN（对象缺少 method）文件，都会写入 outputs；仅记录问题到 check_list.txt。
