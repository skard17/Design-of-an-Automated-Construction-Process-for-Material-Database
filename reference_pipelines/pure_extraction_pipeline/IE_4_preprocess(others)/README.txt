IE_4_preprocess0 

目的：
对抽取的结果键值对按照schema进行清洗和重排，为后续merge做好准备。

逻辑
1) 读 config.yaml
2) 按 part（s0/s1/...）遍历 inputs/<part>/*.json
3) 按规则处理（保留键、调顺序、必要时补默认值）
4) 写到 outputs/<part>/
5) 记录报告和日志

修改：
- 想改“某个 part 最后保留哪些键、键顺序怎么排”
	- 去改 schema/*.json（最常改）

- 想改“开关、输入输出目录名、part 是否启用、用哪份 schema”
	- 去改 config.yaml（parts 下面）

- 想改 inputs 里的子文件夹名字（比如把 single 改成 single_v2）
	- 去改 config.yaml 里该 part 的 input_subdir
	- 例如：single 这一段把 input_subdir: single 改成 input_subdir: single_v2
	- 注意：如果只改了文件夹名、没改 config，脚本会找不到数据并在报告里记 SKIPPED
    - 目前先保留了子文件夹的名字，可以选择清空inputs直接把之前的放进来然后改config，也可以不改子文件夹名字，把对应的json放进去
    metadata-resources
    resources-resources
    s0-section0
    s1-section1
    s3-section3
    s4-section4
    s5-section5
    single-一文单物分类结果
    此处没有section2是因为section2有自己的merge逻辑，里面就会输出最终的section2的要的键

补充：
- s0/s1/s3/s4/s5/metadata/resources：主要按 schema 清理和重排。
- single：是提取模式，只留配置里指定字段（默认是 primary_signature）。

 config里面的clean_output：
- clean_output 的意思是“运行前要不要先清空 outputs”。
- clean_output: true = 先删掉旧 outputs，再生成本轮结果（推荐，结果最干净）。
- clean_output: false = 不清空，旧文件会保留（适合增量调试）。
- 如果 clean_output: true，会先清空 outputs，再重新创建本轮需要的输出子文件夹。
- 所以你就算手动把 outputs 清空，重新运行后也会自动生成 outputs/<part>/ 这些子目录。

 preprocess_report.txt ：
- 会记录每个输入文件一行处理结果。
- 基本格式（展示）：<part>/<文件名>.json  状态
- 状态有：
	- SUCCESS：该文件处理成功并写入 outputs
	- FAILED：该文件处理失败，后面会带错误信息
	- SKIPPED：该 part 被禁用，或该输入目录没有 json 文件
- 例子：
	- s0/1.json  SUCCESS
	- single/ABC.json  FAILED  missing or invalid root key '...'
	- s3  SKIPPED  no_json_files_in=...

运行：
- conda activate sc_classify
- 在当前目录执行：python preprocess.py