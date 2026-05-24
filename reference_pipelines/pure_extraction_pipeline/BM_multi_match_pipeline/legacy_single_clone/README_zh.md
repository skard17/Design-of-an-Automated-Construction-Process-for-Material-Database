# 多物版旧单物框架副本

这个目录是给多物流程准备的“旧单物框架副本”。

目的只有一个：
- 不改原来的单物代码
- 先把单物各 section 的 prompt 和组织方式复制过来
- 再在这份副本上改成多物版

## 当前内容

`prompts/` 下面已经复制了旧单物各 section 的 prompt 快照：

- `section0_single.md`
- `section1_single.md`
- `section2_1_single.md`
- `section2_2_prefix_single.md`
- `section2_2_suffix_single.md`
- `s2_2_fabrication/`
- `section3_single.md`
- `section4_single.md`
- `section5_single.md`

## 设计原则

多物版不推翻旧单物框架，而是沿用下面这些原则：

1. `IE_0_multi`
   - 负责多物基础识别
   - 输出 `multi_system_result`

2. 后续各 section
   - 继续沿用旧单物 section 的字段和 prompt 结构
   - 但是输入不再是单个 `primary_signature`
   - 而是对 `multi_system_result` 里的每个 counted system 分别抽取

3. `section2`
   - 继续沿用旧单物的两段式
   - 先 `section2_1`
   - 再按 method 拼接 `section2_2`

4. 输出组织
   - 每篇文献一个文件夹
   - 文件夹内多个 system JSON

## 下一步

接下来会继续做这几件事：

1. 把 `multi-example.txt` 作为多物版 `IE_0` 入口 prompt 接进脚本
2. 给每个 counted system 生成单独的 section prompt 输入
3. 让 `section0/1/2/3/4/5` 都沿用旧单物字段
4. 输出到按论文分目录的结果目录中
