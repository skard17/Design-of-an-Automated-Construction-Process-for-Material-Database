#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三级分类系统（Markdown 版）：
1. 代码检查论文文件中的 super-con 分区信息
2. 调用 L1.md prompt + LLM
3. 调用 L2&3.md prompt + LLM
"""

import os
import json
import re
import time
import logging
import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Clear broken CA-bundle env vars inherited from Windows shells before httpx/OpenAI init.
for _env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    os.environ.pop(_env_name, None)

LOCAL_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL",
    os.getenv("LOCAL_DEEPSEEK_BASE_URL", "https://api.siliconflow.cn/v1/chat/completions"),
)
LOCAL_API_KEY = os.getenv(
    "SILICONFLOW_API_KEY",
    os.getenv("LOCAL_DEEPSEEK_API_KEY", ""),
)
LOCAL_API_KEY_2 = os.getenv("SILICONFLOW_API_KEY_2", "").strip()
LOCAL_MODEL = os.getenv("SILICONFLOW_MODEL", os.getenv("LOCAL_DEEPSEEK_MODEL", "Pro/deepseek-ai/DeepSeek-V3.2"))
LOCAL_MODEL_2 = os.getenv("SILICONFLOW_MODEL_2", "deepseek-ai/DeepSeek-V3.2")
LOCAL_PER_KEY_PARALLEL = max(1, int(os.getenv("SILICONFLOW_PER_KEY_PARALLEL", "4")))
LOCAL_MAX_PARALLEL = max(1, int(os.getenv("SILICONFLOW_MAX_PARALLEL", os.getenv("LOCAL_DEEPSEEK_MAX_PARALLEL", "8"))))
VERIFY_SSL = os.getenv("SILICONFLOW_VERIFY_SSL", os.getenv("LOCAL_DEEPSEEK_VERIFY_SSL", "true")).lower() in {"1", "true", "yes"}
REQUEST_INTERVAL_SEC = max(0.0, float(os.getenv("SILICONFLOW_REQUEST_INTERVAL_SEC", "0")))
_REQUEST_LOCK = Lock()
_LAST_REQUEST_AT = 0.0


def _pack_slot(api_key: str, model: str) -> str:
    return f"{api_key}|||{model}"


def _split_slot(slot: str, default_model: str) -> tuple[str, str]:
    raw = (slot or "").strip()
    if "|||" in raw:
        key_part, model_part = raw.split("|||", 1)
        return key_part.strip(), (model_part.strip() or default_model)
    return raw, default_model


def _resolve_chat_url(base_url: str) -> str:
    url = (base_url or "").strip() or LOCAL_BASE_URL
    if url.endswith("/chat/completions"):
        return url
    return url.rstrip("/") + "/chat/completions"


# ============================================================================
# 一、文件I/O模块
# ============================================================================

def read_text(path: Path) -> str:
    """读取文本文件内容，使用UTF-8编码，忽略编码错误。"""
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_arxiv_id(filename: str) -> str:
    """从文件名提取论文编号（去掉扩展名）。"""
    return Path(filename).stem


# ============================================================================
# 二、配置加载模块
# ============================================================================

def load_config(config_path: Path) -> Dict[str, Any]:
    """加载并验证YAML配置文件。"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    if not isinstance(cfg, dict):
        raise ValueError("Config file root must be a dict/mapping")
    
    return cfg


def setup_logging(root: Path) -> logging.Logger:
    """
    Setup file+console logging under root/logs.
    """
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"run_{ts}.log"

    logger = logging.getLogger("sc_classify")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_file}")
    return logger


def init_failed_writer(output_dir: Path) -> Tuple[Path, Lock, set]:
    """
    Prepare output/failed.txt writer.
    Returns: (failed_path, lock, existing_ids_set)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_path = output_dir / "failed.txt"
    existing: set = set()
    if failed_path.exists():
        try:
            for line in failed_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                existing.add(line.split("\t", 1)[0].strip())
        except Exception:
            pass
    return failed_path, Lock(), existing


def record_failed(
    failed_path: Path,
    failed_lock: Lock,
    failed_seen: set,
    arxiv_id: str,
    stage: str,
    reason: str
) -> None:
    """Append a failure record once per arxiv_id."""
    with failed_lock:
        if arxiv_id in failed_seen:
            return
        failed_seen.add(arxiv_id)
        with failed_path.open("a", encoding="utf-8") as f:
            f.write(f"{arxiv_id}\t{stage}\t{reason}\n")


def init_client(api_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """初始化本地 DeepSeek 请求配置。"""
    api_key = (os.getenv("LOCAL_DEEPSEEK_API_KEY") or api_cfg.get("api_key", "") or LOCAL_API_KEY).strip()
    if not api_key:
        raise ValueError("API key not found for local DeepSeek service")
    return {
        "api_key": api_key,
        "base_url": os.getenv("LOCAL_DEEPSEEK_BASE_URL", api_cfg.get("base_url", LOCAL_BASE_URL)),
        "timeout": api_cfg.get("timeout_seconds", 300),
    }


# ============================================================================
# 三、分区检查模块
# ============================================================================

def check_supercon_category(tex_content: str, filename: str) -> bool:
    """
    检查 .tex 文件中是否有 super-con 分区信息。
    
    检查方法：
    1. 检查文件名中是否包含 cond-mat 或 supr-con
    2. 检查文件内容中是否包含 super-con 相关关键词
    
    Args:
        tex_content: .tex 文件内容
        filename: 文件名
        
    Returns:
        True 如果找到 super-con 分区信息，False 否则
    """
    # 检查文件名
    filename_lower = filename.lower()
    if 'cond-mat' in filename_lower or 'supr-con' in filename_lower or 'super-con' in filename_lower:
        return True
    
    # 检查文件内容中的关键词
    content_lower = tex_content.lower()
    
    # 检查是否包含 supercon 相关关键词
    supercon_keywords = [
        'cond-mat.supr-con',
        'cond-mat.super-con',
        'supr-con',
        'super-con',
        'superconductivity',
        'superconductor'
    ]
    
    # 检查是否在注释或特定位置提到分区
    # 通常 arxiv 分区信息可能在注释中，如 % arXiv:cond-mat.supr-con/...
    if any(keyword in content_lower for keyword in supercon_keywords):
        # 进一步检查是否在 arxiv 相关上下文中
        arxiv_patterns = [
            r'arxiv[:\s]+cond-mat\.supr-con',
            r'arxiv[:\s]+cond-mat\.super-con',
            r'category[:\s]+cond-mat\.supr-con',
            r'category[:\s]+cond-mat\.super-con',
        ]
        for pattern in arxiv_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return True
    
    return False


# ============================================================================
# 四、Prompt 处理模块
# ============================================================================

def load_prompt_with_paper_text(prompt_path: Path, paper_text: str) -> str:
    """
    加载 prompt 文件，并将论文全文插入到成对的 <paper_text> 标记之间。
    如果找不到两处标记，则将论文文本追加到文件末尾。
    """
    lines = read_text(prompt_path).split('\n')
    marker_indices = [i for i, line in enumerate(lines) if line.strip() == "<paper_text>"]
    
    if len(marker_indices) >= 2:
        start_idx, end_idx = marker_indices[0], marker_indices[1]
        result_lines = lines[: start_idx + 1]  # 保留第一个标记行
        result_lines.append(paper_text)        # 插入论文全文
        result_lines.extend(lines[end_idx:])   # 从第二个标记行开始保留后续
        return '\n'.join(result_lines)
    
    # 找不到标记时，退化为追加到末尾
    return '\n'.join(lines + ["", paper_text])


def truncate_text(text: str, max_chars: int) -> str:
    """截断文本到指定字符数。"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Text truncated...]"


# ============================================================================
# 五、LLM 调用模块
# ============================================================================

def call_llm(
    client: Dict[str, Any],
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    logger: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    调用 LLM API 并返回 JSON 字符串。
    
    Args:
        client: OpenAI 客户端
        prompt: 完整的 prompt（包含 system 和 user 内容）
        model: 模型名称
        max_tokens: 最大输出 token 数
        temperature: 温度
        max_retries: 最大重试次数
        
    Returns:
        LLM 返回的 JSON 字符串，失败时返回 None
    """
    global _LAST_REQUEST_AT
    for attempt in range(max_retries):
        try:
            if REQUEST_INTERVAL_SEC > 0:
                with _REQUEST_LOCK:
                    now = time.monotonic()
                    wait_sec = REQUEST_INTERVAL_SEC - (now - _LAST_REQUEST_AT)
                    if wait_sec > 0:
                        time.sleep(wait_sec)
                    _LAST_REQUEST_AT = time.monotonic()
            response = requests.post(
                _resolve_chat_url(str(client["base_url"])),
                json={
                    "model": str(client.get("model") or os.getenv("SILICONFLOW_MODEL", os.getenv("LOCAL_DEEPSEEK_MODEL", model or LOCAL_MODEL))),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                    "enable_thinking": True,
                },
                headers={
                    "Authorization": f"Bearer {client['api_key']}",
                    "Content-Type": "application/json",
                },
                verify=VERIFY_SSL,
                timeout=float(client["timeout"]),
            )
            response.raise_for_status()
            payload = response.json()
            content = str(payload["choices"][0]["message"]["content"]).strip()
            
            if content:
                return content
            else:
                return None
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                if logger:
                    logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}; retry in {wait_time}s")
                time.sleep(wait_time)
            else:
                # 最后一次尝试失败，记录错误信息
                if logger:
                    logger.error(f"API error (final attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
                else:
                    print(f"  [API错误] 尝试 {attempt + 1}/{max_retries} 失败: {type(e).__name__}: {e}")
                return None
    
    return None


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """
    解析 LLM 返回的 JSON 响应。
    
    Args:
        response: LLM 返回的字符串
        
    Returns:
        解析后的 JSON 字典，失败时返回 None
    """
    if not response:
        return None
    
    # 尝试提取 JSON 部分（可能包含 markdown 代码块）
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = response
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


# ============================================================================
# 六、分类流程模块
# ============================================================================

def classify_paper(
    client: Dict[str, Any],
    paper_text: str,
    arxiv_id: str,
    filename: str,
    prompt_l1_path: Path,
    prompt_l2l3_path: Path,
    model: str,
    max_tokens: int,
    max_chars: int,
    temperature: float,
    max_retries: int,
    output_l1_dir: Path,
    output_l2l3_dir: Path,
    key_lock: Lock,
    logger: logging.Logger,
    failed_path: Path,
    failed_lock: Lock,
    failed_seen: set
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    对单篇论文进行三级分类。
    
    Args:
        filename: 原始文件名（用于检查分区信息）
    
    Returns:
        (value0, value1, value2, value3) 元组
        - value0: super-con 分区检查结果（1 或 None）
        - value1: L1 的值（1/0 或 None）
        - value2: L2 的值（1/0 或 None）
        - value3: L3 的值（1/0 或 None）
    """
    # 截断文本
    paper_text = truncate_text(paper_text, max_chars)
    
    # 步骤1: 检查 super-con 分区信息
    has_supercon = check_supercon_category(paper_text, filename)
    value0 = 1 if has_supercon else None
    
    # 步骤2: 调用 L1.md prompt（单key串行，避免同key并发）
    try:
        l1_prompt = load_prompt_with_paper_text(prompt_l1_path, paper_text)
        with key_lock:
            l1_response = call_llm(client, l1_prompt, model, max_tokens, temperature, max_retries, logger=logger)
        
        if l1_response:
            l1_json = parse_json_response(l1_response)
            if l1_json and 'L1' in l1_json:
                value1 = l1_json['L1']
                
                # 保存 L1 结果
                l1_output_file = output_l1_dir / f"{arxiv_id}.json"
                l1_output_file.parent.mkdir(parents=True, exist_ok=True)
                with l1_output_file.open('w', encoding='utf-8') as f:
                    json.dump(l1_json, f, ensure_ascii=False, indent=2)
                
                # 如果 L1=0，跳过 L2&L3
                if value1 == 0:
                    return (value0, value1, None, None)
                
                # 步骤3: 调用 L2&3.md prompt（只有 L1=1 时才调用；同key串行）
                try:
                    l2l3_prompt = load_prompt_with_paper_text(prompt_l2l3_path, paper_text)
                    with key_lock:
                        l2l3_response = call_llm(client, l2l3_prompt, model, max_tokens, temperature, max_retries, logger=logger)
                    
                    if l2l3_response:
                        l2l3_json = parse_json_response(l2l3_response)
                        if l2l3_json:
                            value2 = l2l3_json.get('L2')
                            value3 = l2l3_json.get('L3')
                            
                            # 保存 L2&L3 结果
                            l2l3_output_file = output_l2l3_dir / f"{arxiv_id}.json"
                            l2l3_output_file.parent.mkdir(parents=True, exist_ok=True)
                            with l2l3_output_file.open('w', encoding='utf-8') as f:
                                json.dump(l2l3_json, f, ensure_ascii=False, indent=2)
                            
                            return (value0, value1, value2, value3)
                    # L2&L3 调用未获得有效结果（避免返回 None 导致上层解包失败）
                    record_failed(failed_path, failed_lock, failed_seen, arxiv_id, "L2L3", "no_valid_response")
                    return (value0, value1, None, None)
                except Exception as e:
                    # 记录L2&L3调用异常
                    logger.error(f"[L2&L3错误] {arxiv_id}: {type(e).__name__}: {e}")
                    record_failed(failed_path, failed_lock, failed_seen, arxiv_id, "L2L3", f"{type(e).__name__}: {e}")
                    return (value0, value1, None, None)
            else:
                record_failed(failed_path, failed_lock, failed_seen, arxiv_id, "L1", "json_parse_failed_or_missing_L1")
                return (value0, None, None, None)
        else:
            record_failed(failed_path, failed_lock, failed_seen, arxiv_id, "L1", "empty_response")
            return (value0, None, None, None)
    except Exception as e:
        # 记录L1调用异常
        logger.error(f"[L1错误] {arxiv_id}: {type(e).__name__}: {e}")
        record_failed(failed_path, failed_lock, failed_seen, arxiv_id, "L1", f"{type(e).__name__}: {e}")
        return (value0, None, None, None)


# ============================================================================
# 七、结果保存模块
# ============================================================================

def load_existing_results(output_l1_dir: Path, output_l2l3_dir: Path) -> Dict[str, List[Optional[int]]]:
    """
    从已有的输出文件中加载结果（用于断点续传）。
    
    Args:
        output_l1_dir: L1输出目录
        output_l2l3_dir: L2&L3输出目录
        
    Returns:
        字典，键为arxiv_id，值为[value0, value1, value2, value3]列表
    """
    results: Dict[str, List[Optional[int]]] = {}
    
    # 从L1目录加载
    if output_l1_dir.exists():
        for json_file in output_l1_dir.glob("*.json"):
            arxiv_id = json_file.stem
            try:
                with json_file.open('r', encoding='utf-8') as f:
                    l1_data = json.load(f)
                    value1 = l1_data.get('L1')
                    
                    # 初始化结果
                    if arxiv_id not in results:
                        results[arxiv_id] = [None, None, None, None]
                    results[arxiv_id][1] = value1
            except Exception:
                continue
    
    # 从L2&L3目录加载
    if output_l2l3_dir.exists():
        for json_file in output_l2l3_dir.glob("*.json"):
            arxiv_id = json_file.stem
            try:
                with json_file.open('r', encoding='utf-8') as f:
                    l2l3_data = json.load(f)
                    value2 = l2l3_data.get('L2')
                    value3 = l2l3_data.get('L3')
                    
                    # 初始化结果（如果不存在）
                    if arxiv_id not in results:
                        results[arxiv_id] = [None, None, None, None]
                    results[arxiv_id][2] = value2
                    results[arxiv_id][3] = value3
            except Exception:
                continue
    
    return results

def save_results_to_txt(
    results: Dict[str, List[Optional[int]]],
    output_dir: Path,
    chunk_size: int = 1000,
    ordered_ids: Optional[List[str]] = None
) -> None:
    """
    将结果保存到 .txt 文件，每 1000 篇文章一个文件。
    
    Args:
        results: 字典，键为 arxiv_id，值为 [value0, value1, value2, value3] 列表
        output_dir: 输出目录
        chunk_size: 每个文件包含的文章数量
        ordered_ids: 可选的有序 arxiv_id 列表，如果提供则按此顺序保存
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果提供了有序列表，使用它；否则按 arxiv_id 排序
    if ordered_ids:
        sorted_ids = [aid for aid in ordered_ids if aid in results]
    else:
        sorted_ids = sorted(results.keys())
    total = len(sorted_ids)
    
    for i in range(0, total, chunk_size):
        chunk_ids = sorted_ids[i:i + chunk_size]
        start_num = i + 1
        end_num = min(i + chunk_size, total)
        filename = f"{start_num}-{end_num}.txt"
        filepath = output_dir / filename
        
        with filepath.open('w', encoding='utf-8') as f:
            for arxiv_id in chunk_ids:
                values = results[arxiv_id]
                # 格式化列表：空值用空字符串表示
                values_str = ','.join(str(v) if v is not None else '' for v in values)
                f.write(f"{arxiv_id} -> [{values_str}]\n")
        
        print(f"  已保存: {filename} ({len(chunk_ids)} 篇文章)")


# ============================================================================
# 八、主流程模块
# ============================================================================

def process_single_paper(
    paper_file: Path,
    idx: int,
    total: int,
    client: Dict[str, Any],
    key_lock: Lock,
    logger: logging.Logger,
    failed_path: Path,
    failed_lock: Lock,
    failed_seen: set,
    output_l1_dir: Path,
    output_l2l3_dir: Path,
    prompt_l1_path: Path,
    prompt_l2l3_path: Path,
    model: str,
    max_tokens: int,
    max_chars: int,
    temperature: float,
    max_retries: int,
    print_lock: Lock
) -> Tuple[int, str, List[Optional[int]]]:
    """处理单篇论文（用于并行处理）。"""
    arxiv_id = extract_arxiv_id(paper_file.name)
    
    try:
        paper_text = read_text(paper_file)
        if not paper_text.strip():
            record_failed(failed_path, failed_lock, failed_seen, arxiv_id, "READ", "empty_file")
            with print_lock:
                print(f"[{idx}/{total}] X {arxiv_id} -> [,,,] (empty_file)")
            return (idx, arxiv_id, [None, None, None, None])
        
        value0, value1, value2, value3 = classify_paper(
            client=client,
            paper_text=paper_text,
            arxiv_id=arxiv_id,
            filename=paper_file.name,
            prompt_l1_path=prompt_l1_path,
            prompt_l2l3_path=prompt_l2l3_path,
            model=model,
            max_tokens=max_tokens,
            max_chars=max_chars,
            temperature=temperature,
            max_retries=max_retries,
            output_l1_dir=output_l1_dir,
            output_l2l3_dir=output_l2l3_dir,
            key_lock=key_lock,
            logger=logger,
            failed_path=failed_path,
            failed_lock=failed_lock,
            failed_seen=failed_seen
        )
        
        # 每篇都输出一行，便于观察进度
        values_str = ','.join(str(v) if v is not None else '' for v in [value0, value1, value2, value3])
        with print_lock:
            status = "OK" if value1 is not None else "X"
            print(f"[{idx}/{total}] {status} {arxiv_id} -> [{values_str}]")
        logger.info(f"[{idx}/{total}] {arxiv_id} -> [{values_str}]")
        
        return (idx, arxiv_id, [value0, value1, value2, value3])
        
    except Exception as e:
        with print_lock:
            print(f"[{idx}/{total}] X ERROR: {arxiv_id} - {e}")
        logger.error(f"[{idx}/{total}] ERROR {arxiv_id}: {type(e).__name__}: {e}")
        record_failed(failed_path, failed_lock, failed_seen, arxiv_id, "PROCESS", f"{type(e).__name__}: {e}")
        return (idx, arxiv_id, [None, None, None, None])


def process_all_papers(
    api_keys: List[str],
    input_dir: Path,
    output_general_dir: Path,
    output_l1_dir: Path,
    output_l2l3_dir: Path,
    prompt_l1_path: Path,
    prompt_l2l3_path: Path,
    model: str,
    max_tokens: int,
    max_chars: int,
    temperature: float,
    max_retries: int,
    timeout_seconds: float,
    logger: logging.Logger,
    failed_path: Path,
    failed_lock: Lock,
    failed_seen: set,
    base_url: str = "https://api.deepseek.com/v1"
) -> None:
    """批量处理所有论文文件（并行版本，使用多个API key）。"""
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    allowed_exts = {'.md', '.markdown'}
    if os.getenv("IE_CLASSIFICATION_RECURSIVE_INPUT", "0").lower() in {"1", "true", "yes"}:
        all_paper_files = sorted([f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in allowed_exts])
    else:
        all_paper_files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in allowed_exts])
    total_all = len(all_paper_files)
    
    if total_all == 0:
        print(f"警告: {input_dir} 目录下没有找到 Markdown 文件（支持 .md/.markdown）")
        return
    
    # 断点续传：检查已处理的文件（通过L1输出目录中的JSON文件判断）
    output_l1_dir.mkdir(parents=True, exist_ok=True)
    processed_ids = set()
    if output_l1_dir.exists():
        for json_file in output_l1_dir.glob("*.json"):
            processed_ids.add(json_file.stem)  # stem是文件名（不含扩展名）
    
    # 过滤掉已处理的文件
    paper_files = []
    skipped_count = 0
    for paper_file in all_paper_files:
        arxiv_id = extract_arxiv_id(paper_file.name)
        if arxiv_id in processed_ids:
            skipped_count += 1
        else:
            paper_files.append(paper_file)
    
    total = len(paper_files)
    num_workers = len(api_keys)
    
    print(f"\n找到 {total_all} 个文件（已处理 {skipped_count} 个，剩余 {total} 个待处理）")
    print(f"并行workers: {num_workers}")
    print(f"模型: {model}")
    print(f"输出目录:")
    print(f"  - General: {output_general_dir}")
    print(f"  - L1: {output_l1_dir}")
    print(f"  - L2&L3: {output_l2l3_dir}")
    
    # 为每个worker创建客户端（尽量配置超时，避免网络半开导致永久阻塞）
    clients: List[OpenAI] = []
    for key in api_keys:
        try:
            # openai>=1.x 常见签名支持 timeout=...
            clients.append(OpenAI(api_key=key, base_url=base_url, timeout=timeout_seconds))
            logger.info(f"Init OpenAI client with timeout={timeout_seconds}s")
        except TypeError:
            # 兼容：某些版本通过 http_client (httpx) 配置 timeout
            try:
                import httpx  # type: ignore
                http_client = httpx.Client(timeout=httpx.Timeout(timeout_seconds))
                clients.append(OpenAI(api_key=key, base_url=base_url, http_client=http_client))
                logger.info(f"Init OpenAI client with httpx timeout={timeout_seconds}s")
            except Exception as e:
                logger.warning(f"Failed to set timeout on OpenAI client; may hang on network issues: {type(e).__name__}: {e}")
                clients.append(OpenAI(api_key=key, base_url=base_url))
    clients = [{"api_key": key, "base_url": base_url, "timeout": timeout_seconds} for key in api_keys]
    key_locks = [Lock() for _ in api_keys]
    if LOCAL_API_KEY.strip():
        if LOCAL_API_KEY_2:
            api_slots = [_pack_slot(LOCAL_API_KEY.strip(), LOCAL_MODEL)] * LOCAL_PER_KEY_PARALLEL
            api_slots += [_pack_slot(LOCAL_API_KEY_2, LOCAL_MODEL_2)] * LOCAL_PER_KEY_PARALLEL
        else:
            api_slots = [_pack_slot(LOCAL_API_KEY.strip(), LOCAL_MODEL)] * LOCAL_MAX_PARALLEL
        clients = []
        for slot in api_slots:
            slot_key, slot_model = _split_slot(slot, model or LOCAL_MODEL)
            clients.append({"api_key": slot_key, "base_url": base_url, "timeout": timeout_seconds, "model": slot_model})
        key_locks = [Lock() for _ in clients]  # 每个key一个锁，避免同key并发

    num_workers = len(clients)
    
    # 用于线程安全的打印
    print_lock = Lock()
    
    # 存储结果（按索引排序）
    results_dict: Dict[int, Tuple[str, List[Optional[int]]]] = {}
    processed = 0
    failed = 0
    
    # 提前获取所有文件的顺序（用于保存时按顺序）
    all_ordered_ids = [extract_arxiv_id(f.name) for f in all_paper_files]
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = []
        for idx, paper_file in enumerate(paper_files, 1):
            # 轮询分配API key（同key的任务串行化）
            key_idx = (idx - 1) % num_workers
            client = clients[key_idx]
            key_lock = key_locks[key_idx]
            future = executor.submit(
                process_single_paper,
                paper_file,
                idx,
                total,
                client,
                key_lock,
                logger,
                failed_path,
                failed_lock,
                failed_seen,
                output_l1_dir,
                output_l2l3_dir,
                prompt_l1_path,
                prompt_l2l3_path,
                model,
                max_tokens,
                max_chars,
                temperature,
                max_retries,
                print_lock
            )
            futures.append((idx, future))
        
        # 收集结果（按完成顺序，真正并行）
        completed = 0
        for future in as_completed([f for _, f in futures]):
            try:
                result_idx, arxiv_id, values = future.result()
                results_dict[result_idx] = (arxiv_id, values)
                completed += 1
                if values != [None, None, None, None]:
                    processed += 1
                else:
                    failed += 1
                
                # 每100个文件打印一次进度
                if completed % 100 == 0 or completed == total:
                    with print_lock:
                        print(f"\n进度: {completed}/{total} ({completed*100//total}%) | 成功: {processed}, 失败: {failed}")
                
                # 每1000个文件保存一次结果到 general 目录
                if completed % 1000 == 0 or completed == total:
                    # 按原始顺序整理当前已收集的新结果
                    current_new_results: Dict[str, List[Optional[int]]] = {}
                    for idx in sorted(results_dict.keys()):
                        arxiv_id, values = results_dict[idx]
                        current_new_results[arxiv_id] = values
                    
                    # 加载所有已有结果（包括L1和L2&L3目录中的JSON文件）
                    current_all_results = load_existing_results(output_l1_dir, output_l2l3_dir)
                    # 合并当前新结果
                    current_all_results.update(current_new_results)
                    
                    # 保存到 general 目录
                    with print_lock:
                        print(f"\n保存当前结果到 general 目录（已处理 {len(current_all_results)} 篇）...")
                    save_results_to_txt(current_all_results, output_general_dir, chunk_size=1000, ordered_ids=all_ordered_ids)
                    
                    # 更新 experimental 列表
                    experimental_ids = []
                    for aid, vals in current_all_results.items():
                        if len(vals) >= 4 and vals[1] == 1 and vals[2] == 1 and vals[3] == 1:
                            experimental_ids.append(aid)
                    if experimental_ids:
                        experimental_file = output_general_dir.parent / "experimental.txt"
                        with experimental_file.open('w', encoding='utf-8') as f:
                            for aid in sorted(experimental_ids):
                                f.write(f"{aid}\n")
                        with print_lock:
                            print(f"  已更新 experimental.txt ({len(experimental_ids)} 篇)")
            except Exception as e:
                with print_lock:
                    print(f"Future异常: {e}")
                failed += 1
                completed += 1
    
    # 最终保存一次，确保所有结果都已保存（虽然循环中已经保存过，但这里作为最终确认）
    print(f"\n最终保存所有结果...")
    # 按原始顺序整理所有新处理的结果
    new_results: Dict[str, List[Optional[int]]] = {}
    for idx in sorted(results_dict.keys()):
        arxiv_id, values = results_dict[idx]
        new_results[arxiv_id] = values
    
    # 加载所有已有结果（包括L1和L2&L3目录中的JSON文件）
    all_results = load_existing_results(output_l1_dir, output_l2l3_dir)
    # 合并新结果
    all_results.update(new_results)
    print(f"  共有 {len(all_results)} 个结果（本次新增 {len(new_results)} 个）")
    
    # 保存所有结果到 .txt 文件（按原始文件顺序）
    save_results_to_txt(all_results, output_general_dir, chunk_size=1000, ordered_ids=all_ordered_ids)
    
    # 筛选出 experimental 文章（L1=1, L2=1, L3=1，不管第0项是什么）
    experimental_ids = []
    for arxiv_id, values in all_results.items():
        # values = [value0, value1, value2, value3]
        # 只要 value1=1, value2=1, value3=1 就算 experimental
        if len(values) >= 4 and values[1] == 1 and values[2] == 1 and values[3] == 1:
            experimental_ids.append(arxiv_id)
    
    # 保存 experimental 列表
    if experimental_ids:
        experimental_file = output_general_dir.parent / "experimental.txt"
        with experimental_file.open('w', encoding='utf-8') as f:
            for arxiv_id in sorted(experimental_ids):
                f.write(f"{arxiv_id}\n")
        print(f"\n已保存 experimental 列表: {experimental_file.name} ({len(experimental_ids)} 篇文章)")
    else:
        print(f"\n未找到 experimental 文章（L1=1, L2=1, L3=1）")
    
    print(f"\n处理完成!")
    print(f"  成功: {processed}/{total}")
    print(f"  失败: {failed}/{total}")
    print(f"  Experimental: {len(experimental_ids)} 篇")
    print(f"  结果已保存到: {output_general_dir}")


# ============================================================================
# 九、主函数
# ============================================================================

def main() -> None:
    """主函数：加载配置，使用8个API key并行处理论文。"""
    root = Path(__file__).resolve().parent
    logger = setup_logging(root)
    cfg = load_config(root / "config.yaml")
    
    api_cfg = cfg.get("api", {})
    run_cfg = cfg.get("run", {})
    
    # 从配置文件读取API keys
    local_api_key = (os.getenv("LOCAL_DEEPSEEK_API_KEY") or LOCAL_API_KEY).strip()
    if local_api_key:
        api_keys = [local_api_key] * LOCAL_MAX_PARALLEL
    else:
        api_keys = api_cfg.get("api_keys", [])
        if not api_keys:
            raise ValueError("api_keys not found in config.yaml. Please add api_keys list under api section.")
        if not isinstance(api_keys, list) or len(api_keys) == 0:
            raise ValueError("api_keys must be a non-empty list in config.yaml")
    
    # 设置路径
    input_dir_cfg = os.getenv("IE_CLASSIFICATION_INPUT_DIR", run_cfg.get("input_dir", "papers"))
    if Path(input_dir_cfg).is_absolute():
        input_dir = Path(input_dir_cfg)
    else:
        input_dir = root / input_dir_cfg    
    output_dir_cfg = os.getenv("IE_CLASSIFICATION_OUTPUT_DIR", run_cfg.get("output_dir", "output"))
    output_dir = Path(output_dir_cfg) if Path(output_dir_cfg).is_absolute() else root / output_dir_cfg
    output_general_dir = output_dir / "general"
    output_l1_dir = output_dir / "L1"
    output_l2l3_dir = output_dir / "L2&L3"
    
    prompt_dir = root / run_cfg.get("prompt_dir", "prompts")
    prompt_l1_path = prompt_dir / "L1.md"
    prompt_l2l3_path = prompt_dir / "L2&3.md"
    
    # API 配置
    model = api_cfg.get("model", "deepseek-reasoner")  # 默认使用 reasoner
    temperature = api_cfg.get("temperature", 0.4)
    max_retries = api_cfg.get("max_retries", 5)
    timeout_seconds = float(api_cfg.get("timeout_seconds", 300))
    max_tokens = run_cfg.get("max_tokens", 5000)
    max_chars = run_cfg.get("max_chars", 200000)
    base_url = api_cfg.get("base_url", "https://api.deepseek.com/v1")
    
    model = os.getenv("SILICONFLOW_MODEL", os.getenv("LOCAL_DEEPSEEK_MODEL", model or LOCAL_MODEL))
    base_url = os.getenv("SILICONFLOW_BASE_URL", os.getenv("LOCAL_DEEPSEEK_BASE_URL", base_url or LOCAL_BASE_URL))

    print("=" * 60)
    print("三级分类系统（并行版本）")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"并行workers: {len(api_keys)}")
    print(f"模型: {model}")
    print(f"温度: {temperature}")
    print(f"最大 tokens: {max_tokens}")
    print(f"最大字符数: {max_chars}")
    print(f"请求超时(秒): {timeout_seconds}")
    print("=" * 60)
    logger.info(f"Run config: input_dir={input_dir}, output_dir={output_dir}, workers={len(api_keys)}, model={model}, temp={temperature}, max_tokens={max_tokens}, max_chars={max_chars}, timeout_seconds={timeout_seconds}, max_retries={max_retries}")
    
    # 检查 prompt 文件
    if not prompt_l1_path.exists():
        raise FileNotFoundError(f"L1 prompt file not found: {prompt_l1_path}")
    if not prompt_l2l3_path.exists():
        raise FileNotFoundError(f"L2&L3 prompt file not found: {prompt_l2l3_path}")
    
    print(f"\n已配置 {len(api_keys)} 个API key，准备并行处理...")

    failed_path, failed_lock, failed_seen = init_failed_writer(output_dir)
    logger.info(f"Failed list path: {failed_path}")
    
    # 处理所有论文（并行）
    process_all_papers(
        api_keys=api_keys,
        input_dir=input_dir,
        output_general_dir=output_general_dir,
        output_l1_dir=output_l1_dir,
        output_l2l3_dir=output_l2l3_dir,
        prompt_l1_path=prompt_l1_path,
        prompt_l2l3_path=prompt_l2l3_path,
        model=model,
        max_tokens=max_tokens,
        max_chars=max_chars,
        temperature=temperature,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        logger=logger,
        failed_path=failed_path,
        failed_lock=failed_lock,
        failed_seen=failed_seen,
        base_url=base_url
    )


if __name__ == "__main__":
    main()
