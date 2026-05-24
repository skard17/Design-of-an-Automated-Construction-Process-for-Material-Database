import requests
import json
import time
from tqdm import tqdm
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict
import urllib3

# 忽略 SSL 证书警告 (对应 curl 的 -k)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# DeepSeek R1 API 配置
API_KEY = os.getenv("DEEPSEEK_R1_API_KEY", "")
R1_URL = "https://10.140.158.153:1020/dsr1/all/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 全局共享锁（多线程写文件时使用）
lock = Lock()

def get_answer_threadsafe(idx, item, max_retries=100):
    context = item.get("context", "") if item.get("context", "") else ""
    question = item.get("question", "") if item.get("question", "") else ""
    question = context + question
    
    symbol_dict = item['symbol']
    symbol_prompt = "Here are the relevant symbols:\n" + "\n".join([f"{k}: {v}" for k, v in symbol_dict.items()])
    solution = item["answer"]
    gt_answer = item["final_answer"]

    
    for attempt in range(1, max_retries + 1):
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a condensed matter physics expert. Please read the following question and provide a step-by-step solution using only the given symbols. Do not introduce any new symbols that are not provided in the problem statement. Your final answer must be presented as a readable LaTeX formula, enclosed in a \\boxed{} environment."
                },
                {
                    "role": "user",
                    "content": question + symbol_prompt
                }
            ],
            "stream": False
        }
        
        start_time = time.time()
        try:
            # verify=False 对应 curl -k，忽略 SSL 证书验证
            response = requests.post(R1_URL, json=payload, headers=headers, verify=False, timeout=3000)
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - start_time
            llm_answer = data['choices'][0]['message']['content']
            break  # 成功就跳出循环
        except Exception as e:
            elapsed = time.time() - start_time
            llm_answer = f"Error (attempt {attempt}): {str(e)}"

            if attempt == max_retries:
                # 如果最后一次仍然失败，则保留错误记录
                break
            else:
                time.sleep(30)  # 简单延迟，避免快速重试导致接口异常

    record = {
        "index": idx,
        "question": question,
        "symbol": symbol_dict,
        "gt_answer": gt_answer,
        "solution" : solution,
        "llm_answer": llm_answer,
        "modified": item["modified"],
        "type": item["type"],
        "answer_type":item["answer_type"],
        "topic": item.get("topic", ""),
        "elapsed_time": round(elapsed, 2)
    }

    # 多线程写文件需要加锁
    with lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    return idx

def file_is_complete(output_file, dataset_len):
    """检查文件是否完成且无Error"""
    if not os.path.exists(output_file):
        return False
    
    line_count = 0
    has_error = False
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            line_count += 1
            try:
                data = json.loads(line)
                if str(data.get("llm_answer", "")).startswith("Error"):
                    has_error = True
            except:
                has_error = True
    
    # 只有当行数达到数据集长度且无Error时，才认为文件完成
    return line_count >= dataset_len and not has_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepSeek R1 on CMPhysBench dataset")
    parser.add_argument("--model", type=str, default="deepseek-r1-huawei-910b", help="LLM model name")
    parser.add_argument("--workers", type=int, default=64, help="Number of concurrent threads")
    args = parser.parse_args()
    model_name = args.model
    num_workers = args.workers

    with open('/Users/wangweida/Desktop/PJ/CMPhysBench/CMPhysBench1106.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(len(dataset))
    # 替换模型名中的 / 为 _，防止路径冲突
    safe_model_name = model_name.replace("/", "_")
    output_file = f"result/results_{safe_model_name}.jsonl"
    print(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    final = list(range(1,521))

    # ---------- Step 1: 建立 index -> item 映射 ----------
    index_to_item = {}
    for item in dataset:
        idx = item.get("id")
        if idx is not None:
            index_to_item[idx] = item

    # ---------- Step 2: 读取已处理的记录 ----------
    cleaned_records = {}
    duplicate_indexes = defaultdict(list)

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    idx = data.get("index")
                    answer = data.get("llm_answer", "")
                    if answer.startswith("Error") or idx in cleaned_records:
                        duplicate_indexes[idx].append(line_num)
                        continue
                    cleaned_records[idx] = data
                except json.JSONDecodeError:
                    continue

        with open(output_file, 'w', encoding='utf-8') as f:
            for record in cleaned_records.values():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---------- Step 3: 构造待处理任务 ----------
    processed_indexes = set(cleaned_records.keys())
    print(processed_indexes)
    tasks = [(idx, item) for idx, item in index_to_item.items()
                if idx not in processed_indexes and idx in final]
    print(f"待处理数量: {len(tasks)}")

    # ---------- 并发处理 ----------
    # 多线程并发处理任务
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(get_answer_threadsafe, idx, item) for idx, item in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"并发处理 ({model_name})"):
            pass  # 仅用于显示进度

    print(f"\n✅ 所有处理完成，结果保存在 {output_file}")
