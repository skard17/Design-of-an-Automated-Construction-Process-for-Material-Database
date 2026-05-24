import os
import sys
import glob

def extract_ids(input_dir, output_file):
    supplement_ids = []
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    if not txt_files:
        print(f"Warning: No txt files found in {input_dir}")
        return

    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if " -> [" not in line:
                        continue
                    
                    parts_split = line.split(" -> ")
                    if len(parts_split) < 2: continue
                    
                    paper_id = parts_split[0].strip()
                    result_str = parts_split[1].strip()
                    
                    inner = result_str.strip("[]")
                    parts = [p.strip() for p in inner.split(",")]
                    
                    # 修改后的条件：匹配 [,1,1,0]
                    if len(parts) >= 4:
                        if parts[1] == '1' and parts[2] == '0' and parts[3] == '0':
                            supplement_ids.append(paper_id)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    unique_ids = sorted(list(set(supplement_ids)))
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for pid in unique_ids:
            out_f.write(f"{pid}\n")
    print(f"Successfully extracted {len(unique_ids)} IDs to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 extract_supplement.py <input_dir> <output_file>")
        sys.exit(1)
    
    extract_ids(sys.argv[1], sys.argv[2])