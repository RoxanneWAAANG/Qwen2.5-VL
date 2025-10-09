import re, json, os

CONTROL_CHARS_RE = re.compile(r'[\x00-\x1F\x7F]')

def _json_sanitize_control_chars(s: str) -> str:
    # 将所有控制字符转成 \u00xx（包括未转义的 \t、\n、\r 等）
    return CONTROL_CHARS_RE.sub(lambda m: '\\u%04x' % ord(m.group(0)), s)

def load_test_data_robust(path: str, out_dir: str, max_samples: int = None):
    os.makedirs(out_dir, exist_ok=True)
    bad_lines_path = os.path.join(out_dir, "bad_lines.txt")

    data = []

    # 先判断是 JSON 数组还是 JSONL
    with open(path, "rb") as f:
        head = f.read(1)
        # 回到文件开头
        f.seek(0)

        if head == b'[':
            # 整体 JSON 数组
            try:
                text = f.read().decode("utf-8", errors="replace")
                try:
                    # 先直接尝试
                    arr = json.loads(text)
                except json.JSONDecodeError:
                    # 若数组内也混入了控制字符，做一次全量清洗再 parse
                    text = _json_sanitize_control_chars(text)
                    arr = json.loads(text)
                if not isinstance(arr, list):
                    raise ValueError("Top-level JSON is not a list.")
                data = arr if max_samples is None else arr[:max_samples]
                return data
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSON array file: {e}")
        else:
            # 按 JSONL 逐行读取
            bad_lines = []
            with open(path, "rb") as fin:
                for idx, raw in enumerate(fin, start=1):
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or line in (",", "[", "]"):
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # 清洗控制字符后再试
                        safe = _json_sanitize_control_chars(line)
                        try:
                            obj = json.loads(safe)
                        except Exception as e2:
                            # 仍然失败：记录坏行并跳过
                            bad_lines.append((idx, str(e2), line[:400]))
                            continue
                    data.append(obj)
                    if max_samples and len(data) >= max_samples:
                        break

            if bad_lines:
                with open(bad_lines_path, "w", encoding="utf-8") as fout:
                    for ln, err, preview in bad_lines:
                        fout.write(f"[line {ln}] {err}\n{preview}\n---\n")
                print(f"⚠️  {len(bad_lines)} bad lines were skipped. See {bad_lines_path}")

            return data
