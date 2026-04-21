
import csv, os, time, threading, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from lxml import etree
from tqdm import tqdm
from openai import OpenAI

# ---------- 1. OpenAI ----------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    # base_url="https://ai.nengyongai.cn/v1"
)

# ---------- 2. QPS 令牌桶 ----------
QPS_LIMIT = 15
_call_times, _lock = collections.deque(), threading.Lock()

def enforce_qps():
    with _lock:
        now = time.time()
        while _call_times and now - _call_times[0] >= 1:
            _call_times.popleft()
        if len(_call_times) >= QPS_LIMIT:
            time.sleep(1 - (now - _call_times[0]))
        _call_times.append(time.time())

# ---------- 3. GraphML 解析 ----------
NS = {"g": "http://graphml.graphdrawing.org/xmlns"}

@lru_cache(maxsize=1024)
def parse_graphml(path: str):
    if not path or not Path(path).exists():
        return [f"[图文件不存在] {path}"], ""
    try:
        root = etree.parse(path).getroot()
        id2info = {}
        for node in root.xpath(".//g:node", namespaces=NS):
            nid = node.attrib["id"]
            label = node.xpath("g:data[@key='d0']/text()", namespaces=NS)
            ntype = node.xpath("g:data[@key='d1']/text()", namespaces=NS)
            id2info[nid] = (label[0].strip() if label else nid,
                            ntype[0].strip() if ntype else "")
        sents = []
        for edge in root.xpath(".//g:edge", namespaces=NS):
            src, tgt = edge.attrib["source"], edge.attrib["target"]
            rel = edge.xpath("g:data[@key='d2' or @key='d3']/text()", namespaces=NS)
            relation = rel[0].strip() if rel else "存在关联"
            sl, st = id2info.get(src, (src, "")); tl, tt = id2info.get(tgt, (tgt, ""))
            sents.append(f"{sl}（{st}） 与 {tl}（{tt}） 的关系：{relation}。")
        return sents, "\n".join(sents)
    except Exception as e:
        return [f"[图解析失败: {e}]"], ""

# ---------- 4. OpenAI 调用 ----------
# ---------- 4. OpenAI call (Biography QA prompt, English) ----------
# ---------- 4. OpenAI call (Encyclopedia QA prompt, English) ----------
def call_openai(question: str, edge_sents: list[str]) -> str:
    """
    Return format:
    Answer: <entity name or "Unknown">
    Reason: <brief justification that cites relations from the paragraph>
    """
    try:
        enforce_qps()
        paragraph = "\n".join(edge_sents[:100])          # use first 100 sentences as context

        prompt = f"""You will be given a paragraph written in natural language.  
The paragraph is automatically generated from an **encyclopedic knowledge graph** and therefore describes relationships among people, places, events, works, scientific concepts, and dates.  
Hidden somewhere in that paragraph is the entity that best answers the user’s question, but the question itself does **not** mention this entity by name.

Your tasks:

1. **First, output the most likely Answer (an entity name).**  
2. **Then, explain your reasoning**  by citing the entities and relationship words that appear in the paragraph.  
   *If the paragraph is insufficient to identify the entity, reply with “Unknown” and omit the reason.*

---
📖 **Relationship paragraph**  
{paragraph}

---
❓ **User question**  
{question}

*Respond **exactly** in the following template (do not add extra lines):*

Answer: <entity name or "Unknown">  
Reason: <concise explanation citing relations from the paragraph>"""

        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an encyclopedia question-answering assistant skilled at reasoning over structured relationships."
                },
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return rsp.choices[0].message.content.strip()

    except Exception as e:
        return f"[API call failed] {e}"



# ---------- 5. GraphML 索引 ----------
def build_gml_index(base_dir: str):
    """一次性扫描目录构建 {filename: abs_path} 索引。"""
    mapping = {}
    for r, _, fs in os.walk(base_dir):
        for f in fs:
            if f.endswith(".graphml"):
                mapping[f] = os.path.join(r, f)
    return mapping

# ---------- 6. 主处理（单个 CSV） ----------
def process_csv(input_csv: str, output_csv: str,
                graphml_dir: str, workers: int = 15):
  
    gml_index = build_gml_index(graphml_dir)

    with open(input_csv, newline='', encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    pool    = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    for r in rows:
        gml_path = gml_index.get(r["file"], "")
        edge_sents, _ = parse_graphml(gml_path)
        futures[pool.submit(call_openai, r["query"], edge_sents)] = r

    # 提前打开输出文件，写表头
    with open(output_csv, "w", newline='', encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["file", "query", "gold_answer", "LLM_answer"])

        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"🤖 处理 {Path(input_csv).name}"):
            row = futures[fut]
            try:
                ans = fut.result()
            except Exception as e:
                ans = f"[线程异常] {e}"

            writer.writerow([row["file"], row["query"], row["answer"], ans])
            fout.flush()               # 立即落盘，实时可见

    print(f"✅ 输出完成：{output_csv}")


# ---------- 7. 手动设定路径 ----------
if __name__ == "__main__":
    # 只需改这三行
    INPUT_CSV   = r".\answer_with_edge\unique_answer_rows_all_combined.csv"                  # 指定输入 CSV
    OUTPUT_CSV  = r""          # 指定输出 CSV
    GRAPHML_DIR = r".\subgraph"        # GraphML 根目录

    process_csv(INPUT_CSV, OUTPUT_CSV, GRAPHML_DIR, workers=15)
