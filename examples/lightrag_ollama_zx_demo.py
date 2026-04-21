import asyncio
import os
import inspect
import hashlib
import logging
import re
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# 配置路径
WORKING_DIR = r"D:\lightrag\dic\dickens7-CPubMed-KG"
TEXT_PATH = r"D:\lightrag\book\CPubMed-KGv2_0_-shaixuan-NL-wukuohao.txt"

# 设置日志
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# ✅ 批量 embedding 函数（建议内部支持批处理）
async def batch_embed(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = await ollama_embed(batch, embed_model="nomic-embed-text", host="http://localhost:11434")
        results.extend(embeddings)
    return results


# ✅ 初始化 LightRAG 实例
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2m",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=1024,  # 控制单段最大 token 长度，利于切段处理
        func=batch_embed,
    ),
)

# ✅ 工具函数：文本切分 + 去重
def split_text(text, max_chars=800):
    sents = re.split(r'(?<=[。！？])', text)
    chunks, buf = [], ""
    for sent in sents:
        if len(buf) + len(sent) <= max_chars:
            buf += sent
        else:
            chunks.append(buf)
            buf = sent
    if buf:
        chunks.append(buf)
    return chunks

def get_text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# ✅ 加载文本、切分、去重、嵌入
with open(TEXT_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()
chunks = split_text(raw_text, max_chars=800)

# 去重处理
unique_chunks, seen = [], set()
for chunk in chunks:
    h = get_text_hash(chunk)
    if h not in seen:
        unique_chunks.append(chunk)
        seen.add(h)

print(f"📄 原始段落数：{len(chunks)}，去重后：{len(unique_chunks)}，开始嵌入...")
rag.insert(unique_chunks)

print(f"✅ 嵌入完成，使用模型：{rag.llm_model_name}")

# ✅ 查询测试
question = "主要内容？"
modes = ["naive", "local", "global", "hybrid"]

for mode in modes:
    print(f"\n[{mode.upper()}]")
    print(rag.query(question, param=QueryParam(mode=mode)))

# ✅ 流式输出
resp = rag.query(question, param=QueryParam(mode="hybrid", stream=True))

async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)

if inspect.isasyncgen(resp):
    asyncio.run(print_stream(resp))
else:
    print(resp)
