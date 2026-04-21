import asyncio
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

WORKING_DIR = r"D:\lightrag\dic\dickens7-CPubMed-KG"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2m",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

with open(r"D:\lightrag\book\CPubMed-KGv2_0_-shaixuan-NL.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

print(f"Using LLM model: {rag.llm_model_name}")

# 原始问题：What are the top themes in this story?
# Perform naive search
print(
    rag.query("哪种疾病在钢板外露和化瘀止痛的情况下，适合采用中西医结合治疗和常规护理？", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("哪种疾病在钢板外露和化瘀止痛的情况下，适合采用中西医结合治疗和常规护理？", param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query("哪种疾病在钢板外露和化瘀止痛的情况下，适合采用中西医结合治疗和常规护理？", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("哪种疾病在钢板外露和化瘀止痛的情况下，适合采用中西医结合治疗和常规护理？", param=QueryParam(mode="hybrid"))
)

# stream response
resp = rag.query(
    "哪种疾病在钢板外露和化瘀止痛的情况下，适合采用中西医结合治疗和常规护理？",
    param=QueryParam(mode="hybrid", stream=True),
)


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


if inspect.isasyncgen(resp):
    asyncio.run(print_stream(resp))
else:
    print(resp)
