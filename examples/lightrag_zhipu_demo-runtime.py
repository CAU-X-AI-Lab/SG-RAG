import os
import logging
import time

from lightrag import LightRAG, QueryParam
from lightrag.llm.zhipu import zhipu_complete, zhipu_embedding
from lightrag.utils import EmbeddingFunc

WORKING_DIR = r"D:\lightrag\dic\dickens7-CPubMed-KG"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

api_key = os.environ.get("ZHIPUAI_API_KEY")
if api_key is None:
    raise Exception("Please set ZHIPU_API_KEY in your environment")

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=zhipu_complete,
    llm_model_name="glm-4-flash-250414",
    llm_model_max_async=30,
    llm_model_max_token_size=32768,
    embedding_func=EmbeddingFunc(
        embedding_dim=2048,
        max_token_size=8192,
        func=lambda texts: zhipu_embedding(texts),
    ),
)

with open(r"D:\lightrag\book\CPubMed-KGv2_0_-shaixuan-NL-wukuohao.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

query_text = "哪种疾病在进行超声诊断时常伴有横纹肌溶解症，并且实验室检查中会出现Ep阳性，同时患者可能伴有骨质疏松症？"

# 定义一个函数用于统计时间
def timed_query(mode_name: str):
    param = QueryParam(mode=mode_name)
    start = time.time()
    result = rag.query(query_text, param=param)
    end = time.time()
    print(f"\n[{mode_name.upper()}] 用时：{end - start:.2f} 秒")
    print(result)

# 执行四种模式
for mode in ["naive", "local", "global", "hybrid"]:
    timed_query(mode)
