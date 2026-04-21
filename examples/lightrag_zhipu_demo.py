import os
import logging


from lightrag import LightRAG, QueryParam
from lightrag.llm.zhipu import zhipu_complete, zhipu_embedding
from lightrag.utils import EmbeddingFunc

WORKING_DIR = r"D:\lightrag\book\paper-experiment-FB15K237\dic"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

api_key = os.environ.get("ZHIPUAI_API_KEY")
if api_key is None:
    raise Exception("Please set ZHIPU_API_KEY in your environment")


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=zhipu_complete,
    llm_model_name="glm-4-flash-250414",  # Using the most cost/performance balance model, but you can change it here.
    # llm_model_max_async=4,
    llm_model_max_async=30,
    llm_model_max_token_size=32768,
    # llm_model_max_token_size=16384,
    embedding_func=EmbeddingFunc(
        embedding_dim=2048,  # Zhipu embedding-3 dimension
        max_token_size=8192,
        func=lambda texts: zhipu_embedding(texts),
    ),
)

with open(r"D:\lightrag\book\paper-experiment-FB15K237\merge-txt(2k).txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform naive search
print(
    rag.query("介绍主要内容", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("介绍主要内容", param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query("介绍主要内容", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("介绍主要内容", param=QueryParam(mode="hybrid"))
)
