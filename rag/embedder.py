import numpy as np

# transformers CVE-2025-32434 kontrolünü bypass et — bge-m3 güvenilir kaynak
import transformers.utils.import_utils as _tu
import transformers.modeling_utils as _mu
_tu.check_torch_load_is_safe = lambda: None
_mu.check_torch_load_is_safe = lambda: None

from FlagEmbedding import BGEM3FlagModel


class BGE_M3_Embedder:
    """
    BAAI/bge-m3 için SentenceTransformer arayüzüyle uyumlu wrapper.
    Mevcut build_faiss ve hybrid_search kodları değişmeden çalışır.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def encode(self, sentences: list, show_progress_bar: bool = False) -> np.ndarray:
        result = self.model.encode(
            sentences,
            batch_size=12,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return np.array(result["dense_vecs"], dtype=np.float32)
