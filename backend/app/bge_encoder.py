"""BGE-M3 encoder adapter.

FlagEmbedding is optional for the local keyword/SQLite development path. Install
it in the production retrieval image to generate all BGE-M3 representations.
"""

from typing import Iterable, List

from .hybrid import HybridEmbedding


class BGEM3Encoder:
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as error:
            raise RuntimeError("Install FlagEmbedding to enable BGE-M3 hybrid encoding.") from error
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def encode(self, texts: Iterable[str]) -> List[HybridEmbedding]:
        output = self.model.encode(
            list(texts),
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        lexical_weights = output["lexical_weights"]
        return [
            HybridEmbedding(
                dense=list(dense),
                sparse_indices=[int(index) for index in weights.keys()],
                sparse_values=[float(value) for value in weights.values()],
                colbert=[list(token_vector) for token_vector in colbert],
            )
            for dense, weights, colbert in zip(output["dense_vecs"], lexical_weights, output["colbert_vecs"])
        ]
