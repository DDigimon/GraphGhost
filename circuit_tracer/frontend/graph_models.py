import math
from typing import List

from pydantic import BaseModel


class Metadata(BaseModel):
    slug: str
    scan: str
    transcoder_list: List[str]
    prompt_tokens: List[str]
    prompt: str
    node_threshold: float | None = None


class QParams(BaseModel):
    pinnedIds: List[str]
    supernodes: List[List[str]]
    linkType: str
    clickedId: str
    sg_pos: str


class Node(BaseModel):
    node_id: str
    feature: int
    layer: str
    ctx_idx: int
    feature_type: str
    token_prob: float = 0.0
    is_target_logit: bool = False
    run_idx: int = 0
    reverse_ctx_idx: int = 0
    jsNodeId: str
    clerp: str = ""
    influence: float | None = None
    activation: float | None = None

    def __init__(self, **data):
        if "node_id" in data and "jsNodeId" not in data:
            data["jsNodeId"] = data["node_id"]
        super().__init__(**data)

    @classmethod
    def feature_node(cls, layer, pos, feat_idx, num_features=None, influence=None, activation=None):
        """Create a feature node."""
        offset = 10 ** math.ceil(math.log10(num_features)) if num_features is not None else 0
        reverse_ctx_idx = 0
        return cls(
            node_id=f"{layer}_{feat_idx}_{pos}",
            feature=feat_idx + offset * int(layer),
            layer=str(layer),
            ctx_idx=pos,
            feature_type="cross layer transcoder",
            jsNodeId=f"{layer}_{feat_idx}-{reverse_ctx_idx}",
            influence=influence,
            activation=activation,
        )

    @classmethod
    def error_node(cls, layer, pos, influence=None):
        """Create an error node."""
        reverse_ctx_idx = 0
        return cls(
            node_id=f"0_{layer}_{pos}",
            feature=-1,
            layer=str(layer),
            ctx_idx=pos,
            feature_type="mlp reconstruction error",
            jsNodeId=f"{layer}_{pos}-{reverse_ctx_idx}",
            influence=influence,
        )

    @classmethod
    def token_node(cls, pos, vocab_idx, influence=None):
        """Create a token node."""
        return cls(
            node_id=f"E_{vocab_idx}_{pos}",
            feature=pos,
            layer="E",
            ctx_idx=pos,
            feature_type="embedding",
            jsNodeId=f"E_{vocab_idx}-{pos}",
            influence=influence,
        )

    @classmethod
    def logit_node(
        cls,
        pos,
        vocab_idx,
        token,
        num_layers,
        target_logit=False,
        token_prob=0.0,
    ):
        """Create a logit node."""
        layer = str(num_layers + 1)
        return cls(
            node_id=f"{layer}_{vocab_idx}_{pos}",
            feature=vocab_idx,
            layer=layer,
            ctx_idx=pos,
            feature_type="logit",
            token_prob=token_prob,
            is_target_logit=target_logit,
            jsNodeId=f"L_{vocab_idx}-{pos}",
            clerp=f'Output "{token}" (p={token_prob:.3f})',
        )


class Link(BaseModel):
    source: str
    target: str
    weight: float


class Model(BaseModel):
    metadata: Metadata
    qParams: QParams
    nodes: List[Node]
    links: List[dict]
