from typing import List

from pydantic import BaseModel


class Example(BaseModel):
    tokens_acts_list: List[float]
    train_token_ind: int
    is_repeated_datapoint: bool
    tokens: List[str]


class ExamplesQuantile(BaseModel):
    quantile_name: str
    examples: List[Example]


class Model(BaseModel):
    transcoder_id: str
    index: int
    examples_quantiles: List[ExamplesQuantile]
    top_logits: List[str]
    bottom_logits: List[str]
    act_min: float
    act_max: float
    quantile_values: List[float]
    histogram: List[float]
    activation_frequency: float
