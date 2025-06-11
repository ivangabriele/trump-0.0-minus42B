from typing import List

from pydantic import BaseModel


class FeedbackDataStfPair(BaseModel):
    id: str
    "Unique post hash."
    input: str
    output: str


class FeedbackDataComparisonPair(BaseModel):
    id: str
    "Unique post hash."
    input: str
    accepted: str
    rejected: List[str]


class FeedbackData(BaseModel):
    sft_pairs: List[FeedbackDataStfPair]
    "For supervised fine-tuning."
    comparison_pairs: List[FeedbackDataComparisonPair]
    "For reward modeling."
