from typing import List

from pydantic import BaseModel


class PreferenceDatasetComparisonPair(BaseModel):
    """
    A single pair of inputs for comparison in RM (Reward Modeling).

    Can also be used for SFT (Supervised Fine-Tuning) with `input` and `accepted`.
    """

    id: str
    "Unique post hash."
    input: str
    accepted: str
    rejected: List[str]


class PreferenceDataset(BaseModel):
    """The RLHF (Reinforcement Learning from Human Feedback) preference dataset."""

    comparison_pairs: List[PreferenceDatasetComparisonPair]
