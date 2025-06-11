from typing import List

from pydantic import BaseModel


class PreferenceDatasetStfPair(BaseModel):
    """
    A single pair of input and output for SFT (Supervised Fine-Tuning).
    """

    id: str
    "Unique post hash."
    input: str
    output: str


class PreferenceDatasetComparisonPair(BaseModel):
    """
    A single pair of inputs for comparison in RM (Reward Modeling).
    """

    id: str
    "Unique post hash."
    input: str
    accepted: str
    rejected: List[str]


class PreferenceDataset(BaseModel):
    sft_pairs: List[PreferenceDatasetStfPair]
    "For SFT (Supervised Fine-Tuning)."
    comparison_pairs: List[PreferenceDatasetComparisonPair]
    "For RM (Reward Modeling)."
