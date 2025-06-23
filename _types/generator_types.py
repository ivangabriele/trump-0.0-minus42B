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


class PpoDatasetPick(BaseModel):
    content: str
    role: str


class PpoDatasetPair(BaseModel):
    """
    Dataset format for Reward Model training.
    """

    prompt: str
    "The original prompt (input) for context."
    chosen: List[PpoDatasetPick]
    "The human-approved output."
    rejected: List[PpoDatasetPick]
    "The first rejected output for comparison."
    # Note: 'input' (original prompt) is not included, as RewardTrainer expects only chosen/rejected columns.


type PpoDataset = List[PpoDatasetPair]
