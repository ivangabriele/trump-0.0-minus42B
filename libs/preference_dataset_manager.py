import json
from os import path

from _types.generator_types import PreferenceDataset
from constants import PREFERENCE_DATASET_PATH


class _PreferenceDatasetManager:
    """
    A class to manage the Preference Dataset.
    """

    _dataset_path: str

    def __init__(self):
        self._dataset_path = path.join(path.dirname(__file__), "..", PREFERENCE_DATASET_PATH)

    def read(self) -> PreferenceDataset:
        if not path.exists(self._dataset_path):
            print(f"Warning: Preference Dataset file `{self._dataset_path}` does not exist.")

            return PreferenceDataset(comparison_pairs=[])

        with open(self._dataset_path, "r") as file:
            data = json.load(file)

        return PreferenceDataset.model_validate(data)

    def write(self, new_preference_dataset: PreferenceDataset) -> None:
        with open(self._dataset_path, "w") as file:
            file.write(new_preference_dataset.model_dump_json(indent=2, by_alias=True))


preference_dataset_manager = _PreferenceDatasetManager()
