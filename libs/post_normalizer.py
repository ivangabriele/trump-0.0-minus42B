from copy import copy
from dotenv import load_dotenv
import os
from os import path
from peft import PeftModel
from pydantic import BaseModel
from pydantic_yaml import parse_yaml_raw_as
from transformers.generation.configuration_utils import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import List
import warnings

from constants import NORMALIZER_PROMPT_CONFIG_PATH
from prepare_common import QUANTIZATION_CONFIG
from .database import database


# Filter out specific torch warnings that can be safely ignored.
warnings.filterwarnings("ignore", message=".*default values have been modified to match model-specific defaults.*")
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm mode.*")
warnings.filterwarnings("ignore", message=".*does not support bfloat16 compilation natively.*")
warnings.filterwarnings("ignore", message=".*for open-end generation.*")


load_dotenv()
NORMALIZER_MODEL_BASE = os.getenv("NORMALIZER_MODEL_BASE")
if not NORMALIZER_MODEL_BASE:
    raise ValueError("Missing `NORMALIZER_MODEL_BASE` env var. Please set it in your .env file.")
NORMALIZER_MODEL_PATH = os.getenv("NORMALIZER_MODEL_PATH")
if not NORMALIZER_MODEL_PATH:
    raise ValueError("Missing `NORMALIZER_MODEL_PATH` env var. Please set it in your .env file.")


class _GeneratorPromptConfigExample(BaseModel):
    input: str
    output: str


class _GeneratorPromptConfig(BaseModel):
    role: str
    task: str
    rules: List[str]
    examples: List[_GeneratorPromptConfigExample]


class PostNormalizer:
    _instruction_lines: List[str]
    _model: PreTrainedModel | PeftModel
    _tokenizer: PreTrainedTokenizerBase

    def __init__(self, with_base_model: bool = False) -> None:
        self._init_instruction_lines()
        self._init_model(with_base_model)

    def normalize(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        prompt = self._get_cleaning_prompt(text)

        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        assert isinstance(formatted_prompt, str), "`formatted_prompt` should be of type `str`."

        inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to(self._model.device)  # type: ignore[union-attr]

        generation_config = GenerationConfig(
            do_sample=False,  # Disable sampling for deterministic output
            num_beams=1,  # Greedy search
            renormalize_logits=False,
        )
        output_tokens = self._model.generate(**inputs, max_new_tokens=512, generation_config=generation_config)  # type: ignore[operator]

        new_output_tokens = output_tokens[0, inputs.input_ids.shape[1] :]
        output = self._tokenizer.decode(new_output_tokens, skip_special_tokens=True)
        if output.startswith("`"):
            output = output[1:]
        if output.endswith("`"):
            output = output[:-1]

        return output

    def _init_instruction_lines(self) -> None:
        prompt_config_path = path.join(path.dirname(__file__), "..", NORMALIZER_PROMPT_CONFIG_PATH)
        with open(prompt_config_path, "r", encoding="utf-8") as prompt_config_file:
            prompt_config = parse_yaml_raw_as(_GeneratorPromptConfig, prompt_config_file.read())

        prompt_lines = [
            prompt_config.role,
            "",
            f"{prompt_config.task} You MUST follow these rules:",
            *[f"{i + 1}. {rule}" for i, rule in enumerate(prompt_config.rules)],
            "",
            "Here are some examples:",
        ]

        for example in prompt_config.examples:
            if not database.has_post_with_raw_text(example.input):
                print(f"Warning: Example input `{example.input}` not found in the database. Skipping...")
                continue
            prompt_lines.extend(["", f"RAW TEXT:\n`{example.input}`", f"NORMALIZED TEXT:\n`{example.output}`\n"])

        self._instruction_lines = prompt_lines

    def _init_model(self, with_base_model: bool = False) -> None:
        print("Info: Initializing LLM...")

        if with_base_model:
            model_name_or_path = NORMALIZER_MODEL_BASE
        else:
            model_name_or_path = NORMALIZER_MODEL_PATH

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right", trust_remote_code=False)
        assert isinstance(tokenizer, PreTrainedTokenizerBase), (
            "`tokenizer` should be of type `PreTrainedTokenizerBase`."
        )
        # Add a padding token if not already present (especially for GPT/OPT models)
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation="eager",
            device_map="auto",
            quantization_config=QUANTIZATION_CONFIG,
            trust_remote_code=False,
        ).eval()  # Set the model to evaluation mode

        if with_base_model:
            model = base_model.eval()
            assert isinstance(model, PreTrainedModel), "`model` should be of type `PreTrainedModel`."
        else:
            model = PeftModel.from_pretrained(base_model, NORMALIZER_MODEL_PATH).eval()  # type: ignore[call-arg]

        self._tokenizer = tokenizer
        self._model = model

    def _get_cleaning_prompt(self, raw_text: str) -> str:
        prompt_lines = copy(self._instruction_lines)
        prompt_lines.extend(
            [
                "---",
                "Now, normalize the following text according to these rules without any additional explanations or comments:",
                f"RAW TEXT:\n`{raw_text}`",
                "NORMALIZED TEXT:\n`",
            ]
        )

        prompt = "\n".join(prompt_lines)

        return prompt
