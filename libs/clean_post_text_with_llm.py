import warnings
from os import path
from typing import Optional, Tuple
from pydantic import BaseModel, ConfigDict
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerationMixin

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from constants import GENERATOR_LLM_MODEL, GENERATOR_PROMPT_CONFIG_PATH


class _Cache(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Optional[PreTrainedTokenizerBase]
    model: Optional[PreTrainedModel | GenerationMixin]


class _GeneratorPromptConfig(BaseModel):
    role: str
    task: str
    rules: list[str]
    examples: list[Tuple[str, str]]


_CACHE: _Cache = _Cache(tokenizer=None, model=None)


def _initialize_llm() -> tuple[PreTrainedTokenizerBase, PreTrainedModel | GenerationMixin]:
    if _CACHE.tokenizer and _CACHE.model:
        return _CACHE.tokenizer, _CACHE.model

    # Filter out specific torch warnings that can be safely ignored.
    warnings.filterwarnings(
        "ignore",
        message=".*`generation_config` default values have been modified to match model-specific defaults.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Not enough SMs to use max_autotune_gemm mode.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*does not support bfloat16 compilation natively.*",
    )

    print("Info: Initializing LLM...")
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_LLM_MODEL)
    assert isinstance(tokenizer, PreTrainedTokenizerBase), "`tokenizer` should be of type `PreTrainedTokenizerBase`."
    # The `device_map="auto"` will intelligently use the GPU if available.
    # Using bfloat16 for a smaller memory footprint.
    model = AutoModelForCausalLM.from_pretrained(GENERATOR_LLM_MODEL, device_map="auto", torch_dtype=torch.bfloat16)
    assert isinstance(model, GenerationMixin), "`model` should be of type `GenerationMixin`."
    assert isinstance(model, PreTrainedModel), "`model` should be of type `PreTrainedModel`."

    # Llama 3 Chat Template
    # tokenizer.chat_template = (
    #     "<|begin_of_text|>{% for message in messages %}"
    #     "{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}"
    #     "{% endfor %}"
    #     # This final part tells the model it's now its turn to generate a response.
    #     "<|start_header_id|>assistant<|end_header_id|>\n\n"
    # )

    _config = AutoConfig.from_pretrained(GENERATOR_LLM_MODEL)
    # print("=" * 120)
    # print("CONFIGURATION:")
    # print("-" * 120)
    # print(repr(config))
    # print("-" * 120)

    _CACHE.tokenizer = tokenizer
    _CACHE.model = model

    return tokenizer, model


def _get_cleaning_prompt(raw_text: str) -> str:
    prompt_config_path = path.join(path.dirname(__file__), "..", GENERATOR_PROMPT_CONFIG_PATH)
    with open(prompt_config_path, "r", encoding="utf-8") as prompt_data_file:
        prompt_config = _GeneratorPromptConfig.model_validate_json(prompt_data_file.read())

    prompt_lines = [
        prompt_config.role,
        "",
        f"{prompt_config.task} You MUST follow these rules:",
        *[f"{i + 1}. {rule}" for i, rule in enumerate(prompt_config.rules)],
        "",
        "Here are some examples:",
    ]

    for example_input, example_output in prompt_config.examples:
        prompt_lines.extend(["", f"RAW TEXT:\n`{example_input}`", f"NORMALIZED TEXT:\n`{example_output}`\n"])

    prompt_lines.extend(
        [
            "---",
            "Now, clean the following text according to these rules:",
            f"RAW TEXT:\n`{raw_text}`",
            "NORMALIZED TEXT:\n`",
        ]
    )

    prompt = "\n".join(prompt_lines)

    # print("=" * 120)
    # print("PROMPT:")
    # print("-" * 120)
    # print(prompt)
    # print("-" * 120)

    return prompt


def clean_post_text_with_llm(text: str, attempt: int = 0) -> str:
    if not text or not text.strip():
        return ""

    tokenizer, model = _initialize_llm()
    prompt = _get_cleaning_prompt(text)

    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    assert isinstance(formatted_prompt, str), "`formatted_prompt` should be of type `str`."

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)  # type: ignore[union-attr]

    # generation_config = GenerationConfig(
    #     do_sample=False,  # Disable sampling for determinism
    #     num_beams=1,  # Greedy search
    #     renormalize_logits=False,  # Unused when sampling is disabled
    #     temperature=None,  # Unused when sampling is disabled
    #     top_k=None,  # Unused when sampling is disabled
    #     top_p=None,  # Unused when sampling is disabled
    # )
    generation_config = GenerationConfig(
        do_sample=True,
        num_beams=1,  # Greedy search
        renormalize_logits=False,
        temperature=min(0.7 + (attempt * 0.3), 1.5),
        top_k=50,
        top_p=0.9,
    )
    output_tokens = model.generate(**inputs, max_new_tokens=512, generation_config=generation_config)  # type: ignore[operator]

    new_output_tokens = output_tokens[0, inputs.input_ids.shape[1] :]
    output = tokenizer.decode(new_output_tokens, skip_special_tokens=True)
    if output.startswith("`"):
        output = output[1:]
    if output.endswith("`"):
        output = output[:-1]

    # print("=" * 120)
    # print("OUTPUT (BEFORE):")
    # print("-" * 120)
    # print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))
    # print("=" * 120)
    # print("OUTPUT (AFTER):")
    # print("-" * 120)
    # print(output)
    # print("-" * 120)

    return output
