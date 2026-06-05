from llm4rec.llm.llama_backbone import load_llama, freeze_model, print_trainable_params
from llm4rec.llm.tokenizer_utils import (
    setup_tokenizer,
    get_answer_token_ids,
    tokenize_prompt,
    build_training_input,
)
from llm4rec.llm.frozen_projector import FrozenSoftPromptProjector
from llm4rec.llm.injector import SoftPromptInjector, prepend_soft_prompt
from llm4rec.llm.lora import build_lora_model, save_lora, load_lora
from llm4rec.llm.scoring import score_from_logits, score_batch, compute_sequence_logprob
from llm4rec.llm.collate import LLMCollator
