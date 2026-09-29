#!/usr/bin/env python3

import os
import torch

from dataclasses import dataclass, field

from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)

from peft import (
    prepare_model_for_kbit_training,
    PeftModel,
)

from trl import (
    DPOTrainer,
    DPOConfig,
)

from tqdm import tqdm

import huggingface_hub

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def model_family(model_id):

    m = model_id.lower()

    if "llama" in m:
        return "llama"

    if "mistral" in m:
        return "mistral"

    if "gemma" in m:
        return "gemma"

    if "phi4" in m:
        return "phi4"

    return "other"


def apply_chat_template_llama(
    tokenizer,
    system_prompt,
    user_content,
):

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )


def apply_chat_template_mistral(
    tokenizer,
    system_prompt,
    user_content,
):

    merged_user = (
        system_prompt
        + "\n\n"
        + user_content
    )

    messages = [
        {
            "role": "user",
            "content": merged_user,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )


def apply_chat_template_gemma(
    tokenizer,
    system_prompt,
    user_content,
):

    merged_user = (
        system_prompt
        + "\n\n"
        + user_content
    )

    messages = [
        {
            "role": "user",
            "content": merged_user,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )


def apply_chat_template_phi4(
    tokenizer,
    system_prompt,
    user_content,
):

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )



@dataclass
class ScriptArguments:

    hf_token: str = field(
        metadata={"help": "HF token"}
    )

    base_model_name: str = field(
        metadata={"help": "Base model name"}
    )

    adapter_path: str = field(
        metadata={"help": "Path to SFT LoRA adapter"}
    )

    dataset_path: str = field(
        metadata={"help": "Dataset root"}
    )

    output_dir: str = field(
        default="dpo_output"
    )

    per_device_train_batch_size: int = field(
        default=1
    )

    gradient_accumulation_steps: int = field(
        default=1
    )

    learning_rate: float = field(
        default=2e-6
    )

    num_train_epochs: int = field(
        default=3
    )

    beta: float = field(
        default=0.1
    )

    max_prompt_length: int = field(
        default=2000
    )

    max_target_length: int = field(
        default=512
    )

    trust_remote_code: bool = field(
        default=True
    )


parser = HfArgumentParser(
    ScriptArguments
)

args = parser.parse_args_into_dataclasses()[0]

huggingface_hub.login(
    token=args.hf_token
)


tokenizer = AutoTokenizer.from_pretrained(
    args.base_model_name,
    trust_remote_code=args.trust_remote_code,

    padding_side="right",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

family = model_family(
    args.base_model_name
)


SYSTEM_PROMPT = """
You are an expert in Indian law and simplifying legal content. Your task is to summarize complex Indian legal documents so that a reader with up to a high school education can easily understand the summary. Avoid domain-specific complex legal terms and when necessary, provide simple explanations of legal terms in parentheses next to the specific term. Keep the response sentences short and simple. Respond only in English.
""".strip()

USER_PROMPT_TEMPLATE = """
Please write a simplified summary for the following Indian legal judgment so that the summary is suitable for readers up to high school(11th - 12th grade level) education. Please ensure the summary retains the key facts, core legal issues, arguments and the final decision of the court, without adding any new information or altering the meaning. Do not include any closing remarks or disclaimers. Avoid providing offers for further assistance. Neither add any section headers nor give any personal opinion. Do not include any closing remark. Respond only in English.

### Judgment:
{judgment}

### Simplified Summary:

""".strip()


def load_dataset_custom(
    base_path: str
) -> Dataset:

    prompts = []

    chosen = []

    rejected = []

    for folder in tqdm(
        sorted(os.listdir(base_path)),
        desc="Loading dataset",
    ):

        fpath = os.path.join(
            base_path,
            folder,
        )

        if not os.path.isdir(fpath):
            continue

        try:

            with open(
                os.path.join(
                    fpath,
                    "Judgement.txt",
                ),
                encoding="utf-8",
            ) as f:

                judgement = f.read().strip()

            with open(
                os.path.join(
                    fpath,
                    "Preferred.txt",
                ),
                encoding="utf-8",
            ) as f:

                chosen_summary = (
                    f.read().strip()
                )

            with open(
                os.path.join(
                    fpath,
                    "Rejected.txt",
                ),
                encoding="utf-8",
            ) as f:

                rejected_summary = (
                    f.read().strip()
                )

            if (
                not judgement
                or not chosen_summary
                or not rejected_summary
            ):
                continue

            user_prompt = (
                USER_PROMPT_TEMPLATE.format(
                    judgment=judgement
                )
            )

            if family == "llama":

                prompt = apply_chat_template_llama(
                    tokenizer,
                    SYSTEM_PROMPT,
                    user_prompt,
                )

            elif family == "mistral":

                prompt = apply_chat_template_mistral(
                    tokenizer,
                    SYSTEM_PROMPT,
                    user_prompt,
                )

            elif family == "gemma":

                prompt = apply_chat_template_gemma(
                    tokenizer,
                    SYSTEM_PROMPT,
                    user_prompt,
                )

            elif family == "phi4":

                prompt = apply_chat_template_phi4(
                    tokenizer,
                    SYSTEM_PROMPT,
                    user_prompt,
                )

            else:
                raise ValueError(
                    "Unsupported model family"
                )

            prompts.append(prompt)

            chosen.append(
                chosen_summary
            )

            rejected.append(
                rejected_summary
            )

        except Exception as e:

            print(
                f"Skipping {folder}: {e}"
            )

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected,
        }
    )


dataset = load_dataset_custom(
    args.dataset_path
)

print(
    f"Loaded {len(dataset)} samples"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=(
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    ),
)


base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_name,

    quantization_config=bnb_config,

    device_map="auto",

    trust_remote_code=args.trust_remote_code,
)

base_model.config.use_cache = False

base_model.gradient_checkpointing_enable()

base_model = prepare_model_for_kbit_training(
    base_model
)

model = PeftModel.from_pretrained(
    base_model,

    args.adapter_path,

    is_trainable=True,
)


ref_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_name,

    quantization_config=bnb_config,

    device_map="auto",

    trust_remote_code=args.trust_remote_code,
)

ref_model.eval()

for p in ref_model.parameters():
    p.requires_grad = False


dpo_config = DPOConfig(
    output_dir=args.output_dir,

    beta=args.beta,

    per_device_train_batch_size=(
        args.per_device_train_batch_size
    ),

    gradient_accumulation_steps=(
        args.gradient_accumulation_steps
    ),

    learning_rate=args.learning_rate,

    num_train_epochs=args.num_train_epochs,

    max_prompt_length=args.max_prompt_length,

    max_target_length=args.max_target_length,

    max_length=(
        args.max_prompt_length
        + args.max_target_length
    ),

    logging_steps=10,

    save_strategy="epoch",

    evaluation_strategy="no",

    bf16=torch.cuda.is_bf16_supported(),

    report_to="none",

    remove_unused_columns=False,
)


trainer = DPOTrainer(
    model=model,

    ref_model=ref_model,

    args=dpo_config,

    train_dataset=dataset,

    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model(
    args.output_dir
)

tokenizer.save_pretrained(
    args.output_dir
)

print(
    f"\n DPO-on-SFT training complete — "
    f"saved to {args.output_dir}"
)