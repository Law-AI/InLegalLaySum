#!/usr/bin/env python3

import os
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
    EarlyStoppingCallback,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

from trl import (
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
)


def model_family(model_id):
    m = model_id.lower()

    if "llama" in m:
        return "llama"

    if "mistral" in m:
        return "mistral"

    if "gemma" in m:
        return "gemma"
    
    return "other"


def apply_chat_template_llama(
    tokenizer,
    system_prompt,
    user_content,
    assistant_target,
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
        {
            "role": "assistant",
            "content": assistant_target,
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
    assistant_target,
):
    merged_user = system_prompt + "\n\n" + user_content

    messages = [
        {
            "role": "user",
            "content": merged_user,
        },
        {
            "role": "assistant",
            "content": assistant_target,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )


def apply_chat_template_gemma(
    tokenizer,
    system_prompt,
    user_content,
    assistant_target,
):
    merged_user = system_prompt + "\n\n" + user_content

    messages = [
        {
            "role": "user",
            "content": merged_user,
        },
        {
            "role": "model",
            "content": assistant_target,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )


def get_target_modules_for_lora(model_short_name: str):

    name = model_short_name.lower()

    if any(x in name for x in ["llama", "mistral", "gemma"]):
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    return ["q_proj", "v_proj"]


def load_tokenizer(model_name: str, hf_token: str):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=hf_token,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_token = tokenizer.eos_token

    return tokenizer, eos_token


def load_bnb_config():

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def load_lora_config(
    model_short_name: str,
    r: int,
    alpha: int,
    dropout: float,
):

    lora_target_modules = get_target_modules_for_lora(
        model_short_name
    )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    return lora_config


def load_model(
    model_name: str,
    hf_token: str,
    bnb_config: BitsAndBytesConfig,
    lora_config: LoraConfig,
):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        lora_config,
    )

    return model


def get_response_template(family):

    if family == "llama":
        return "<|start_header_id|>assistant<|end_header_id|>"

    elif family == "mistral":
        return "[/INST]"

    elif family == "gemma":
        return "<start_of_turn>model"
    
    else:
        raise ValueError("Unsupported model family")


def load_training_arguments(args):

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        gradient_accumulation_steps=args.gradient_accumulation_steps,

        gradient_checkpointing=args.gradient_checkpointing,

        optim="paged_adamw_32bit",

        bf16=args.bf16,

        learning_rate=args.learning_rate,

        warmup_ratio=args.warmup_ratio,

        weight_decay=args.weight_decay,

        lr_scheduler_type="cosine",

        num_train_epochs=args.num_epochs,

        evaluation_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,

        logging_strategy="steps",
        logging_steps=args.logging_steps,

        save_total_limit=args.save_total_checkpoints,

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",
        greater_is_better=False,

        save_safetensors=True,

        report_to="none",
    )

    return training_args


def main(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    set_seed(args.seed)

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    tokenizer, eos_token = load_tokenizer(
        args.model_name,
        args.hf_token,
    )

    family = model_family(args.model_name)

    SYSTEM_PROMPT = """You are an expert in Indian law and simplifying legal content. Your task is to summarize complex Indian legal documents so that a reader with up to a high school education can easily understand the summary. Avoid domain-specific complex legal terms and when necessary, provide simple explanations of legal terms in parentheses next to the specific term. Keep the response sentences short and simple. Respond only in English.""".strip()

    USER_PROMPT_TEMPLATE = """Please write a simplified summary for the following Indian legal judgment so that the summary is suitable for readers up to high school(11th - 12th grade level) education. Please ensure the summary retains the key facts, core legal issues, arguments and the final decision of the court, without adding any new information or altering the meaning. Do not include any closing remarks or disclaimers. Avoid providing offers for further assistance. Neither add any section headers nor give any personal opinion. Do not include any closing remark. Respond only in English. 

    ### Judgment:
    {judgment}

    ### Simplified Summary:

    """.strip()

    texts = []

    root = Path(args.dataset_path)

    for case_dir in sorted(root.iterdir()):

        if not case_dir.is_dir():
            continue

        j_path = case_dir / "Judgement.txt"
        s_path = case_dir / "Simplified_Summary.txt"

        if not j_path.exists() or not s_path.exists():
            continue

        judgement = j_path.read_text(
            encoding="utf-8"
        ).strip()

        simplified = s_path.read_text(
            encoding="utf-8"
        ).strip()

        # IMPORTANT FIX
        simplified = simplified + eos_token

        user_prompt = USER_PROMPT_TEMPLATE.format(
            judgment=judgement
        )

        if family == "llama":

            text = apply_chat_template_llama(
                tokenizer,
                SYSTEM_PROMPT,
                user_prompt,
                simplified,
            )

        elif family == "mistral":

            text = apply_chat_template_mistral(
                tokenizer,
                SYSTEM_PROMPT,
                user_prompt,
                simplified,
            )

        elif family == "gemma":

            text = apply_chat_template_gemma(
                tokenizer,
                SYSTEM_PROMPT,
                user_prompt,
                simplified,
            )

        else:
            raise ValueError("Unsupported model family")

        # Optional debugging
        tokenized = tokenizer(text)["input_ids"]

        if len(tokenized) > args.max_seq_length:
            print(
                f"WARNING: Sample exceeds max_seq_length "
                f"({len(tokenized)} > {args.max_seq_length})"
            )

        texts.append(
            {
                "text": text
            }
        )

    print(f"\nTotal samples: {len(texts)}")

    dataset = Dataset.from_list(texts)

    dataset = dataset.train_test_split(
        test_size=0.1,
        seed=args.seed,
    )

    train_dataset = dataset["train"]

    eval_dataset = dataset["test"]

    model = load_model(
        args.model_name,
        args.hf_token,
        load_bnb_config(),
        load_lora_config(
            args.model_short_name,
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
        ),
    )

    model.print_trainable_parameters()

    response_template = get_response_template(
        family
    )

    # IMPORTANT FIX
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    training_args = load_training_arguments(args)

    trainer = SFTTrainer(
        model=model,

        tokenizer=tokenizer,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        args=training_args,

        data_collator=data_collator,

        max_seq_length=args.max_seq_length,

        dataset_text_field="text",

        packing=False,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3
            )
        ],
    )

    trainer.train()

    peft_id = "Q_LoRA_Model"

    output_model_id = args.output_dir

    peft_model_id = os.path.join(
        output_model_id,
        peft_id,
    )

    os.makedirs(
        peft_model_id,
        exist_ok=True,
    )

    trainer.model.save_pretrained(
        peft_model_id
    )

    tokenizer.save_pretrained(
        peft_model_id
    )

    print(
        f"\nQLoRA fine-tuned model saved to "
        f"{peft_model_id}"
    )

    output_dir_new = os.path.join(
        args.output_dir,
        "final_model",
    )

    os.makedirs(
        output_dir_new,
        exist_ok=True,
    )

    trainer.save_model(
        output_dir_new
    )

    tokenizer.save_pretrained(
        output_dir_new
    )

    print(
        f"\nFinal QLoRA fine-tuned model saved to "
        f"{output_dir_new}"
    )

    train_losses = []
    eval_losses = []

    for log in trainer.state.log_history:

        if "loss" in log and "eval_loss" not in log:
            train_losses.append(log["loss"])

        if "eval_loss" in log:
            eval_losses.append(log["eval_loss"])

    plt.figure()

    if train_losses:
        plt.plot(
            train_losses,
            label="Training Loss",
        )

    if eval_losses:
        plt.plot(
            np.linspace(
                0,
                len(train_losses),
                len(eval_losses),
            ),
            eval_losses,
            label="Validation Loss",
        )

    plt.xlabel("Training Steps")

    plt.ylabel("Loss")

    plt.title(
        "Training vs Validation Loss"
    )

    plt.legend()

    plt.savefig(
        os.path.join(
            args.output_dir,
            "loss_curve.png",
        )
    )

    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fine-tune a language model using QLoRA"
    )

    parser.add_argument(
        "--model_name",
        required=True,
    )

    parser.add_argument(
        "--model_short_name",
        required=True,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--save_total_checkpoints",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--dataset_path",
        required=True,
    )

    parser.add_argument(
        "--bf16",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--hf_token",
        required=True,
    )

    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    main(args)