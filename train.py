import logging
import os
import json
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset
from bert_score import score

import transformers
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    GenerationConfig,
    set_seed,
    EvalPrediction,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

meteor = evaluate.load('meteor')

@dataclass
class ModelArguments:
    """
    Arguments related to which model/config/tokenizer we are going to fine-tune or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Leave unset if training a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, specify a model type. Choices: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some default config settings when training from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path, if different from model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path, if different from model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to store the downloaded pretrained models from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use a fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Specific model version to use (can be a branch name, tag, or commit ID)."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, "
                "the dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "Reduce CPU memory usage when loading models by creating an empty model shell and only "
                "materializing its parameters when pretrained weights are loaded."
            )
        },
    )
    alpha: Optional[float] = field(
        default=0.2, metadata={"help": "Alpha value for step-by-step distillation."}
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides cannot be used with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments related to the data we are going to use for training and evaluation.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text, csv, or json file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate perplexity (a text, csv, or json file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging or faster training, truncate the number of training examples to this value, if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging or faster evaluation, truncate the number of evaluation examples to this value, if set."
            )
        },
    )
    streaming: bool = field(
        default=False, metadata={"help": "Enable streaming mode for large datasets."}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. The training dataset will be truncated into "
                "blocks of this size. Defaults to the model max input length."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the training set to use as a validation set if no validation split is provided."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of workers to use for data preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to retain line breaks when using TXT files."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`.")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Either a dataset name or training/validation file must be provided.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt", "jsonl"], "`train_file` should be a csv, json, or txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt", "jsonl"], "`validation_file` should be a csv, json, or txt file."


def preprocess_function(examples, tokenizer):
    """_summary_

    Args:
        examples (_type_): _description_
        tokenizer (_type_): _description_

    Returns:
        _type_: _description_
    """
    system_messgage = "You are a medical assistant specializing in providing expert consultations for medical inquiries. Your role is to assist users by offering reliable medical information, clarifying symptoms, explaining possible medical conditions, and recommending appropriate next steps. You are empathetic, professional, and always focused on delivering accurate and helpful responses tailored to the user's concerns."

    inputs = examples['input']
    outputs = examples['output']

    qa_texts = [f'{system_messgage} \nQuestion: {inp} \nAnswer: {out}' for inp, out in zip(inputs, outputs)]

    tokenized_examples = tokenizer(qa_texts, truncation=True, padding='max_length', max_length=4096)

    return tokenized_examples


def compute_metrics(p: EvalPrediction, tokenizer):
    predictions = p.predictions[0]
    labels = p.label_ids

    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate METEOR
    meteor_score = meteor.compute(predictions=pred_str, references=label_str)
    meteor_score = round(meteor_score['meteor'], 3)

    # Calculate BERTScore
    _, _, f1 = score(pred_str, label_str, lang='en', verbose=False, device='cpu')
    f1 = round(f1.mean().item(), 3)

    return {
        'meteor': meteor_score,
        'f1': f1
    }


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = logits.argmax(-1)

    return pred_ids, labels


def main():
    set_seed(42)

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        with open(os.path.abspath(sys.argv[1]), "r") as f:
            config_data = json.load(f)

        generation_config_dict = config_data.pop("generation_config", None)
        model_args, data_args, training_args = parser.parse_dict(config_data)

        if generation_config_dict:
            gen_config = GenerationConfig(**generation_config_dict)
        else:
            gen_config = None
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        gen_config = None

    send_example_telemetry('run_clm', model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.setLevel(logging.INFO)
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        
        raw_datasets = datasets.load_dataset(extension, data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        torch_dtype=getattr(torch, model_args.torch_dtype) if model_args.torch_dtype is not None else None,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        device_map='auto'
    )

    if gen_config is not None:
        model.genetation_config = gen_config

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    logger.info("Tokenizing dataset...")
    if "input" in raw_datasets["train"].column_names and "output" in raw_datasets["train"].column_names:
        logger.info("Dataset has the correct columns.")
        tokenized_datasets = raw_datasets.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache
        )
    else:
        raise ValueError("Dataset does not have the correct columns.")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'] if training_args.do_train else None,
        eval_dataset=tokenized_datasets['validation'] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    if training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        logger.info(metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
