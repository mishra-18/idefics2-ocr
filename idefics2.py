import torch
from peft import LoraConfig
from transformers import (
    AutoProcessor, 
    BitsAndBytesConfig, 
    AutoModelForVision2Seq,
    TrainingArguments, 
    Trainer 
    )
from datacollator import MyDataCollator
from datasets import load_dataset
import os
from dotenv import load_dotenv, find_dotenv
import wandb

logging = True

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Set up the training parameters.')
    parser.add_argument('--wandb', dest='wandb', type=bool, default=False, help="Log with Wandb")    
    return parser


class Idefics2FT:
    def __init__(self):
        pass
    def _load_model(self, model_id="HuggingFaceM4/idefics2-8b"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,
            low_cpu_mem_usage=True
            )
        
        return model
    
    def _load_dataset(self, dataset_id= "nielsr/docvqa_1200_examples", model_id="HuggingFaceM4/idefics2-8b"):
        train_dataset = load_dataset(dataset_id, split="train")
        eval_dataset = load_dataset(dataset_id, split="test")

        train_dataset = train_dataset.remove_columns(["id", "bounding_boxes", "answer"])
        eval_dataset = eval_dataset.remove_columns(["id", "bounding_boxes", "answer"])

        processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)

        data_collator = MyDataCollator(processor)

        return train_dataset, eval_dataset, data_collator
    
if __name__ == '__main__':

    # Setting wandb for logging
    parser = get_parser()
    opts = parser.parse_args()
    logging=opts.wandb
    if logging:
        # Lak the api keu from .env
        load_dotenv(find_dotenv())
        wandb.login(key=os.environ["WANDB_API"])

    # Initialize the Finetuning class
    id2ft = Idefics2FT()

    print("Loading the model...")
    # Load Model
    model = id2ft._load_model()

    print("Adding model adapters...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*|lm_head$',
        init_lora_weights="gaussian"
    )

    model.add_adapter(lora_config)
    model.enable_adapters()

    print("Finished.")

    # DataLoading
    print('Loading Dataset...')
    train_dataset, _, data_collator = id2ft._load_dataset()

    print("Setting up training arguments...")
    # TrainingArguments
    training_args = TrainingArguments(
    num_train_epochs=3,
    max_steps=178,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    learning_rate = 1e-4,
    weight_decay=0.01,
    logging_steps=25,
    output_dir = "Idefics2-OCR",
    save_strategy = "steps",
    save_steps = 25,
    save_total_limit = 1,
    fp16 = True,
    remove_unused_columns=False,
    report_to= "wandb" if logging == True else "none"
    )

    print("Initializing Trainer.")
    trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = train_dataset
    )
    
    print("Starting training")
    trainer.train()
