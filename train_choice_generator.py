from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pytorch_lightning as pl
import argparse
from pytorch_lightning import Trainer, seed_everything
from modules import ActionDataModule, Model
import os
import json
from peft import get_peft_model, LoraConfig, TaskType


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    torch.set_float32_matmul_precision("high")

    datamodule = ActionDataModule(args, tokenizer)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    model = Model(model, tokenizer, args.lr, args.seed)

    trainer = Trainer(
        default_root_dir=args.output_dir,
        devices=args.devices,
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        deterministic=True,
        accumulate_grad_batches=args.accumulate_grad_batches,
        accelerator="gpu",
        log_every_n_steps=10,
        enable_checkpointing=False,
    )
    
    trainer.fit(model, datamodule=datamodule)
    if len(args.devices) == 1:
        model.save_pretrained(os.path.join(args.output_dir, args.model.split("/")[-1]))
        print("Done")
    elif torch.distributed.get_rank() in [-1, 0]:
        model.save_pretrained(os.path.join(args.output_dir, args.model.split("/")[-1]))
        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/work/frink/models/llama3-8B-Instruct-HF")
    parser.add_argument("--output_dir", type=str, default="./model/action_model")
    parser.add_argument("--input_dir", type=str, default="./processed_data_v1.2")

    parser.add_argument("--devices", type=list, default=[0])

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)
    
    parser.add_argument("--eval_set", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", type=bool, default=False)
    
    args = parser.parse_args()

    seed_everything(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train(args)
