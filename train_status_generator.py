from transformers import RobertaForSequenceClassification, AutoTokenizer
import torch
import pytorch_lightning as pl
import argparse
from pytorch_lightning import Trainer, seed_everything
from modules import StatusDataModule, Model
import os
import json


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = RobertaForSequenceClassification.from_pretrained(args.model, num_labels=5, torch_dtype=torch.bfloat16)
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    torch.set_float32_matmul_precision("high")

    datamodule = StatusDataModule(args, tokenizer)
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
        enable_checkpointing=True,
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
    parser.add_argument("--model", type=str, default="FacebookAI/xlm-roberta-large")
    parser.add_argument("--output_dir", type=str, default="./model/status_model")
    parser.add_argument("--input_dir", type=str, default="./processed_data")

    parser.add_argument("--devices", type=list, default=[0])

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", type=bool, default=True)
    
    args = parser.parse_args()

    seed_everything(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train(args)
