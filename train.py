import argparse
import logging
import os
import json
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score

from loader import BirdDataset
from wav2vec_classifier import wav2vecClassifier, wav2vecConfig
from logger import setup_logging

def parse_args():
    """ 
    User defined arguments
    """

    parser = argparse.ArgumentParser()

    # Dataset

    # Model
    parser.add_argument("--n_classes", type=int, default=206)
    parser.add_argument("--n_ffn", type=int, default=1024)
    parser.add_argument("--n_model", type=int, default=512)
    parser.add_argument("--n_query", type=int, default=1)
    parser.add_argument("--n_head", type=int, default=8)

    # Dataset
    parser.add_argument("--audio_path", type=str, default="./train_audio")
    parser.add_argument("--csv_path", type=str, default="./train.csv")
    parser.add_argument("--id_mapping_file", type=str, default="./id_mapping.json")
    parser.add_argument("--sample_file", type=str, default="./sample_submission.csv")

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False)
    parser.add_argument("--report_to", type=str, default="tensorboard")

    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_model_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="steps")

    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--log_dir", type=str, default="./logs")
    
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--logging_strategy", type=str, default="steps")

    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=100)
    
    opt = parser.parse_args()

    return opt

def outputForSubmission(predictions, mapping_file, sample_file="sample_submission.csv"):
    """
    Output the predictions for submission
    """

    # Load the mapping file
    with open(mapping_file, "r") as f:
        id_tables = json.load(f)

    # Load the sample submission file
    sample_df = pd.read_csv(sample_file)
    col_order = sample_df.columns.tolist()[1:]
    col_idx = [id_tables[c] for c in col_order]

    with open("submission.csv", "w") as f:
        f.write(f"row_id,{','.join(col_order)}\n")
        
        for idx, pred in enumerate(predictions):
            distribution = pred[col_idx].tolist()
            distribution = [str(d) for d in distribution]
            
            f.write(f"{idx},{','.join(distribution)}\n")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=1)
    
    return {"accuracy": accuracy_score(labels, preds)}

def train(args):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        report_to=[args.report_to],
        save_strategy=args.save_strategy,
        save_steps=args.save_model_steps,
        metric_for_best_model="loss",
        logging_steps=args.logging_steps,
        logging_strategy=args.logging_strategy,
        logging_dir=args.log_dir,
        remove_unused_columns=False,
        fp16=args.fp16,
    )

    config = wav2vecConfig(
        n_classes=args.n_classes, 
        n_ffn=args.n_ffn, 
        n_model=args.n_model, 
        n_query=args.n_query,
    )
    
    model = wav2vecClassifier(config)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    logging.info(f"Total parameters: {total:,}")
    logging.info(f"Trainable parameters: {trainable:,}")

    trainset = BirdDataset(
        audio_path=args.audio_path,
        csv_path=args.csv_path,
        subset="train",
        id_mapping_file=args.id_mapping_file,
        train_ratio=args.train_ratio,
    )
    devset = BirdDataset(
        audio_path=args.audio_path,
        csv_path=args.csv_path,
        subset="valid",
        id_mapping_file=args.id_mapping_file,
        train_ratio=args.train_ratio,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        compute_metrics=compute_metrics
    )

    # logging.info("Starting training...")
    # trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # logging.info("Finish training...")

    logging.info("Starting evaluation...")
    
    pred_output = trainer.predict(devset)
    pred_distribution = F.softmax(torch.tensor(pred_output.predictions), dim=1)
    
    acc = compute_metrics((pred_distribution, pred_output.label_ids))

    logging.info(f"Accuracy: {acc['accuracy']:.4f}")
    logging.info("Finish evaluation...")

    return pred_distribution

def main(args):
    setup_logging(args.log_dir)

    # ----- Tensorboard Setup -----
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # ----- Start training -----
    pred_output = train(args)

    # ----- Submission -----
    outputForSubmission(pred_output, args.id_mapping_file, args.sample_file)

if __name__ == "__main__":
    opt = parse_args()

    main(opt)