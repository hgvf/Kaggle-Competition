import evaluate
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from datasets import Dataset as HFDataset
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DefaultDataCollator
from sklearn.metrics import accuracy_score
from dataloader import CLSDataset

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    result = accuracy.compute(predictions=predictions, references=labels)
   
    return {'eval_accuracy': result['accuracy']}

def submission(pred, filename="submission.csv"):
    out = pred.predictions
    out = np.argmax(out, axis=1)

    with open(filename, "w") as f:
        f.write('Image,Label\n')
        for i in range(946, len(out)+946):
            f.write(f"{i}.jpg,{out[i-946]}\n")
    
if __name__ == "__main__":
    train_set = CLSDataset(
        data_dir="./train/train",
        label_data="./train.csv"
    )

    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size

    train_set, val_set = random_split(train_set, [train_size, val_size])
    test_set = CLSDataset(
        data_dir="./test/test",
        label_data="./test.csv",
        test=True
    )

    # OOM... QQ
    # def train_generator():
    #     for item in train_set:
    #         yield item

    # def test_generator():
    #     for item in test_set:
    #         yield item

    # train_hf_dataset = HFDataset.from_generator(train_generator)
    # test_hf_dataset = HFDataset.from_generator(test_generator)

    checkpoint = "google/vit-base-patch16-224-in21k"
    labels = [0, 1] 
    id2label = {0: "fake", 1: "real"}
    label2id = {"fake": 0, "real": 1}
    # data_collator = DefaultDataCollator(return_tensors="tf")

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    trainer.train()

    predictions = trainer.predict(test_set)
    submission(predictions)