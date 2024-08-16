# -*- coding: utf-8 -*-
"""UrduCorpus-TransformerFineTuning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VgscuwtmZSzNh_YgTBBYYHNtCtA4JmCP
"""

print("hello")

!pip install transformers torch datasets

import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import AutoTokenizer
# Load the tokenizer
# tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
# Load the unlabelled dataset
dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='/kaggle/input/urdudatabulk/urduDataBulk.txt', block_size=128)

# Create the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Load the pre-trained XLM-RoBERTa model and modify the top layer for MLM
model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base')

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_total_limit=2,
)

# Set up the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# Train the model
trainer.train()

# Save the pre-trained model
trainer.save_model('./Adeel-xlmRoberta-Pretrained-BULK')

print("congrats")

