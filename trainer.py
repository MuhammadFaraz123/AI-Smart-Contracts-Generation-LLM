from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
from datetime import datetime
import torch
import pandas as pd


device = ("cuda" if torch.cuda.is_available() else "cpu" )

# Load model
tokenizer = AutoTokenizer.from_pretrained("")

tokenizer.add_eos_token = True
tokenizer.pad_token_id = 0

tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

device = "cuda" # the device to load the model onto

# Load Tokenizer
model = AutoModelForCausalLM.from_pretrained("")

# Load data
df = pd.read_csv("")
df = df.sample(n=100)


data = [(i, inst, expl, resp) for i, (inst, expl, resp) in enumerate(zip(df['prompt'], df['code']))]

train_df= data[ :int(len(df)*0.90)]
test_df= data[int(len(df)*0.90):]

max_length = 2048
def tokenizerr(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding='max_length',
       
    )

    result["labels"] = result["input_ids"].copy()

    return result

tokenized_Train = [tokenizerr(data) for data in train_df]
tokenized_Test = [tokenizerr(data) for data in test_df]

project = "base"
base_model_name = "model-name"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_Train,
    eval_dataset=tokenized_Test,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        # gradient_checkpointing=True,
        max_steps=50000,
        learning_rate=2.5e-5, 
        bf16=True,
        optim="adamw_hf",
        logging_steps=250,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=25000,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=250,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"         
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False


if __name__ == "__main__":
    trainer.train()
