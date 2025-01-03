import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW

def init_process(rank, size, backend="gloo"):
    """
    Initialize the distributed environment.
    """
    dist.init_process_group(backend, rank=rank, world_size=size)
    print(f"Process {rank} initialized for distributed training.")

def load_data(tokenizer, batch_size=8):
    """
    Load and tokenize dataset.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: {
        "input_ids": torch.tensor([sample["input_ids"] for sample in x], dtype=torch.long),
        "attention_mask": torch.tensor([sample["attention_mask"] for sample in x], dtype=torch.long),
    })
    return dataloader

def train(rank, size, tokenizer, model):
    """
    Distributed training function.
    """
    # Load dataset and move it to DataLoader
    dataloader = load_data(tokenizer)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Move model to appropriate device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else f"cpu")
    model.to(device)

    # Training loop
    model.train()
    for epoch in range(1):
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient synchronization
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= size  # Average gradients across processes

            optimizer.step()

            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

    print(f"Training completed for process {rank}.")

if __name__ == "__main__":
    # Get rank and size from environment variables set by the launcher
    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Local process rank

    # Initialize the process group
    init_process(rank, size)

    # Load the model and tokenizer
    model_name = "microsoft/phi-1_5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Start training
    train(rank, size, tokenizer, model)

    # Cleanup the process group
    dist.destroy_process_group()
