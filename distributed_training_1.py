import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model

from transformers import BitsAndBytesConfig


def init_process(rank, size, backend="gloo"):
    """
    Initialize the distributed environment.
    """
    dist.init_process_group(backend, rank=rank, world_size=size)
    print(f"Process {rank} initialized for distributed training.")

def load_data(tokenizer, batch_size=1):
    """
    Load and tokenize dataset.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize_function(examples):
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(examples["text"], truncation=True, padding="max_length", return_tensors="pt")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: {
        "input_ids": torch.tensor([sample["input_ids"] for sample in x], dtype=torch.long),
        "attention_mask": torch.tensor([sample["attention_mask"] for sample in x], dtype=torch.long),
    }, num_workers=0)

   


    return dataloader

def train(rank, size, tokenizer, model):
    """
    Distributed training function.
    """
    # Set device to CPU (Codespaces do not have GPUs)
    device = torch.device("cpu")

    # Move model to the CPU
    model.to(device)

    # Load dataset and move it to DataLoader
    dataloader = load_data(tokenizer)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Training loop
    model.train()
    for epoch in range(1):  # Replace 1 with your desired number of epochs
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient synchronization across processes
            for param in model.parameters():
                if param.grad is not None:
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

    # Load the tokenizer
    
    #model_name = "microsoft/phi-1_5"
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize model with empty weights
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_name)

    # Since this is a CPU-only environment, shard the model across CPUs (optional)
    device_map = infer_auto_device_map(
        model,
        max_memory={"cpu": "6GB"}  # Adjust based on the memory available in Codespaces
    )

    # Dispatch model to devices using the device map (all on CPU in this case)
    #quant_config = BitsAndBytesConfig(load_in_4bit=True) 

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        offload_folder="/tmp",
        low_cpu_mem_usage=True   # Offload unused layers to disk
    )

    # Start training
    train(rank, size, tokenizer, model)

    # Cleanup the process group
    dist.destroy_process_group()
