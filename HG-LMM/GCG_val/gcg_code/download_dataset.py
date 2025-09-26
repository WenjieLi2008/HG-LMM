from datasets import load_dataset

val_dataset = load_dataset("MBZUAI/GranD", split="validation")

print(val_dataset[0])
print(f"valiadation size:{len(val_dataset)}")
