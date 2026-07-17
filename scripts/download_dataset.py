from datasets import load_dataset
import os 

print("="*60)
print("Downloading tiny stories")
print("="*60)

train_dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:1000]"
)

validation_dataset = load_dataset(
    "roneneldan/TinyStories",
    split="validation"
)

os.makedirs("../data/tinystories", exist_ok=True)

train_file = "../data/tinystories/train.txt"
validation_file = "../data/tinystories/validation.txt"

print("Saving training stories...")

with open(train_file, "w", encoding="utf-8") as f:
    for story in train_dataset:
        f.write(story["text"])
        f.write("\n")


print("Saving validation stories...")

with open(validation_file, "w", encoding="utf-8") as f:
    for story in validation_dataset:
        f.write(story["text"])
        f.write("\n")


print("=" * 60)
print("Dataset Saved Successfully")
print("=" * 60)

print(train_file)
print(validation_file)

print()

print("Frist Story")
print('-'*60)

print(train_dataset[0]["text"])