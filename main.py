from src.dataset.reader import TextReader
from src.dataset.tokenizer import Tokenizer
from src.dataset.vocabulary import Vocabulary

from src.model.gpt_model import GPTModel
from src.trainer.trainer import Trainer

from src.generation.generator import Generator

from src.dataset.dataset_loader import DatasetLoader
from src.dataset.dataloader import Dataloader


import torch


# ==============================
# Configuration
# ==============================

TRAIN = True          # True = Train model
                      # False = Only load model

EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_PATH = "checkpoints/mini_gpt.pt"


def main():

    # ==============================
    # Read Dataset
    # ==============================

    # reader = TextReader("data/raw/dictionary.txt")
    # text = reader.read()
    loader = DatasetLoader("tinystories")
    text = loader.load()

    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)

    vocabulary = Vocabulary()
    vocabulary.build(tokens)

    vocabulary.save("data/vocabulary/vocab.json")

    vocabulary.load("data/vocabulary/vocab.json")


    encoded = vocabulary.encode(tokens)
    token_tensor = torch.tensor(encoded, dtype=torch.long)

    #--------------------------------------------
    # Train/ Validation Split
    # here we divide data into train and validation in 80% and 20% percentage 
    # 80% - train and 20 5 for validation 
    # we do this so that model should not only memorize the dataset and answer (overfitting), instead it should learn from it and answer  
    #--------------------------------------------
   
    split_index = int(len(token_tensor) * 0.80)

    train_tokens = token_tensor[:split_index] # 80%

    val_tokens = token_tensor[split_index:] # 20%

    train_loader = Dataloader(train_tokens,context_length=8)

    val_loader = Dataloader(val_tokens,context_length=8)

    print(f"Train Tokens      : {len(train_tokens)}")
    print(f"Validation Tokens : {len(val_tokens)}")

    inputs, targets = train_loader.get_batch(0)

    print("Batch Input Shape :", inputs.shape)
    print("Batch Target Shape:", targets.shape)

    print()

    print("First Sample Input IDs:")
    print(inputs[0].tolist())

    print()

    print("First Sample Target IDs:")
    print(targets[0].tolist())

    print()

    print("First Sample Input Words:")
    print(vocabulary.decode(inputs[0].tolist()))

    print()

    print("First Sample Target Words:")
    print(vocabulary.decode(targets[0].tolist()))




    # ==============================
    # Build Model
    # ==============================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    model = GPTModel(
        embedding_dim=8,
        hidden_dim=32,
        num_heads=2,
        num_layers=4,
        vocab_size=vocabulary.size()
    ).to(device)

    trainer = Trainer(model,device,learning_rate=LEARNING_RATE)

    generator = Generator(model,vocabulary,temperature=0.5,top_k=5,top_p=0.90)

    # ==============================
    # Train or Load
    # ==============================

    if TRAIN:

        logits, losses = trainer.train(
            train_loader,
            val_loader,
            epochs=EPOCHS
        )

        print("LOGIT SHAPE" , logits.shape)

        

        trainer.save_model(MODEL_PATH)

        sample_inputs = inputs[0]

        next_token = generator.generate_next_token(sample_inputs)

        predicted_word = vocabulary.decode([next_token.item()])

        print("=" * 60)
        print("NEXT TOKEN")
        print("=" * 60)
        print(predicted_word)

        

    else:

        trainer.load_model(MODEL_PATH)

        prompt = input("\nAsk here : ")

        prompt_tokens = tokenizer.tokenize(prompt)

        prompt_ids = vocabulary.encode(prompt_tokens)

        prompt_tensor = torch.tensor(prompt_ids,dtype=torch.long)

        generated_text = generator.generate_text(prompt_tensor,max_new_tokens=50)

        next_token = generator.generate_next_token(prompt_tensor)

        predicted_word = vocabulary.decode([next_token.item()])

        print("=" * 60)
        print("NEXT TOKEN")
        print("=" * 60)
        print(predicted_word)

        # ==============================
        # Debug Output
        # ==============================

        print("=" * 60)
        print("GENERATED TEXT")
        print("=" * 60)
        print(generated_text)


if __name__ == "__main__":
    main()