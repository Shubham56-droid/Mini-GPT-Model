import torch
import torch.nn as nn 

class Trainer:
    def __init__(self,model,device,learning_rate=0.001,warmup_epochs=10):

        self.model = model
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        # Learning Rate Scheduler

        # StepLR
        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer,
        #     step_size=50, #every 50 eopchs
        #     gamma=0.5 #reduce LR by half
        # )
        self.loss_history = []

    def train_step(self,inputs,targets):

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Clear old gradient
        self.optimizer.zero_grad()

        # forward pass
        logits = self.model(inputs)

        vocab_size = logits.size(-1)

        logits = logits.view(-1,vocab_size)

        targets = targets.view(-1)

        # calculate loss
        loss = self.criterion(logits,targets)

        # compute gradient - how much to blame or loss contibution for each guess
        loss.backward()

        # gradient clipping - outlier removal 
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1.0)

        # Update model weight
        self.optimizer.step()

        return logits,loss
    
    def train(self,loader,epochs=1000):

        # Cosine Annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=epochs,eta_min=1e-6)

        
        for epoch in range(epochs):

            epoch_indices = loader.get_epoch_indices()

            epoch_loss = 0
            num_batches = 0

            for start_index  in epoch_indices:

                if start_index  + loader.batch_size > len(loader):
                    continue

                inputs,targets = loader.get_batch(start_index)

                logits,loss = self.train_step(inputs,targets)

                epoch_loss += loss.item()
                num_batches += 1

            # num_batches = (len(loader) - loader.batch_size + 1) // loader.batch_size

            average_loss = epoch_loss / num_batches

            self.loss_history.append(average_loss)

            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            #print(f"Epoch: {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

            # if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch+1}/{epochs} | "f"Average Loss: {average_loss:.4f} | "f"LR: {current_lr:.6f}")

           
        return logits, self.loss_history
    
    def save_model(self,path):

        torch.save(self.model.state_dict(),path)

        print(f"\nModel saved to {path}")

    def load_model(self,path):

        self.model.load_state_dict(torch.load(path,map_location=self.device))

        self.model.eval()

        print(f"\nModel loaded from {path}")

# Forward
# ↓

# Loss
# ↓

# Backward
# ↓

# Update weights