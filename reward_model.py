import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import DataLoader
from preference_dataset import PreferenceDataset
from preference_dataloader import PreferenceDataloader
import reward_metrics as metrics
from datasets import load_dataset
import random

class RewardModel(nn.Module):
    def __init__(self, encoder_name="distilbert-base-uncased", hidden_dim=256, device="cpu"):
        super().__init__()
        self.device = device
        # pretrained model, returns last hidden state
        self.encoder = AutoModel.from_pretrained(encoder_name)
        enc_out_dim = self.encoder.config.hidden_size
        # reward modeling head
        self.head = nn.Sequential(
            nn.Linear(enc_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward_text_pair(self, input_ids, attention_mask):
        reward = torch.tensor([0])
        # expects input_ids of shape (batch size, sequence length)
        # get [CLS] / first token embedding as pooled representation
        # https://huggingface.co/transformers/v3.0.2/model_doc/distilbert.html#distilbertmodel
        # feed the pooled representation through the reward modeling head to ge the actual reward
        # BEGIN STUDENT CODE (~3 lines)
        # END STUDENT CODE
        return reward

    def forward_pair_batch(self, batch):
        # batch contains prompt+resp A and B tokenized
        rA = self.forward_text_pair(batch['input_ids_A'].to(self.device), batch['attn_A'].to(self.device))
        rB = self.forward_text_pair(batch['input_ids_B'].to(self.device), batch['attn_B'].to(self.device))
        return rA, rB
        
    def pairwise_loss(self, rA, rB, label):
        # label: 1 if A preferred, 0 if B preferred
        # compute logits (batchwise reward difference)
        logits = rA - rB
        return nn.BCEWithLogitsLoss()(logits, label.float())
        
    def evaluate(self, data_loader):
        preds = []
        labels = []
        
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                rA, rB = self.forward_pair_batch(batch)
                preds.append((rA > rB).long().to(self.device))
                labels.append(batch['label'].view(-1).long().to(self.device))
                
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
            
        return preds, labels

def main():
    torch.manual_seed(42)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("CUDA and MPS not available, using CPU")

    model = RewardModel(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    TRAIN_SIZE = 720
    VAL_SIZE = 180
    TEST_SIZE = 360

    # Load subset of CNN/Daily Mail summarization dataset
    train_data = load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True)
    train_data = [x for _, x in zip(range(TRAIN_SIZE), train_data)]

    val_data = load_dataset("cnn_dailymail", "3.0.0", split="validation", streaming=True)
    val_data = [x for _, x in zip(range(VAL_SIZE), val_data)]

    test_data = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
    test_data = [x for _, x in zip(range(TEST_SIZE), test_data)]

    print("Loading train")
    train_loader = PreferenceDataloader(train_data)
    train_loader.loader = train_loader.get_dataloaders(train_data, device, batch_size=8, num_examples=TRAIN_SIZE)
    print()

    print("Loading validation")
    val_loader = PreferenceDataloader(val_data)
    val_loader.loader = val_loader.get_dataloaders(val_data, device, batch_size=8, num_examples=VAL_SIZE)
    print()

    print("Loading test")
    test_loader = PreferenceDataloader(test_data)
    test_loader.loader = test_loader.get_dataloaders(test_data, device, batch_size=8, num_examples=TEST_SIZE, shuffle=False)
    print()

    train_rewards = []
    
    # pick a random validation example to watch through training
    example = random.choice(val_loader.loader.dataset)
         
    NUM_EPOCHS = 10
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for batch in train_loader.loader:
            # move to device
            for k in batch:
                batch[k] = batch[k].to(device)

            rA, rB = model.forward_pair_batch(batch)
            train_rewards.append(zip(rA.tolist(), rB.tolist(), batch['label'].tolist()))
            loss = model.pairwise_loss(rA, rB, batch['label'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # validate, compute pairwise accuracy
        avg_loss = running_loss / len(train_loader.loader)
        preds, labels = model.evaluate(val_loader.loader)
        val_acc = metrics.compute_pairwise_accuracy(preds, labels)

        print(f"Epoch {epoch+1} | Train loss: {avg_loss:.4f} | Val pairwise acc: {val_acc:.3f}")

        # compute mean reward of train samples
        win_mean, lose_mean = metrics.compute_mean_reward(train_rewards)
        print(f"Mean train winning response reward: {win_mean:.3f} | Mean train losing response reward: {lose_mean:.3f}")
        
        # display example
        metrics.display_example(model, example)
        
    preds, labels = model.evaluate(test_loader.loader)
    test_acc = metrics.compute_pairwise_accuracy(preds, labels)
    print(f"Final test pairwise acc: {test_acc:.3f}")
        
    correct_mask = (preds == labels).cpu()
    incorrect_mask = (preds != labels).cpu()

    # Get indices
    correct_indices = correct_mask.nonzero(as_tuple=True)[0].tolist()
    incorrect_indices = incorrect_mask.nonzero(as_tuple=True)[0].tolist()
    
    # Sample 5 correct and 5 incorrect test cases (handle case where fewer than 5 exist)
    sampled_correct = random.sample(correct_indices, min(5, len(correct_indices)))
    sampled_incorrect = random.sample(incorrect_indices, min(5, len(incorrect_indices)))
    
    print("\n***********************************")
    print("* CORRECTLY CLASSIFIED TEST CASES *")
    print("***********************************")
    for idx in sampled_correct:
        test_ex = test_loader.loader.dataset[idx]
        metrics.display_example(model, test_ex)

    print("\n*************************************")
    print("* INCORRECTLY CLASSIFIED TEST CASES *")
    print("*************************************")
    for idx in sampled_incorrect:
        test_ex = test_loader.loader.dataset[idx]
        metrics.display_example(model, test_ex)
        
if __name__ == "__main__":
    main()



