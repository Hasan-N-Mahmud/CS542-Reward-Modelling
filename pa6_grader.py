# Sample autograder for PA6
# =========================
#
# Place in same directory as reward_model.py files to run
#
import numpy as np
import time
from datetime import datetime
import os
import torch, sys
import torch.nn as nn
import transformers
import random
import reward_metrics as metrics
print(f"[env check] python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}")

# Dummy encoder that mimics DistilBERT output structure
# For forward pass testing
class DummyOutput:
    def __init__(self, hidden_state):
        self.last_hidden_state = hidden_state

class DummyEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        hidden = input_ids.unsqueeze(-1).repeat(1, 1, self.hidden_dim).float()
        return DummyOutput(hidden)
        
# Track sumbission score
score = 0
max_score = 60

# Prompt for manual input
def points(prompt='Points? (0-10): '):
    while "the answer is invalid":
        pts = int(input(prompt).lower().strip())
        if pts >= 0 and pts <= 10:
            return pts

def print_score():
   print(f'Score: {score}/{max_score}\n')

print('1. Importing RewardModel from student solution...\n')

# import from student solution
from reward_model import RewardModel
try:
    rm = RewardModel()
    print("PASSED\n")
    score += 3
except Exception as e:
    print("FAILED\n",e)

print('2. Checking sample_preference_pair implementation...\n')
from preference_dataloader import PreferenceDataloader

dataset = [
        {"article": "A0", "highlights": "S0"},
        {"article": "A1", "highlights": "S1"},
        {"article": "A2", "highlights": "S2"},
        {"article": "A3", "highlights": "S3"},
    ]
    
pdl = PreferenceDataloader(dataset)

emb_summaries = torch.tensor([
        [1.0, 0.0],     # index 0 (the good summary)
        [0.9, 0.0],     # highest cosine similarity (should be chosen)
        [0.2, 0.0],
        [-0.5, 0.0],
    ])
    
example = dataset[0]     # good summary is S0
idx = 0                  # index of example
    
prompt, good_summary, bad_summary = pdl.sample_preference_pair(idx, example, dataset, None, emb_summaries)

if prompt == "A0" and good_summary == "S0" and bad_summary == "S1":
    print("PASSED\n")
    score += 10
else:
    print("FAILED")
    print(f"Correct:\n\tprompt: A0, good_summary: S0, bad_summary: S1\n")
    print(f"Output:\n\tprompt: {prompt}, good_summary: {good_summary}, bad_summary: {bad_summary}\n")
    
print('3. Checking randomize_positions implementation...\n')

good = 1
bad = 0

original_random = random.random
random.random = lambda: 0.1

r1 = np.array(pdl.randomize_positions(good, bad))

random.random = lambda: 0.9
r2 = np.array(pdl.randomize_positions(good, bad))

condition = (
    (r1 != r2).all()
    and (r1[0] == r1[-1])
    and (r2[0] == r2[-1])
)

if condition:
    print("PASSED\n")
    score += 10
else:
    print("FAILED")
    print(f"Correct:\n\tr1 = [1 0 1], r2 = [0 1 0] OR\n\
        r1 = [0 1 0], r2 = [1 0 1]")
    print(f"Output:\n\tr1 = {r1}, r2 = {r2}")

random.random = original_random

print('4. Checking forward_text_pair implementation...\n')

torch.manual_seed(0)

hidden_dim = 4
rm.encoder = DummyEncoder(hidden_dim)
rm.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
with torch.no_grad():
    rm.head[0].weight.fill_(0.25)
    rm.head[2].weight.fill_(0.25)

rA = rm.forward_text_pair(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.tensor([[1, 1, 1]])).long().item()
rB = rm.forward_text_pair(input_ids=torch.tensor([[4, 5, 6]]), attention_mask=torch.tensor([[1, 1, 1]])).long().item()

if rA == 1 and rB == 4:
    print("PASSED\n")
    score += 15
else:
    print("FAILED")
    print(f"Correct:\n\trA = 1, rB = 4")
    print(f"Output:\n\trA = {rA}, rB = {rB}")
    
print('5. Checking compute pairwise accuracy implementation...\n')

preds  = torch.tensor([1, 0, 1, 1])
labels = torch.tensor([1, 0, 0, 1])

acc = metrics.compute_pairwise_accuracy(preds, labels)
if abs(acc - 0.75) < 1e-6:
    print("PASSED\n")
    score += 10
else:
    print("FAILED")
    print(f"Correct: 0.75")
    print(f"Output: {acc}")
    
print('6. Checking compute_mean_reward...\n')

rewards = [
        [(1, 3), (4, 2)],      # batch 1
        [(10, 5), (7, 8)]      # batch 2
    ]
    
win_mean, lose_mean = metrics.compute_mean_reward(rewards)

if win_mean == 6.25 and lose_mean == 3.75:
    print("PASSED\n")
    score += 10
else:
    print("FAILED")
    print(f"Correct:\n\twin_mean = 6.25, lose_mean = 3.75")
    print(f"Output:\n\twin_mean = {win_mean}, lose_mean = {lose_mean}\n")

print('7. Points for correct submission format (2 points if  a single .zip file submitted)...\n')

pts = points()
print(f'\n{pts}/2')
score += pts
           
print_score()
