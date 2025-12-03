import numpy as np
import torch
from transformers import AutoTokenizer

def compute_pairwise_accuracy(preds, labels):
    correct = 0
    total = 0
    accuracy = 0
    
    with torch.no_grad():
        # compute pairwise accuracy over all training samples
        # a correct sample is one where the predicted preference label equals the true preference label
        # BEGIN STUDENT CODE (~3 lines)
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total if total > 0 else 0.0
        # END STUDENT CODE
    return accuracy

def compute_mean_reward(rewards):
    winners = []
    losers = []
    
    win_mean = 0
    lose_mean = 0

    # compute the mean reward of "winning" and "losing" responses over the train samples
    # remember that winning response according to the model is the one with the higher reward,
    #  but that this sample may occur in either the first OR second position in the pair
    # BEGIN STUDENT CODE (~10 lines)
    # END STUDENT CODE
    
    return win_mean, lose_mean
    
    # just a helper function
def decode_prompt_responses(model, ids_A, ids_B):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    sep_id = tokenizer.sep_token_id
    
    if sep_id in ids_A:
        sep_index = ids_A.index(sep_id)
        prompt_ids = ids_A[1:sep_index]      # skip [CLS]
        response_ids = ids_A[sep_index+1:]
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        print("PROMPT:\t\t", prompt)
        print("RESPONSE A:\t", response)
        
    if sep_id in ids_B:
        sep_index = ids_B.index(sep_id)
        response_ids = ids_B[sep_index+1:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        print("RESPONSE B:\t", response)
        
# just a helper function
def display_example(model, example):
    with torch.no_grad():
        ids_A = example["input_ids_A"]
        ids_B = example["input_ids_B"]
        decode_prompt_responses(model, ids_A.tolist(), ids_B.tolist())
        rA = model.forward_text_pair(ids_A.unsqueeze(0).to(model.device), example['attn_A'].unsqueeze(0).to(model.device))
        rB = model.forward_text_pair(ids_B.unsqueeze(0).to(model.device), example['attn_B'].unsqueeze(0).to(model.device))
        print(f"Reward A: {rA.item():.3f} | Reward B: {rB.item():.3f}")
        pred = 'A' if (rA > rB).long().item() == 1 else 'B'
        print(f"Predicted: {pred}")
        correct = 'A' if example["label"].item() == 1 else 'B'
        print(f"Correct: {correct}")
        print()
