import torch
from torch.utils.data import DataLoader
from preference_dataset import PreferenceDataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import random

class PreferenceDataloader():
    def __init__(self,data_split):
        self.data_split = data_split
        self.loader = None
        
    def precompute_embeddings(self, dataset, device):
        embedder = SentenceTransformer("all-MiniLM-L6-v2").to(device)
        texts = [ex["article"][:400] for ex in dataset]
        summaries = [ex["highlights"] for ex in dataset]

        print("Encoding prompts...")
        emb_prompts = embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        print("Encoding summaries...")
        emb_summaries = embedder.encode(summaries, convert_to_tensor=True, show_progress_bar=True)

        return emb_prompts, emb_summaries
        
    def sample_preference_pair(self, idx, example, dataset, emb_prompts, emb_summaries):
        prompt = example["article"]
        good_summary = example["highlights"]
        bad_summary = None

        emb_good = emb_summaries[idx]

        # Compute cosine similarity to all summaries
        # Exclude the good summary itself
        # Then select the *closest* different summary (hard negative)
        # BEGIN STUDENT CODE (~4 lines)
        # END STUDENT CODE

        return prompt, good_summary, bad_summary
        
    def randomize_positions(self, good_summary, bad_summary):
        response_A, response_B, label = None, None, None

        # Randomly decide if good summary is A or B to prevent positional bias
        # Use a 50% probability threshold
        # Label the sample "1" if "A" is the good summary
        # Label the sample "0" if "B" is the good summary
        # BEGIN STUDENT CODE (~6 lines)
        # END STUDENT CODE
            
        return response_A, response_B, label
        
    def get_dataloaders(self, data_split, device, batch_size=8, tokenizer_name="distilbert-base-uncased", num_examples=200, shuffle=True):
        data_split = data_split
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        seq_len = 128
        
        emb_prompts, emb_summaries = self.precompute_embeddings(data_split,device)

        data = []
        for i, ex in enumerate(data_split):
            if i >= num_examples:
                break
            prompt, good_summary, bad_summary = self.sample_preference_pair(i,ex,data_split,emb_prompts,emb_summaries)
            response_A, response_B, label = self.randomize_positions(good_summary, bad_summary)

            # Tokenize prompt + response pairs
            inputs_A = tokenizer(prompt, response_A, truncation=True, padding="max_length", max_length=seq_len, return_tensors="pt")
            inputs_B = tokenizer(prompt, response_B, truncation=True, padding="max_length", max_length=seq_len, return_tensors="pt")

            example = {
                "input_ids_A": inputs_A["input_ids"].squeeze(0),
                "attn_A": inputs_A["attention_mask"].squeeze(0),
                "input_ids_B": inputs_B["input_ids"].squeeze(0),
                "attn_B": inputs_B["attention_mask"].squeeze(0),
                "label": torch.tensor(label, dtype=torch.long)
            }
            data.append(example)

        dataset = PreferenceDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
