from gensim import models
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig

def get_bert_embed(phrase_list, m, tok, max_len=32, normalize=True, summary_method="CLS"):
    batch_size = 64
    input_ids = []
    for phrase in phrase_list:
        input_ids.append(tok.encode_plus(
            phrase, max_length=max_len, add_special_tokens=True,
            truncation=True, pad_to_max_length=True)['input_ids'])
    m.eval()

    count = len(input_ids)
    now_count = 0
    with torch.no_grad():
        while now_count < count:
            input_gpu_0 = torch.LongTensor(input_ids[now_count:min(
                now_count + batch_size, count)])
            if summary_method == "CLS":
                embed = m(input_gpu_0)[1]
            if summary_method == "MEAN":
                embed = torch.mean(m(input_gpu_0)[0], dim=1)
            if normalize:
                embed_norm = torch.norm(
                    embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                embed = embed / embed_norm
            embed_np = embed.cpu().detach().numpy()
            if now_count == 0:
                output = embed_np
            else:
                output = np.concatenate((output, embed_np), axis=0)
            now_count = min(now_count + batch_size, count)
    return output

model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')
tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')
all_entities = []
with open('./data/all_entities') as f:
    for line in f:
        all_entities.append(line.strip())

all_entities_emb = get_bert_embed(all_entities, model, tokenizer, max_len=512, normalize=True, summary_method="CLS")

np.save('./all_entities_pretrain_emb.npy', all_entities_emb)
