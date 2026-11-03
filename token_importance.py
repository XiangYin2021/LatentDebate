import os
os.environ['TRANSFORMERS_CACHE'] = "/data/parietal/store3/soda/lihu/hf_model/"
import argparse
import tqdm
import torch
import pandas as pd
import sklearn
import sklearn.metrics
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer
import numpy as np

model_id = 'cross-encoder/stsb-roberta-large'
measure_model = CrossEncoder(model_name=model_id, num_labels=1)



def compute_positional_attention(max_len, center_idx, w=0.2, s=1):
    """
    Compute a delta vector using the formula in the image, given a center index.

    Args:
        max_len (int): Total length of the sequence.
        center_idx (int): The center index i for which to compute delta[i,j] across j.
        w (float): Locality control parameter (w > 0).
        s (float): Symmetry control parameter.

    Returns:
        np.ndarray: A 1D array of length `max_len` representing delta_i over j.
    """
    deltas = np.zeros(max_len)
    for j in range(max_len):
        l2 = (center_idx - j) ** 2
        if center_idx <= j:
            alpha = -s * w * l2
        else:
            alpha = -w * l2
        deltas[j] = np.exp(alpha)

    # Normalize
    deltas /= np.sum(deltas)
    return deltas


def softmax_with_temperature(x, pe=False, temperature=0.25):
    x = np.array(x)
    if pe:
        pe_value = compute_positional_attention(len(x), len(x), w=0.1)
        print(pe_value)
        x *= pe
    x = x / temperature
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / e_x.sum()


def get_tokenwise_importance(generations, tokenizer_model):
    generations = [generations.split('??')[-1]]
    
    print(generations)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=False)


    scores = []
    token_importance_list, pe_token_importance_list = [], []
    for sample_idx, gen in enumerate(tqdm.tqdm(generations)):
        # likelihoods = gen['token_wise_entropy']
        gen_scores = []
        for k in range(len(generations)):
            generated_text = generations[k]
            #question = gen['question']
            tokenized = torch.tensor(tokenizer.encode(generated_text, add_special_tokens=False))
            sub_tokens = tokenizer.tokenize(generated_text)
            # likelihood = likelihoods[k]['original_token_wise_entropy']
            token_importance = []
            # measure cosine similarity by removing each token and compare the similarity
            for token in tokenized:
                similarity_to_original = measure_model.predict([generated_text,
                                                                generated_text.replace(
                                                                    tokenizer.decode(token, skip_special_tokens=True),
                                                                    '')])
                token_importance.append(1 - torch.tensor(similarity_to_original))

            token_importance = torch.tensor(token_importance).reshape(-1).tolist()
            #token_importance = softmax_with_temperature(token_importance)
            #pe_token_importance = softmax_with_temperature(token_importance, pe=True)
            
            token_importance_list.append(token_importance)
            #pe_token_importance_list.append(pe_token_importance)
            
            
        scores.append(torch.tensor(gen_scores).mean())
    scores = torch.tensor(scores)
    if torch.isnan(scores).sum() > 0:
        scores[torch.isnan(scores).nonzero(as_tuple=True)] = 0

    
    return list(zip(sub_tokens,token_importance_list[0]))
    #print(list(zip(sub_tokens,pe_token_importance_list[0])))


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--measurement-model", default='cross-encoder/stsb-roberta-large',
                   choices=['cross-encoder/stsb-roberta-large',
                            'cross-encoder/stsb-distilroberta-base'],
                   help="desc")
    p.add_argument('--tokenizer-model', default='meta-llama/Llama-3.1-8B')
    p.add_argument('--run-name', default='')

    return (p.parse_args())


if __name__ == '__main__':
    #args = cmdline_args()
    #generations = 'The city of Tokyo is not in Japan. This statement is: False. The city of Hanoi is not in Poland. This statement is: True. United Arab Emirates is in Antarctica, literally maybe. This statement is:'
    import json
    
    wf = open('probs/llama-3.1-8b-instruct_trivia_qa_prob_new_token_score.jsonl', 'w')
    
    for line in open('/data/parietal/store3/soda/lihu/code/fast_uncertainty/hallucination_detection/probs/llama-3.1-8b-instruct_trivia_qa_prob.jsonl'):
        obj = json.loads(line)
        generations = obj['result']['query']['query']
        
        token_score = get_tokenwise_importance(generations, tokenizer_model='meta-llama/Llama-3.1-8B-Instruct')
        print(token_score)
        obj['result']['query']['token_score'] = token_score
        
        json.dump(obj, wf)
        wf.write('\n')
        wf.flush()