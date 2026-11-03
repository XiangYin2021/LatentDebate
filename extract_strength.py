import os
os.environ['TRANSFORMERS_CACHE'] = "/data/parietal/store3/soda/lihu/hf_model/"
import torch
import json
import pandas
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from token_importance import get_tokenwise_importance

task= 'trivia_qa'
model_name = 'meta-llama/Llama-3.1-8B-Instruct'
#model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
#model_name = 'meta-llama/Llama-2-13b-hf'

model_tokens = {
    'meta-llama/Llama-3.1-8B-Instruct': [[[3082], [3641]], [[837],[905]], [[2575], [4139]], [[1904], [3934]]]
}

short_model_name = model_name.split('/')[-1].lower()
label_file_name = f'label/{task}_{short_model_name}.json'

mt = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="cuda:2"
    )
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
mt.eval()
head_fn = mt.lm_head

target_tokens = model_tokens[model_name]



true_token = tokenizer.decode(target_tokens[0][0])
false_token = tokenizer.decode(target_tokens[0][1])
print(true_token, false_token)

max_new_token = 10


def load_data():
    data = list()
    for line in open(label_file_name):
        obj = json.loads(line)
        query, generated_texts, label = obj['query'], obj['generated_texts'], obj['label']
        data.append((query, generated_texts, label))
    return data

def gen_prob(query):
    
    query_results = {'query':query}
    
    # forward
    with torch.no_grad():
        inputs = tokenizer(
            query, padding=True, return_tensors="pt", return_offsets_mapping=True
        ).to(mt.device)
        
        texutal_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        query_results['tokenized_query'] = texutal_tokens
        
        print(f"tokens = {texutal_tokens}")
        
        forward_dict = mt.forward(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
        
    hidden = forward_dict["hidden_states"]
    
    # (num_layers, batch_size, num_heads, seq_len, seq_len)
    attentions = forward_dict.attentions
    attn_tensor = torch.stack(attentions)  # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
    #(num_layers, seq_len, seq_len)
    avg_attn_per_layer = attn_tensor.mean(dim=2).squeeze(1)
    
    # token 被关注程度
    layer_token_attn = avg_attn_per_layer.mean(dim=1).T         # (L, T)
    
    layer_token_attn = layer_token_attn.cpu().tolist()
    layer_token_attn = [[round(e, 4) for e in layer] for layer in layer_token_attn]

    
    for i, target_token in enumerate(target_tokens):
        debate_matrix = []

        # 预先把 token group 转成 tensor（避免反复 Python 循环）
        token_groups = [
            torch.tensor(tokens, device=hidden[0].device)
            for tokens in target_token
        ]

        with torch.no_grad():
            for layer, hidden_layer in enumerate(hidden):
                # hidden_layer: (1, seq_len, hidden_dim)

                # 一次性算所有 step 的 logits
                # logits_all: (1, seq_len, vocab_size)
                logits_all = head_fn(hidden_layer)

                # 收集每个 token group 的 logits
                # 每个元素 shape: (1, seq_len)
                group_logits = []
                for tokens in token_groups:
                    # index_select 比 Python for 快很多
                    logits = logits_all.index_select(
                        dim=-1, index=tokens
                    ).sum(dim=-1)
                    group_logits.append(logits)

                # (1, seq_len, num_groups)
                stacked_layer_logits = torch.stack(group_logits, dim=-1)

                # softmax over groups
                layer_norm_probs = stacked_layer_logits.softmax(dim=-1)

                # 取第一个 group 的概率
                true_probs = layer_norm_probs[..., 0]  # (1, seq_len)

                # 转成 Python list（最后一步再转）
                norm_probs = torch.round(true_probs * 1e4) / 1e4
                debate_matrix.append(norm_probs.squeeze(0).tolist())
        query_results[f'query_probs_{i+1}'] = [list(row) for row in zip(*debate_matrix)]
    
    
    query_results['token_score'] = get_tokenwise_importance(query, model_name)
    query_results['attentions'] = layer_token_attn
    

    
    answer_results = dict()

    # Step 2: Generate the response while capturing hidden states
    # generation_config = GenerationConfig(
    #     output_hidden_states=True,
    #     return_dict_in_generate=True,
    #     max_new_tokens=max_new_token,  # Adjust as needed
    #     do_sample=False,
    #     num_beams=1,
    #     output_scores=True,
        
    # )

    # with torch.no_grad():
    #     generation_outputs = mt.generate(
    #         inputs.input_ids,
    #         generation_config=generation_config,
    #     )

    # # Extract generated token IDs and their corresponding hidden states
    # generated_ids = generation_outputs.sequences[:, inputs.input_ids.shape[-1]:]  # Exclude input tokens
    # generated_hidden_states = generation_outputs.hidden_states  # Tuple of tensors for each generated token

    # # Decode the generated response
    # generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # print("Generated Response:", generated_text)
    # #print(len(generated_hidden_states[1:]))
    
    # answer_results['answer'] = generated_text
    # answer_results['tokenized_answer'] = tokenizer.convert_ids_to_tokens(generated_ids[0])

    
    return {'query':query_results, 'answer':answer_results}



def store_probs(data, outfile, examples):
    wf = open(outfile, 'w', encoding='utf8')

    
    for index, (query, generated_texts, label) in enumerate(data):
        print(f"process {index} sample")
        statement = f"{query}? {generated_texts}"
        prompt = f"""{examples} {statement}. This statement is:"""
        probs = gen_prob(prompt)
        json.dump({'label':label, 'result':probs}, wf)
        wf.write("\n")
        wf.flush()
    


# examples = "Who was the man behind The Chipmunks? David Seville. This statement is: True. Who was the man behind The Chipmunks? Tritons. This statement is: False." 
# data = load_data()
# store_probs(data, outfile=f'probs/{short_model_name}_{task}_prob.jsonl', examples=examples)

# examples = "Who is the spouse of the Green performer? Miquette Giraudy. This statement is: True. Who is the spouse of the Green performer? South Central Coast. This statement is: False." 
# data = pandas.read_csv('/data/parietal/store3/soda/lihu/code/internal_debate/input/musique.csv', delimiter='\t')
# store_probs(data, outfile=f'../output/{model_folder}/musique_prob.jsonl', examples=examples)

examples = "What happens to you if you eat watermelon seeds? The watermelon seeds pass through your digestive system. This statement is: True.What happens to you if you eat watermelon seeds? You grow watermelons in your stomach. This statement is: False." 
data = load_data()
store_probs(data, outfile=f'probs/{short_model_name}_{task}_prob.jsonl', examples=examples)

