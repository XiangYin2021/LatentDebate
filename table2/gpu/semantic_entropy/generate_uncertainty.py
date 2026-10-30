import datasets
import json
from uncertainty_estimator import Uncertainty
import time
import re

methods = [
    #'Perplexity',
    # 'MSP',
    'SemanticEntropy',
    # 'SAR',
    # 'LexicalSimilarity',
    #'EigValLaplacian'
    #'Eccentricity',
    #'Verbalized1S',
    # 'NumSemSets',
    # 'CCP',
    #'MSP',
    # 'TokenSAR',
    #'DegMat',
    #'EigValLaplacian',
]

def get_statement(text):
    match = re.search(r"False\.\s*(.*)$", text)

    if match:
        result = match.group(1)
        return result
    return None
model_name = 'meta-llama/Llama-3.1-8B-Instruct'

def load_data(label_file_name):
    data = list()
    for line in open(label_file_name):
        obj = json.loads(line)
        query, label = obj['result']['query']['query'],  obj['label']
        query = get_statement(query) + 'True or False?'
        label = 'True' if label == 1 else 'False'
        answer_token = obj['result']['answer']['tokenized_answer'][0]
        data.append((query, label))
    return data

examples = [
    {'question': 'Paris is the capital of France', 'answer': 'True'},
    {'question': 'London is the capital of France', 'answer': 'False'},
    
]

for estimator_name in methods:
    print("*" * 30)
    print(f"estimator_name = {estimator_name}")
    print("*" * 30)

    estimator = Uncertainty(model_name, estimator_name, few_shot_examples=examples)

    tasks = ['triviaqa', 'truthfulqa']

    for task in tasks:
        for seed in range(1):
            label_file_name = f'../../data/{task}_prob.jsonl'
            output_file_name = (
                f"output2/{task}_seed{seed}_{estimator_name}_"
                f"{model_name.split('/')[-1].lower()}.json"
            )

            print(f"Input : {label_file_name}")
            print(f"Output: {output_file_name}")

            with open(output_file_name, "w", encoding="utf8") as wf:
                for index, (question, answer) in enumerate(load_data(label_file_name)):
                    if index >= 250:
                        continue

                    print(index, question, answer)

                    start = time.perf_counter()
                    result = estimator.run(question, answer)
                    end = time.perf_counter()

                    running_time = end - start
                    result["running_time"] = running_time

                    print("#" * 30)
                    print(f"my_function took {running_time:.6f} seconds")
                    print("#" * 30)

                    json.dump(result, wf)
                    wf.write("\n")
                    wf.flush()
        