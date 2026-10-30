import os
os.environ['TRANSFORMERS_CACHE'] = "/data/parietal/store3/soda/lihu/hf_model/"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph import estimate_uncertainty
from lm_polygraph.estimators import Perplexity, MeanTokenEntropy, PTrue, PointwiseMutualInformation, MaximumSequenceProbability
from lm_polygraph.estimators import LexicalSimilarity, MonteCarloSequenceEntropy, SemanticEntropy, EigValLaplacian, SAR, RenyiNeg, FisherRao
from lm_polygraph.estimators import NumSemSets, Eccentricity, DegMat, Verbalized1S, Verbalized2S, ClaimConditionedProbabilityClaim, TokenSAR
import evaluate


def create_qwen_prompt(question, examples):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides short answers to questions. Please provide a very short answer (less than 5 tokens) \
         Please strictly follow the format of below examples to reply:\n"}
    ]

    for example in examples:
        messages.append({"role": "user", "content": f"Question: {example['question']}"})
        messages.append({"role": "assistant", "content": f"{example['answer']}"})

    messages.append({"role": "user", "content": f"Question: {question}"})


    return messages


class Uncertainty():
    def __init__(self, model_name, estimator_name, few_shot_examples=[], rouge_alpha=0.3):
        #model_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="eager", device_map="cuda:1")
        self.base_model.config.output_attentions = True
        self.base_model.config.return_dict = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.rouge_alpha = rouge_alpha
        self.max_new_token = 1000
        self.gen_params = GenerationParameters(max_new_tokens=self.max_new_token)

        self.few_shot_examples = few_shot_examples
        self.model = WhiteboxModel(self.base_model, self.tokenizer, model_path=model_name, generation_parameters=self.gen_params)
        self.rouge = evaluate.load('rouge')
        
        if estimator_name == 'SemanticEntropy':
            self.estimator = SemanticEntropy()
        elif estimator_name == 'MSP':
            self.estimator = MaximumSequenceProbability()
        elif estimator_name == 'Perplexity':
            self.estimator = Perplexity()
        elif estimator_name == 'MeanTokenEntropy':
            self.estimator = MeanTokenEntropy()
        elif estimator_name == 'RenyiNeg':
            self.estimator = RenyiNeg()
        elif estimator_name == 'FisherRao':
            self.estimator = FisherRao()
        elif estimator_name == 'PTrue':
            self.estimator = PTrue()
        elif estimator_name == 'PointwiseMutualInformation':
            self.estimator = PointwiseMutualInformation()
        elif estimator_name == 'SAR':
            self.estimator = SAR()
        elif estimator_name == 'TokenSAR':
            self.estimator = TokenSAR()
        elif estimator_name == 'LexicalSimilarity':
            self.estimator = LexicalSimilarity()
        elif estimator_name == 'MonteCarloSequenceEntropy':
            self.estimator = MonteCarloSequenceEntropy()
        elif estimator_name == 'EigValLaplacian':
            self.estimator = EigValLaplacian()
        elif estimator_name == 'NumSemSets':
            self.estimator = NumSemSets()
        elif estimator_name == 'Eccentricity':
            self.estimator = Eccentricity()
        elif estimator_name == 'DegMat':
            self.estimator = DegMat()
        elif estimator_name == 'CCP':
            self.estimator = ClaimConditionedProbabilityClaim()
        elif estimator_name == 'Verbalized1S':
             self.estimator = Verbalized1S()
        # elif estimator_name == 'Verbalized2S':
        #     self.estimator = Verbalized2S()
        
        

    def run(self, query, answer):
        prompt = create_qwen_prompt(query, self.few_shot_examples)
        
        prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        en_output = estimate_uncertainty(self.model, self.estimator, input_text=prompt)
        prediction = en_output.generation_text
        
        
        # score = self.rouge.compute(predictions=[prediction], references=[answer])
        # label = 1 if score["rougeL"] > self.rouge_alpha else 0
        #label = judge(answer, prediction) 
        result =  {
            'query': query,
            'answer': answer,
            'generated_texts': prediction,
            #'rouge': score["rougeL"],
            'uncertainty': en_output.uncertainty,
            #'label': label
            
        }
        return result