import argparse
import os
import json
from pathlib import Path
import openai
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from eval import *
from data_process import write_jsonl, load_jsonl
from model_config import MODEL_ABBREV



def load_model(name1, device=None):
    if device is None:
        device = 'auto'
        
    if "davinci" in name1:
        model1 = None
        tokenizer1 = None
    else:
        model1 = AutoModelForCausalLM.from_pretrained(name1, return_dict=True, device_map="auto", trust_remote_code=True)
        model1.eval()
        tokenizer1 = AutoTokenizer.from_pretrained(name1)
    return model1, tokenizer1

def calculatePerplexity_gpt3(prompt, modelname):
    prompt = prompt.replace('\x00','')
    responses = None
    # Put your API key here
    openai.api_key = "YOUR_API_KEY" # YOUR_API_KEY
    while responses is None:
        try:
            responses = openai.Completion.create(
                        engine=modelname, 
                        prompt=prompt,
                        max_tokens=0,
                        temperature=1.0,
                        logprobs=5,
                        echo=True)
        except openai.error.InvalidRequestError:
            print("too long for openai API")
    data = responses["choices"][0]["logprobs"]
    all_prob = [d for d in data["token_logprobs"] if d is not None]
    p1 = np.exp(-np.mean(all_prob))
    return p1, all_prob, np.mean(all_prob)

     
def calculatePerplexity(sentence, model, tokenizer, gpu, max_tokens=1024):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence, max_length=max_tokens, truncation=True)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()


def inference(model1, tokenizer1, text, ex, modelname1):
    pred = {}

    if "davinci" in modelname1:
        p1, all_prob, p1_likelihood = calculatePerplexity_gpt3(text, modelname1) 
        p_lower, _, p_lower_likelihood = calculatePerplexity_gpt3(text.lower(), modelname1)
    else:
        p1, all_prob, p1_likelihood = calculatePerplexity(text, model1, tokenizer1, gpu=model1.device)
        p_lower, _, p_lower_likelihood = calculatePerplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

   # ppl
    pred["ppl"] = p1
    pred["ppl_lower"] = p_lower


    # Ratio of log ppl of lower-case and normal-case
    pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    pred["ppl/zlib"] = np.log(p1)/zlib_entropy
    
    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()

    ex["pred"] = pred
    return ex

def evaluate_data(test_data, model1, tokenizer1, col_name, modelname1, output_folder):
    print(f"all data size: {len(test_data)}")
    
    with open(output_folder, 'w') as f:
        for ex in tqdm(test_data): 
            text = ex[col_name]
            new_ex = inference(model1, tokenizer1, text, ex, modelname1)
            new_ex.update(ex)

            # add to list
            # score_list.append(new_ex)

            # write to file
            json.dump(new_ex, f)
            f.write('\n')


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--target_model', type=str, default="WizardLM/WizardCoder-15B-V1.0", help="the model to attack: WizardLM/WizardCoder-15B-V1.0, bigcode/starcoder")
        self.parser.add_argument('--output_dir', type=str, default="")
        self.parser.add_argument('--data', type=str, default="data/seen_unseen/question_toy.jsonl", help="the dataset to evaluate.")
        self.parser.add_argument('--length', type=int, default=64, help="the length of the input text to evaluate. Choose from 32, 64, 128, 256")
        self.parser.add_argument('--key_name', type=str, default="content", help="the key name corresponding to the input text. Selecting from: input, paraphrase")
        
        
if __name__ == '__main__':
    args = Options()
    args = args.parser.parse_args()
    
    if len(args.output_dir) == 0:
        output_folder = os.path.join(os.path.dirname(args.data), 'model_score')
    else:
        output_folder = os.path.join(args.output_dir, 'model_score')
    os.makedirs(output_folder, exist_ok=True)

    try:
        dataset_name = args.data.split("/")[-2]
    except:
        dataset_name = "dataset"

    # load model and data
    model, tokenizer = load_model(args.target_model)
    
    if "jsonl" in args.data:
        data = load_jsonl(f"{args.data}")

    evaluate_data(data, model, tokenizer, args.key_name, args.target_model, output_folder=f"{output_folder}/{MODEL_ABBREV[args.target_model]}.jsonl")

    # fig_fpr_tpr(all_output, args.output_dir)