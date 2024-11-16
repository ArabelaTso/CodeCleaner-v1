import os, re, sys
import requests
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from collections import defaultdict
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd
from src.json_util import load_jsonl
from concurrent.futures import ThreadPoolExecutor, TimeoutError


def send_json(url, data):
    response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Fail. Status Code: {response.status_code}")
        return None

def work(code, seq_id=0):
    MIN_LENGTH = 50;
    MAX_LENGTH = 2500
    
    if len(code) < MIN_LENGTH:
        print('Too few characters.')
        return "", 0.0
    if len(code) >= MAX_LENGTH:
        print('Input is too long! Truncate the code.')
        code = code[:MAX_LENGTH-1]
    
    url = "https://stack.dataportraits.org/q"
    report = send_json(url, {
                            "document": code,
                            "seq_id": seq_id
                        })
    while report is None:
        time.sleep(1)
        print("Retrying ...")
        report = send_json(url, {
                                "document": code,
                                "seq_id": seq_id
                            })
    
    if len(report['chains']) == 0:
        if len(code) >= MAX_LENGTH:
            print('No results found.')
            return "", 0.0
        else:
            print('No results found, try adding more text.')
            return "", 0.0
    
    sketch_width = report['width'];
    
    chains = report['chain_idxs']
    flgs = [False for _ in range(len(report['doc']))]
    for chain in chains:
        for idx in range(chain[0], min(len(flgs), chain[-1] + sketch_width), 1):
            flgs[idx] = True
    
    flgtrueidxs = [idx for idx, val in enumerate(flgs) if val == True]
    overlap = "".join([report['doc'][idx] for idx, val in enumerate(flgs) if val == True])#report['doc'][flgtrueidxs[0]:flgtrueidxs[-1]+1]
    
    return overlap, len(flgtrueidxs) / len(report['doc'])

    
def write_one_block(f, start_id, id, qid, origin, overlap, overlap_ratio):
    f.write(f'--- {start_id + id} --\n')
    f.write(f'Question ID: {qid}\n')
    f.write(f'Original Code:\n```\n{origin}\n```\n\n\n')
    
    f.write(f'Overlapping Code:\n```\n{overlap}\n```\n')
    f.write(f'<Overlap Ratio: {overlap_ratio}>\n\n')
        
    f.write('---\n\n')
    

def enumerate_query_code_groups(jsonl_file, group_name, output_folder='data/overlap/', start_id=0):
    os.makedirs(output_folder, exist_ok=True)
    
    input_data = load_jsonl(jsonl_file)
    
    output_file = os.path.join(output_folder, f"overlap{group_name}.txt")
    
    # executor = ThreadPoolExecutor(max_workers=40)
    # futures = []

    with open(output_file, 'a+') as f:
        for id, data in enumerate(input_data[start_id:], 0):
            print(f'Processing {start_id + id}th code...')
        
            origin = data['content']
            qid = data['question_id']
            # print(f'Original Code: {origin}\n\n')
            
            overlap, overlap_ratio = work(origin)

        #     futures.append(executor.submit(work, origin))
            
        # for id, data in enumerate(input_data[start_id:], 0):
        #     overlap, overlap_ratio = futures[id].result()

            write_one_block(f, start_id, id, qid, origin, overlap, overlap_ratio)
            f.flush()
            

def extract_overlap_rate(file, remove_zero=False):
    with open(file, 'r') as f:
        lines = f.readlines()
        
        overlap_rate = []
        for line in lines:
            if line.startswith('<Overlap Ratio: '):
                cur_rate = float(line.split('<Overlap Ratio: ')[1].split('>')[0])
                if remove_zero and cur_rate <= 0.2:
                    continue
                else:
                    overlap_rate.append(cur_rate)
        return overlap_rate
    


if __name__ == '__main__':
    input_dir = 'data/'
    output_dir = 'data/overlap_v1'
    
    for group_name in ['infill_year2021']:   #'infill_refactor_np_apd', 'infill_refactor_np_shuffles', 'infill_refactor_pd_apd', 'infill_refactor_pd_shuffles', 'infill_refactor_fn-args', 'infill_refactor_fn-decorator'
        enumerate_query_code_groups(jsonl_file=f'./{input_dir}/{group_name}/question.jsonl', group_name=group_name, output_folder=output_dir, start_id=0)
    