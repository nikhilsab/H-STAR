"""
Text based Column Extraction
"""

import time
import json
import argparse
import copy
import os
import regex as re

from typing import List
import platform
import multiprocessing

from generation.generator_gemini import Generator
from utils.utils import load_data_split
from nsql.database import NeuralDB
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import vertexai
from transformers import AutoTokenizer


ROOT_DIR = os.path.join(os.path.dirname(__file__), "../..")

def worker_annotate(
        pid: int,
        args,
        generator: Generator,
        g_eids: List,
        col_dict,
        tokenizer
):
    """
    A worker process for annotating.
    """
    g_dict = dict()
    built_few_shot_prompts = []
    pattern_col = '(f_col\(\[(.*?)\]\))'
    pattern_col = re.compile(pattern_col, re.S)

    for g_eid in g_eids:
        pred_col = ""
        try:
            pred_cols=[]
            new_df = {}
            # Extract Columns
            for n in range(2):
                try:
                    pred_col = re.findall(pattern_col,col_dict[str(g_eid)]['output'][n])[0][1]
                    pred_col = pred_col.replace("'","")
                    pred_col = pred_col.split(', ')
                    pred_cols.append(pred_col)
                    print(pred_cols)

                except:
                    pass
            pred_col = list(set().union(*pred_cols))
            g_data_item = col_dict[str(g_eid)]['data_item']
            g_dict[g_eid] = {
                'generations': [],
                'cols' : [],
                'ori_data_item': copy.deepcopy(g_data_item)
            }
            df = pd.DataFrame(data=g_data_item['table']['rows'], columns=g_data_item['table']['header'])
            if new_df is not None and len(new_df) == len(df):
                df = df.join(new_df)

            g_dict[g_eid]['ori_data_item']['table']['header']= df.columns.tolist()
            g_dict[g_eid]['ori_data_item']['table']['rows']= df.values.tolist()

            db = NeuralDB(
                tables=[{'title': g_data_item['table']['page_title'], 'table': g_dict[g_eid]['ori_data_item']['table']}]
            )
            
            g_data_item['table'] = db.get_table_df()
            g_data_item['title'] = db.get_table_title()
            if pred_col != []:
                filtered_pred = [value for value in pred_col if value in g_data_item['table'].columns]
                g_dict[g_eid]['cols'] = filtered_pred


            n_shots = args.n_shots
            few_shot_prompt = generator.build_few_shot_prompt_from_file(
                file_path=args.prompt_file,
                n_shots=n_shots
            )
            # Take table transpose
            generate_prompt = generator.build_generate_prompt(
                data_item=g_data_item,
                generate_type=(args.generate_type,)
            )
            prompt = few_shot_prompt + "\n\n" + generate_prompt
            # Ensure the input length fit max input tokens by shrinking the number of rows
            max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens
            
            num_rows = (g_data_item['table'].shape[0])
            while len(tokenizer.tokenize(prompt)) >= max_prompt_tokens:
                num_rows = 5
                generate_prompt = generator.build_generate_prompt(
                data_item=g_data_item,
                generate_type=(args.generate_type,),
                num_rows = num_rows
                )

                prompt = few_shot_prompt + "\n\n" + generate_prompt
            print(f"Process#{pid}: Building prompt for eid#{g_eid}, original_id#{g_data_item['id']}")
            built_few_shot_prompts.append((g_eid, prompt))
            if len(built_few_shot_prompts) < args.n_parallel_prompts:
                continue

            print(f"Process#{pid}: Prompts ready with {len(built_few_shot_prompts)} parallels. Run openai API.")
            response_dict = generator.generate_one_pass(
                prompts=built_few_shot_prompts,
                verbose=args.verbose
            )
            for eid, g_pairs in response_dict.items():
                g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
                g_dict[eid]['generations'] = g_pairs
            
            built_few_shot_prompts = []
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Process#{pid}: eid#{g_eid}, wtqid#{g_data_item['id']} generation error: {e}")

    # Final generation inference
    if len(built_few_shot_prompts) > 0:
        response_dict = generator.generate_one_pass(
            prompts=built_few_shot_prompts,
            verbose=args.verbose
        )
        for eid, g_pairs in response_dict.items():
            g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
            g_dict[eid]['generations'] = g_pairs
    
    return g_dict


def main():
    # Build paths
    args.api_keys_file = os.path.join(ROOT_DIR, args.api_keys_file)
    args.prompt_file = os.path.join(ROOT_DIR, args.prompt_file)
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    print(args.save_dir, args.api_keys_file)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    start_time = time.time()
    dataset = load_data_split(args.dataset, args.dataset_split)

    if args.dataset == "fetaqa" and args.dataset_split == "test":
        dataset = []
        with open(os.path.join(ROOT_DIR, "utils", "fetaqa", "fetaQA-v1_test.jsonl"), "r") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                dic = json.loads(line)
                # print(dic)
                feta_id = dic['feta_id']
                caption = dic['table_page_title']
                question = dic['question']
                answer = dic["answer"]
                header = dic['table_array'][0]
                rows = dic['table_array'][1:]
                data = {
                "id": feta_id,
                "table": {
                    "id": feta_id,
                    "header": header,
                    "rows": rows,
                    "page_title": caption
                },
                "question": question,
                "answer": answer
                }
                dataset.append(data)

    # For TabFact test split, we load the small test set (about 2k examples) to test,
    # since it is expensive to test on full set
    if args.dataset == "tab_fact" and args.dataset_split == "test":
        dataset = []
        with open(os.path.join(ROOT_DIR, "utils", "tab_fact", "small_test.jsonl"), "r") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                dic = json.loads(line)
                id = dic['table_id']
                caption = dic['table_caption']
                question = dic['statement']
                answer_text = dic['label']
                header = dic['table_text'][0]
                rows = dic['table_text'][1:]
                
                data = {
                    "id": i,
                    "table": {
                        "id": id,
                        "header": header,
                        "rows": rows,
                        "page_title": caption
                    },
                    "question": question,
                    "answer_text": answer_text
                }
                dataset.append(data)

    # Load keys
    keys = ""
    key_path = ''
    credentials = Credentials.from_service_account_file(
    key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform'])

    if credentials.expired:
        credentials.refresh(Request())
    PROJECT_ID = ""
    vertexai.init(project=PROJECT_ID, location="us-central1",credentials=credentials)

    with open(os.path.join(args.save_dir, args.input_program_file), 'r') as f:
        data = json.load(f)
    col_dict = dict()
    for eid, _ in data.items():
        data_item = data[eid]['ori_data_item']
        if data[eid]['generations']:
            col_gen = data[eid]['generations']
        else:
            col_gen = []
        col_dict[eid] = {'output': col_gen, 'data_item': data_item}

    # Split by processes
    col_dict_group = [dict() for _ in range(args.n_processes)]
    for idx, eid in enumerate(col_dict.keys()):
        col_dict_group[idx % args.n_processes][eid] = col_dict[eid]

    # Annotate
    print(len(col_dict))
    generator = Generator(args, keys=keys)
    # Enter dataset size for inference: range(0, len(dataset))
    generate_eids = list(range(0,2))
    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[int(g_eid) % args.n_processes].append(g_eid)
    print('\n******* Annotating *******')
    g_dict = dict()
    worker_results = []
    pool = multiprocessing.Pool(processes=args.n_processes)
    for pid in range(args.n_processes):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(ROOT_DIR, "utils", "gpt2"))
        worker_results.append(pool.apply_async(worker_annotate, args=(
            pid,
            args,
            generator,
            generate_eids_group[pid],
            col_dict_group[pid],
            tokenizer
        )))

    # Merge annotation results
    for r in worker_results:
        worker_g_dict = r.get()
        g_dict.update(worker_g_dict)
    pool.close()
    pool.join()

    # Save annotation results
    save_file_name = f'{args.dataset}_{args.dataset_split}_col_text.json'
    with open(os.path.join(args.save_dir, save_file_name), 'w') as f:
        dict_keys = list(g_dict.keys())
        dict_keys.sort()
        sorted_g_dict = {i: g_dict[i] for i in dict_keys}
        json.dump(sorted_g_dict, f, indent=4)

    print(f"Elapsed time: {time.time() - start_time}")


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # File path or name
    parser.add_argument('--dataset', type=str, default='wikitq',
                        choices=['wikitq', 'tab_fact', 'fetaqa'])
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'validation', 'test'])
    parser.add_argument('--api_keys_file', type=str, default='key.txt')
    parser.add_argument('--prompt_file', type=str, default='templates/prompts/col.txt')
    parser.add_argument('--input_program_file', type=str,
                        default='wikitq_test_col_augment.json')
    parser.add_argument('--save_dir', type=str, default='results/model_gemini/')

    # Multiprocess options
    parser.add_argument('--n_processes', type=int, default=1)

    # Program generation options
    parser.add_argument('--prompt_style', type=str, default='text_full_table',
                        choices=['create_table_select_3_full_table',
                                 'create_table_select_full_table',
                                 'create_table_select_3',
                                 'create_table',
                                 'create_table_select_3_full_table_w_all_passage_image',
                                 'create_table_select_3_full_table_w_gold_passage_image',
                                 'transpose',
                                 'text_full_table',
                                 'no_table'])
    parser.add_argument('--generate_type', type=str, default='col',
                        choices=['col', 'sql', 'answer', 'row', 'python'])
    parser.add_argument('--n_shots', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    # LLM options
    parser.add_argument('--engine', type=str, default="gemini-1.5-flash-001")
    parser.add_argument('--n_parallel_prompts', type=int, default=1)
    parser.add_argument('--max_generation_tokens', type=int, default=256)
    parser.add_argument('--max_api_total_tokens', type=int, default=8001)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--sampling_n', type=int, default=3)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--stop_tokens', type=str, default='\n\n',
                        help='Split stop tokens by ||')

    # debug options
    parser.add_argument('-v', '--verbose', action='store_false')

    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split('||')
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    main()
