import os

ROOT_DIR = os.path.join(os.path.dirname(__file__))
print(ROOT_DIR)
# Disable the TOKENIZERS_PARALLELISM
TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"

"""WIKITQ"""
# ### sql column select ###
# os.system(fr"""{TOKENIZER_FALSE}python ./scripts/model_gemini/col_sql.py --dataset wikitq \
# --dataset_split test \
# --prompt_file prompts/col_select_sql.txt \
# --n_parallel_prompts 3 \
# --max_generation_tokens 512 \
# --temperature 0.2 \
# --sampling_n 2 \
# -v""")

# ### text column select ###
# os.system(fr"""{TOKENIZER_FALSE}python ./scripts/model_gemini/col_text.py --dataset wikitq \
# --dataset_split test \
# --prompt_file prompts/col_select_text_v2.txt \
# --n_parallel_prompts 3 \
# --input_program_file wikitq_test_col_sql.json \
# --max_generation_tokens 512 \
# --temperature 0.4 \
# --sampling_n 2 \
# -v""")

# ### sql row select ###
# os.system(fr"""{TOKENIZER_FALSE}python ./scripts/model_gemini/row_sql.py --dataset wikitq \
# --dataset_split test \
# --prompt_file prompts/row_sql_v2.txt \
# --n_parallel_prompts 3 \
# --input_program_file wikitq_test_col_text.json \
# --max_generation_tokens 512 \
# --temperature 0.2 \
# --sampling_n 2 \
# -v""")

# ### text row select ###
# os.system(fr"""{TOKENIZER_FALSE}python ./scripts/model_gemini/row_text.py --dataset wikitq \
# --dataset_split test \
# --prompt_file prompts/row_text_v2.txt \
# --n_parallel_prompts 3 \
# --input_program_file wikitq_test_row_sql.json \
# --max_generation_tokens 512 \
# --temperature 0.4 \
# --sampling_n 2 \
# -v""")

# ### sql reason ###
# os.system(fr"""{TOKENIZER_FALSE}python ./scripts/model_gemini/reason_sql.py --dataset wikitq \
# --dataset_split test \
# --prompt_file prompts/sql_reasoning_wtq.txt \
# --n_parallel_prompts 1 \
# --input_program_file wikitq_test_row_text.json \
# --max_generation_tokens 512 \
# --temperature 0.1 \
# --sampling_n 1 \
# -v""")

# ### final text reason ###
# os.system(fr"""{TOKENIZER_FALSE}python ./scripts/model_gemini/reason_text.py --dataset wikitq \
# --dataset_split test \
# --prompt_file prompts/text_reason_wtq.txt \
# --n_parallel_prompts 1 \
# --input_program_file wikitq_test_sql_reason.json \
# --max_generation_tokens 512 \
# --temperature 0.0 \
# --sampling_n 1 \
#  -v""")

"""*******************************************"""

"""TABFACT"""
# ## sql column select ###
# os.system(fr"""{TOKENIZER_FALSE}python ./scripts/model_gemini/col_sql.py --dataset tab_fact \
# --dataset_split test \
# --prompt_file prompts/col_select_sql.txt \
# --n_parallel_prompts 3 \
# --max_generation_tokens 512 \
# --temperature 0.4 \
# --sampling_n 2 \
# -v""")

# ## text column select ###
# os.system(fr"""{TOKENIZER_FALSE}python ./scripts/model_gemini/col_text.py --dataset tab_fact \
# --dataset_split test \
# --prompt_file prompts/col_select_text_v2.txt \
# --n_parallel_prompts 3 \
# --input_program_file tab_fact_test_col_sql.json \
# --max_generation_tokens 512 \
# --temperature 0.7 \
# --sampling_n 2 \
# -v""")

# ## sql row select ###
# os.system(fr"""{TOKENIZER_FALSE}python ./scripts/model_gemini/row_sql.py --dataset tab_fact \
# --dataset_split test \
# --prompt_file prompts/row_sql.txt \
# --n_parallel_prompts 1 \
# --input_program_file tab_fact_test_col_text.json \
# --max_generation_tokens 512 \
# --temperature 0.4 \
# --sampling_n 2 \
# -v""")

# ## text row select ###
# os.system(fr"""{TOKENIZER_FALSE}python ./scripts/model_gemini/row_text.py --dataset tab_fact \
# --dataset_split test \
# --prompt_file prompts/row_text.txt \
# --n_parallel_prompts 1 \
# --input_program_file tab_fact_test_row_sql.json \
# --max_generation_tokens 512 \
# --temperature 0.7 \
# --sampling_n 2 \
# -v""")

# ## sql reason ###
# os.system(fr"""{TOKENIZER_FALSE}python ./scripts/model_gemini/reason_sql.py --dataset tab_fact \
# --dataset_split test \
# --prompt_file prompts/sql_reasoning_tabfact.txt \
# --n_parallel_prompts 1 \
# --input_program_file tab_fact_test_row_text.json \
# --max_generation_tokens 512 \
# --temperature 0.1 \
# --sampling_n 1 \
# -v""")

# ## final text reason ###
# os.system(fr"""{TOKENIZER_FALSE}python ./scripts/model_gemini/reason_text.py --dataset tab_fact \
# --dataset_split test \
# --prompt_file prompts/text_reason_tabfact.txt \
# --n_parallel_prompts 1 \
# --input_program_file tab_fact_test_sql_reason.json \
# --max_generation_tokens 256 \
# --temperature 0.0 \
# --sampling_n 1 \
# -v""")

"""***********************************************"""