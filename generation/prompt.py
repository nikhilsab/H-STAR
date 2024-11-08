"""
Build prompt for LLMs.
"""

import random
from typing import Dict, Tuple
import pandas as pd
from utils.errors import DuplicateColumnsError
from utils.normalizer import prepare_df_for_neuraldb_from_table


def _create_table_prompt(df: pd.DataFrame, title: str):
    """
    Return the CREATE TABLE clause as prompt.
    """
    string = "CREATE TABLE {}(\n".format(title)
    for header in df.columns:
        column_type = 'text'
        try:
            if df[header].dtype == 'int64':
                column_type = 'int'
            elif df[header].dtype == 'float64':
                column_type = 'real'
            elif df[header].dtype == 'datetime64':
                column_type = 'datetime'
        except AttributeError as e:
            raise DuplicateColumnsError(e)

        string += '\t{} {},\n'.format(header, column_type)
    string = string.rstrip(',\n') + ')\n'
    return string


class PromptBuilder(object):
    def __init__(self, args):
        self.args = args
        self.prompt_style = args.prompt_style
        random.seed(args.seed)

    def _select_x_prompt(self, df: pd.DataFrame, num_rows: int,
                         few_shot_demonstration=True):
        """
        Return the first X rows table contents as prompt.
        """
        # Full table in text format with pipe separators
        if self.prompt_style == 'text_full_table':
            string = '/*\n'
            col_list = df.columns.values.tolist()
            string += 'col : ' + ' | '.join(df.columns) + '\n'
            for row_id, row in df.iloc[:num_rows].iterrows():
                string += f'row {row_id} : '
                for column_id, header in enumerate(df.columns):
                    string += str(row[header])
                    if column_id != len(df.columns) - 1:
                        string += ' | '
                string += '\n'
            string += '*/\n'
            string += f'columns:{col_list}\n'
            return string

        elif self.prompt_style == 'create_table_select_full_table':
            string = '/*\nAll rows of the table:\nSELECT * FROM w;\n'
        elif self.prompt_style == 'create_table_select_3':
            string = '/*\nAll rows of the table:\nSELECT * FROM w;\n'

        # Transposed table format (rows become columns)
        elif self.prompt_style == 'transpose':
            col_list = df.columns.values.tolist()
            df = df.T
            string = '/*\n'   
            string+= 'row : '
            
            # Add header row with row numbers (limited to 15 rows)
            for idx,i in enumerate(df.columns.tolist()):
                if idx == 15 or idx == len(df.columns):
                    break
                string += 'row {}'.format(i)
                if idx != 14 and idx != len(df.columns)-1:
                    string += ' | '
            string += '\n'
            for row_id, row in df.iloc[:14].iterrows():
                string += f'{row_id} : '
                for column_id, header in enumerate(df.columns):
                    string += str(row[header])
                    if column_id == 14 or column_id == len(df.columns)-1:
                        break
                    if column_id != 15 or column_id != len(df.columns)-1:
                        string += ' | '
                string += '\n'
            string += '*/\n'
            string += f'columns:{col_list}\n'
            return string

        # SQL-like formats with different headers
        elif self.prompt_style == 'create_table_select_3_hidden':
            string = '/*\n{} example rows:\n'.format(num_rows)

        elif few_shot_demonstration is True and self.prompt_style in \
                ["create_table_select_3_full_table"]:
            string = '/*\n{} example rows:\nSELECT * FROM w LIMIT {};\n'.format(num_rows, num_rows)

        elif few_shot_demonstration is False and self.prompt_style in \
                ["create_table_select_3_full_table"]:
            string = '/*\n'
        
        else:
            raise ValueError(f"Select x prompt style {self.prompt_style} is not supported.")

        for column_id, header in enumerate(df.columns):
            string += str(header)
            if column_id != len(df.columns) - 1:
                string += '\t'
        string += '\n'
        for row_id, row in df.iloc[:num_rows].iterrows():
            for column_id, header in enumerate(df.columns):
                string += str(row[header])
                if column_id != len(df.columns) - 1:
                    string += '\t'
            string += '\n'
        string += '*/\n'
        col_list = df.columns.values.tolist()
        string += f'columns:{col_list}\n'

        return string

    def build_one_shot_prompt(
            self,
            prompt_type: Tuple,
            table: pd.DataFrame,
            question: str,
            answer_text: str,
            nsql: str,
            passages: Dict = None,
            images: Dict = None,
            title: str = None,
            only_title: bool = False,
            **kwargs
    ):
        """
        Build one-shot prompt with table-question-nsql.
        """
        one_shot_prompt = ""
        if self.prompt_style == 'create_table_select_full_table':
            one_shot_prompt += _create_table_prompt(table, title)
            one_shot_prompt += self._select_x_prompt(
                df=table,
                num_rows=table.shape[0]
            )
        elif self.prompt_style in ['create_table_select_3_full_table', 'create_table_select_3']:
            one_shot_prompt += _create_table_prompt(table, title)
            one_shot_prompt += self._select_x_prompt(
                df=table,
                num_rows=3,
            )
        elif self.prompt_style == 'create_table':
            one_shot_prompt += _create_table_prompt(table, title)
        elif self.prompt_style == 'no_table':
            pass
        else:
            raise ValueError('{} is not supported.'.format(self.prompt_style))

        # question and nsql pairs
        if prompt_type == ('question', 'nsql'):
            one_shot_prompt += 'Q: {}\n'.format(question)
            one_shot_prompt += 'NeuralSQL: {}\n'.format(nsql)
        elif prompt_type == ('question', 'sql'):
            one_shot_prompt += 'Q: {}\n'.format(question)
            one_shot_prompt += 'SQL: {}\n'.format(nsql)
        elif prompt_type == ('question', 'answer'):
            one_shot_prompt += 'Q: {}\n'.format(question)
            one_shot_prompt += 'A: {}\n'.format(', '.join(answer_text))
        else:
            raise ValueError(f'Prompt type {prompt_type} is not supported.')

        return one_shot_prompt

    def build_generate_prompt(
            self,
            generate_type: Tuple,
            table: pd.DataFrame,
            question: str = None,
            title: str = None,
            # sub_title: str = None,
            supporting_context: Dict = None,
            **kwargs
    ):
        """
        Build the prompt of the generation sample.
        """
        generate_prompt = ""

        # SECTION 1: Set initial prompt based on generation type
        # Different prompt templates for answer, column, row, or verification tasks
        if generate_type == ('answer',):
            generate_prompt += """\n-- Answer the question based on the given table below.\n\n"""
        elif generate_type == ('col',):
            generate_prompt += """Here is a new table with its corresponding statement:\n<input>\n"""
            if title is not None:
                generate_prompt += f"""table caption: {title}\n"""
            # if sub_title is not None:
            #     generate_prompt+= f"""sub title: {sub_title}\n"""

        elif generate_type == ('row',):
            generate_prompt += """Here is a new table with its corresponding question:\n<input>\n"""
            if title is not None:
                generate_prompt += f"""table caption: {title}\n"""

        elif generate_type == ('verification',):
            generate_prompt += """Here is a new table with its corresponding question:\n<input>\n"""
            if title is not None:
                generate_prompt += f"""table caption: {title}\n"""

        else:
            generate_prompt += """\n-- Generate NeuralSQL and question pairs based on the given table below.\n\n"""

        # SECTION 2: Handle table formatting based on prompt_style
        # Three main styles: create_table_select_full_table, text_full_table, create_table_select_3
        if self.prompt_style in ['create_table_select_full_table', 'create_table_select_3_full_table', 'transpose']:
            if self.prompt_style in ['create_table_select_full_table', 'create_table_select_3_full_table']:
                generate_prompt += _create_table_prompt(table, title)
            if 'num_rows' in kwargs.keys():
                num_rows = kwargs['num_rows']
                generate_prompt += self._select_x_prompt(
                df=table,
                num_rows=num_rows,
                few_shot_demonstration=False
            )
            else:
                generate_prompt += self._select_x_prompt(
                    df=table,
                    num_rows=table.shape[0],
                    few_shot_demonstration=False
                )

        elif self.prompt_style in ['text_full_table', 'transpose']:
            if 'num_rows' in kwargs.keys():
                num_rows = kwargs['num_rows']
                generate_prompt += self._select_x_prompt(
                df=table,
                num_rows=num_rows,
                few_shot_demonstration=False
            )
            else:
                generate_prompt += self._select_x_prompt(
                    df=table,
                    num_rows=table.shape[0],
                    few_shot_demonstration=False
                )

        elif self.prompt_style in ['create_table_select_3']:
            generate_prompt += _create_table_prompt(table, title)
            generate_prompt += self._select_x_prompt(
                df=table,
                num_rows=3,
                few_shot_demonstration=False
            )
        else:
            raise ValueError('{} is not supported.'.format(self.prompt_style))

        # SECTION 3: Add final prompt elements based on generation type
        # Appends appropriate output markers and formatting for each task type
        if generate_type == ('answer',):
            generate_prompt += 'statement: {}\n'.format(question)
            generate_prompt += 'A: '
        
        elif generate_type == ('col',):
            generate_prompt += 'statement: {}\n'.format(question)
            generate_prompt += '<output>:\n'
        
        elif generate_type == ('sql',):
            generate_prompt += 'Q: {}\n'.format(question)
            generate_prompt += 'SQL: '
        
        elif generate_type == ('row',):
            generate_prompt += 'statement: {}\n'.format(question)
            generate_prompt += '<output>\n'

        elif generate_type == ('verification',):
            generate_prompt += 'statement: {}\n'.format(question)
            generate_prompt += '<initial response>\n'
            generate_prompt += '<output>\n'
        else:
            raise ValueError(f'Generate type {generate_type} is not supported.')

        return generate_prompt