from nsql.database import NeuralDB

def extract_rows(sub_table):
    if not sub_table or sub_table['header'] is None:
        return []
    answer = []
    if 'row_id' and 'index' in sub_table['header']:
        for _row in sub_table['rows']:
            dummy = []
            dummy.extend(_row[1:])
            answer.append(f'row {dummy[0]}')
        return answer
    else:
        for _row in sub_table['rows']:
            dummy = []
            dummy.extend(_row)
            answer.append(f'row {dummy[0]}')
        return answer

def extract_answers(sub_table):
    if not sub_table or sub_table['header'] is None:
        return []
    answer = []
    if 'row_id' in sub_table['header']:
        for _row in sub_table['rows']:
            answer.extend((_row[1:]))
        return answer
    else:
        for _row in sub_table['rows']:
            answer.extend((_row))
        return answer

class Executor(object):
    def __init__(self, args, keys=None):
        self.new_col_name_id = 0

    def generate_new_col_names(self, number):
        col_names = ["col_{}".format(i) for i in range(self.new_col_name_id, self.new_col_name_id + number)]
        self.new_col_name_id += number
        return col_names

    def sql_exec(self, sql: str, db: NeuralDB, verbose=True):
        if verbose:
            print("Exec SQL '{}' with additional row_id on {}".format(sql, db))
        result = db.execute_query(sql)
        return result
