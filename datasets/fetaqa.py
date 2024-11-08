import os
import datasets
from utils.wtq.utils import _load_table_w_page as _load_table
import json

_CITATION = """\
@article{nan2022fetaqa,
  title={Fetaqa\: Free-form table question answering},
  author={Nan, Linyong and Hsieh, Chiachun and Mao, Ziming and Lin, Xi Victoria and Verma, Neha and Zhang, Rui and Kry{\'s}ci{\'n}ski, Wojciech and Schoelkopf, Hailey and Kong, Riley and Tang, Xiangru and others},
  journal={Transactions of the Association for Computational Linguistics},
  volume={10},
  pages={35--49},
  year={2022},
  publisher={MIT Press One Broadway, 12th Floor, Cambridge, Massachusetts 02142, USA~…}
}
"""

_LICENSE = "CC-BY-SA-4.0 License"

_URL = "https://github.com/Yale-LILY/FeTaQA/archive/refs/heads/main.zip"

class FetaQA(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features = datasets.Features({
                "id": datasets.Value("int32"),
                "table": {
                "id": datasets.Value("string"),
                "header": datasets.features.Sequence(datasets.Value("string")),
                "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
                "page_title": datasets.Value("string"),
                },
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
            }
        ),
        supervised_keys=None,
        # homepage=_HOMEPAGE,
        license=_LICENSE,
        citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # data_dir = os.path.join(dl_manager.download_and_extract(_URL), 'FeTaQA-main')
        data_dir = 'datasets'
        
        # print(os.getcwd()) #'utils/'
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "data/fetaQA-v1_test.json"),
                            "data_dir": data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "data/fetaQA-v1_test.json"),
                            "data_dir": data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "data/fetaQA-v1_test.json"),
                            "data_dir": data_dir},
            ),

        ]

    def _generate_examples(self, filepath, data_dir):
            with open(filepath, encoding="utf-8") as f:
                lines  = json.load(f)
                for i,dic in enumerate(lines['data']):
                    feta_id = dic['feta_id']
                    caption = dic['table_page_title']
                    question = dic['question']
                    answer = dic["answer"]
                    header = dic['table_array'][0]
                    rows = dic['table_array'][1:]
                    yield f"{i}", {
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
