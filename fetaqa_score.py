from datasets import load_metric
import nltk
nltk.download('punkt')

#  the code below refers to the https://github.com/Yale-LILY/FeTaQA/blob/main/end2end/train.py
def postprocess_text(preds, labels, metric_name):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    if metric_name == "rouge":
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    elif metric_name == "sacrebleu":  # sacrebleu
        labels = [[label] for label in labels]
    elif metric_name == "bleu":
        preds = [pred.split(' ') for pred in preds]
        labels = [[label.split(' ')] for label in labels]
    else:
        pass

    return preds, labels

class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, gold_text):
        summary = {}
        # print(gold_text,preds)
        # gold_text = [item["answer"] for item in golds]

        assert len(preds) == len(gold_text)
        metric_list = ["rouge"]#, "bleu"]
        # metric_list = ["sacrebleu", "rouge", "meteor", "bertscore", "bleurt"]

        for metric_name in metric_list:
            metric = load_metric(metric_name)
            processed_preds, processed_golds = postprocess_text(preds, gold_text, metric_name)

            if metric_name == "bertscore":
                res = metric.compute(predictions=processed_preds, references=processed_golds, lang="en")
                for k, v in res.items():
                    if k == "hashcode":
                        continue
                    summary[f"{metric_name}_{k}"] = round(1.0 * sum(v) / len(v), 2)

            else:
                res = metric.compute(predictions=processed_preds, references=processed_golds)
                # print(res)
                if metric_name == "sacrebleu":
                    summary[metric_name] = res["score"] * 0.01  # limit it to range of [0, 1] for unifying
                elif metric_name == "bleurt":
                    summary["bleurt"] = round(1.0 * sum(res["scores"]) / len(res["scores"]), 2)
                elif metric_name == 'rouge':
                    for sub_metric_name in res.keys():
                        for i, key in enumerate(['precision', 'recall', 'fmeasure']):
                            summary["{}_{}".format(sub_metric_name, key)] = res[sub_metric_name][1][i]
                        # this the the fmeasure('f-score') from the mid('mean aggregation')
                else:
                    summary[metric_name] = res[metric_name]
        return summary

if __name__ == '__main__':
    import json
    import argparse
    
    test_data = []
    preds = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate")
    args = parser.parse_args()
    model_name = args.model_name

    with open(f"results/model_{model_name}/fetaqa_test_exec_results.json") as f:
        data = json.load(f)
    for i in range(0,len(data)):
        try:
            it = data[str(i)]
            ans = str(it['generations'][0].split('Answer: ')[1])
            ans = ans.replace("\n```", "")
            preds.append(ans)
            test_data.append(it['ori_data_item']['answer'])
        except:
            pass
    evaluator = EvaluateTool(args=None)
    score = evaluator.evaluate(preds, test_data)

    print(score)