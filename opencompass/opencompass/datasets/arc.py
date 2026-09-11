import json
import os.path as osp
from os import environ

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ARCDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(path,
                                     split='validation',
                                     subset_name=name)
            rows = []
            for row in dataset:
                answerKey = row['answerKey']
                question = row['question']
                choices = row['choices']
                if len(choices['text']) != 4:
                    continue
                labels = row['choices']['label']
                answerKey = 'ABCD'[labels.index(answerKey)]

                rows.append({
                    'question': question,
                    'answerKey': answerKey,
                    'textA': choices['text'][0],
                    'textB': choices['text'][1],
                    'textC': choices['text'][2],
                    'textD': choices['text'][3],
                })
        else:
            with open(path, 'r', errors='ignore') as in_f:
                rows = []
                for line in in_f:
                    item = json.loads(line.strip())
                    question = item['question']
                    if len(question['choices']) != 4:
                        continue
                    labels = [c['label'] for c in question['choices']]
                    answerKey = 'ABCD'[labels.index(item['answerKey'])]
                    rows.append({
                        'question': question['stem'],
                        'answerKey': answerKey,
                        'textA': question['choices'][0]['text'],
                        'textB': question['choices'][1]['text'],
                        'textC': question['choices'][2]['text'],
                        'textD': question['choices'][3]['text'],
                    })
        dataset = Dataset.from_list(rows)
        return dataset


@LOAD_DATASET.register_module()
class ARCDatasetAllChoices(BaseDataset):
    """ARC loader that keeps every question (3/4/5 choices alike).

    Choices are rendered into a single ``options_str`` column so the
    prompt adapts to the per-question option count; ``answerKey`` is the
    ABCDE letter at the answer's position, matching lm-eval convention.
    For 4-choice questions ``options_str`` renders byte-identical to the
    fixed ``A. {textA}\\n...D. {textD}`` template used with ARCDataset.
    """

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        letters = 'ABCDE'
        rows = []
        with open(path, 'r', errors='ignore') as in_f:
            for line in in_f:
                item = json.loads(line.strip())
                question = item['question']
                choices = question['choices']
                labels = [c['label'] for c in choices]
                rows.append({
                    'id':
                    item['id'],
                    'question':
                    question['stem'],
                    'options_str':
                    '\n'.join(f'{letters[i]}. {c["text"]}'
                              for i, c in enumerate(choices)),
                    'answerKey':
                    letters[labels.index(item['answerKey'])],
                })
        return Dataset.from_list(rows)


class ARCDatasetClean(BaseDataset):

    # load the contamination annotations of CEval from
    # https://github.com/liyucheng09/Contamination_Detector
    @staticmethod
    def load_contamination_annotations(path, split='val'):
        import requests

        assert split == 'test', 'We only have test set annotation for ARC'
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope.utils.config_ds import MS_DATASETS_CACHE
            annotation_cache_path = osp.join(
                MS_DATASETS_CACHE,
                f'ARC_c_{split}_contamination_annotations.json')
            link_of_annotations = 'https://modelscope.cn/datasets/opencompass/Contamination_Detector/resolve/master/ARC_annotations.json'  # noqa
        else:
            annotation_cache_path = osp.join(
                path, f'ARC_c_{split}_contamination_annotations.json')
            link_of_annotations = 'https://github.com/liyucheng09/Contamination_Detector/releases/download/v0.1.1rc/ARC_annotations.json'  # noqa

        if osp.exists(annotation_cache_path):
            with open(annotation_cache_path, 'r') as f:
                annotations = json.load(f)
            return annotations

        annotations = json.loads(requests.get(link_of_annotations).text)
        with open(annotation_cache_path, 'w') as f:
            json.dump(annotations, f)
        return annotations

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        annotations = ARCDatasetClean.load_contamination_annotations(
            osp.dirname(path), 'test')
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(path, split='test', subset_name=name)
            rows = []
            for row in dataset:
                answerKey = row['answerKey']
                question = row['question']
                choices = row['choices']
                if len(choices['text']) != 4:
                    continue
                labels = row['choices']['label']
                answerKey = 'ABCD'[labels.index(answerKey)]
                id_ = row['id']
                if id_ in annotations:
                    is_clean = annotations[id_][0]
                else:
                    is_clean = 'not labeled'
                rows.append({
                    'question': question,
                    'answerKey': answerKey,
                    'textA': choices['text'][0],
                    'textB': choices['text'][1],
                    'textC': choices['text'][2],
                    'textD': choices['text'][3],
                    'is_clean': is_clean,
                })
        else:
            with open(path, 'r', errors='ignore') as in_f:
                rows = []
                for line in in_f:
                    item = json.loads(line.strip())
                    id_ = item['id']
                    question = item['question']
                    if id_ in annotations:
                        is_clean = annotations[id_][0]
                    else:
                        is_clean = 'not labeled'
                    if len(question['choices']) != 4:
                        continue
                    labels = [c['label'] for c in question['choices']]
                    answerKey = 'ABCD'[labels.index(item['answerKey'])]
                    rows.append({
                        'question': question['stem'],
                        'answerKey': answerKey,
                        'textA': question['choices'][0]['text'],
                        'textB': question['choices'][1]['text'],
                        'textC': question['choices'][2]['text'],
                        'textD': question['choices'][3]['text'],
                        'is_clean': is_clean,
                    })
        return Dataset.from_list(rows)
