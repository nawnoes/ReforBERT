import os
import random
import logging

import torch
import numpy as np

from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from finetuning.squad_model import ReforBertForQA
from finetuning.tokenization_reforbert import ReforBertTokenizer
from finetuning.config_reforBert import ReforBertConfig


CONFIG_CLASSES = {
    # "kobert": BertConfig,
    "reforbert": ReforBertConfig,
}

TOKENIZER_CLASSES = {
    # "kobert": KoBertTokenizer,
    "reforbert": ReforBertTokenizer,
}

# MODEL_FOR_SEQUENCE_CLASSIFICATION = {
#     "kobert": BertForSequenceClassification,
#     "reforbert": BertForSequenceClassification,
# }
#
# MODEL_FOR_TOKEN_CLASSIFICATION = {
#     "kobert": BertForTokenClassification,
#     "reforbert": BertForTokenClassification,
# }

MODEL_FOR_QUESTION_ANSWERING = {
    # "kobert": BertForQuestionAnswering,
    "reforbert": ReforBertForQA,
}


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }


def pearson_and_spearman(labels, preds):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def f1_pre_rec(labels, preds):
    return {
        "precision": precision_score(labels, preds, suffix=True),
        "recall": recall_score(labels, preds, suffix=True),
        "f1": f1_score(labels, preds, suffix=True)
    }


def show_ner_report(labels, preds):
    return classification_report(labels, preds, suffix=True)


def compute_metrics(task_name, labels, preds):
    assert len(preds) == len(labels)
    if task_name == "kornli":
        return acc_score(labels, preds)
    elif task_name == "nsmc":
        return acc_score(labels, preds)
    elif task_name == "paws":
        return acc_score(labels, preds)
    elif task_name == "korsts":
        return pearson_and_spearman(labels, preds)
    elif task_name == "question-pair":
        return acc_score(labels, preds)
    elif task_name == 'naver-ner':
        return f1_pre_rec(labels, preds)
    else:
        raise KeyError(task_name)
