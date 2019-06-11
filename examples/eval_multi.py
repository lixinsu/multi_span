#!/usr/bin/env python
# coding: utf-8

import json
import sys
import os
import argparse
import statistics
from collections import defaultdict
import ipdb


def load_prediction_file(filename):
    return json.load(open(filename))


def load_reference_file(filename):
    records = [json.loads(line) for line in open(filename)]
    print('{} samples in dataset'.format(len(records)))
    return {datum['query_id']:datum['answers'] for datum in records}


def lengths(gts):
    return sum([len(x) for x in gts]) * 1.0 / len(gts)



def exact_match(qid2answers, qid2preditions):
    qid2em = defaultdict(dict)
    for qid, answers in qid2answers.items():
        if qid not in qid2predictions:
            print('{} is not answered'.format(qid))
            continue
        predictions = qid2predictions[qid]
        if isinstance( predictions, str ):
            predictions = [predictions ]
        if len(answers) != len(predictions):
            qid2em[qid]['em'] = 0.0
            continue
        if isinstance(predictions[0], dict):
            predictions = [pred['text'] for pred in predictions]
        if set(answers) == set(predictions):
            qid2em[qid]['em'] = 1.0
        else:
            qid2em[qid]['em'] = 0.0
    return 100*statistics.mean([x['em'] for x in qid2em.values()]), qid2em


def _f1(preds, answers):
    if len(preds) == 0:
        return 0.0
    tp = len(set(preds) & set(answers))
    if tp == 0:
        return 0.0
    p = float(tp) / len(preds)
    r = float(tp) / len(answers)
    return 2*p*r/(p+r)


def f_measure(qid2answer, qid2prediction):
    qid2f1 = defaultdict(dict)
    for qid, answers in qid2answers.items():
        if qid not in qid2predictions:
            print('{} is not answered'.format(qid))
            continue
        predictions = qid2predictions[qid]
        if isinstance(predictions[0], dict):
            predictions = [pred['text'] for pred in predictions]
        if isinstance( predictions, str ):
            predictions = [predictions ]
        qid2f1[qid]['f1'] = _f1(predictions, answers)
    return 100*statistics.mean([x['f1'] for x in qid2f1.values()]), qid2f1


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('dataset', default=None)
   parser.add_argument('pred', default=None)
   args = parser.parse_args()
   qid2answers = load_reference_file(args.dataset)
   qid2predictions = load_prediction_file(args.pred)
   print('{} samples in prediction'.format(len(qid2predictions)))
   em, qid2em = exact_match(qid2answers, qid2predictions)
   f1, qid2f1 = f_measure(qid2answers, qid2predictions)
   print('em: {:.2f}\nf1: {:.2f}'.format(em, f1))
