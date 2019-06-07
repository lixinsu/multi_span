#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import collections
from utils import write_predictions_couple_labeling

RawResult = collections.namedtuple("RawResult",
        ["unique_id", "start_logits", "end_logits"])
eval_examples, eval_features, all_results = pickle.load(open(sys.argv[1],'rb'))

write_predictions_couple_labeling(eval_examples, eval_features, all_results,
        20, 30, True, 'output.json', 'output_nbest.json', None, False, False, 0)
