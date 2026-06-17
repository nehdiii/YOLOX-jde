#!/usr/bin/env python3
# encoding: utf-8
"""Preflight for JDE motion-only tracking."""

from yolox.exp import get_exp
from yolox.evaluators.mot_evaluator_dance_jde_motion import MOTEvaluatorJDEMotion

CONFIGS = [
    ("JDE V1 motion-only", "exps/example/dancetrack/yolox_x_dancetrack_jde_v1.py"),
    ("JDE V2 motion-only", "exps/example/dancetrack/yolox_x_dancetrack_jde_v2.py"),
    ("JDE V2+Deform motion-only", "exps/example/dancetrack/yolox_x_dancetrack_jde_v2_deform.py"),
]

print("Evaluator:", MOTEvaluatorJDEMotion)
for name, exp_file in CONFIGS:
    exp = get_exp(exp_file, None)
    model = exp.get_model()
    head = model.head
    print("=" * 80)
    print(name)
    print("exp_file:", exp_file)
    print("head:", type(head))
    print("reid_dim:", getattr(head, "reid_dim", None))
    print("num_ids:", getattr(head, "num_ids", None))
    print("deform_reid:", getattr(head, "deform_reid", None))
    print("tracking mode: MOTION ONLY, no JDE embedding association")
print("=" * 80)
print("JDE MOTION-ONLY PREFLIGHT OK")