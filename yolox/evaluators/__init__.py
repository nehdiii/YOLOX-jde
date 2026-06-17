#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Safe evaluator imports for detector/JDE training.

Detector-only training only needs COCOEvaluator.
MOT/DanceTrack evaluators can import trackers, FastReID, and faiss.
We keep those imports optional so detector training does not crash when faiss is absent.
"""

from .coco_evaluator import COCOEvaluator

try:
    from .voc_evaluator import VOCEvaluator
except Exception:
    VOCEvaluator = None

try:
    from .mot_evaluator import MOTEvaluator
except Exception:
    MOTEvaluator = None

try:
    from .mot_evaluator_dance import MOTEvaluator as MOTEvaluatorDance
except Exception:
    MOTEvaluatorDance = None

try:
    from .mot_evaluator_public import MOTEvaluatorPublic
except Exception:
    MOTEvaluatorPublic = None

try:
    from .mot_evaluator_dance_jde_v1 import MOTEvaluator as MOTEvaluatorDanceJDEV1
except Exception:
    MOTEvaluatorDanceJDEV1 = None

try:
    from .mot_evaluator_dance_jde_v1 import MOTEvaluatorJDE
except Exception:
    MOTEvaluatorJDE = None