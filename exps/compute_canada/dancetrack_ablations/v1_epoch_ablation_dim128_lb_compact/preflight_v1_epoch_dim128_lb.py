#!/usr/bin/env python3
# encoding: utf-8
import os
from yolox.exp import get_exp
EXP_FILE = "exps/compute_canada/dancetrack_ablations/v1_epoch_ablation_dim128_lb_compact/exp_v1_epoch_dim128_lb_param.py"
for ep in [12, 16, 20, 25, 30]:
    os.environ["MAX_EPOCH"] = str(ep)
    os.environ["EPOCH_NAME"] = f"{ep:03d}"
    os.environ["EXP_NAME"] = f"preflight_v1_epoch{ep:03d}_dim128_lb"
    exp = get_exp(EXP_FILE, None)
    model = exp.get_model()
    h = model.head
    print("=" * 100)
    print("MAX_EPOCH:", ep)
    print("exp.max_epoch:", getattr(exp, "max_epoch", None))
    print("exp.no_aug_epochs:", getattr(exp, "no_aug_epochs", None))
    print("reid_dim:", getattr(h, "reid_dim", None))
    print("reid_weight:", getattr(h, "reid_weight", None))
    print("use_uncertainty:", getattr(h, "use_uncertainty", None))
    print("has s_det:", hasattr(h, "s_det"))
    print("has s_id:", hasattr(h, "s_id"))
    assert getattr(exp, "max_epoch", None) == ep
    assert getattr(exp, "no_aug_epochs", None) == 1
    assert getattr(h, "reid_dim", None) == 128
    assert bool(getattr(h, "use_uncertainty", False)) is True
    assert hasattr(h, "s_det")
    assert hasattr(h, "s_id")
print("=" * 100)
print("JDE V1 EPOCH ABLATION PREFLIGHT OK")