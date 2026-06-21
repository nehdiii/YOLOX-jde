#!/usr/bin/env python3
import os
from yolox.exp import get_exp

EXP_FILE = "exps/compute_canada/dancetrack_repeated_runs/v1_loss_balance_dim_bs48_5runs/exp_v1_loss_balance_dim_param.py"
for dim in [64, 128, 256, 512]:
    os.environ["REID_DIM"] = str(dim)
    os.environ["EXP_NAME"] = f"preflight_v1_lb_dim{dim}"
    exp = get_exp(EXP_FILE, None)
    model = exp.get_model()
    h = model.head
    print("="*90)
    print("dim:", dim)
    print("head:", type(h))
    print("reid_dim:", getattr(h, "reid_dim", None))
    print("reid_weight:", getattr(h, "reid_weight", None))
    print("use_uncertainty:", getattr(h, "use_uncertainty", None))
    print("has s_det:", hasattr(h, "s_det"))
    print("has s_id:", hasattr(h, "s_id"))
    assert getattr(h, "reid_dim", None) == dim
    assert abs(float(getattr(h, "reid_weight", -1)) - 1.0) < 1e-9
    assert bool(getattr(h, "use_uncertainty", False)) is True
    assert hasattr(h, "s_det")
    assert hasattr(h, "s_id")
print("="*90)
print("V1 LOSS-BALANCE DIM REPEATED-RUN PREFLIGHT OK")