#!/usr/bin/env python3
import os
from yolox.exp import get_exp
EXP_FILE = "exps/compute_canada/dancetrack_repeated_runs/v2_loss_balance_dim128_rmw_bs48_5runs_narval_narval/exp_v2_lb_dim128_rmw_param.py"
values = [0.05, 0.10, 0.20, 0.30, 0.40, 1.00]
for rmw in values:
    os.environ["REID_MATCH_WEIGHT"] = str(rmw)
    os.environ["EXP_NAME"] = f"preflight_v2_lb_dim128_rmw{int(round(rmw*100)):03d}"
    exp = get_exp(EXP_FILE, None)
    model = exp.get_model()
    h = model.head
    print("="*100)
    print("REID_MATCH_WEIGHT:", rmw)
    print("head:", type(h))
    print("reid_dim:", getattr(h, "reid_dim", None))
    print("reid_weight:", getattr(h, "reid_weight", None))
    print("use_uncertainty:", getattr(h, "use_uncertainty", None))
    print("has s_det:", hasattr(h, "s_det"))
    print("has s_id:", hasattr(h, "s_id"))
    print("reid_match_weight:", getattr(h, "reid_match_weight", None))
    print("reid_match_max_cost:", getattr(h, "reid_match_max_cost", None))
    print("use_reid_in_dynamic_k:", getattr(h, "use_reid_in_dynamic_k", None))
    assert getattr(h, "reid_dim", None) == 128
    assert abs(float(getattr(h, "reid_weight", -1)) - 1.0) < 1e-9
    assert bool(getattr(h, "use_uncertainty", False)) is True
    assert hasattr(h, "s_det")
    assert hasattr(h, "s_id")
    assert abs(float(getattr(h, "reid_match_weight", -1)) - rmw) < 1e-9
    assert abs(float(getattr(h, "reid_match_max_cost", -1)) - 2.0) < 1e-9
    assert bool(getattr(h, "use_reid_in_dynamic_k", False)) is True
print("="*100)
print("JDE-V2 LB DIM128 RMW REPEATED-RUN PREFLIGHT OK")