#!/usr/bin/env python3
import os
from yolox.exp import get_exp

EXP_FILE = "exps/compute_canada/dancetrack_ablations/v2_stage1_no_lb_rw025_dim128_compact/exp_v2_stage1_no_lb_rw025_dim128_param.py"
for rmw in [0.00, 0.05, 0.10, 0.20, 0.30]:
    os.environ["REID_MATCH_WEIGHT"] = str(rmw)
    os.environ["EXP_NAME"] = f"preflight_v2_no_lb_rmw{int(round(rmw*100)):03d}"
    exp = get_exp(EXP_FILE, None)
    model = exp.get_model()
    h = model.head
    print("=" * 90)
    print("rmw:", rmw)
    print("head:", type(h))
    print("reid_dim:", getattr(h, "reid_dim", None))
    print("reid_weight:", getattr(h, "reid_weight", None))
    print("use_uncertainty:", getattr(h, "use_uncertainty", None))
    print("reid_match_weight:", getattr(h, "reid_match_weight", None))
    print("reid_match_max_cost:", getattr(h, "reid_match_max_cost", None))
    print("use_reid_in_dynamic_k:", getattr(h, "use_reid_in_dynamic_k", None))
    assert getattr(h, "reid_dim", None) == 128
    assert abs(float(getattr(h, "reid_weight", -1)) - 0.25) < 1e-9
    assert bool(getattr(h, "use_uncertainty", True)) is False
    assert abs(float(getattr(h, "reid_match_weight", -1)) - rmw) < 1e-9
    assert abs(float(getattr(h, "reid_match_max_cost", -1)) - 2.0) < 1e-9
    assert bool(getattr(h, "use_reid_in_dynamic_k", False)) is True
print("=" * 90)
print("COMPACT JDE V2 NO-LB STAGE 1 PREFLIGHT OK")