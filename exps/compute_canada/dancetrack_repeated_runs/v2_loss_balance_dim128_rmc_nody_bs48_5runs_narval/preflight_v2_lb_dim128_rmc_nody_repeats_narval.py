#!/usr/bin/env python3
import os

from yolox.exp import get_exp


EXP_FILE = "exps/compute_canada/dancetrack_repeated_runs/v2_loss_balance_dim128_rmc_nody_bs48_5runs_narval/exp_v2_lb_dim128_rmc_nody_param.py"
values = [0.5, 1.0, 1.5, 2.5, 3.0, 3.5]

for rmc in values:
    os.environ["REID_MATCH_MAX_COST"] = str(rmc)
    os.environ["EXP_NAME"] = f"preflight_v2_lb_dim128_rmw010_rmc{int(round(rmc * 100)):03d}_nody_narval"
    exp = get_exp(EXP_FILE, None)
    model = exp.get_model()
    h = model.head
    print("=" * 100)
    print("REID_MATCH_MAX_COST:", rmc)
    print("head:", type(h))
    print("reid_dim:", getattr(h, "reid_dim", None))
    print("reid_weight:", getattr(h, "reid_weight", None))
    print("use_uncertainty:", getattr(h, "use_uncertainty", None))
    print("reid_match_weight:", getattr(h, "reid_match_weight", None))
    print("reid_match_max_cost:", getattr(h, "reid_match_max_cost", None))
    print("use_reid_in_dynamic_k:", getattr(h, "use_reid_in_dynamic_k", None))
    assert getattr(h, "reid_dim", None) == 128
    assert abs(float(getattr(h, "reid_weight", -1)) - 1.0) < 1e-9
    assert bool(getattr(h, "use_uncertainty", False)) is True
    assert abs(float(getattr(h, "reid_match_weight", -1)) - 0.1) < 1e-9
    assert abs(float(getattr(h, "reid_match_max_cost", -1)) - rmc) < 1e-9
    assert bool(getattr(h, "use_reid_in_dynamic_k", True)) is False

print("=" * 100)
print("NARVAL JDE-V2 LB DIM128 RMC NODY REPEATED-RUN PREFLIGHT OK")
