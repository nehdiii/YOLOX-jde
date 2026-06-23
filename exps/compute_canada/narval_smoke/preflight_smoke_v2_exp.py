#!/usr/bin/env python3
from yolox.exp import get_exp

EXP_FILE = "exps/compute_canada/narval_smoke/exp_v2_lb_dim128_rmw010_smoke.py"
exp = get_exp(EXP_FILE, None)
model = exp.get_model()
h = model.head

print("=" * 100)
print("Narval smoke V2 exp preflight")
print("exp_file:", EXP_FILE)
print("exp_name:", exp.exp_name)
print("max_epoch:", getattr(exp, "max_epoch", None))
print("data_num_workers:", getattr(exp, "data_num_workers", None))
print("head:", type(h))
print("reid_dim:", getattr(h, "reid_dim", None))
print("reid_weight:", getattr(h, "reid_weight", None))
print("use_uncertainty:", getattr(h, "use_uncertainty", None))
print("has s_det:", hasattr(h, "s_det"))
print("has s_id:", hasattr(h, "s_id"))
print("reid_match_weight:", getattr(h, "reid_match_weight", None))
print("reid_match_max_cost:", getattr(h, "reid_match_max_cost", None))
print("use_reid_in_dynamic_k:", getattr(h, "use_reid_in_dynamic_k", None))

assert getattr(exp, "max_epoch", None) == 1
assert getattr(h, "reid_dim", None) == 128
assert abs(float(getattr(h, "reid_weight", -1)) - 1.0) < 1e-9
assert bool(getattr(h, "use_uncertainty", False)) is True
assert hasattr(h, "s_det")
assert hasattr(h, "s_id")
assert abs(float(getattr(h, "reid_match_weight", -1)) - 0.10) < 1e-9
assert abs(float(getattr(h, "reid_match_max_cost", -1)) - 2.0) < 1e-9
assert bool(getattr(h, "use_reid_in_dynamic_k", False)) is True

print("=" * 100)
print("NARVAL SMOKE V2 EXP PREFLIGHT OK")