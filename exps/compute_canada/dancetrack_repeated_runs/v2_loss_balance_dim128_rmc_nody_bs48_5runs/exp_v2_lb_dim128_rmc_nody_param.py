# encoding: utf-8
"""JDE-V2 repeated-run config for sweeping reid_match_max_cost.

This bundle reuses the JDE-V2 DanceTrack repeated-run config and only changes
the matching hyperparameters under study:

- swept by REID_MATCH_MAX_COST: 0.5, 1.0, 1.5, 2.5, 3.0, 3.5
- fixed reid_dim=128
- fixed reid_weight=1.0
- fixed use_uncertainty=True
- fixed reid_match_weight=0.1
- fixed use_reid_in_dynamic_k=False
"""

import importlib.util
import os


BASE_EXP_FILE = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "v2_loss_balance_dim128_rmw_bs48_5runs",
        "exp_v2_lb_dim128_rmw_param.py",
    )
)

spec = importlib.util.spec_from_file_location("_jde_v2_rmw_base_exp", BASE_EXP_FILE)
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)


class Exp(base_module.Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.environ.get(
            "EXP_NAME",
            "yolox_x_dancetrack_jde_v2_lb_dim128_rmc_nody_param",
        )

        self.reid_dim = 128
        self.reid_weight = 1.0
        self.use_uncertainty = True

        self.reid_match_weight = 0.1
        self.reid_match_max_cost = float(os.environ.get("REID_MATCH_MAX_COST", "2.0"))
        self.use_reid_in_dynamic_k = False
