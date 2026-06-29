# encoding: utf-8
"""Narval JDE-V2 repeated-run config for sweeping reid_match_max_cost."""

import importlib.util
import os


BASE_EXP_FILE = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "v2_loss_balance_dim128_rmc_nody_bs48_5runs",
        "exp_v2_lb_dim128_rmc_nody_param.py",
    )
)

spec = importlib.util.spec_from_file_location("_jde_v2_rmc_nody_base_exp", BASE_EXP_FILE)
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)


class Exp(base_module.Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.environ.get(
            "EXP_NAME",
            "yolox_x_dancetrack_jde_v2_lb_dim128_rmc_nody_narval_param",
        )
