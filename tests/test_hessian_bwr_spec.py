"""BWR 解析特化（二阶差分旁路）对照测试。

computeCustomHessianKernel 对符合资格的 BWR（param_count==2, q0 链子质量与 θ
无关）走 9 点中心差分二阶（bwrSpecEval），替代 ~1.7 万条字节码展开（实测 UH
phsp 段提速 ~16×）。本测试保证特化与字节码数值一致:
  1. simple（BWR）: CTPWA_NO_BWR_SPEC=1（字节码） vs 默认（特化）hessian 逐元素
     相对差 < 1e-4（实测 ~2e-6, 差分截断误差）;
  2. Flatte/Interp 模型不受影响（无特化资格 → 两模式 hessian 完全一致）。
"""
import os

import numpy as np
import pytest
import torch

from conftest import make_params


def _hess(ana_key, use_bytecode: bool):
    import ctpwa
    if use_bytecode:
        os.environ["CTPWA_NO_BWR_SPEC"] = "1"
    else:
        os.environ.pop("CTPWA_NO_BWR_SPEC", None)
    ana = ctpwa.analysis(f"configs/{ana_key}.yml")
    p = make_params(ana, torch.device("cuda:0")).detach()
    return ana.getHessian(p).cpu().numpy()


@pytest.mark.parametrize("key", ["simple", "flatte"])
def test_bwr_spec_matches_bytecode(key):
    h_bc = _hess(key, True)
    h_sp = _hess(key, False)
    denom = max(float(np.abs(h_bc).max()), 1e-30)
    rel = float(np.abs(h_bc - h_sp).max()) / denom
    if key == "simple":
        # BWR 走特化: 允许差分截断误差（实测 ~2e-6）
        assert rel < 1e-4, f"{key}: 特化 vs 字节码 rel={rel:.2e}"
    else:
        # Flatte 无特化资格 → 应完全一致
        assert rel < 1e-12, f"{key}: 非特化模型被误伤 rel={rel:.2e}"
