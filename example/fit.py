import torch
import numpy as np
import time
import os
import sys
import argparse
import logging
import ctpwa

log = logging.getLogger("pwa-fit")

# ============================================================
# 初始化分析对象
# ============================================================
int_time1 = int(time.time())
ana = ctpwa.analysis()
int_time2 = int(time.time())
print(f"振幅初始化耗时: {int_time2 - int_time1} 秒")

# 参数信息
conjugate_pairs = ana.getConstraintsIndex()
params_names = ana.getParamNames()          # 前 n_coupling_free 个耦合名 + 后 n_res_free 个共振态名
n_coupling_free = ana.getNVector()          # 自由耦合复数参数数

# 共振态参数信息
free_res_info = ana.getFreeResParams()      # [3, n_res] float64 CPU
n_res_free = free_res_info.shape[1]
HAS_FREE_RES = n_res_free > 0

n_params_total = 2 * n_coupling_free + n_res_free
print(f"耦合参数数量: {n_coupling_free}")
print(f"共振态参数数量: {n_res_free}")

# ============================================================
# 生成初始参数（已移入 UnifiedPWAOptimizer.generate_initial_params）
# 保留模块级函数作为向后兼容包装器
# ============================================================
def generate_initial_params(n_coupling_free, free_res_info,
                            seed=42, device="cuda"):
    """向后兼容包装器。新代码请用 optimizer.generate_initial_params(seed)。"""
    import warnings
    warnings.warn(
        "generate_initial_params() 已移入 UnifiedPWAOptimizer 类方法，"
        "请直接使用 optimizer.generate_initial_params(seed=...)",
        DeprecationWarning, stacklevel=2,
    )
    n_res = free_res_info.shape[1]
    n_total = 2 * n_coupling_free + n_res
    params = torch.zeros(n_total, dtype=torch.float64, device=device)
    torch.manual_seed(seed)
    params[0] = 1.0
    for idx in range(1, n_coupling_free):
        amplitude = torch.rand(1, device=device).item() * 0.5
        phase = torch.rand(1, device=device).item() * 2 * torch.pi
        params[idx] = amplitude * np.cos(phase)
    params[n_coupling_free] = 0.0
    for idx in range(1, n_coupling_free):
        amplitude = torch.rand(1, device=device).item() * 0.5
        phase = torch.rand(1, device=device).item() * 2 * torch.pi
        params[n_coupling_free + idx] = amplitude * np.sin(phase)
    if n_res > 0:
        init_vals = free_res_info[0].to(device=device, dtype=torch.float64)
        if seed > 42:
            torch.manual_seed(seed)
            lower = free_res_info[1].to(device=device, dtype=torch.float64)
            upper = free_res_info[2].to(device=device, dtype=torch.float64)
            noise = (torch.rand(n_res, device=device, dtype=torch.float64) - 0.5) \
                    * 0.1 * (upper - lower)
            init_vals = torch.clamp(init_vals + noise,
                                    lower + 1e-7 * (upper - lower),
                                    upper - 1e-7 * (upper - lower))
        params[2 * n_coupling_free:] = init_vals
    return params


# ============================================================
# 构建自由耦合参数 -> 振幅下标的映射
# ============================================================
# ============================================================
# 优化器
# ============================================================
class UnifiedPWAOptimizer:
    def __init__(self, ana, free_res_info, params_names,
                 v_max=None, project_grad=None):
        self.analysis = ana
        self.params_names = params_names
        self.device = "cuda"
        self.best_nll = float("inf")
        self.best_params = None
        self.all_results = []

        self.n_coupling_free = ana.getNVector()
        self.n_res_free = free_res_info.shape[1]
        self.n_params = 2 * self.n_coupling_free + self.n_res_free
        self.has_free_res = self.n_res_free > 0
        # 保存 free_res_info 供 generate_initial_params 使用
        self._free_res_info = free_res_info
        # 耦合幅度上界（防 LBFGS 放飞，实测随机初值放开共振态参数时
        # 耦合会无界增长到 1e14 → 振幅溢出）+ 投影梯度（防边界伪收敛）。
        # ⚠️ 默认 10000 而不能更小: 不同波的振幅归一化差异极大
        # （ONE 模型/弱归一化波如 PHSP 需要 |v|~300-1000，实测本模型
        # 最佳解 |A| 达 297；v_max=50 会把它们掐死在墙上，模型形状被
        # 强制扭曲 → 拟合直接失败/正 NLL）。
        # 优先级: 构造参数 > FIT_VMAX/FIT_PROJECT 环境变量 > 默认值
        self.v_max = v_max if v_max is not None else float(os.environ.get("FIT_VMAX", "10000.0"))
        _pg = project_grad if project_grad is not None else os.environ.get("FIT_PROJECT", "1")
        self.project_grad = _pg if isinstance(_pg, bool) else str(_pg) == "1"
        # 统一 Hessian 缓存: 同参数点只在第一次真正计算一步 getHessian，
        # 后续（正定性判定/参数误差/分支比误差）直接复用。
        self._hess_cache = None  # (params.clone(), hessian_full)

        # 共振态参数bounds (GPU)
        if self.has_free_res:
            self._lower = free_res_info[1].to(dtype=torch.float64, device=self.device)
            self._upper = free_res_info[2].to(dtype=torch.float64, device=self.device)

    # --------------------------------------------------------
    def generate_initial_params(self, seed=42):
        """生成统一参数向量 [real_coupling | imag_coupling | theta]。

        Layout: [real_0,...,real_{n_free-1}, imag_0,...,imag_{n_free-1},
                 theta_0,...,theta_{n_res-1}]
        real_0=1.0, imag_0=0.0 为固定参考振幅。
        seed=42 时共振态参数取 PDG 初值；seed>42 时在 bounds 内加 10% 噪声。
        """
        free_res_info = self._free_res_info
        n_coupling_free = self.n_coupling_free
        device = self.device

        n_res = free_res_info.shape[1]
        n_total = 2 * n_coupling_free + n_res
        params = torch.zeros(n_total, dtype=torch.float64, device=device)

        torch.manual_seed(seed)

        # 耦合实部
        params[0] = 1.0  # 固定参考
        for idx in range(1, n_coupling_free):
            amplitude = torch.rand(1, device=device).item() * 0.5
            phase = torch.rand(1, device=device).item() * 2 * torch.pi
            params[idx] = amplitude * np.cos(phase)

        # 耦合虚部
        params[n_coupling_free] = 0.0  # 固定参考
        for idx in range(1, n_coupling_free):
            amplitude = torch.rand(1, device=device).item() * 0.5
            phase = torch.rand(1, device=device).item() * 2 * torch.pi
            params[n_coupling_free + idx] = amplitude * np.sin(phase)

        # 共振态参数
        if n_res > 0:
            init_vals = free_res_info[0].to(device=device, dtype=torch.float64)
            if seed > 42:
                torch.manual_seed(seed)
                lower = free_res_info[1].to(device=device, dtype=torch.float64)
                upper = free_res_info[2].to(device=device, dtype=torch.float64)
                noise = (torch.rand(n_res, device=device, dtype=torch.float64) - 0.5) \
                        * 0.1 * (upper - lower)
                init_vals = torch.clamp(init_vals + noise,
                                        lower + 1e-7 * (upper - lower),
                                        upper - 1e-7 * (upper - lower))
            params[2 * n_coupling_free:] = init_vals

        print(f"生成初始参数 (seed={seed}): n_coupling={n_coupling_free}, n_res={n_res}")
        return params

    # --------------------------------------------------------
    def compute_loss_and_grad(self, params):
        """计算 NLL 和梯度。params: float64, [n_params]"""
        nc = self.n_coupling_free
        # 固定第一个耦合参数 (1+0j) + 耦合 clamp
        with torch.no_grad():
            params.data[0] = 1.0
            params.data[nc] = 0.0
            params.data[1:nc].clamp_(-self.v_max, self.v_max)
            params.data[nc + 1:2 * nc].clamp_(-self.v_max, self.v_max)

        # 共振态参数有界约束: clamp
        if self.has_free_res:
            with torch.no_grad():
                start = 2 * nc
                params.data[start:] = torch.clamp(
                    params.data[start:], self._lower, self._upper
                )

        nll = self.analysis.getNLL(params)
        grad = torch.autograd.grad(nll, params, retain_graph=False)[0]

        # 固定参数的梯度清零
        with torch.no_grad():
            grad[0] = 0.0
            grad[nc] = 0.0
            # 投影梯度: clamp 边界处指向边界外的梯度置零。
            # 否则 LBFGS 在边界处"参数不动、梯度非零" → tolerance_change
            # 伪收敛（实测停在正 NLL 的垃圾点）。FIT_PROJECT=0 可关闭。
            if self.project_grad:
                g_c = grad[1:nc]
                c = params[1:nc]
                g_c[(c <= -self.v_max) & (g_c < 0)] = 0.0
                g_c[(c >= self.v_max) & (g_c > 0)] = 0.0
                g_i = grad[nc + 1:2 * nc]
                ci = params[nc + 1:2 * nc]
                g_i[(ci <= -self.v_max) & (g_i < 0)] = 0.0
                g_i[(ci >= self.v_max) & (g_i > 0)] = 0.0
                if self.has_free_res:
                    res_start = 2 * nc
                    g_r = grad[res_start:]
                    phys = params[res_start:]
                    g_r[(phys <= self._lower) & (g_r < 0)] = 0.0
                    g_r[(phys >= self._upper) & (g_r > 0)] = 0.0

        return nll, grad

    # --------------------------------------------------------
    def optimize_single_run(
        self,
        initial_params,
        run_id=0,
        max_iter=500,
        lr=1.0,
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        history_size=100,
    ):
        """单次优化"""
        params = initial_params.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [params],
            lr=lr,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn="strong_wolfe",
        )

        nll_history = []

        def closure():
            optimizer.zero_grad()
            nll, grad = self.compute_loss_and_grad(params)
            params.grad = grad
            nll_history.append(nll.item())
            return nll

        start_time = time.time()
        optimizer.step(closure)
        end_time = time.time()

        # 最终clamp一次确保共振态参数在界内 + 耦合在 ±v_max 内
        with torch.no_grad():
            params.data[0] = 1.0
            params.data[self.n_coupling_free] = 0.0
            params.data[1:self.n_coupling_free].clamp_(-self.v_max, self.v_max)
            params.data[self.n_coupling_free + 1:2 * self.n_coupling_free].clamp_(
                -self.v_max, self.v_max)
            if self.has_free_res:
                start = 2 * self.n_coupling_free
                params.data[start:] = torch.clamp(params.data[start:], self._lower, self._upper)

        final_nll = nll_history[-1] if nll_history else float("inf")
        final_params = params.clone().detach()

        # Hessian
        hessian_start = time.time()
        hessian_full = self.analysis.getHessian(final_params)
        # print("hessian矩阵：", hessian_full)
        hessian_time = time.time() - hessian_start
        # 播种缓存: 同一最佳点后续要 Hessian（正定性/误差/分支比）直接复用
        self._hess_cache = (final_params.clone(), hessian_full)

        # 去除固定参数 (index 0 和 n_coupling_free)
        fixed_mask = torch.ones(self.n_params, dtype=torch.bool, device=self.device)
        fixed_mask[0] = False
        fixed_mask[self.n_coupling_free] = False
        hessian = hessian_full[fixed_mask][:, fixed_mask]

        # 特征值分析
        try:
            eigenvalues = torch.linalg.eigvalsh(hessian)
            # print("Hessian特征值：", eigenvalues)
            is_pos_def = bool(torch.all(eigenvalues > 0).item())
            min_eig = eigenvalues[0].item()
            max_eig = eigenvalues[-1].item()
            cond_num = max_eig / min_eig if min_eig > 0 else float("inf")
        except Exception:
            is_pos_def = False
            min_eig = max_eig = cond_num = float("nan")

        # 参数误差
        coupling_real_errors = None
        coupling_imag_errors = None
        res_errors = None
        if is_pos_def:
            try:
                covariance = torch.linalg.inv(hessian)
                std_dev = torch.sqrt(torch.diag(covariance))

                # 耦合参数误差: 前 2*(n_coupling_free-1) 个元素
                n_c_var = self.n_coupling_free - 1  # 扣除固定的
                coupling_real_errors = torch.zeros(
                    self.n_coupling_free, dtype=torch.float32, device=self.device
                )
                coupling_imag_errors = torch.zeros(
                    self.n_coupling_free, dtype=torch.float32, device=self.device
                )

                # std_dev 前 2*n_c_var 个元素: 实部误差和虚部误差交替
                for i in range(n_c_var):
                    coupling_real_errors[i + 1] = std_dev[2 * i].float()
                    coupling_imag_errors[i + 1] = std_dev[2 * i + 1].float()

                # 共振态参数误差
                if self.has_free_res:
                    res_start = 2 * n_c_var
                    res_errors = std_dev[res_start:].float()
            except Exception as e:
                log.error(f"计算参数误差时出错: {e}")

        result = {
            "run_id": run_id,
            "final_params": final_params,
            "final_nll": final_nll,
            "nll_history": nll_history,
            "time": end_time - start_time,
            "hessian_time": hessian_time,
            "iterations": len(nll_history),
            "initial_params": initial_params.clone().detach(),
            "hessian_full": hessian_full,
            "hessian": hessian,
            "is_positive_definite": is_pos_def,
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "condition_number": cond_num,
            "coupling_real_errors": coupling_real_errors,
            "coupling_imag_errors": coupling_imag_errors,
            "res_errors": res_errors,
        }

        if final_nll < self.best_nll:
            self.best_nll = final_nll
            self.best_params = final_params.clone()
            self.best_result = result

        return result

    # --------------------------------------------------------
    def _project_params_(self, p):
        """把参数向量投影回可行域（固定参考 + 耦合 ±v_max + 共振态 bounds）"""
        with torch.no_grad():
            p.data[0] = 1.0
            p.data[self.n_coupling_free] = 0.0
            p.data[1:self.n_coupling_free].clamp_(-self.v_max, self.v_max)
            p.data[self.n_coupling_free + 1:2 * self.n_coupling_free].clamp_(
                -self.v_max, self.v_max)
            if self.has_free_res:
                res_start = 2 * self.n_coupling_free
                p.data[res_start:].clamp_(self._lower, self._upper)

    # --------------------------------------------------------
    def polish_damped_newton(self, params_phys, max_steps=40, tol=1e-6, lam0=1.0,
                             verbose=True):
        """LBFGS 之后的精确 Hessian 抛光（damped Newton / Levenberg-Marquardt 式）。

        params_phys: 完整参数 [Re_v, Im_v, θ_phys]（例: fit.py 的物理空间约定）。
        在自由参数子空间（mask 掉固定参考方向）解 (H + λI)·d = -g：
          - H 不正定时 λ 抬到 -λ_min + δ，保证方向可下降；
          - 每步后投影回可行域；目标不下降 → 增大 λ 重试（信任域语义）；
          - 收敛后 H 正定则再做一次纯 Newton 步收尾。
        实测: LBFGS（宽容差）停在 NLL=-673 的伪平坦点，抛光可到 -1175；
        对发散垃圾点也能救回（+903 → -1402）。
        返回 (params, nll, is_pos_def)。
        """
        dev = self.device
        mask = torch.ones(self.n_params, dtype=torch.bool, device=dev)
        mask[0] = False
        mask[self.n_coupling_free] = False
        n_free = int(mask.sum().item())

        best = params_phys.clone().detach()
        self._project_params_(best)
        nll_best = self.analysis.getNLL(best).item()
        lam = lam0

        for step in range(max_steps):
            H_full = self.analysis.getHessian(best)
            H = H_full[mask][:, mask]
            eig = torch.linalg.eigvalsh(H)
            eig_min = eig[0].item()

            p = best.clone().requires_grad_(True)
            nll = self.analysis.getNLL(p)
            g = torch.autograd.grad(nll, p)[0]
            g_m = g[mask]

            if eig_min > 1e-8:
                # 已正定：纯 Newton 一步收尾
                d_m = torch.linalg.solve(H, -g_m)
                cand = best.clone()
                cand[mask] = best[mask] + d_m
                self._project_params_(cand)
                nll_cand = self.analysis.getNLL(cand).item()
                if nll_cand < nll_best - tol:
                    best, nll_best = cand, nll_cand
                    if verbose:
                        log.debug(f"[polish] step{step}: Newton accept, NLL={nll_best:.6f}")
                break

            lam = max(lam, -eig_min + 1e-6)
            I = torch.eye(n_free, dtype=torch.float64, device=dev)
            try:
                d_m = torch.linalg.solve(H + lam * I, -g_m)
            except Exception:
                d_m = torch.linalg.lstsq(H + lam * I, -g_m).solution
            cand = best.clone()
            cand[mask] = best[mask] + d_m
            self._project_params_(cand)
            nll_cand = self.analysis.getNLL(cand).item()
            if nll_cand < nll_best - tol:
                best, nll_best = cand, nll_cand
                lam = max(lam * 0.3, 1e-6)
                if verbose:
                    log.debug(f"[polish] step{step}: accept λ={lam:.2e}, NLL={nll_best:.6f}")
            else:
                lam *= 10.0
                if verbose:
                    log.debug(f"[polish] step{step}: reject, λ={lam:.2e}")
                if lam > 1e10:
                    break

        H_final = self.analysis.getHessian(best)
        H_f = H_final[mask][:, mask]
        eig_f = torch.linalg.eigvalsh(H_f)
        pd = bool((eig_f[0] > 1e-8).item())
        # 播种缓存: 抛光终点的 Hessian 供误差/分支比复用
        self._hess_cache = (best.clone(), H_final)
        if verbose:
            print(f"[polish] done: NLL={nll_best:.6f}, PD={pd}, "
                  f"min_eig={eig_f[0].item():.3e}, steps={step + 1}")
        return best, nll_best, pd

    # --------------------------------------------------------
    def _get_hessian_cached(self, params):
        """统一 Hessian（带缓存）: 参数与上次完全相同时直接复用，否则计算。
        fit.py 在多处（正定性/参数误差/分支比误差）会在同一最佳点上要 Hessian,
        只算一次即可。"""
        if (self._hess_cache is not None
                and self._hess_cache[0].shape == params.shape
                and torch.equal(self._hess_cache[0], params)):
            return self._hess_cache[1]
        h = self.analysis.getHessian(params)
        self._hess_cache = (params.clone(), h)
        return h

    # --------------------------------------------------------
    def compute_param_errors(self, params_phys):
        """在给定参数点用精确 Hessian 求参数误差（H 正定时有效）。
        返回 (coupling_real_errors, coupling_imag_errors, res_errors)。
        """
        hessian_full = self._get_hessian_cached(params_phys)
        fixed_mask = torch.ones(self.n_params, dtype=torch.bool, device=self.device)
        fixed_mask[0] = False
        fixed_mask[self.n_coupling_free] = False
        hessian = hessian_full[fixed_mask][:, fixed_mask]
        eig = torch.linalg.eigvalsh(hessian)
        if eig[0].item() <= 1e-8:
            return None, None, None
        try:
            covariance = torch.linalg.inv(hessian)
            std_dev = torch.sqrt(torch.diag(covariance))
            n_c_var = self.n_coupling_free - 1
            coupling_real_errors = torch.zeros(
                self.n_coupling_free, dtype=torch.float32, device=self.device)
            coupling_imag_errors = torch.zeros(
                self.n_coupling_free, dtype=torch.float32, device=self.device)
            for i in range(n_c_var):
                coupling_real_errors[i + 1] = std_dev[2 * i].float()
                coupling_imag_errors[i + 1] = std_dev[2 * i + 1].float()
            res_errors = None
            if self.has_free_res:
                res_start = 2 * n_c_var
                res_errors = std_dev[res_start:].float()
            return coupling_real_errors, coupling_imag_errors, res_errors
        except Exception as e:
            log.error(f"计算参数误差时出错: {e}")
            return None, None, None

    # --------------------------------------------------------
    def extract_coupling_complex(self, params):
        """从统一参数中提取复数耦合向量 (complex64, n_coupling_free)"""
        real = params[:self.n_coupling_free].float()
        imag = params[self.n_coupling_free:2 * self.n_coupling_free].float()
        return torch.complex(real, imag)

    def extract_theta_phys(self, params):
        """从统一参数中提取共振态物理参数"""
        if not self.has_free_res:
            return None
        return params[2 * self.n_coupling_free:]

    # --------------------------------------------------------
    def compute_fit_fractions(self, params=None):
        """拟合分数 (fit fractions): FF_i = ∫|A_i|² / Σ_j ∫|A_j|²。
        只用 phsp_truth（无效率 MC）→ 与效率/归一化无关，跨实验可比。
        误差传播用拟合同源的统一 Hessian（缓存命中直接复用）。"""
        if params is None:
            if self.best_params is None:
                log.warning("没有优化结果!")
                return None, None
            params = self.best_params

        coupling = self.extract_coupling_complex(params)
        # 传拟合同源的全量 Hessian → FF 误差的正定性判定与拟合完全一致
        # （旧版内部用独立的 computeCouplingHessian, 近平坦方向
        #  min_eig~1e-5 时正定判定翻脸 → 误差被跳过变全 0）。
        hessian_full = self._get_hessian_cached(params)
        # getFitFractions 在配置无 phsp_truth 时天然返回空张量 [0,2]
        # （早期拟合不带 mctruth）→ 这里自然跳过, 不抛异常、不触碰相关 kernel
        ff_result = self.analysis.getFitFractions(coupling, hessian_full)
        if ff_result is None or ff_result.numel() == 0:
            log.warning("跳过拟合分数: 配置没有 phsp_truth (无效率相空间 MC), "
                        "需要时再加入并重跑")
            return None, None
        return ff_result[:, 0], ff_result[:, 1]

    # --------------------------------------------------------
    def compute_efficiency(self, params=None):
        """分波效率: ε_i = (Σ_{phsp}|A_i|²/N_phsp) / (Σ_{phsp_truth}|A_i|²/N_truth)。
        phsp(如 cut 后 MC)=带效率样本, phsp_truth=无效率 MC truth,
        即分波加权的探测/选择效率 ∫|A_i|²ε(x)dΦ / ∫|A_i|²dΦ (tf-pwa get_efficiency 语义)。
        效率依赖拟合结果(振幅形状含共振参数), 用拟合后 params 计算。
        误差 = 参数误差(拟合同源 Hessian) ⊕ MC 统计误差(tf-pwa add_int_error 同款)。
        配置缺 phsp 或 phsp_truth 时返回空张量 [0,2] → 这里自然跳过。"""
        if params is None:
            if self.best_params is None:
                log.warning("没有优化结果!")
                return None, None
            params = self.best_params

        coupling = self.extract_coupling_complex(params)
        hessian_full = self._get_hessian_cached(params)
        eff_result = self.analysis.getEfficiency(coupling, hessian_full)
        if eff_result is None or eff_result.numel() == 0:
            log.warning("跳过分波效率: 配置缺 phsp (带效率 MC) 或 phsp_truth, "
                  "需要时再加入并重跑")
            return None, None
        return eff_result[:, 0], eff_result[:, 1]

    # --------------------------------------------------------
    def save_parameters(self, params, coupling_real_err, coupling_imag_err,
                        res_errors, run_id, filename_base):
        """保存所有参数到文件"""
        try:
            coupling = self.extract_coupling_complex(params)
            params_np = coupling.cpu().numpy()                         # [n_coupling_free]
            real_err_np = (coupling_real_err.cpu().numpy()
                           if coupling_real_err is not None
                           else np.zeros(self.n_coupling_free))
            imag_err_np = (coupling_imag_err.cpu().numpy()
                           if coupling_imag_err is not None
                           else np.zeros(self.n_coupling_free))

            if self.has_free_res:
                theta = self.extract_theta_phys(params)
                theta_np = theta.cpu().numpy()
                lower_np = self._lower.cpu().numpy()
                upper_np = self._upper.cpu().numpy()
                res_err_np = res_errors.cpu().numpy() if res_errors is not None else None

            txt_filename = f"{filename_base}.txt"
            if run_id == 0:
                with open(txt_filename, "w") as f:
                    f.write("# PWA Unified Parameters - All Runs\n")
                    f.write(f"# n_coupling_free: {self.n_coupling_free}\n")
                    f.write(f"# n_res_free: {self.n_res_free}\n")
                    f.write(f"# File generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("#" * 120 + "\n")
                    f.write("# Index  ParameterName                      ")
                    f.write("RealPart        RealError        ")
                    f.write("ImagPart        ImagError        ")
                    f.write("Magnitude       Phase(rad)      Phase(deg)\n")
                    f.write("#" * 120 + "\n")

            with open(txt_filename, "a") as f:
                f.write(f"# RUN: run_{run_id}\n")
                f.write("#" * 120 + "\n")
                for fi in range(self.n_coupling_free):
                    name = self.params_names[fi]
                    value = params_np[fi]
                    re_err = real_err_np[fi]
                    im_err = imag_err_np[fi]
                    magnitude = np.abs(value)
                    phase_rad = np.angle(value)
                    phase_deg = np.degrees(phase_rad)
                    f.write(
                        f"{fi:4d}  {name:50s}  "
                        f"{value.real:12.8f} ± {re_err:12.8f}  "
                        f"{value.imag:12.8f} ± {im_err:12.8f}  "
                        f"{magnitude:12.8f}  {phase_rad:12.8f}  {phase_deg:12.8f}\n"
                    )

                if self.has_free_res:
                    for i in range(self.n_res_free):
                        idx = self.n_coupling_free + i
                        name = self.params_names[idx]
                        err_str = f"± {res_err_np[i]:12.8f}" if res_err_np is not None else "             "
                        f.write(
                            f"{idx:4d}  {name:50s}  "
                            f"{theta_np[i]:12.8f}  {err_str}  "
                            f"bounds=[{lower_np[i]:.6g}, {upper_np[i]:.6g}]\n"
                        )
                f.write("#" * 120 + "\n")

            if run_id == 0:
                print(f"参数文件已创建: {txt_filename}")
            return True
        except Exception as e:
            log.error(f"保存参数失败: {e}")
            return False

    # --------------------------------------------------------
    def save_nll_history(self, nll_history, run_id, filename_base):
        try:
            txt_filename = f"{filename_base}.txt"
            if run_id == 0:
                with open(txt_filename, "w") as f:
                    f.write("# NLL History - All Runs\n")
                    f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("#" * 60 + "\n")
                    f.write("# Iteration  NLL\n")
                    f.write("#" * 60 + "\n")

            with open(txt_filename, "a") as f:
                f.write(f"# RUN: run_{run_id}\n")
                f.write("#" * 60 + "\n")
                for j, nll_val in enumerate(nll_history):
                    f.write(f"{j:8d}  {nll_val:15.8f}\n")
                f.write("#" * 60 + "\n")
            return True
        except Exception as e:
            log.error(f"保存NLL历史失败: {e}")
            return False

    # --------------------------------------------------------
    def save_weight_file(self, params, filename, waves=None, event_data=None):
        """保存权重文件。先reCalcAmp再writeResult。
        waves: 可选分波下标子集（如 [6,7]）, 只画 |Σ_{i∈S}A_i·v_i|² 的分布,
        空=全部。优先级: 参数 > FIT_WAVES 环境变量 > 默认(全部)。
        event_data: True 时 TTree 额外含末态四动量(任意分布按需现算)。
        优先级: 参数 > FIT_EVENT_DATA 环境变量 > 默认(False)。
        逐事件干涉 (interf_<i>_<j>) 请用 write_interf_result(pairs) 按需导出。"""
        try:
            if waves is None:
                w = os.environ.get("FIT_WAVES", "").strip()
                waves = [int(x) for x in w.split(",")] if w else []
            if self.has_free_res:
                theta = self.extract_theta_phys(params)
                self.analysis.reCalcAmp(theta)
            if event_data is None:
                event_data = os.environ.get("FIT_EVENT_DATA", "0") == "1"
            flag = 1 if event_data else 0
            self.analysis.writeResult(params, filename, flag, waves)
            if waves:
                print(f"权重文件已保存: {filename} (waves 子集: {waves}, "
                      f"直方图为 |Σ_{waves}A_i·v_i|²)")
            else:
                print(f"权重文件已保存: {filename}")
            return True
        except Exception as e:
            log.error(f"保存权重文件失败 {filename}: {e}")
            return False

    # --------------------------------------------------------
    def run_multiple_optimizations(self, num_runs=10, warm_start=None,
                                    output_dir="results",
                                    checkpoint_interval=1,
                                    resume_from=None, **kwargs):
        """多次优化运行。

        Args:
            checkpoint_interval: 每 N 轮保存一次 checkpoint（默认 1 = 每轮都存）。
            resume_from: dict，含 'start_run' 和 'all_nlls' 等续跑状态；
                         None 则从头开始。
        """
        results = []
        os.makedirs(output_dir, exist_ok=True)

        params_filename = os.path.join(output_dir, "parameters.txt")
        nll_filename = os.path.join(output_dir, "nll_history.txt")
        checkpoint = os.path.join(output_dir, "best_params.pt")
        resume_file = os.path.join(output_dir, "checkpoint.pt")

        # 续跑: 恢复已有结果
        start_run = 0
        if resume_from is not None:
            start_run = resume_from.get("start_run", 0)
            self.best_nll = resume_from.get("best_nll", float("inf"))
            bp = resume_from.get("best_params")
            if bp is not None:
                self.best_params = bp.to(self.device)
            print(f"续跑: 从 run {start_run} 继续，已有 best_nll={self.best_nll:.6f}")

        for i in range(start_run, num_runs):
            print(f"\n{'='*80}")
            print(f"开始第 {i}/{num_runs-1} 次优化")
            print(f"{'='*80}")

            seed = 42 if i == 0 else 42 + i
            initial_params = self.generate_initial_params(seed=seed)
            # warm start: run 0 的耦合取自收敛解（共振态参数保持 PDG 初值）。
            # 实测: 随机初值 + 放开共振态参数时 LBFGS 沿平坦方向放飞
            # （NLL 正值/撞边界）；从已收敛耦合出发则稳定收敛。
            if i == 0 and warm_start is not None:
                w = warm_start.to(self.device)
                initial_params[:2 * self.n_coupling_free] = w[:2 * self.n_coupling_free]
                print("warm start: 耦合来自收敛解 (NLL 优于随机初值的放飞解)")

            try:
                result = self.optimize_single_run(initial_params, run_id=i, **kwargs)
                results.append(result)
                self.all_results.append(result)

                print(f"第 {i} 次优化完成!")
                print(f"  NLL = {result['final_nll']:.6f}")
                print(f"  正定性 = {result['is_positive_definite']}")
                print(f"  耗时 = {result['time']:.2f}s, Hessian = {result['hessian_time']:.2f}s")
                print(f"  迭代次数 = {result['iterations']}")

                self.save_parameters(
                    result["final_params"],
                    result["coupling_real_errors"],
                    result["coupling_imag_errors"],
                    result["res_errors"],
                    i, params_filename.replace(".txt", ""),
                )

                self.save_nll_history(result["nll_history"], i,
                                      nll_filename.replace(".txt", ""))

                # checkpoint: 按间隔保存 best_params + 续跑状态
                if result["final_nll"] <= self.best_nll:
                    torch.save(result["final_params"].cpu(), checkpoint)
                if (i + 1) % checkpoint_interval == 0 or i == num_runs - 1:
                    torch.save({
                        "start_run": i + 1,
                        "best_nll": self.best_nll,
                        "best_params": self.best_params.cpu() if self.best_params is not None else None,
                        "all_nlls": [r["final_nll"] for r in self.all_results],
                    }, resume_file)
                    log.debug(f"checkpoint 已保存: {resume_file}")

            except Exception as e:
                log.error(f"第 {i} 次优化失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        return results

    # --------------------------------------------------------
    def print_optimized_parameters(self, params=None, coupling_real_err=None,
                                   coupling_imag_err=None, res_errors=None,
                                   run_id=None):
        if params is None:
            if self.best_params is None:
                log.warning("没有优化结果!")
                return
            params = self.best_params
            run_info = "最佳"
        else:
            run_info = f"第 {run_id} 次运行"

        coupling = self.extract_coupling_complex(params)
        params_np = coupling.cpu().numpy()                         # [n_coupling_free]
        real_err_np = (coupling_real_err.cpu().numpy()
                       if coupling_real_err is not None
                       else np.zeros(self.n_coupling_free))
        imag_err_np = (coupling_imag_err.cpu().numpy()
                       if coupling_imag_err is not None
                       else np.zeros(self.n_coupling_free))

        print(f"\n{'='*80}")
        print(f"{run_info}优化结果:")
        print(f"{'='*80}")
        print(f"固定参数: {self.params_names[0]} = 1.000000 + 0.000000i")

        for fi in range(1, self.n_coupling_free):
            name = self.params_names[fi]
            value = params_np[fi]
            re_err = real_err_np[fi]
            im_err = imag_err_np[fi]
            magnitude = np.abs(value)
            phase = np.angle(value)
            x, y = value.real, value.imag
            dx, dy = re_err, im_err
            mag_err = np.sqrt((x**2 * dx**2 + y**2 * dy**2) / (x**2 + y**2)) if magnitude > 0 else 0.0
            phase_err = np.sqrt((y**2 * dx**2 + x**2 * dy**2) / (x**2 + y**2)**2) if magnitude > 0 else 0.0
            print(
                f"{fi:3d}: {name:50s} = "
                f"({value.real:10.6f} ± {re_err:10.6f}) + "
                f"({value.imag:10.6f} ± {im_err:10.6f})i  "
                f"(|A|={magnitude:.6f} ± {mag_err:.6f}, "
                f"φ={np.degrees(phase):.2f}° ± {np.degrees(phase_err):.2f}°)"
            )

        # 共振态参数
        if self.has_free_res:
            theta = self.extract_theta_phys(params)
            print()
            theta_np = theta.cpu().numpy()
            lower_np = self._lower.cpu().numpy()
            upper_np = self._upper.cpu().numpy()
            res_err_np = res_errors.cpu().numpy() if res_errors is not None else None
            for j in range(self.n_res_free):
                idx = self.n_coupling_free + j
                name = self.params_names[idx]
                err_str = f" ± {res_err_np[j]:.6f}" if res_err_np is not None else ""
                print(f"{idx:3d}: {name:50s} = {theta_np[j]:12.8f}{err_str}"
                      f"  (bounds=[{lower_np[j]:.6g}, {upper_np[j]:.6g}])")

    # --------------------------------------------------------
    def save_all_results_summary(self, fit_values=None, fit_errors=None,
                                 fit_attempted=False,
                                 eff_values=None, eff_errors=None,
                                 output_dir="results"):
        if not self.all_results:
            log.warning("没有结果!")
            return

        sorted_results = sorted(self.all_results, key=lambda x: x["final_nll"])
        os.makedirs(output_dir, exist_ok=True)
        summary_file = os.path.join(output_dir, "optimization_summary.txt")
        with open(summary_file, "w") as f:
            f.write("PWA优化结果\n")
            f.write("=" * 100 + "\n")
            f.write(f"总运行次数: {len(self.all_results)}\n")
            f.write(f"耦合参数数量: {self.n_coupling_free}\n")
            f.write(f"自由共振态参数: {self.n_res_free}\n")
            f.write(f"总参数维度: {self.n_params}\n")
            f.write(f"最佳NLL: {self.best_nll:.6f}\n")
            f.write(f"参数文件: parameters.txt\n")
            f.write(f"NLL历史: nll_history.txt\n\n")

            f.write("=" * 100 + "\n")
            f.write("运行结果 (按NLL排序):\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'排名':<4} {'运行ID':<6} {'NLL':<12} {'迭代':<8} "
                    f"{'耗时':<10} {'Hessian耗时':<12} {'正定':<6}\n")
            f.write("-" * 100 + "\n")

            for rank, res in enumerate(sorted_results):
                f.write(f"{rank+1:<4} {res['run_id']:<6} {res['final_nll']:<12.6f} "
                        f"{res['iterations']:<8} {res['time']:<10.2f} "
                        f"{res['hessian_time']:<12.2f} "
                        f"{str(res['is_positive_definite']):<6}\n")

            if self.best_result.get("is_positive_definite", False):
                f.write("=" * 100 + "\n")
                f.write("最佳拟合分数 (fit fractions, 无效率/MC无关):\n")
                f.write("=" * 100 + "\n")
                try:
                    # 传入主程序已算好的结果，避免重复跑 truth 积分;
                    # fit_attempted=True 时主程序已处理(含 phsp_truth 缺失的跳过), 不再重试
                    if fit_values is None and not fit_attempted:
                        fit_values, fit_errors = self.compute_fit_fractions(self.best_params)
                    if fit_values is not None:
                        for i in range(len(fit_values)):
                            f.write(f"{i:2d}: {fit_values[i]:.6e} ± {fit_errors[i]:.6e}\n")
                except Exception as e:
                    f.write(f"计算拟合分数失败: {e}\n")

            if eff_values is not None:
                f.write("=" * 100 + "\n")
                f.write("分波效率 (ε_i, phsp带效率/phsp_truth无效率 加权比值):\n")
                f.write("=" * 100 + "\n")
                for i in range(len(eff_values)):
                    f.write(f"{i:2d}: {eff_values[i]:.6e} ± {eff_errors[i]:.6e}\n")

            if self.best_params is not None and self.has_free_res:
                f.write("=" * 100 + "\n")
                f.write("最佳共振态参数:\n")
                f.write("=" * 100 + "\n")
                theta = self.extract_theta_phys(self.best_params)
                theta_np = theta.cpu().numpy()
                lower_np = self._lower.cpu().numpy()
                upper_np = self._upper.cpu().numpy()
                f.write(f"{'Index':<6} {'Name':<30} {'Value':<16} {'Lower':<16} {'Upper':<16}\n")
                for i in range(self.n_res_free):
                    name = self.params_names[self.n_coupling_free + i]
                    f.write(f"{i:<6} {name:<30} {theta_np[i]:<16.8f} "
                            f"{lower_np[i]:<16.8f} {upper_np[i]:<16.8f}\n")

        print(f"优化结果摘要已保存到: {summary_file}")


# ============================================================
# CLI 参数解析
# ============================================================
def _env_float(name, default):
    """从环境变量读 float，不存在则返回 default。"""
    v = os.environ.get(name)
    return float(v) if v is not None else default


def _env_int(name, default):
    """从环境变量读 int，不存在则返回 default。"""
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _env_bool(name, default):
    """从环境变量读 bool（"1"/"true"/"yes" → True），不存在则返回 default。"""
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes")


def _env_str(name, default=""):
    """从环境变量读字符串，不存在则返回 default。"""
    return os.environ.get(name, default)


def build_parser():
    """构建 argparse 解析器。所有参数均支持同名 FIT_* 环境变量作为 fallback。"""
    p = argparse.ArgumentParser(
        description="ctpwa PWA 拟合驱动",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
环境变量 fallback:
  每个 CLI 参数都有对应的 FIT_* 环境变量，优先级: CLI > 环境变量 > 默认值。
  FIT_RUNS, FIT_NITER, FIT_LR, FIT_TOL_GRAD, FIT_TOL_CHANGE,
  FIT_HISTORY_SIZE, FIT_VMAX, FIT_PROJECT, FIT_WAVES, FIT_EVENT_DATA,
  FIT_WARM, FIT_POLISH, FIT_CHECKPOINT_INTERVAL

示例:
  # 快速扫描（早期调试）
  python fit.py --runs 3 --niter 100 --no-polish

  # 正式拟合
  python fit.py --runs 10 --niter 500

  # warm start 继续优化
  python fit.py --runs 5 --niter 1000 --warm-start results/best_params.pt

  # 只画特定波
  python fit.py --runs 1 --niter 500 --waves 6,7

  # 断点续跑（从上次中断处继续）
  python fit.py --runs 20 --niter 500 --resume

  # 安静模式 + 指定配置
  python fit.py -q --config /path/to/config.yml --runs 10
""",
    )

    # --- 配置 ---
    p.add_argument("--config", type=str, default="config.yml",
                   help="config.yml 路径 (默认: 当前目录下的 config.yml)")

    # --- 日志 ---
    g_log = p.add_mutually_exclusive_group()
    g_log.add_argument("-v", "--verbose", action="count", default=0,
                       help="增加日志详细度 (-v info, -vv debug)")
    g_log.add_argument("-q", "--quiet", action="store_true", default=False,
                       help="安静模式，只输出最终结果")

    # --- 核心运行参数 ---
    p.add_argument("--runs", type=int, default=None,
                   help="优化运行次数 (env: FIT_RUNS, 默认: 10)")
    p.add_argument("--niter", type=int, default=None,
                   help="LBFGS 最大迭代次数 (env: FIT_NITER, 默认: 500)")
    p.add_argument("--lr", type=float, default=None,
                   help="LBFGS 学习率 (env: FIT_LR, 默认: 0.9)")
    p.add_argument("--tol-grad", type=float, default=None,
                   help="LBFGS 梯度收敛阈值 (env: FIT_TOL_GRAD, 默认: 1e-7)")
    p.add_argument("--tol-change", type=float, default=None,
                   help="LBFGS 参数变化收敛阈值 (env: FIT_TOL_CHANGE, 默认: 1e-9)")
    p.add_argument("--history-size", type=int, default=None,
                   help="LBFGS history 大小 (env: FIT_HISTORY_SIZE, 默认: 100)")

    # --- 约束 / 数值稳定 ---
    p.add_argument("--vmax", type=float, default=None,
                   help="耦合幅度上界 |v| <= vmax (env: FIT_VMAX, 默认: 10000)")
    p.add_argument("--no-project", action="store_true", default=None,
                   help="关闭投影梯度 (env: FIT_PROJECT=0)")

    # --- Warm start ---
    p.add_argument("--warm-start", nargs="?", const="auto", default=None,
                   help="Warm start 路径。无参数时自动用 results/best_params.pt "
                        "(env: FIT_WARM=1)")

    # --- Polish ---
    g_polish = p.add_mutually_exclusive_group()
    g_polish.add_argument("--polish", dest="polish", action="store_true", default=None,
                          help="启用 damped Newton 抛光 (env: FIT_POLISH=1, 默认开启)")
    g_polish.add_argument("--no-polish", dest="polish", action="store_false",
                          help="关闭抛光")

    # --- 权重文件选项 ---
    p.add_argument("--waves", type=str, default=None,
                   help="分波下标子集，逗号分隔 (env: FIT_WAVES, 如 '6,7')")
    p.add_argument("--event-data", action="store_true", default=None,
                   help="TTree 额外含末态四动量 (env: FIT_EVENT_DATA=1)")

    # --- Checkpoint / Resume ---
    p.add_argument("--checkpoint-interval", type=int, default=None,
                   help="每 N 轮保存一次 checkpoint (env: FIT_CHECKPOINT_INTERVAL, 默认: 1)")
    p.add_argument("--resume", action="store_true", default=False,
                   help="从 output-dir/checkpoint.pt 续跑")

    # --- 输出 ---
    p.add_argument("--output-dir", type=str, default="results",
                   help="输出目录 (默认: results)")

    return p


def resolve_args(args):
    """将 argparse Namespace 与环境变量合并，返回最终配置 dict。
    优先级: CLI 显式传入 > FIT_* 环境变量 > 硬编码默认值。"""
    cfg = {}

    cfg["num_runs"] = (args.runs if args.runs is not None
                       else _env_int("FIT_RUNS", 10))
    cfg["max_iter"] = (args.niter if args.niter is not None
                       else _env_int("FIT_NITER", 500))
    cfg["lr"] = (args.lr if args.lr is not None
                 else _env_float("FIT_LR", 0.9))
    cfg["tolerance_grad"] = (args.tol_grad if args.tol_grad is not None
                             else _env_float("FIT_TOL_GRAD", 1e-7))
    cfg["tolerance_change"] = (args.tol_change if args.tol_change is not None
                               else _env_float("FIT_TOL_CHANGE", 1e-9))
    cfg["history_size"] = (args.history_size if args.history_size is not None
                           else _env_int("FIT_HISTORY_SIZE", 100))

    cfg["v_max"] = (args.vmax if args.vmax is not None
                    else _env_float("FIT_VMAX", 10000.0))

    # project_grad: CLI --no-project → False; 否则看 FIT_PROJECT
    if args.no_project is True:
        cfg["project_grad"] = False
    else:
        cfg["project_grad"] = _env_bool("FIT_PROJECT", True)

    # polish: CLI --polish/--no-polish > FIT_POLISH > 默认 True
    if args.polish is not None:
        cfg["polish"] = args.polish
    else:
        cfg["polish"] = _env_bool("FIT_POLISH", True)

    # warm start
    if args.warm_start is not None:
        if args.warm_start == "auto":
            cfg["warm_start_path"] = os.path.join(cfg.get("_output_dir", "results"),
                                                  "best_params.pt")
        else:
            cfg["warm_start_path"] = args.warm_start
    elif _env_bool("FIT_WARM", False):
        cfg["warm_start_path"] = os.path.join("results", "best_params.pt")
    else:
        cfg["warm_start_path"] = None

    # waves
    if args.waves is not None:
        cfg["waves"] = [int(x.strip()) for x in args.waves.split(",") if x.strip()]
    else:
        w = _env_str("FIT_WAVES", "").strip()
        cfg["waves"] = [int(x) for x in w.split(",") if x.strip()] if w else []

    # event data
    if args.event_data is not None:
        cfg["event_data"] = args.event_data
    else:
        cfg["event_data"] = _env_bool("FIT_EVENT_DATA", False)

    # checkpoint interval
    cfg["checkpoint_interval"] = (args.checkpoint_interval if args.checkpoint_interval is not None
                                  else _env_int("FIT_CHECKPOINT_INTERVAL", 1))

    cfg["resume"] = args.resume
    cfg["config"] = args.config
    cfg["verbose"] = args.verbose
    cfg["quiet"] = args.quiet
    cfg["output_dir"] = args.output_dir

    return cfg


# ============================================================
# 主程序
# ============================================================
def _setup_logging(verbose, quiet):
    """根据 -v/-q 设置日志级别。"""
    if quiet:
        level = logging.WARNING
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-5s %(message)s",
        force=True,  # 覆盖可能已有的 basicConfig
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = resolve_args(args)

    # ---- P4.3: 日志分级 ----
    _setup_logging(cfg["verbose"], cfg["quiet"])

    # ---- P4.2: 配置文件路径 ----
    config_path = os.path.abspath(cfg["config"])
    if not os.path.isfile(config_path):
        log.error(f"配置文件不存在: {config_path}")
        sys.exit(1)
    config_dir = os.path.dirname(config_path)
    if os.getcwd() != config_dir:
        print(f"切换到配置目录: {config_dir}")
        os.chdir(config_dir)

    # ---- P4.1: GPU 前置检查 ----
    if not torch.cuda.is_available():
        log.error("未检测到 CUDA GPU。ctpwa 需要 GPU 加速。")
        log.error(f"  PyTorch version: {torch.__version__}")
        log.error(f"  CUDA available:  {torch.cuda.is_available()}")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 打印最终配置
    print("=" * 60)
    print("PWA 拟合配置:")
    print(f"  config={config_path}")
    print(f"  runs={cfg['num_runs']}, niter={cfg['max_iter']}, lr={cfg['lr']}")
    print(f"  tol_grad={cfg['tolerance_grad']:.1e}, "
          f"tol_change={cfg['tolerance_change']:.1e}, "
          f"history_size={cfg['history_size']}")
    print(f"  vmax={cfg['v_max']}, project_grad={cfg['project_grad']}")
    print(f"  polish={cfg['polish']}")
    print(f"  warm_start={cfg['warm_start_path']}")
    print(f"  waves={cfg['waves'] if cfg['waves'] else '(all)'}")
    print(f"  event_data={cfg['event_data']}")
    print(f"  checkpoint_interval={cfg['checkpoint_interval']}")
    print(f"  resume={cfg['resume']}")
    print(f"  output_dir={output_dir}")
    print("=" * 60)

    # 复用模块级已初始化的分析对象（避免二次初始化浪费显存）
    # 模块级 ana / params_names / free_res_info 在 import 时已创建
    print(f"耦合参数数量: {n_coupling_free}, "
          f"共振态参数数量: {n_res_free}")

    # 初始化优化器
    optimizer = UnifiedPWAOptimizer(
        ana, free_res_info, params_names,
        v_max=cfg["v_max"],
        project_grad=cfg["project_grad"],
    )

    # ---- P4.4: Resume ----
    resume_from = None
    resume_file = os.path.join(output_dir, "checkpoint.pt")
    if cfg["resume"]:
        if os.path.exists(resume_file):
            ckpt = torch.load(resume_file, weights_only=False)
            resume_from = ckpt
            print(f"Resume: 加载 checkpoint ({resume_file}), "
                  f"从 run {ckpt.get('start_run', 0)} 继续")
        else:
            log.warning(f"Resume 请求但 checkpoint 不存在: {resume_file}，从头开始")

    # warm start
    warm = None
    ws_path = cfg["warm_start_path"]
    if ws_path and os.path.exists(ws_path):
        warm = torch.load(ws_path, weights_only=True)
        print(f"Warm start: 从 {ws_path} 载入耦合作为 run 0 初值")
    elif ws_path:
        log.warning(f"Warm start 文件不存在: {ws_path}，使用随机初值")

    # 运行优化
    results = optimizer.run_multiple_optimizations(
        num_runs=cfg["num_runs"],
        max_iter=cfg["max_iter"],
        lr=cfg["lr"],
        tolerance_grad=cfg["tolerance_grad"],
        tolerance_change=cfg["tolerance_change"],
        history_size=cfg["history_size"],
        warm_start=warm,
        output_dir=output_dir,
        checkpoint_interval=cfg["checkpoint_interval"],
        resume_from=resume_from,
    )

    # ---- 分析结果 ----
    if not optimizer.all_results:
        log.error("没有任何成功的优化结果!")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("所有优化结果总结:")
    print(f"{'='*80}")

    sorted_results = sorted(optimizer.all_results, key=lambda x: x["final_nll"])
    for i, res in enumerate(sorted_results):
        print(f"运行 {res['run_id']:2d}: NLL = {res['final_nll']:12.6f}, "
              f"迭代 = {res['iterations']:3d}, "
              f"耗时 = {res['time']:6.2f}s, Hessian = {res['hessian_time']:6.2f}s, "
              f"正定 = {res['is_positive_definite']}")

    print(f"\n{'='*80}")
    print("最佳结果:")
    print(f"{'='*80}")

    best_res = sorted_results[0]
    print(f"最佳NLL: {best_res['final_nll']:.6f} (来自第 {best_res['run_id']} 次运行)")

    # ---- 精确 Hessian 抛光 ----
    if cfg["polish"]:
        try:
            p2, nll2, pd2 = optimizer.polish_damped_newton(best_res["final_params"])
            if nll2 < best_res["final_nll"]:
                print(f"抛光: NLL {best_res['final_nll']:.6f} → {nll2:.6f} "
                      f"(Δ={nll2 - best_res['final_nll']:.3f}), 正定={pd2}")
                best_res["final_params"] = p2.clone()
                best_res["final_nll"] = nll2
                best_res["is_positive_definite"] = pd2
                if pd2:
                    (best_res["coupling_real_errors"],
                     best_res["coupling_imag_errors"],
                     best_res["res_errors"]) = optimizer.compute_param_errors(p2)
                else:
                    best_res["coupling_real_errors"] = None
                    best_res["coupling_imag_errors"] = None
                    best_res["res_errors"] = None
                torch.save(p2.cpu(), os.path.join(output_dir, "best_params.pt"))
        except Exception as e:
            log.error(f"抛光失败: {e}")

    # ---- 打印最佳参数 ----
    if best_res["is_positive_definite"] and best_res["coupling_real_errors"] is not None:
        optimizer.print_optimized_parameters(
            best_res["final_params"],
            best_res["coupling_real_errors"],
            best_res["coupling_imag_errors"],
            best_res["res_errors"],
            best_res["run_id"],
        )
    else:
        log.warning("Hessian矩阵不正定，无法提供参数误差估计")
        optimizer.print_optimized_parameters(best_res["final_params"])
    print(f"{'='*80}")

    # ---- 保存最佳权重文件 ----
    best_weight_file = os.path.join(output_dir, "weight_best.root")
    optimizer.save_weight_file(
        best_res["final_params"], best_weight_file,
        waves=cfg["waves"],
        event_data=cfg["event_data"],
    )

    # ---- 拟合分数 ----
    ff_values = ff_errors = None
    if best_res["is_positive_definite"]:
        try:
            ff_values, ff_errors = optimizer.compute_fit_fractions(
                best_res["final_params"]
            )
            if ff_values is not None:
                print(f"\n{'='*80}")
                print("最佳结果的拟合分数 (fit fractions, Σ=1, 无效率/MC无关):")
                print(f"{'='*80}")
                for i in range(len(ff_values)):
                    print(f"{i:2d}: {ff_values[i]:.6f} ± {ff_errors[i]:.6f}")
        except Exception as e:
            log.error(f"计算拟合分数失败: {e}")

    # ---- 分波效率 ----
    eff_values = eff_errors = None
    if best_res["is_positive_definite"]:
        try:
            eff_values, eff_errors = optimizer.compute_efficiency(
                best_res["final_params"]
            )
            if eff_values is not None:
                print(f"\n{'='*80}")
                print("最佳结果的分波效率 (ε_i, phsp/phsp_truth 加权比值):")
                print(f"{'='*80}")
                for i in range(len(eff_values)):
                    print(f"{i:2d}: {eff_values[i]:.6f} ± {eff_errors[i]:.6f}")
        except Exception as e:
            log.error(f"计算分波效率失败: {e}")

    # ---- 保存摘要 ----
    optimizer.save_all_results_summary(
        ff_values, ff_errors, fit_attempted=True,
        eff_values=eff_values, eff_errors=eff_errors,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
