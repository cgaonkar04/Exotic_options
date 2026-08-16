"""
Crank-Nicolson Finite-Difference Option Pricer
================================================

Prices:
    1) American Call   (PSOR-projected Crank-Nicolson)
    2) American Put     (PSOR-projected Crank-Nicolson)
    3) Up-and-Out Barrier Call (European exercise, knock-out ABOVE the
       barrier -> payoff is forced to 0 the instant S touches/crosses B, CN)

Benchmarks:
    - Vanilla Black-Scholes closed-form European price (call/put)
    - Reiner & Rubinstein (1991) / Haug closed-form up-and-out call price
      (continuous monitoring, valid for barrier B > strike K)

Outputs:
    - Console table of CN prices vs analytical/benchmark prices and deviations
    - results.csv  with the same table
    - convergence_american_call.png
    - convergence_american_put.png
    - convergence_barrier_call.png
    - price_surface_comparison.png  (bar chart of all prices side by side)

Author: generated for user's CN pricing project
"""

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import os

# All output files (CSV + PNGs) are written next to this script, wherever it
# is run from -- no hardcoded absolute paths.
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# 0. Market / contract parameters (edit these to change the experiment)
# ----------------------------------------------------------------------
S0    = 100.0     # spot
K     = 100.0     # strike
T     = 1.0       # maturity (years)
r     = 0.05      # risk-free rate
sigma = 0.20      # volatility
B     = 175.0     # up-and-out barrier level: option knocks out (payoff = 0)
                   # the instant S >= B. Requires B > K for a non-trivial price.
BARRIER_TYPE = "up"   # "up" -> knock out above B (this project); "down" -> knock out below B

Smax_mult = 4.0   # only used when barrier_type="down" or no barrier: Smax = Smax_mult * K


# ----------------------------------------------------------------------
# 1. Analytical benchmarks
# ----------------------------------------------------------------------
def bs_price(S, K, T, r, sigma, kind="call"):
    """Closed-form Black-Scholes European price."""
    if T <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def merton_down_and_out_call(S, K, T, r, sigma, B):
    """
    Merton (1973) closed-form price of a continuously-monitored
    down-and-out call, valid for barrier B <= K, no dividends, S > B.
    C_do(S) = C(S,K) - (S/B)^(1 - 2r/sigma^2) * C(B^2/S, K)
    (Kept for reference / re-use if you switch BARRIER_TYPE back to "down".)
    """
    if S <= B:
        return 0.0
    lam = 1.0 - 2.0 * r / sigma ** 2
    C1 = bs_price(S, K, T, r, sigma, "call")
    C2 = bs_price(B ** 2 / S, K, T, r, sigma, "call")
    return C1 - (S / B) ** lam * C2


def haug_up_and_out_call(S, K, T, r, sigma, H):
    """
    Reiner & Rubinstein (1991) / Haug closed-form price of a continuously-
    monitored up-and-out call with zero rebate, no dividends.
    Valid for H > K (barrier above the strike -- the standard, non-trivial
    case). If H <= K the option is worthless (can't pay off without first
    knocking out), and if S >= H it has already knocked out.
    """
    if S >= H:
        return 0.0
    if H <= K:
        return 0.0
    sqT = sigma * np.sqrt(T)
    mu = (r - 0.5 * sigma ** 2) / sigma ** 2

    x1 = np.log(S / K) / sqT + (1 + mu) * sqT
    x2 = np.log(S / H) / sqT + (1 + mu) * sqT
    y1 = np.log(H ** 2 / (S * K)) / sqT + (1 + mu) * sqT
    y2 = np.log(H / S) / sqT + (1 + mu) * sqT

    A = S * norm.cdf(x1) - K * np.exp(-r * T) * norm.cdf(x1 - sqT)
    Bp = S * norm.cdf(x2) - K * np.exp(-r * T) * norm.cdf(x2 - sqT)
    C = S * (H / S) ** (2 * (mu + 1)) * norm.cdf(-y1) \
        - K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(-y1 + sqT)
    D = S * (H / S) ** (2 * (mu + 1)) * norm.cdf(-y2) \
        - K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(-y2 + sqT)

    return A - Bp + C - D


# ----------------------------------------------------------------------
# 2. Tridiagonal (Thomas) solver
# ----------------------------------------------------------------------
def thomas_solve(a, b, c, d):
    """
    Solve a tridiagonal system:
        a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    a[0] and c[-1] are unused. Returns x.
    """
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ----------------------------------------------------------------------
# 3. Crank-Nicolson PDE engine
#    Solves the Black-Scholes PDE on S in [Slow, Shigh] backward in time.
#    Supports:
#       - European payoff (no early exercise, no barrier)
#       - American early exercise via PSOR projection
#       - Knock-out barrier (Dirichlet V=0 at the barrier edge of the domain)
# ----------------------------------------------------------------------
def crank_nicolson(kind="call", american=False, barrier=None, barrier_type="down",
                    M=200, N=200, Smax_mult=Smax_mult,
                    S0=S0, K=K, T=T, r=r, sigma=sigma,
                    omega=1.2, tol=1e-8, max_psor_iter=10000):
    """
    Returns (S0_price, S_grid, V_grid_at_t0) using Crank-Nicolson finite
    differences on a uniform asset-price grid.

    kind         : "call" or "put"
    american     : True -> enforce early-exercise constraint via PSOR
    barrier      : None, or a float barrier level
    barrier_type : "down" -> knock-out below the barrier; domain becomes
                       [barrier, Smax] with V(barrier,t)=0, far end untouched
                   "up"   -> knock-out above the barrier; domain becomes
                       [0, barrier] with V(barrier,t)=0, near end untouched
                   (ignored if barrier is None)
    M            : number of asset-price steps (grid has M+1 nodes)
    N            : number of time steps
    """
    if barrier is not None and barrier_type == "up":
        Slow, Shigh = 0.0, barrier
    elif barrier is not None and barrier_type == "down":
        Slow, Shigh = barrier, Smax_mult * K
    else:
        Slow, Shigh = 0.0, Smax_mult * K
    Smax = Shigh  # kept as "Smax" for the far-field formulas below
    dS = (Shigh - Slow) / M
    dt = T / N
    S = Slow + dS * np.arange(M + 1)

    # Terminal payoff
    if kind == "call":
        V = np.maximum(S - K, 0.0)
    else:
        V = np.maximum(K - S, 0.0)

    # A path sitting exactly on the barrier at maturity has already knocked out
    if barrier is not None and barrier_type == "down":
        V[0] = 0.0
    if barrier is not None and barrier_type == "up":
        V[-1] = 0.0

    payoff = V.copy()  # needed for American constraint (payoff is time independent)

    # interior indices 1..M-1 ; i is index in S array, i.e. asset value S[i] = Slow + i*dS
    i_idx = np.arange(1, M)

    # local variance/drift coefficients use the *asset value*, not the grid index,
    # since the domain may not start at 0 (barrier case)
    Si = S[i_idx]
    sig2 = sigma ** 2

    a_coef = 0.25 * dt * (sig2 * Si ** 2 / dS ** 2 - r * Si / dS)
    b_coef = -0.5 * dt * (sig2 * Si ** 2 / dS ** 2 + r)
    c_coef = 0.25 * dt * (sig2 * Si ** 2 / dS ** 2 + r * Si / dS)

    # LHS tridiagonal (implicit half): M1 * V^n = M2 * V^{n+1} + boundary terms
    A_lo = -a_coef
    A_di = 1.0 - b_coef
    A_up = -c_coef

    B_lo = a_coef
    B_di = 1.0 + b_coef
    B_up = c_coef

    n_int = M - 1

    for n in range(N):           # step backward from t_{n+1} -> t_n ; n=0 is closest to maturity
        tau_next = n * dt        # time-to-maturity at level n+1 (V^{n+1})
        tau_curr = (n + 1) * dt  # time-to-maturity at level n   (V^{n})

        # Boundary values (time-to-maturity parametrization: tau = T - t)
        is_down_barrier = barrier is not None and barrier_type == "down"
        is_up_barrier = barrier is not None and barrier_type == "up"

        # --- lower boundary (S = Slow) ---
        if is_down_barrier:
            # knocked out below the barrier: value is 0 for all times
            V0_next, V0_curr = 0.0, 0.0
        elif kind == "call":
            V0_next, V0_curr = 0.0, 0.0  # standard call: worthless at S=0
        else:
            V0_next = K * np.exp(-r * tau_next)
            V0_curr = K * np.exp(-r * tau_curr)
            if american:
                V0_next = K  # American put: intrinsic value at S=0
                V0_curr = K

        # --- upper boundary (S = Shigh) ---
        if is_up_barrier:
            # knocked out above the barrier: value is 0 for all times
            VM_next, VM_curr = 0.0, 0.0
        elif kind == "call":
            VM_next = Smax - K * np.exp(-r * tau_next)
            VM_curr = Smax - K * np.exp(-r * tau_curr)
        else:
            VM_next, VM_curr = 0.0, 0.0

        V_old = V.copy()  # this is V^{n+1} (known); V_old[0], V_old[M] already hold
                          # the correct boundary values from the previous step

        # Build RHS = B_mat * V_old_interior  (boundary values already sit inside V_old)
        rhs = B_lo * V_old[0:M - 1] + B_di * V_old[1:M] + B_up * V_old[2:M + 1]

        # boundary correction terms added on LHS side (these come from unknowns
        # V0_curr, VM_curr multiplying the off-diagonal LHS coefficients of the
        # *new* time level, which are not part of the interior unknown vector)
        rhs[0] += a_coef[0] * V0_curr
        rhs[-1] += c_coef[-1] * VM_curr

        if not american:
            V_new_interior = thomas_solve(A_lo, A_di, A_up, rhs)
        else:
            # PSOR projected solve: (A_di, A_lo, A_up) system with constraint
            # V_i >= payoff_i
            x = V_old[1:M].copy()  # warm start
            for it in range(max_psor_iter):
                max_diff = 0.0
                for i in range(n_int):
                    left = A_lo[i] * (x[i - 1] if i > 0 else V0_curr)
                    right = A_up[i] * (x[i + 1] if i < n_int - 1 else VM_curr)
                    y = (rhs[i] - left - right) / A_di[i]
                    y = x[i] + omega * (y - x[i])
                    y = max(y, payoff[i + 1])
                    max_diff = max(max_diff, abs(y - x[i]))
                    x[i] = y
                if max_diff < tol:
                    break
            V_new_interior = x

        V = np.empty(M + 1)
        V[0] = V0_curr
        V[1:M] = V_new_interior
        V[M] = VM_curr

    price = np.interp(S0, S, V)
    return price, S, V


# ----------------------------------------------------------------------
# 4. Base-case pricing (a reasonably fine grid) + deviation from vanilla BS
# ----------------------------------------------------------------------
M_base, N_base = 400, 400

bs_call = bs_price(S0, K, T, r, sigma, "call")
bs_put = bs_price(S0, K, T, r, sigma, "put")
haug_barrier = haug_up_and_out_call(S0, K, T, r, sigma, B)

am_call_price, _, _ = crank_nicolson(kind="call", american=True, M=M_base, N=N_base)
am_put_price, _, _ = crank_nicolson(kind="put", american=True, M=M_base, N=N_base)
barrier_price, _, _ = crank_nicolson(kind="call", american=False, barrier=B,
                                      barrier_type=BARRIER_TYPE, M=M_base, N=N_base)

# sanity check: European CN call/put should match BS closely
eu_call_price, _, _ = crank_nicolson(kind="call", american=False, M=M_base, N=N_base)
eu_put_price, _, _ = crank_nicolson(kind="put", american=False, M=M_base, N=N_base)

results = [
    ("American Call",        am_call_price, bs_call,       am_call_price - bs_call),
    ("American Put",         am_put_price,  bs_put,        am_put_price - bs_put),
    ("Up-and-Out Call",      barrier_price, haug_barrier,  barrier_price - haug_barrier),
    ("European Call (CN, check)", eu_call_price, bs_call,  eu_call_price - bs_call),
    ("European Put (CN, check)",  eu_put_price,  bs_put,   eu_put_price - bs_put),
]

print(f"Grid used for base-case prices: M={M_base} (space), N={N_base} (time)")
print(f"Params: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}, B={B}\n")
header = f"{'Instrument':30s} {'CN Price':>12s} {'Benchmark':>12s} {'Deviation':>12s}"
print(header)
print("-" * len(header))
for name, cn, bench, dev in results:
    print(f"{name:30s} {cn:12.6f} {bench:12.6f} {dev:12.6f}")

with open(os.path.join(OUTPUT_DIR, "results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Instrument", "CN_Price", "Benchmark_Price", "Deviation"])
    for name, cn, bench, dev in results:
        w.writerow([name, cn, bench, dev])


# ----------------------------------------------------------------------
# 5. Convergence studies
#    Refine the grid (M=N=n) and track error against the benchmark price.
#    - American call/put: benchmark = European BS price is NOT the right target
#      (American has an early-exercise premium), so we use a very fine CN grid
#      as the "true" reference solution instead, and additionally show the
#      early-exercise premium over BS for context.
#    - Barrier call: Merton closed-form is exact, so we use it directly.
# ----------------------------------------------------------------------
grid_sizes = [10, 20, 40, 80, 160, 320, 640]

# Fine reference solution for American options (since no closed form exists)
M_ref, N_ref = 1200, 1200
ref_am_call, _, _ = crank_nicolson(kind="call", american=True, M=M_ref, N=N_ref)
ref_am_put, _, _ = crank_nicolson(kind="put", american=True, M=M_ref, N=N_ref)

def convergence_errors(kind, american, barrier, benchmark, barrier_type="down"):
    errs = []
    for n in grid_sizes:
        p, _, _ = crank_nicolson(kind=kind, american=american, barrier=barrier,
                                  barrier_type=barrier_type, M=n, N=n)
        errs.append(abs(p - benchmark))
    return np.array(errs)

err_am_call = convergence_errors("call", True, None, ref_am_call)
err_am_put = convergence_errors("put", True, None, ref_am_put)
err_barrier = convergence_errors("call", False, B, haug_barrier, barrier_type=BARRIER_TYPE)

def plot_convergence(errs, title, fname, ref_note):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(grid_sizes, errs, "o-", color="#2563eb", linewidth=2, markersize=6, label="CN absolute error")
    # reference O(h^2) slope for visual comparison
    ref_slope = errs[0] * (np.array(grid_sizes[0]) / np.array(grid_sizes)) ** 2
    ax.loglog(grid_sizes, ref_slope, "--", color="#9ca3af", linewidth=1.5, label="O(1/N^2) reference slope")
    ax.set_xlabel("Grid size N (= M, time and space steps)")
    ax.set_ylabel("Absolute error vs. " + ref_note)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)

plot_convergence(err_am_call, "Convergence: American Call (CN)",
                  os.path.join(OUTPUT_DIR, "convergence_american_call.png"),
                  "fine-grid CN reference (N=1200)")
plot_convergence(err_am_put, "Convergence: American Put (CN)",
                  os.path.join(OUTPUT_DIR, "convergence_american_put.png"),
                  "fine-grid CN reference (N=1200)")
plot_convergence(err_barrier, "Convergence: Up-and-Out Barrier Call (CN)",
                  os.path.join(OUTPUT_DIR, "convergence_barrier_call.png"),
                  "Reiner-Rubinstein/Haug closed form")

# ----------------------------------------------------------------------
# 6. Summary bar chart of prices
# ----------------------------------------------------------------------
labels = ["Euro Call\n(BS)", "Amer Call\n(CN)", "Euro Put\n(BS)", "Amer Put\n(CN)",
          "Vanilla Call\n(BS)", "UO Barrier\nCall (CN)"]
values = [bs_call, am_call_price, bs_put, am_put_price, bs_call, barrier_price]
colors = ["#93c5fd", "#2563eb", "#fca5a5", "#dc2626", "#93c5fd", "#059669"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, values, color=colors)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Price")
ax.set_title(f"Option Prices Comparison (S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}, B={B})")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "price_comparison.png"), dpi=150)
plt.close(fig)

print("\nSaved: results.csv, convergence_american_call.png, convergence_american_put.png,")
print("       convergence_barrier_call.png, price_comparison.png")
