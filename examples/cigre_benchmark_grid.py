"""
CIGRE HV Case 0 baseline, then VHL between Bus 7 and Bus 8.

Case 0 is CSV-only AC PF (no Link/VSC), checked against thesis Tables 4.6 / 5.1 / 5.2.
Combined P/Q control goes through VSCController. P-opt is P_Optimizer_V2 (SCIP),
not linopy/Gurobi. N-1 is the heuristic in n1_guard.py, switched in ControllerConfig.

Run from the repository root:

    uv run python examples/cigre_benchmark_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pypsa  # noqa: E402
from combined_control.VSCController import (  # noqa: E402
    ControllerConfig,
    VSCController,
)

CIGRE_CSV_DIR = Path(__file__).resolve().parent / "networks" / "cigre-hv-benchmark"

# Thesis Table 4.6 — voltage magnitude [p.u.] and angle [deg]
TABLE_46_V = pd.Series(
    {
        "Bus 1": 1.032,
        "Bus 2": 1.006,
        "Bus 3": 0.996,
        "Bus 4": 0.950,
        "Bus 5": 0.952,
        "Bus 6": 0.986,
        "Bus 7": 1.043,
        "Bus 8": 1.042,
        "Bus 9": 1.030,
        "Bus 10": 1.030,
        "Bus 11": 1.030,
        "Bus 12": 1.030,
    }
)
TABLE_46_ANG = pd.Series(
    {
        "Bus 1": -3.7,
        "Bus 2": -2.0,
        "Bus 3": -34.4,
        "Bus 4": -39.5,
        "Bus 5": -29.1,
        "Bus 6": -32.1,
        "Bus 7": -6.2,
        "Bus 8": -31.9,
        "Bus 9": 0.0,
        "Bus 10": 1.6,
        "Bus 11": -33.0,
        "Bus 12": -27.7,
    }
)

# Thesis Table 5.1 — Case 0 line loadings [%]; 3-4a/b → combined Line 3-4 at s_nom=500
TABLE_51_LINE = pd.Series(
    {
        "Line 1-2": 21.80,
        "Line 1-6": 67.40,
        "Line 2-5": 73.20,
        "Line 3-4": 45.58,
        "Line 4-5": 27.78,
        "Line 4-6": 21.63,
        "Line 7-8": 74.21,
    }
)

# Thesis Table 5.2 — Case 0 transformer loadings [%]
TABLE_52_TRAFO = pd.Series(
    {
        "Transformer 1-7": 37.10,
        "Transformer 3-8": 34.90,
        "Transformer 9-1": 52.92,
        "Transformer 10-2": 52.79,
        "Transformer 11-3": 33.88,
        "Transformer 12-6": 70.29,
    }
)


def ac_loading(n: pypsa.Network, snapshot: object) -> pd.Series:
    """
    Line loading [%] from max(|S0|, |S1|) / s_nom [MVA].

    Matches thesis Tables 5.1 (uses the higher-loaded line end).
    """
    S0 = np.hypot(n.lines_t.p0.loc[snapshot], n.lines_t.q0.loc[snapshot])
    S1 = np.hypot(n.lines_t.p1.loc[snapshot], n.lines_t.q1.loc[snapshot])
    return 100.0 * np.maximum(S0, S1) / n.lines.s_nom


def trafo_loading(n: pypsa.Network, snapshot: object) -> pd.Series:
    """Transformer loading [%] from max(|S0|, |S1|) / s_nom [MVA]."""
    S0 = np.hypot(n.transformers_t.p0.loc[snapshot], n.transformers_t.q0.loc[snapshot])
    S1 = np.hypot(n.transformers_t.p1.loc[snapshot], n.transformers_t.q1.loc[snapshot])
    return 100.0 * np.maximum(S0, S1) / n.transformers.s_nom


def build_cigre_case0() -> pypsa.Network:
    """CIGRE HV CSV only — Case 0 AC network, no Link/VSC."""
    n = pypsa.Network()
    n.import_from_csv_folder(CIGRE_CSV_DIR)
    n.name = "CIGRE HV Case 0"
    return n


def build_cigre_vhl() -> pypsa.Network:
    """CIGRE HV CSV plus one VHL on Bus 7–8 (in parallel with Line 7-8)."""
    n = build_cigre_case0()
    n.name = "CIGRE HV + VHL 7-8"

    n.add(
        "Link",
        "Link 7-8",
        bus0="Bus 7",
        bus1="Bus 8",
        p_set=0.0,
        efficiency=0.9,
        p_nom=500.0,
    )
    n.add(
        "ControllableVSC",
        "VSC 1",
        bus="Bus 7",
        q_set=0.0,
        link="Link 7-8",
        side="bus0",
    )
    n.add(
        "ControllableVSC",
        "VSC 2",
        bus="Bus 8",
        q_set=0.0,
        link="Link 7-8",
        side="bus1",
    )
    return n


def print_case0_vs_tables(n: pypsa.Network, snapshot: object) -> None:
    """Print Case 0 PF results and deltas vs thesis Tables 4.6 / 5.1 / 5.2."""
    v = n.buses_t.v_mag_pu.loc[snapshot]
    ang = n.buses_t.v_ang.loc[snapshot] * 180.0 / np.pi
    line_ld = ac_loading(n, snapshot)
    trafo_ld = trafo_loading(n, snapshot)

    bus_cmp = pd.DataFrame(
        {
            "V [pu]": v.reindex(TABLE_46_V.index),
            "V ref": TABLE_46_V,
            "dV": v.reindex(TABLE_46_V.index) - TABLE_46_V,
            "ang [deg]": ang.reindex(TABLE_46_ANG.index),
            "ang ref": TABLE_46_ANG,
            "dang": ang.reindex(TABLE_46_ANG.index) - TABLE_46_ANG,
        }
    )
    line_cmp = pd.DataFrame(
        {
            "Loading [%]": line_ld.reindex(TABLE_51_LINE.index),
            "ref [%]": TABLE_51_LINE,
            "d [%]": line_ld.reindex(TABLE_51_LINE.index) - TABLE_51_LINE,
        }
    )
    trafo_cmp = pd.DataFrame(
        {
            "Loading [%]": trafo_ld.reindex(TABLE_52_TRAFO.index),
            "ref [%]": TABLE_52_TRAFO,
            "d [%]": trafo_ld.reindex(TABLE_52_TRAFO.index) - TABLE_52_TRAFO,
        }
    )

    print("\n=== Case 0 vs Table 4.6 (voltages) ===")
    print(bus_cmp.round(4).to_string())
    print(f"\nmax |dV| [pu]: {bus_cmp['dV'].abs().max():.4f}")
    print(f"max |dang| [deg]: {bus_cmp['dang'].abs().max():.2f}")

    print("\n=== Case 0 vs Table 5.1 (line loadings) ===")
    print(line_cmp.round(2).to_string())
    print(f"\nmax |d loading| [pp]: {line_cmp['d [%]'].abs().max():.2f}")

    print("\n=== Case 0 vs Table 5.2 (trafo loadings) ===")
    print(trafo_cmp.round(2).to_string())
    print(f"\nmax |d loading| [pp]: {trafo_cmp['d [%]'].abs().max():.2f}")


def print_state(n: pypsa.Network, snapshot: object, title: str) -> None:
    print(f"\n=== {title} ===")
    print("Line loading [%]:")
    print(ac_loading(n, snapshot).sort_values(ascending=False).round(2).to_string())
    print("\nTrafo loading [%]:")
    print(trafo_loading(n, snapshot).sort_values(ascending=False).round(2).to_string())
    print("\nBus voltage [p.u.]:")
    print(n.buses_t.v_mag_pu.loc[snapshot].round(4).to_string())
    if not n.links.empty:
        print("\nLink p_set [MW]:")
        print(n.links["p_set"].round(2).to_string())


def main() -> None:
    # --- Case 0: pure AC, no VHL ---
    n0 = build_cigre_case0()
    snap0 = n0.snapshots[0]
    n0.pf()
    print_case0_vs_tables(n0, snap0)

    # --- VHL path: Link 7-8 + VSCs + combined P/Q ---
    n = build_cigre_vhl()
    snap = n.snapshots[0]
    n.pf()
    loading_initial = ac_loading(n, snap)

    v_initial = n.buses_t.v_mag_pu.loc[snap]
    print_state(n, snap, "Initial AC power flow (with VHL, p_set=0)")

    cfg = ControllerConfig(
        angle_limit_deg=25.0,
        max_line_loading=0.95,
        S_rated=400.0,
        n1_guard_enable=False,
        n1_guard_margin=0.95,
        n1_guard_max_passes=3,
        slack_bus="Bus 9",
        distributed_slack=False,
    )
    ctl = VSCController(n, config=cfg)
    ctl.run_mode(mode="combined")

    n.pf()
    loading_opt = ac_loading(n, snap)
    v_opt = n.buses_t.v_mag_pu.loc[snap]
    print_state(n, snap, "After combined P/Q")

    df_loadings = pd.DataFrame(
        {
            "Initial [%]": loading_initial,
            "After P/Q [%]": loading_opt,
        }
    ).sort_values(by="After P/Q [%]", ascending=False)

    df_voltages = pd.DataFrame(
        {
            "Initial [p.u.]": v_initial,
            "After P/Q [p.u.]": v_opt,
        }
    )

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    ax = df_loadings.plot(kind="bar", figsize=(12, 7))
    ax.set_ylabel("Loading [% of s_nom]")
    ax.set_title("CIGRE HV line loadings before and after VSC control")
    ax.grid(axis="y", linestyle=":")
    plt.tight_layout()
    loadings_path = out_dir / "cigre_line_loadings.png"
    plt.savefig(loadings_path, dpi=150)
    plt.close()

    ax = df_voltages.plot(marker="o", figsize=(12, 7))
    ax.axhline(1.0, linestyle="--")
    ax.set_ylabel("Voltage magnitude [p.u.]")
    ax.set_title("CIGRE HV voltages before and after VSC control")
    ax.grid(axis="y", linestyle=":")
    plt.tight_layout()
    voltages_path = out_dir / "cigre_voltages.png"
    plt.savefig(voltages_path, dpi=150)
    plt.close()

    print(f"\nSaved plots to:\n  {loadings_path}\n  {voltages_path}")


if __name__ == "__main__":
    main()
