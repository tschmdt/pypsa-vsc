"""CIGRE HV with a VHL between Bus 7 and Bus 8.

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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

from combined_control.VSCController import ControllerConfig, VSCController

CIGRE_CSV_DIR = Path(__file__).resolve().parent / "networks" / "cigre-hv-benchmark"


def ac_loading(n: pypsa.Network, snapshot: object) -> pd.Series:
    """Line loading [%] from |S| / s_nom. S from p0, q0 [MW, MVAr]."""
    P = n.lines_t.p0.loc[snapshot]
    Q = n.lines_t.q0.loc[snapshot]
    return 100.0 * np.hypot(P, Q) / n.lines.s_nom


def trafo_loading(n: pypsa.Network, snapshot: object) -> pd.Series:
    """Transformer loading [%] from |S| / s_nom."""
    P = n.transformers_t.p0.loc[snapshot]
    Q = n.transformers_t.q0.loc[snapshot]
    return 100.0 * np.hypot(P, Q) / n.transformers.s_nom


def build_cigre_vhl() -> pypsa.Network:
    """CIGRE HV CSV plus one lossless-rated VHL on Bus 7–8 (in parallel with Line 7-8)."""
    n = pypsa.Network()
    n.import_from_csv_folder(CIGRE_CSV_DIR)
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
    n = build_cigre_vhl()
    snap = n.snapshots[0]
    n.pf()
    loading_initial = ac_loading(n, snap)
    v_initial = n.buses_t.v_mag_pu.loc[snap]
    print_state(n, snap, "Initial AC power flow")

    cfg = ControllerConfig(
        angle_limit_deg=25.0,
        max_line_loading=0.95,
        S_rated=400.0,
        n1_guard_enable=False,
        n1_guard_margin=0.95,
        n1_guard_max_passes=3,
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
