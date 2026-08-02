"""
README Demo 1: 4-bus network with HVDC-VSC link (VHL) and combined P/Q control.

Run from the repository root:

    uv run python examples/vsc_demo1.py
"""

from pathlib import Path
import sys

# Allow `uv run python examples/vsc_demo1.py` even if the editable install
# has not picked up combined_control yet.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

# Non-interactive backend so the script finishes cleanly in Cursor/CI.
# Comment this out if you want interactive plt.show() windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

from combined_control.VSCController import ControllerConfig, VSCController


def build_demo_network() -> pypsa.Network:
    n = pypsa.Network()

    bus_names = [f"Bus {i}" for i in range(1, 5)]
    n.add("Bus", bus_names, v_nom=110)

    n.add("Line", "1-2", bus0="Bus 1", bus1="Bus 2", x=10, r=0.5, s_nom=250)
    n.add("Line", "1-4", bus0="Bus 1", bus1="Bus 4", x=15, r=0.25, s_nom=250)
    n.add("Line", "2-3", bus0="Bus 2", bus1="Bus 3", x=12, r=0.2, s_nom=300)
    n.add("Line", "3-4", bus0="Bus 3", bus1="Bus 4", x=10, r=0.2, s_nom=250)

    n.add("Generator", "Gen 1", bus="Bus 1", p_set=200, control="Slack")
    n.add("Generator", "Gen 2", bus="Bus 2", p_set=200, q_set=50, control="PQ")
    n.add("Generator", "Gen 4", bus="Bus 4", p_set=150, control="PQ")

    n.add("Load", "Load 2", bus="Bus 2", p_set=150)
    n.add("Load", "Load 3", bus="Bus 3", p_set=250, q_set=100)

    n.generators.loc["Gen 1", ["p_nom", "marginal_cost", "p_min_pu", "p_max_pu"]] = [
        400,
        30,
        0,
        1,
    ]
    n.generators.loc["Gen 2", ["p_nom", "marginal_cost", "p_min_pu", "p_max_pu"]] = [
        300,
        30,
        0,
        1,
    ]
    n.generators.loc["Gen 4", ["p_nom", "marginal_cost", "p_min_pu", "p_max_pu"]] = [
        300,
        5,
        0,
        1,
    ]

    n.add(
        "Link",
        "Link 3-4",
        bus0="Bus 3",
        bus1="Bus 4",
        p_set=-100,
        efficiency=0.95,
        p_nom=500,
    )

    n.add("ControllableVSC", "VSC 1", bus="Bus 3", link="Link 3-4", side="bus0")
    n.add("ControllableVSC", "VSC 2", bus="Bus 4", link="Link 3-4", side="bus1")

    return n


def main() -> None:
    n = build_demo_network()
    snap = n.snapshots[0]

    # Initial power flow
    n.pf()

    P = n.lines_t.p0.loc[snap]
    Q = n.lines_t.q0.loc[snap]
    S = np.hypot(P, Q)
    s_thermal = n.lines.s_nom * n.lines.s_max_pu

    loading_initial = 100 * S / s_thermal
    v_initial = n.buses_t.v_mag_pu.loc[snap]

    cfg = ControllerConfig(
        angle_limit_deg=25,
        max_line_loading=0.9,
        S_rated=300,
        n1_guard_enable=False,
    )

    ctl = VSCController(n, config=cfg)
    ctl.run_mode(mode="combined")

    # Power flow after optimization
    n.pf()

    P1 = n.lines_t.p0.loc[snap]
    Q1 = n.lines_t.q0.loc[snap]
    S1 = np.hypot(P1, Q1)

    loading_optimal = 100 * S1 / s_thermal
    v_optimal = n.buses_t.v_mag_pu.loc[snap]

    df_loadings = pd.DataFrame(
        {
            "Initial Loading [%]": loading_initial,
            "Optimized Loading [%]": loading_optimal,
        }
    ).sort_values(by="Optimized Loading [%]", ascending=False)

    df_voltages = pd.DataFrame(
        {
            "Initial Voltages [p.u.]": v_initial,
            "Optimized Voltages [p.u.]": v_optimal,
        }
    )

    print("\n=== Line loadings [%] ===")
    print(df_loadings.round(2).to_string())
    print("\n=== Bus voltages [p.u.] ===")
    print(df_voltages.round(4).to_string())

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    ax = df_loadings.plot(kind="bar", figsize=(12, 7))
    ax.set_ylabel("Loading [% of s_nom]")
    ax.set_title("Line Loadings Before and After VSC Optimization")
    ax.grid(axis="y", linestyle=":")
    plt.tight_layout()
    loadings_path = out_dir / "vsc_demo1_line_loadings.png"
    plt.savefig(loadings_path, dpi=150)
    plt.close()

    ax = df_voltages.plot(marker="o", figsize=(12, 7))
    ax.axhline(1.0, linestyle="--")
    ax.set_ylabel("Voltage Magnitude [p.u.]")
    ax.set_title("Voltage Profile Before and After VSC Optimization")
    ax.grid(axis="y", linestyle=":")
    plt.tight_layout()
    voltages_path = out_dir / "vsc_demo1_voltages.png"
    plt.savefig(voltages_path, dpi=150)
    plt.close()

    print(f"\nSaved plots to:\n  {loadings_path}\n  {voltages_path}")


if __name__ == "__main__":
    main()
