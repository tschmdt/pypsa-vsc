"""
IEEE14DC test network, based on:
R. Wiget, E. Iggland, G. Andersson, "Security Constrained Optimal Power Flow
for HVAC and HVDC Grids", PSCC 2014, Section IV-A.

AC part: IEEE 14 bus test case with the branch capacities of the paper
(100 MVA in the lower part with the buses 1-7, 50 MVA in the upper part and
for the transformers in the middle).

Differences to the paper:
- the paper uses a meshed 4-terminal HVDC overlay with an additional DC bus 25;
  here the overlay is built from point-to-point VHLs at the same converter
  buses (1, 2, 3, 13), so DC bus 25 and the two DC side generators are missing
- all buses are modelled at 132 kV, the transformer tap ratios are kept
- the links are lossless

Run with: uv run python examples/ieee14dc.py
"""

# IEEE 14 Bus AC/DC network with three point-to-point VHLs

import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from combined_control.VSCController import ControllerConfig, VSCController


# =============================================================================
# 1. Network
# =============================================================================

n = pypsa.Network()

S_BASE = 100.0   # MVA
V_BASE = 132.0   # kV
Z_BASE = V_BASE**2 / S_BASE


# =============================================================================
# 2. Buses
# =============================================================================

bus_names = [f"Bus {i}" for i in range(1, 15)]

n.add(
    "Bus",
    bus_names,
    v_nom=V_BASE
)

# Voltage setpoints of generator buses
n.buses.loc["Bus 1", "v_mag_pu_set"] = 1.060
n.buses.loc["Bus 2", "v_mag_pu_set"] = 1.045
n.buses.loc["Bus 3", "v_mag_pu_set"] = 1.010
n.buses.loc["Bus 6", "v_mag_pu_set"] = 1.070
n.buses.loc["Bus 8", "v_mag_pu_set"] = 1.090


# =============================================================================
# 3. AC lines
#
# r, x and b originate from the IEEE 14 bus per-unit data.
# PyPSA requires:
#   r, x in ohm
#   b in siemens
# =============================================================================

lines = [
    # name, bus0, bus1, r_pu, x_pu, b_pu, s_nom
    ("1-2",   "Bus 1",  "Bus 2",  0.01938, 0.05917, 0.0528, 100),
    ("1-5",   "Bus 1",  "Bus 5",  0.05403, 0.22304, 0.0492, 100),
    ("2-3",   "Bus 2",  "Bus 3",  0.04699, 0.19797, 0.0438, 100),
    ("2-4",   "Bus 2",  "Bus 4",  0.05811, 0.17632, 0.0340, 100),
    ("2-5",   "Bus 2",  "Bus 5",  0.05695, 0.17388, 0.0346, 100),
    ("3-4",   "Bus 3",  "Bus 4",  0.06701, 0.17103, 0.0128, 100),
    ("4-5",   "Bus 4",  "Bus 5",  0.01335, 0.04211, 0.0000, 100),

    ("6-11",  "Bus 6",  "Bus 11", 0.09498, 0.19890, 0.0000, 50),
    ("6-12",  "Bus 6",  "Bus 12", 0.12291, 0.25581, 0.0000, 50),
    ("6-13",  "Bus 6",  "Bus 13", 0.06615, 0.13027, 0.0000, 50),
    ("7-8",   "Bus 7",  "Bus 8",  0.00000, 0.17615, 0.0000, 50),
    ("7-9",   "Bus 7",  "Bus 9",  0.00000, 0.11001, 0.0000, 50),
    ("9-10",  "Bus 9",  "Bus 10", 0.03181, 0.08450, 0.0000, 50),
    ("9-14",  "Bus 9",  "Bus 14", 0.12711, 0.27038, 0.0000, 50),
    ("10-11", "Bus 10", "Bus 11", 0.08205, 0.19207, 0.0000, 50),
    ("12-13", "Bus 12", "Bus 13", 0.22092, 0.19988, 0.0000, 50),
    ("13-14", "Bus 13", "Bus 14", 0.17093, 0.34802, 0.0000, 50),
]

for name, bus0, bus1, r_pu, x_pu, b_pu, s_nom in lines:
    n.add(
        "Line",
        name,
        bus0=bus0,
        bus1=bus1,
        r=r_pu * Z_BASE,
        x=x_pu * Z_BASE,
        b=b_pu / Z_BASE,
        s_nom=s_nom
    )


# =============================================================================
# 4. Transformers
#
# IEEE 14 transformer branches:
#   Bus 4 -- Bus 7
#   Bus 4 -- Bus 9
#   Bus 5 -- Bus 6
# =============================================================================

transformers = [
    # name, bus0, bus1, x_pu, tap_ratio
    ("T4-7", "Bus 4", "Bus 7", 0.20912, 0.978),
    ("T4-9", "Bus 4", "Bus 9", 0.55618, 0.969),
    ("T5-6", "Bus 5", "Bus 6", 0.25202, 0.932),
]

for name, bus0, bus1, x_pu, tap_ratio in transformers:
    n.add(
        "Transformer",
        name,
        bus0=bus0,
        bus1=bus1,
        r=0.0,
        x=x_pu * 50.0 / S_BASE,
        s_nom=50.0,
        tap_ratio=tap_ratio
    )


# =============================================================================
# 5. Loads
# =============================================================================

loads = {
    "Bus 2":  (21.7, 12.7),
    "Bus 3":  (94.2, 19.0),
    "Bus 4":  (47.8, -3.9),
    "Bus 5":  (7.6, 1.6),
    "Bus 6":  (11.2, 7.5),
    "Bus 9":  (29.5, 16.6),
    "Bus 10": (9.0, 5.8),
    "Bus 11": (3.5, 1.8),
    "Bus 12": (6.1, 1.6),
    "Bus 13": (13.5, 5.8),
    "Bus 14": (14.9, 5.0),
}

for bus, (p_set, q_set) in loads.items():
    number = bus.split()[1]

    n.add(
        "Load",
        f"Load {number}",
        bus=bus,
        p_set=p_set,
        q_set=q_set
    )


# =============================================================================
# 6. Generators and synchronous condensers
# =============================================================================

n.add(
    "Generator",
    "Gen 1",
    bus="Bus 1",
    control="Slack",
    p_set=232.4,
    p_nom=332.4,
    marginal_cost=10,
    p_min_pu=0.0,
    p_max_pu=1.0
)

n.add(
    "Generator",
    "Gen 2",
    bus="Bus 2",
    control="PV",
    p_set=40.0,
    p_nom=140.0,
    marginal_cost=30,
    p_min_pu=0.0,
    p_max_pu=1.0
)

# Synchronous condensers:
# active power is zero, but voltage is controlled through reactive power.

n.add(
    "Generator",
    "SC 3",
    bus="Bus 3",
    control="PV",
    p_set=0.0,
    p_nom=0.0
)

n.add(
    "Generator",
    "SC 6",
    bus="Bus 6",
    control="PV",
    p_set=0.0,
    p_nom=0.0
)

n.add(
    "Generator",
    "SC 8",
    bus="Bus 8",
    control="PV",
    p_set=0.0,
    p_nom=0.0
)


# =============================================================================
# 7. Three point-to-point HVDC links
# =============================================================================

P_NOM_LINK = 100.0

# -------------------------------------------------------------------------
# VHL 1: Bus 1 <-> Bus 13
# -------------------------------------------------------------------------

n.add(
    "Link",
    "VHL 1-13",
    bus0="Bus 1",
    bus1="Bus 13",
    p_nom=P_NOM_LINK,
    p_set=0.0,
    p_min_pu=-1.0,
    p_max_pu=1.0,
    efficiency=1.0
)

n.add(
    "ControllableVSC",
    "VSC 1-13 Bus 1",
    bus="Bus 1",
    link="VHL 1-13",
    side="bus0"
)

n.add(
    "ControllableVSC",
    "VSC 1-13 Bus 13",
    bus="Bus 13",
    link="VHL 1-13",
    side="bus1"
)


# -------------------------------------------------------------------------
# VHL 2: Bus 1 <-> Bus 11
# -------------------------------------------------------------------------

n.add(
    "Link",
    "VHL 1-11",
    bus0="Bus 1",
    bus1="Bus 11",
    p_nom=P_NOM_LINK,
    p_set=0.0,
    p_min_pu=-1.0,
    p_max_pu=1.0,
    efficiency=1.0
)

n.add(
    "ControllableVSC",
    "VSC 1-11 Bus 1",
    bus="Bus 1",
    link="VHL 1-11",
    side="bus0"
)

n.add(
    "ControllableVSC",
    "VSC 1-11 Bus 11",
    bus="Bus 11",
    link="VHL 1-11",
    side="bus1"
)


# -------------------------------------------------------------------------
# VHL 3: Bus 10 <-> Bus 4
# -------------------------------------------------------------------------

n.add(
    "Link",
    "VHL 10-4",
    bus0="Bus 10",
    bus1="Bus 4",
    p_nom=P_NOM_LINK,
    p_set=0.0,
    p_min_pu=-1.0,
    p_max_pu=1.0,
    efficiency=1.0
)

n.add(
    "ControllableVSC",
    "VSC 10-4 Bus 10",
    bus="Bus 10",
    link="VHL 10-4",
    side="bus0"
)

n.add(
    "ControllableVSC",
    "VSC 10-4 Bus 4",
    bus="Bus 4",
    link="VHL 10-4",
    side="bus1"
)


# =============================================================================
# 8. Initial AC power flow
# =============================================================================

snap = n.snapshots[0]
line_limits = n.lines.s_nom * n.lines.s_max_pu

# Reset setpoints so a partial re-run does not treat a previous opt/guard
# state as the "initial" operating point.
n.links["p_set"] = 0.0
if not n.links_t.p_set.empty:
    n.links_t.p_set.loc[:, :] = 0.0
if not n.controllable_vscs.empty:
    n.controllable_vscs["q_set"] = 0.0
    if not n.controllable_vscs_t.q_set.empty:
        n.controllable_vscs_t.q_set.loc[:, :] = 0.0

n.pf()

loading_initial = 100.0 * np.hypot(
    n.lines_t.p0.loc[snap], n.lines_t.q0.loc[snap]
) / line_limits
v_initial = n.buses_t.v_mag_pu.loc[snap].copy()
p_set_initial = (
    n.links_t.p_set.loc[snap].astype(float).copy()
    if not n.links_t.p_set.empty
    else n.links["p_set"].astype(float).copy()
)


# =============================================================================
# 9. Combined VSC optimization
# =============================================================================

cfg = ControllerConfig(
    angle_limit_deg=25,
    max_line_loading=0.90,
    S_rated=P_NOM_LINK,
    n1_guard_enable=True,  # set False to skip preventive guard
)

controller = VSCController(n, config=cfg)

# Capture AC / setpoint stages for both guard=False and guard=True.
stage = {}


def _capture_stage(tag: str) -> None:
    stage[f"p_set_{tag}"] = n.links_t.p_set.loc[snap].astype(float).copy()
    stage[f"loading_{tag}"] = 100.0 * np.hypot(
        n.lines_t.p0.loc[snap], n.lines_t.q0.loc[snap]
    ) / line_limits
    stage[f"v_{tag}"] = n.buses_t.v_mag_pu.loc[snap].copy()


_enforce_n1_guard = controller.enforce_n1_guard
_run_p_control = controller.run_p_control


def _enforce_n1_guard_with_capture(snapshot):
    # After P-opt, before guard
    n.pf()
    _capture_stage("opt")
    ok = _enforce_n1_guard(snapshot)
    # After preventive guard
    n.pf()
    _capture_stage("guard")
    stage["dP_corrective"] = stage["p_set_guard"] - stage["p_set_opt"]
    return ok


def _run_p_control_with_capture(**kwargs):
    result = _run_p_control(**kwargs)
    if not cfg.n1_guard_enable:
        # link_optimization already ran a final pf when guard_active=False
        _capture_stage("opt")
        stage["p_set_guard"] = stage["p_set_opt"].copy()
        stage["loading_guard"] = stage["loading_opt"].copy()
        stage["v_guard"] = stage["v_opt"].copy()
        stage["dP_corrective"] = pd.Series(0.0, index=n.links.index)
    return result


controller.enforce_n1_guard = _enforce_n1_guard_with_capture
controller.run_p_control = _run_p_control_with_capture
controller.run_mode(mode="combined")


# =============================================================================
# 10. Final state after full combined control (P + guard + Q)
# =============================================================================

n.pf()
_capture_stage("final")


# =============================================================================
# 11. Comparison: initial / after P-opt / after guard (+ final after Q)
# =============================================================================

loading_opt = stage["loading_opt"]
loading_guard = stage["loading_guard"]
loading_final = stage["loading_final"]
v_opt = stage["v_opt"]
v_guard = stage["v_guard"]
v_final = stage["v_final"]
p_set_opt = stage["p_set_opt"]
p_set_guard = stage["p_set_guard"]
dP_corrective = stage.get("dP_corrective", pd.Series(0.0, index=n.links.index))

df_loadings = pd.DataFrame(
    {
        "Initial [%]": loading_initial,
        "After P-opt [%]": loading_opt,
        "After guard [%]": loading_guard,
        "Final (P+Q) [%]": loading_final,
    }
).sort_values(by="After P-opt [%]", ascending=False)

df_voltages = pd.DataFrame(
    {
        "Initial [p.u.]": v_initial,
        "After P-opt [p.u.]": v_opt,
        "After guard [p.u.]": v_guard,
        "Final (P+Q) [p.u.]": v_final,
    }
)

df_link_setpoints = pd.DataFrame(
    {
        "p_set initial [MW]": p_set_initial.reindex(n.links.index),
        "p_set opt [MW]": p_set_opt,
        "dP corrective [MW]": dP_corrective,
        "p_set after guard [MW]": p_set_guard,
        "p0 final [MW]": n.links_t.p0.loc[snap],
    }
)

print(f"\nGuard enabled: {cfg.n1_guard_enable}")
print("\n=== Line loadings: initial / P-opt / guard / final ===")
print(df_loadings.round(2))

print("\n=== Link / VSC active-power setpoints ===")
print(df_link_setpoints.round(3))
print(
    f"\nCorrective step size |dP| max = {float(dP_corrective.abs().max()):.3f} MW, "
    f"sum |dP| = {float(dP_corrective.abs().sum()):.3f} MW"
)

print("\n=== VSC reactive setpoints (after Q-opt) ===")
print(
    n.controllable_vscs[["bus", "link", "side", "q_set", "q_min", "q_max"]].round(3)
)

print("\n=== Bus voltages: initial / P-opt / guard / final ===")
print(df_voltages.round(4))


# =============================================================================
# 12. Result plots
# =============================================================================

load_cols = ["Initial [%]", "After P-opt [%]"]
if cfg.n1_guard_enable:
    load_cols.append("After guard [%]")
df_loadings[load_cols].plot(kind="bar", figsize=(13, 7))
plt.axhline(100, linestyle="--")
plt.ylabel("Loading [% of s_nom]")
plt.title(
    "Line Loadings: Initial / After P-opt"
    + (" / After Guard" if cfg.n1_guard_enable else " (guard off)")
)
plt.grid(axis="y", linestyle=":")
plt.tight_layout()
plt.show()

v_cols = ["Initial [p.u.]", "After P-opt [p.u.]"]
if cfg.n1_guard_enable:
    v_cols.append("After guard [p.u.]")
df_voltages[v_cols].plot(marker="o", figsize=(12, 7))
plt.axhline(1.0, linestyle="--")
plt.ylabel("Voltage Magnitude [p.u.]")
plt.title(
    "Voltage Profile: Initial / After P-opt"
    + (" / After Guard" if cfg.n1_guard_enable else " (guard off)")
)
plt.grid(axis="y", linestyle=":")
plt.tight_layout()
plt.show()
