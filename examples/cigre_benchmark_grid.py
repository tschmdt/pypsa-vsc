# Imports
import pypsa
import pandas as pd
import numpy as np

# from combined_control.P_Optimizer_V2 import show_snapshot_report
import matplotlib.pyplot as plt
from combined_control.VSCController import ControllerConfig, VSCController
from combined_control.P_Optimizer_linopy import (
    link_optimization_linopy,
    show_snapshot_report,
)
from pathlib import Path


_EXAMPLES_DIR = Path(__file__).resolve().parent
CIGRE_CSV_DIR = _EXAMPLES_DIR / "networks" / "cigre-hv-benchmark"


# %% Base Network: Cigre HV


# Console Settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

network = pypsa.Network()
network.import_from_csv_folder(CIGRE_CSV_DIR)
snap = network.snapshots[0]

network._whoami = "Network"

# network_basis = network.copy()


# Q given in pu with Sbase=100MVA
network.add("ShuntImpedance", "Shunt 4", bus="Bus 4", b=0.0033, active=True)
network.add("ShuntImpedance", "Shunt 5", bus="Bus 5", b=0.0016, active=True)
network.add("ShuntImpedance", "Shunt 6", bus="Bus 6", b=0.0037, active=True)

print(network.shunt_impedances[["b", "active"]])

# network.add("Generator", "G_Shunt 4", bus="Bus 4", control="PQ", p_set=0, q_set=160)
# network.add("Generator", "G_Shunt 5", bus="Bus 5", control="PQ", p_set=0, q_set=80)
# network.add("Generator", "G_Shunt 6", bus="Bus 6", control="PQ", p_set=0, q_set=180)

network.name = "Cigre HV Benachmark"
print(network)

# AC Load Flow
network.pf()


# -----Output-----

# Bus Voltages and Angles
bus_vmag = network.buses_t.v_mag_pu.iloc[0]
bus_vang_deg = network.buses_t.v_ang.iloc[0] * 180 / np.pi

bus_df = pd.DataFrame({"Voltage [pu]": bus_vmag, "Angle [deg]": bus_vang_deg})
print("\n--- Bus Voltages and Angles ---")
print(bus_df.round(4))

# Line Load Flows
p0 = network.lines_t.p0.iloc[0]
q0 = network.lines_t.q0.iloc[0]
line_df = pd.DataFrame({"P_from [MW]": p0, "Q_from [Mvar]": q0})
print("\n--- Line Flows (From Bus) ---")
print(line_df.round(2))

# Generators
gen_p = network.generators_t.p.iloc[0]
gen_q = network.generators_t.q.iloc[0]
gen_df = pd.DataFrame({"P [MW]": gen_p, "Q [Mvar]": gen_q})
print("\n--- Generator Outputs ---")
print(gen_df.round(2))

# Angle-Stability via Angle Differences
print("\n--- Angle Differences (Δθ) ---")
angle_rows = []
for idx, row in network.lines.iterrows():
    bus0 = row["bus0"]
    bus1 = row["bus1"]
    theta0 = bus_vang_deg[bus0]
    theta1 = bus_vang_deg[bus1]
    delta = theta0 - theta1
    angle_rows.append(
        {"Line": idx, "Bus0": bus0, "Bus1": bus1, "Δθ [deg]": round(delta, 2)}
    )
angle_df = pd.DataFrame(angle_rows)
print(angle_df)

# Summe Shunt-Reaktanzen
total_b = network.shunt_impedances.loc[network.shunt_impedances.active, "b"].sum()
print("\nTotal b active:", total_b)

# Line Loading
P = network.lines_t.p0.loc[snap]
Q = network.lines_t.q0.loc[snap]
S = np.hypot(P, Q) 
s_nom = network.lines.s_nom
loading_S_default = 100 * S / s_nom           # MVA-based (new)
# Trafp Loading
P_T = network.transformers_t.p0.loc[snap]
Q_T = network.transformers_t.q0.loc[snap]
S_T = np.hypot(P_T, Q_T)
loading_trafo_S_default = 100 * S_T / network.transformers.s_nom


print("Leitungsauslastungen: ")
print(loading_S_default.sort_values(ascending=False).round(2))
print("Trafoauslastungen: ")
print(loading_trafo_S_default.sort_values(ascending=False).round(2))

print("Variante 2 berechnung mit hyp")
P1 = network.lines_t.p0.loc[snap]
Q1 = network.lines_t.q0.loc[snap]
S1 = np.hypot(P, Q) 
loading_S_default_V2 = 100 * S / network.lines.s_nom 
print(loading_S_default_V2.sort_values(ascending=False).round(2))


# %% Extended with VSC-Link(s) between Bus 7 and 8

# Console Settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 200)   

network = pypsa.Network()
network.import_from_csv_folder(CIGRE_CSV_DIR)
# network_basis = network.copy()

# Q given in pu with Sbase=100MVA
network.add("ShuntImpedance", "Shunt 4", bus="Bus 4", b=0.0033, active=True)
network.add("ShuntImpedance", "Shunt 5", bus="Bus 5", b=0.0016, active=True)
network.add("ShuntImpedance", "Shunt 6", bus="Bus 6", b=0.0037, active=True)

print(network.shunt_impedances[["b", "active"]])

network.generators.loc["Generator 9", ["p_nom","marginal_cost","p_min_pu","p_max_pu"]] = [600, 50, 0.0, 1.0]
network.generators.loc["Generator 10", ["p_nom","marginal_cost","p_min_pu","p_max_pu"]] = [600, 25.0, 0.0, 0.9]
network.generators.loc["Generator 11", ["p_nom","marginal_cost","p_min_pu","p_max_pu"]] = [300, 50, 0.0, 0.9]
network.generators.loc["Generator 12", ["p_nom","marginal_cost","p_min_pu","p_max_pu"]] = [400, 3, 0.0, 0.9]


network.name = "Cigre HV Benachmark"


network.add(
    "Link", "Link 7-8", bus0="Bus 7", bus1="Bus 8", p_set=0, efficiency=0.9, p_nom=500
)
# Voltage Source Converter (VSC)
network.add(
    "ControllableVSC",
    "VSC 1",
    bus="Bus 7",
    q_min=-0,
    q_max=0,
    q_set=0,
    link="Link 7-8",
    side="bus0",
)
network.add(
    "ControllableVSC",
    "VSC 2",
    bus="Bus 8",
    q_min=-0,
    q_max=0,
    q_set=0,
    link="Link 7-8",
    side="bus1",
)


# network.remove("Link","Link 7-8")
# network.remove("ControllableVSC", "VSC 1")
# network.remove("ControllableVSC", "VSC 2")

#network.remove("Line","Line 7-8")

network.pf()

# -----Output-----
snap = network.snapshots[0]

# Bus Voltages and Angles
bus_vmag = network.buses_t.v_mag_pu.iloc[0]
bus_vang_deg = network.buses_t.v_ang.iloc[0] * 180 / np.pi

bus_df = pd.DataFrame({"Voltage [pu]": bus_vmag, "Angle [deg]": bus_vang_deg})
print("\n--- Bus Voltages and Angles ---")
print(bus_df.round(4))

# Line Load Flow
p0 = network.lines_t.p0.iloc[0]
q0 = network.lines_t.q0.iloc[0]
line_df = pd.DataFrame({"P_from [MW]": p0, "Q_from [Mvar]": q0})
print("\n--- Line Flows (From Bus) ---")
print(line_df.round(2))

# Generators
gen_p = network.generators_t.p.iloc[0]
gen_q = network.generators_t.q.iloc[0]
gen_df = pd.DataFrame({"P [MW]": gen_p, "Q [Mvar]": gen_q})
print("\n--- Generator Outputs ---")
print(gen_df.round(2))

# Angle-Stability via Angle Differences
print("\n--- Angle Differences (Δθ) ---")
angle_rows = []
for idx, row in network.lines.iterrows():
    bus0 = row["bus0"]
    bus1 = row["bus1"]
    theta0 = bus_vang_deg[bus0]
    theta1 = bus_vang_deg[bus1]
    delta = theta0 - theta1
    angle_rows.append(
        {"Line": idx, "Bus0": bus0, "Bus1": bus1, "Δθ [deg]": round(delta, 2)}
    )
angle_df = pd.DataFrame(angle_rows)
print(angle_df)

# Sum Shunt-Reactances
total_b = network.shunt_impedances.loc[network.shunt_impedances.active, "b"].sum()
print("\nTotal b active:", total_b)

# Line & Trafo Loadings based on p0 (deprecated)
# print("Leitungsauslastungen: ")
# line_loading = (np.abs(network.lines_t.p0) / network.lines.s_nom.values) * 100
# trafo_loading = (
#     np.abs(network.transformers_t.p0) / network.transformers.s_nom.values
# ) * 100
# print("Line Loadings:", line_loading)
# print("Trafo Loadings:", trafo_loading)


# Lines: P/Q/S und %-Loading after s_nom, sorted
pL = network.lines_t.p0.loc[snap]
qL = network.lines_t.q0.loc[snap]
sL = np.hypot(pL, qL)  # MVA auf Seite 0
sL_nom = network.lines.s_nom

line_tbl = (
    pd.DataFrame(
        {
            "From": network.lines.bus0,
            "To": network.lines.bus1,
            "v_nom [kV]": network.lines.v_nom,
            "length [km]": network.lines["length"]
            if "length" in network.lines
            else np.nan,
            "S_nom [MVA]": sL_nom,
            "P0 [MW]": pL,
            "Q0 [MVAr]": qL,
            "S0 [MVA]": sL,
            "Loading [%]": 100 * sL / sL_nom,
        },
        index=network.lines.index,
    )
    .sort_values("Loading [%]", ascending=False)
    .round(2)
)

print("\n=== Line Loadings ===")
print(line_tbl.to_string())

# Trafos: P/Q/S and %-Loading, incl. phase_shift/x_pu 
pT = network.transformers_t.p0.loc[snap]
qT = network.transformers_t.q0.loc[snap]
sT = np.hypot(pT, qT)
sT_nom = network.transformers.s_nom

trafo_tbl = pd.DataFrame(
    {
        "HV bus": network.transformers.bus0,
        "LV bus": network.transformers.bus1,
        "S_nom [MVA]": sT_nom,
        "P0 [MW]": pT,
        "Q0 [MVAr]": qT,
        "S0 [MVA]": sT,
        "Loading [%]": 100 * sT / sT_nom,
    },
    index=network.transformers.index,
)

trafo_tbl = trafo_tbl.sort_values("Loading [%]", ascending=False).round(2)

print("\n=== Transformer Loading ===")
print(trafo_tbl.to_string())

# %% Prepare and Export Results for Latex



out_dir = _EXAMPLES_DIR / "output" / "tables"
out_dir.mkdir(parents=True, exist_ok=True)


# ggf. kürzere Spaltennamen
line_out = line_tbl.rename(columns={
    "v_nom [kV]":"V[kV]", "length [km]":"L[km]",
    "S_nom [MVA]":"Snom[MVA]", "Loading [%]":"Load[%]"
})
trafo_out = trafo_tbl.rename(columns={
    "S_nom [MVA]":"Snom[MVA]", "Loading [%]":"Load[%]"
})

# Export
line_out.to_latex(
    out_dir / "line_loading.tex",
    index=True, longtable=True, bold_rows=False, escape=False,
    caption="Leitungsauslastungen (nach S, absteigend).",
    label="tab:line-loading",
)
trafo_out.to_latex(
    out_dir / "trafo_loading.tex",
    index=True, longtable=True, bold_rows=False, escape=False,
    caption="Transformatorauslastungen (absteigend).",
    label="tab:trafo-loading",
)


print("Gespeichert nach:")
print("  ", (out_dir / "line_loading.tex").resolve())
print("  ", (out_dir / "trafo_loading.tex").resolve())



# %% für alle Zeitpunkte

controller = VSCController(network)
result = controller.run_p_control(angle_limit_deg=30)
# controller.run_mode(mode="technical", angle_limit_deg=40)
# controller.show_report(snapshots=network.snapshots[0])

# %% nur für bestimmte Zeitpunkte
# from combined_control.VSCController import VSCController
# controller = VSCController(network)

# result = controller.run_p_control(
#     angle_limit_deg=30,
#     report_snapshots=[
#         pd.Timestamp("2025-07-24 12:00:00"),
#         pd.Timestamp("2025-07-24 15:00:00")
#     ]
# )


p_link_vals = np.linspace(-200, 700, 50)
zf_vals = []

for p_set in p_link_vals:
    network.links.at["Link 7-8", "p_set"] = p_set
    network.lpf()

    # Leitungsauslastung
    f_line = np.sqrt(network.lines_t.p0.loc[0] ** 2)
    s_nom_line = network.lines.s_nom
    rel_line = f_line / s_nom_line
    zf_line = (rel_line**2).sum()

    # Trafoauslastung
    if not network.transformers.empty:
        f_trafo = np.sqrt(network.transformers_t.p0.loc[0] ** 2)
        s_nom_trafo = network.transformers.s_nom
        rel_trafo = f_trafo / s_nom_trafo
        zf_trafo = (rel_trafo**2).sum()
    else:
        zf_trafo = 0

    # Gesamte Zielfunktion wie im Optimierer
    zf_total = zf_line + zf_trafo
    zf_vals.append(zf_total)

plt.figure(figsize=(8, 5))
plt.plot(p_link_vals, zf_vals, marker="o")
plt.xlabel("p_link [MW]")
plt.ylabel("Zielfunktionswert (ZF: ∑(f/s_nom)² inkl. Trafos)")
plt.title("Zielfunktion vs. p_link – wie im Optimierer")
plt.grid(True)
plt.show()
# Minimum der Zielfunktion und zugehöriger p_link-Wert
min_index = np.argmin(zf_vals)
min_p_link = p_link_vals[min_index]
min_zf = zf_vals[min_index]

print(f"Minimum der Zielfunktion: {min_zf:.4f} bei p_link = {min_p_link:.2f} MW")
# %% Q_Optimizer

controller = VSCController(network)
# results = controller.run_q_control(run_vsi=False, snapshot=network.snapshots[0])
# results = controller.run_p_control(angle_limit_deg=30)
# results = controller.run_q_control(run_vsi=True)

# result = controller.run_mode(mode="P_control", angle_limit_deg=25)
# result = controller.run_mode(mode="Q_control")

p_results, q_result = controller.run_mode(
    mode="combined", angle_limit_deg=30, S_rated=400
)

# %% Test Linopy P-Controller

# 1) Sicherstellen, dass mindestens 1 Snapshot existiert
if len(network.snapshots) == 0:
    network.set_snapshots(pd.Index([pd.Timestamp("2000-01-01")]))

# 2) Optional: initialer AC-PF, damit Startwerte/Reports stimmig sind
try:
    network.pf()
except Exception:
    pass

# 3) Einmal laufen lassen (Gurobi empfohlen; alternativ Highs — siehe unten)
results = link_optimization_linopy(
    network,
    angle_limit_deg=25,  # oder dein üblicher Wert
    pf_callback=network.pf,  # pf() nach dem Solve ausführen
    lpf_callback=getattr(network, "lpf", None),  # falls vorhanden
    max_line_loading=0.95,
    detail_level=None,  # kein Auto-Report in der Funktion
    snapshots="all",  # oder eine Liste/Index mit konkreten Snaps
    solver_name="gurobi",  # <<--- wichtig
    solver_io_api="direct",
    solver_options={"MIPGap": 1e-3, "TimeLimit": 300},
)

# 4) Kurzer Report (gleiche Funktion wie bei dir)
show_snapshot_report(results, network, snapshots="all", detail_level=1)

# 5) Minimaler Sanity-Check
for snap, res in results.items():
    print(
        f"\nSnapshot {snap}: status={res.get('status', '?')}, "
        f"Obj={res.get('objective')}, Verletzung={res.get('verletzung')}"
    )
print("Neue Link-Setpoints:\n", network.links["p_set"])


# %% Preparations and Run OPF

network.optimize()
snap = network.snapshots[0]

# Generation by each generator
print(network.generators_t.p.loc[snap].sort_values(ascending=False).to_string())

# System costs (only mc considered)
mc = network.generators["marginal_cost"]            # €/MWh
p  = network.generators_t.p.loc[snap]               # MW
cost_snapshot = p.mul(mc, axis=0).sum()             # € per Snapshot (with 1h considered = €)
print(f"\n Costs Snapshot: {cost_snapshot:,.2f} €")
# print the load active power (P) consumption
print(f"\n Load active power: {network.loads_t.p}")
# print the generator active power (P) dispatch
print(f"\n Generator active power dispatch: {network.generators_t.p}")
# print the clearing price (corresponding to gas)
print(f"\n Clearing price: {network.buses_t.marginal_price}")
# Line Flows/Loadings
flows = network.lines_t.p0.loc[snap]                                # MW
loading = 100 * flows.abs() / network.lines.s_nom                   # %
print("\n Line Loadings DC-approx. [%]:")
print(loading.sort_values(ascending=False).to_string())






# %% Preparations and Run SCLOPF

snap = network.snapshots[0]

branch_outages = network.lines.index[:] # includes all lines in n-1-analysis

network.optimize.optimize_security_constrained(snap, branch_outages=branch_outages)

# Generation by each generator
print(f"\n Generator P Dispatch: {network.generators_t.p.loc[snap].sort_values(ascending=False).to_string()}")

# System costs (only mc considered)
mc = network.generators["marginal_cost"]            # €/MWh
p  = network.generators_t.p.loc[snap]               # MW
cost_snapshot = p.mul(mc, axis=0).sum()             # € per Snapshot (with 1h considered = €)
print(f"\n Costs Snapshot: {cost_snapshot:,.2f} €")
# print the load active power (P) consumption
print(f"\n Load active power: {network.loads_t.p}")

# print the clearing price (corresponding to gas)
print(f"\n Clearing price: {network.buses_t.marginal_price}")
# Line Flows/Loadings
flows = network.lines_t.p0.loc[snap]                                # MW
loading = 100 * flows.abs() / network.lines.s_nom                   # %
print("\n Line Loadings DC-approx. [%]:")
print(loading.sort_values(ascending=False).to_string())


# Set P to econ. optimised P
network.generators_t.p_set = network.generators_t.p_set.reindex(
    columns=network.generators.index)
network.generators_t.p_set.loc[snap] = network.generators_t.p.loc[snap]
network.storage_units_t.p_set = network.storage_units_t.p_set.reindex(
    columns=network.storage_units.index
)
network.storage_units_t.p_set.loc[snap] = network.storage_units_t.p.loc[snap]
print(network.generators_t.p.loc[snap].equals(network.generators_t.p_set.loc[snap]))
print(network.generators_t.p_set.loc[snap].sort_values(ascending=False).to_string())

res = network.lpf_contingency(snap, branch_outages=branch_outages)
p_by_outage = res
loading_c = 100 * p_by_outage.abs().divide(network.lines.s_nom, axis=1)

print(loading_c)
print("\nWorst-case Auslastung je Leitung [%]:")
print(loading_c.max(axis=0).sort_values(ascending=False).head(10))

print("\nWorst-case Auslastung je Ausfall [%]:")
print(loading_c.max(axis=1).sort_values(ascending=False).head(10))

max_loading = (
    abs(res.divide(network.passive_branches().s_nom, axis=0)).describe().loc["max"]
)
print("max loading: ", max_loading)
# %% SCLOPF Variante 2

import pandas as pd
import numpy as np

# ===== 1) SCLOPF ausführen =====
snap = network.snapshots[0]

branch_outages_scopf = network.lines.index.difference(["Line 7-8"])

# SCLOPF
stat = network.optimize.optimize_security_constrained(
    snapshots=[snap],
    branch_outages=branch_outages_scopf,
    solver_name="gurobi",
    assign_all_duals=True)

# ===== 2) Basis-Ergebnisse (N-0) ausgeben =====
print("\nGenerator P-Dispatch [MW]:")
print(network.generators_t.p.loc[snap].sort_values(ascending=False).to_string())

mc = network.generators["marginal_cost"]  # €/MWh
p  = network.generators_t.p.loc[snap]     # MW
cost_snapshot = (p * mc).sum()
print(f"\nKosten Snapshot [€/h]: {cost_snapshot:,.2f}")

print("\nLMPs [€/MWh]:")
print(network.buses_t.marginal_price.loc[snap].sort_values().to_string())

# N-0-Leitungsauslastungen
flows_n0 = network.lines_t.p0.loc[snap]
loading_n0 = 100 * flows_n0.abs() / network.lines.s_nom
print("\nLeitungs-Auslastungen N-0 [%]:")
print(loading_n0.sort_values(ascending=False).to_string())

# ===== 3) Ex-post N-1-Check mit eingefrorenem SCLOPF-Dispatch =====
network.model = None  
n2 = network.copy()

# Generatoren fix fahren lassen (PQ) und p_set-DataFrame anlegen/füllen
#n2.generators["control"] = "PQ"
n2.generators_t.p_set = pd.DataFrame(0.0, index=n2.snapshots, columns=n2.generators.index)
n2.generators_t.p_set.loc[snap, :] = network.generators_t.p.loc[snap]


# # Links (falls vorhanden) fix fahren lassen: p_set = p0 der OPF-Lösung
if not n2.links.empty:
    n2.links_t.p_set = pd.DataFrame(0.0, index=n2.snapshots, columns=n2.links.index)
    n2.links_t.p_set.loc[snap, :] = network.links_t.p0.loc[snap]

# StorageUnits (falls vorhanden) fix fahren lassen
if not n2.storage_units.empty:
    n2.storage_units_t.p_set = pd.DataFrame(0.0, index=n2.snapshots, columns=n2.storage_units.index)
    n2.storage_units_t.p_set.loc[snap, :] = network.storage_units_t.p.loc[snap]

#

# Einheitliche Outage-Liste für den Check (MultiIndex vermeidet Warnungen)
branch_outages_check = pd.MultiIndex.from_product(
    [["Line"], n2.lines.index],
    names=["component", "name"]
)

# DC-Contingency-Lastfluss mit eingefrorenem Dispatch
# Rows=branches & Columns=Cases
res = n2.lpf_contingency(snap, branch_outages=branch_outages_scopf)
# res: Zeilen = passive branches (Lines+Transformer), Spalten = Fälle (base + Ausfälle)

# Effektive Grenzwerte (inkl. s_max_pu) für alle passiven Zweige bilden und sauber ausrichten
pb = n2.passive_branches()  # enthält sowohl Lines als auch Transformer
s_max = (pb.s_max_pu.fillna(1.0) * pb.s_nom)

# Prozent-Auslastungen je (Zweig, Fall)
loading_c = 100.0 * res.abs().divide(s_max, axis=0)

print("\nWorst-case Auslastung je Zweig über alle Ausfälle [%]:")
print(loading_c.max(axis=1).sort_values(ascending=False).head(20).to_string())

print("\nWorst-case Auslastung je Ausfall über alle Zweige [%]:")
print(loading_c.max(axis=0).sort_values(ascending=False).head(20).to_string())

# (Optional) Top (Zweig, Ausfall)-Paare
top_pairs = loading_c.stack().sort_values(ascending=False).head(20)
print("\nTop (Ausfall, Zweig)-Paare nach Auslastung [%]:")
print("Spalte 1: Ausgelastete Leitung, Spalte 2: Ausgefallene Leitung")
print(top_pairs.to_string())

# (Optional) Plausibilitätscheck: alles <= 100 % (kleine Numerik-Toleranzen sind ok)
max_pct = loading_c.max().max()
print(f"\nMaximale Auslastung im N-1-Check [%]: {max_pct:.3f}")




slack_gen = n2.generators.index[0]  # wähle einen Slack (ggf. spezifischen Namen einsetzen)
n2.generators.at[slack_gen, "control"] = "Slack"

out_ac = n2.pf()
print("\nAC-PF Result:", out_ac)

# AC: Spannungen und MVA-Auslastungen
v = n2.buses_t.v_mag_pu.loc[snap]
print("\nBus-Spannungen [p.u.]:")
print(v.sort_values().to_string())

P = n2.lines_t.p0.loc[snap]
Q = n2.lines_t.q0.loc[snap]
S = np.hypot(P, Q) 
s_nom = network.lines.s_nom
loading_S = 100 * S / s_nom 

# S_line = (n2.lines_t.p0.loc[snap]**2 + n2.lines_t.q0.loc[snap]**2)**0.5
# s_line_max = n2.lines.s_nom * n2.lines.s_max_pu.fillna(1.0)
# loading_line_ac = 100.0 * S_line / s_line_max
print("\nLeitungen AC-Loading [%]:")
#print(loading_line_ac.sort_values(ascending=False).head(20).to_string())
print(loading_S.sort_values(ascending=False).round(2))

S_traf = (n2.transformers_t.p0.loc[snap]**2 + n2.transformers_t.q0.loc[snap]**2)**0.5 if not n2.transformers.empty else pd.Series(dtype=float)
s_traf_max = n2.transformers.s_nom * n2.transformers.s_max_pu.fillna(1.0) if not n2.transformers.empty else pd.Series(dtype=float)
if not S_traf.empty:
    loading_traf_ac = 100.0 * S_traf / s_traf_max
    print("\nTrafos AC-Loading [%]:")
    print(loading_traf_ac.sort_values(ascending=False).head(20).to_string())

# %% SCLOPF Variante 3 – Link/VSC deaktiviert, "Line 7-8" NICHT als Ausfall

import pandas as pd
import numpy as np



# ---------- 0) HVDC-Link komplett deaktivieren ----------
if "p_set" not in getattr(network.links_t, "_series", {}):
    network.links_t["p_set"] = pd.DataFrame(0.0, index=network.snapshots,
                                            columns=network.links.index)
if "p_set" not in network.links.columns:
    network.links["p_set"] = 0.0

network.links["p_set"] = 0.0
network.links_t.p_set.loc[:, :] = 0.0
network.links.loc[:, "p_nom"] = 0.0 

# ---------- 1) SCLOPF ohne Ausfall "Line 7-8" ----------
snap = network.snapshots[0]

# exakten Namen "Line 7-8" robust finden (Unicode-Dash etc.)
target = None
for n in network.lines.index:
    if n.replace("–", "-").strip() == "Line 7-8":
        target = n
        break
assert target is not None, "Line 7-8 nicht gefunden – Namen prüfen."

# WICHTIG: hier KEIN MultiIndex übergeben, sondern ein einfacher Index!
branch_outages_scopf = network.lines.index.difference([target])

stat = network.optimize.optimize_security_constrained(
    snapshots=[snap],
    branch_outages=branch_outages_scopf,  # einfacher Index
    solver_name="gurobi",
    assign_all_duals=True,
)
print("SCLOPF-Status:", stat)

# ---------- 2) Basis-Ergebnisse (N-0) ----------
print("\nGenerator P-Dispatch [MW]:")
print(network.generators_t.p.loc[snap].sort_values(ascending=False).to_string())

mc = network.generators["marginal_cost"]
p  = network.generators_t.p.loc[snap]
cost_snapshot = (p * mc).sum()
print(f"\nKosten Snapshot [€/h]: {cost_snapshot:,.2f}")

lmp = network.buses_t.marginal_price.loc[snap]
print("\nLMPs [€/MWh]:")
print("(leer)" if lmp.empty else lmp.sort_values().to_string())

flows_n0 = network.lines_t.p0.loc[snap]
loading_n0 = 100.0 * flows_n0.abs() / network.lines.s_nom
print("\nLeitungs-Auslastungen N-0 [%] (DC, |P|/S_nom):")
print(loading_n0.sort_values(ascending=False).to_string())

# ---------- 3) Ex-post N-1-Check mit eingefrorenem SCLOPF-Dispatch ----------
network.model = None
n2 = network.copy()

# Generator-Dispatch einfrieren
n2.generators_t.p_set = pd.DataFrame(0.0, index=n2.snapshots, columns=n2.generators.index)
n2.generators_t.p_set.loc[snap, :] = network.generators_t.p.loc[snap]

# Link P = 0 halten
if "p_set" not in getattr(n2.links_t, "_series", {}):
    n2.links_t["p_set"] = pd.DataFrame(0.0, index=n2.snapshots, columns=n2.links.index)
if "p_set" not in n2.links.columns:
    n2.links["p_set"] = 0.0
n2.links["p_set"] = 0.0
n2.links_t.p_set.loc[snap, :] = 0.0
if not n2.links_t.p0.empty:
    n2.links_t.p0.loc[snap, :] = 0.0

# Ex-post DC-Contingency mit GENAU derselben (einfachen) Outage-Liste
res = n2.lpf_contingency(snap, branch_outages=branch_outages_scopf)

# Prüfen, ob "Line 7-8" als Ausfall noch auftaucht (sollte False sein)
cols_out = [c[1] for c in res.columns if isinstance(c, tuple)]
print("\nIst 'Line 7-8' in den Ausfallspalten? ->", (target in cols_out))

# ---------- 4) Prozent-Auslastungen je (Zweig, Fall) – MultiIndex-sicher ----------
pb = n2.passive_branches()  # Lines + Trafos
s_max = (pb.s_nom * pb.s_max_pu.fillna(1.0)).astype(float)

# an res.index ausrichten und numerisch teilen
s_vec = s_max.reindex(res.index).to_numpy().reshape(-1, 1)
loading_c = pd.DataFrame(
    100.0 * np.abs(res.to_numpy()) / s_vec,
    index=res.index,
    columns=res.columns,
)

print("\nWorst-case Auslastung je Zweig über alle Ausfälle [%]:")
print(loading_c.max(axis=1).sort_values(ascending=False).head(20).to_string())

print("\nWorst-case Auslastung je Ausfall über alle Zweige [%]:")
print(loading_c.max(axis=0).sort_values(ascending=False).head(20).to_string())

top_pairs = loading_c.stack().sort_values(ascending=False).head(20)
print("\nTop (Ausfall, Zweig)-Paare nach Auslastung [%]:")
print("Spalte 1 = ausgelasteter Zweig (Zeile), Spalte 2 = ausgefallener Zweig (Spalte)")
print(top_pairs.to_string())

max_pct = loading_c.max().max()
print(f"\nMaximale Auslastung im N-1-Check [%]: {max_pct:.3f}")

# ---------- 5) AC-PF (N-0) und AC-Loadings ----------
slack_gen = n2.generators.index[0]
n2.generators.at[slack_gen, "control"] = "Slack"

out_ac = n2.pf()
print("\nAC-PF Result:", out_ac)

v = n2.buses_t.v_mag_pu.loc[snap]
print("\nBus-Spannungen [p.u.]:")
print(v.sort_values().to_string())

P = n2.lines_t.p0.loc[snap]
Q = n2.lines_t.q0.loc[snap]
S = np.hypot(P, Q)
s_line_max = n2.lines.s_nom * n2.lines.s_max_pu.fillna(1.0)
loading_line_ac = 100.0 * S / s_line_max

print("\nLeitungen AC-Loading [%] (|S|/S_max):")
print(loading_line_ac.sort_values(ascending=False).round(2).to_string())

if not n2.transformers.empty:
    PT = n2.transformers_t.p0.loc[snap]
    QT = n2.transformers_t.q0.loc[snap]
    ST = np.hypot(PT, QT)
    s_traf_max = n2.transformers.s_nom * n2.transformers.s_max_pu.fillna(1.0)
    loading_traf_ac = 100.0 * ST / s_traf_max
    print("\nTrafos AC-Loading [%] (|S|/S_max):")
    print(loading_traf_ac.sort_values(ascending=False).round(2).to_string())


# %% SCLOPF – Link/VSC deaktiviert, diesmal MIT Ausfallfall "Line 7-8"

import pandas as pd
import numpy as np

# ---------- 0) HVDC-Link für Lastfluss deaktivieren ----------
# (Hinweis: Für OPF ist der Link damit NICHT deaktiviert. Falls du auch im OPF
# keinen Link haben willst: p_nom=0 für den/die Links setzen oder Link entfernen.)
# %% SCLOPF Variante 3 – Link/VSC deaktiviert, "Line 7-8" NICHT als Ausfallfall

import pandas as pd
import numpy as np


# Sicherstellen, dass eine bool'sche 'status'-Spalte existiert
for comp in ("lines", "transformers"):
    df = getattr(n2, comp, None)
    if df is not None and not df.empty:
        if "status" not in df.columns:
            df["status"] = True  # alles aktiv als Default



# ---------- 0) HVDC-Link komplett deaktivieren ----------

if "p_set" not in getattr(network.links_t, "_series", {}):
    network.links_t["p_set"] = pd.DataFrame(0.0, index=network.snapshots,
                                            columns=network.links.index)
if "p_set" not in network.links.columns:
    network.links["p_set"] = 0.0

network.links["p_set"] = 0.0
network.links_t.p_set.loc[:, :] = 0.0

# ---------- 1) SCLOPF mit ALLEN Leitungs-Ausfällen (inkl. "Line 7-8") ----------
snap = network.snapshots[0]

# WICHTIG: einfacher Index (kein MultiIndex), sonst pandas-Fehler
branch_outages_scopf = network.lines.index.copy()

stat = network.optimize.optimize_security_constrained(
    snapshots=[snap],
    branch_outages=branch_outages_scopf,  # alle Lines als N-1-Fälle
    solver_name="gurobi",
    assign_all_duals=True,
)
print("SCLOPF-Status:", stat)

# ---------- 2) Basis-Ergebnisse (N-0) ----------
print("\nGenerator P-Dispatch [MW]:")
print(network.generators_t.p.loc[snap].sort_values(ascending=False).to_string())

mc = network.generators["marginal_cost"]
p  = network.generators_t.p.loc[snap]
cost_snapshot = (p * mc).sum()
print(f"\nKosten Snapshot [€/h]: {cost_snapshot:,.2f}")

lmp = network.buses_t.marginal_price.loc[snap]
print("\nLMPs [€/MWh]:")
print("(leer)" if lmp.empty else lmp.sort_values().to_string())

flows_n0 = network.lines_t.p0.loc[snap]
loading_n0 = 100.0 * flows_n0.abs() / network.lines.s_nom
print("\nLeitungs-Auslastungen N-0 [%] (DC, |P|/S_nom):")
print(loading_n0.sort_values(ascending=False).to_string())

# ---------- 3) Ex-post N-1-Check mit eingefrorenem SCLOPF-Dispatch ----------
network.model = None
n2 = network.copy()

# Generator-Dispatch einfrieren
n2.generators_t.p_set = pd.DataFrame(0.0, index=n2.snapshots, columns=n2.generators.index)
n2.generators_t.p_set.loc[snap, :] = network.generators_t.p.loc[snap]

# Link-P im (L)PF auf 0 halten
if "p_set" not in getattr(n2.links_t, "_series", {}):
    n2.links_t["p_set"] = pd.DataFrame(0.0, index=n2.snapshots, columns=n2.links.index)
if "p_set" not in n2.links.columns:
    n2.links["p_set"] = 0.0
n2.links["p_set"] = 0.0
n2.links_t.p_set.loc[snap, :] = 0.0
if not n2.links_t.p0.empty:
    n2.links_t.p0.loc[snap, :] = 0.0

# Ex-post DC-Contingency (gleiche Ausfallmenge wie im SCLOPF)
res = n2.lpf_contingency(snap, branch_outages=branch_outages_scopf)

# ---------- 4) Prozent-Auslastungen je (Zweig, Fall) – robust ohne MultiIndex-Quirks ----------
pb = n2.passive_branches()  # Lines + Trafos
s_max = (pb.s_nom * pb.s_max_pu.fillna(1.0)).astype(float)

# an res.index ausrichten und numerisch teilen
s_vec = s_max.reindex(res.index).to_numpy().reshape(-1, 1)
loading_c = pd.DataFrame(
    100.0 * np.abs(res.to_numpy()) / s_vec,
    index=res.index,
    columns=res.columns,
)

print("\nWorst-case Auslastung je Zweig über alle Ausfälle [%]:")
print(loading_c.max(axis=1).sort_values(ascending=False).head(20).to_string())

print("\nWorst-case Auslastung je Ausfall über alle Zweige [%]:")
print(loading_c.max(axis=0).sort_values(ascending=False).head(20).to_string())

top_pairs = loading_c.stack().sort_values(ascending=False).head(20)
print("\nTop (Ausfall, Zweig)-Paare nach Auslastung [%]:")
print("Spalte 1 = ausgelasteter Zweig (Zeile), Spalte 2 = ausgefallener Zweig (Spalte)")
print(top_pairs.to_string())

max_pct = loading_c.max().max()
print(f"\nMaximale Auslastung im N-1-Check [%]: {max_pct:.3f}")

# ---------- 5) AC-PF (N-0) und AC-Loadings ----------
slack_gen = n2.generators.index[0]
n2.generators.at[slack_gen, "control"] = "Slack"

out_ac = n2.pf()
print("\nAC-PF Result:", out_ac)

v = n2.buses_t.v_mag_pu.loc[snap]
print("\nBus-Spannungen [p.u.]:")
print(v.sort_values().to_string())

P = n2.lines_t.p0.loc[snap]
Q = n2.lines_t.q0.loc[snap]
S = np.hypot(P, Q)
s_line_max = n2.lines.s_nom * n2.lines.s_max_pu.fillna(1.0)
loading_line_ac = 100.0 * S / s_line_max

print("\nLeitungen AC-Loading [%] (|S|/S_max):")
print(loading_line_ac.sort_values(ascending=False).round(2).to_string())

if not n2.transformers.empty:
    PT = n2.transformers_t.p0.loc[snap]
    QT = n2.transformers_t.q0.loc[snap]
    ST = np.hypot(PT, QT)
    s_traf_max = n2.transformers.s_nom * n2.transformers.s_max_pu.fillna(1.0)
    loading_traf_ac = 100.0 * ST / s_traf_max
    print("\nTrafos AC-Loading [%] (|S|/S_max):")
    print(loading_traf_ac.sort_values(ascending=False).round(2).to_string())


# %% SCLOPF – Link/VSC AKTIV, N-1 über ALLE AC-Leitungen (inkl. "Line 7-8")

import pandas as pd
import numpy as np

# ===== 0) Voraussetzung: Link + VSC existieren und sind für OPF nutzbar =====
# Falls der Link noch nicht existiert, legen wir ihn an (ansonsten passiert nichts).
if "Link 7-8" not in getattr(network, "links", pd.DataFrame(index=[])).index:
    network.add("Link", "Link 7-8", bus0="Bus 7", bus1="Bus 8",
                p_nom=500.0, efficiency=0.9)  # p_nom > 0 => im OPF nutzbar
    # VSCs optional für Q-Unterstützung
    network.add("ControllableVSC", "VSC 7", bus="Bus 7", q_min=-500, q_max=500,
                q_set=0.0, link="Link 7-8", side="bus0")
    network.add("ControllableVSC", "VSC 8", bus="Bus 8", q_min=-500, q_max=500,
                q_set=0.0, link="Link 7-8", side="bus1")

# WICHTIG: Für das OPF ist p_set irrelevant. Entscheidend ist p_nom.
# Falls du zuvor p_nom=0 gesetzt hattest, setze es wieder >0:
network.links.loc["Link 7-8", "p_nom"] = max(1.0, network.links.loc["Link 7-8", "p_nom"])

# Optional: bidirektionale Nutzung im OPF erlauben (Default ist häufig nur >=0)
# Wenn nötig einkommentieren:
# network.links.loc["Link 7-8", ["p_min_pu", "p_max_pu"]] = [-1.0, 1.0]

# Calcultaion of Initial Values


snap = network.snapshots[0]
n0_before = network.copy()
n0_before.lpf(snap)
# flows_before = n0_before.lines_t.p0.loc[snap]
# print("Flows before", flows_before)
# loading_before = 100.0 * flows_before.abs() / network.lines.s_nom
# print("\nLeitungs-Auslastungen before [%] (DC, ||/S_nom):")
# print(loading_before.sort_values(ascending=False).to_string())

n0_before.pf(snap)
P = n0_before.lines_t.p0.loc[snap]
Q = n0_before.lines_t.q0.loc[snap]
S = np.hypot(P, Q)
s_line_max = n0_before.lines.s_nom * n0_before.lines.s_max_pu.fillna(1.0)
loading_line_ac_before = 100.0 * S / s_line_max
print("\nLeitungs-Auslastungen before [%] (AC, |S|/S_nom):")
print(loading_line_ac_before.sort_values(ascending=False).to_string())

V_initial_AC= n0_before.buses_t.v_mag_pu


# ===== 1) SCLOPF-Fallmenge: ALLE AC-Leitungen (inkl. "Line 7-8") =====
branch_outages_scopf = network.lines.index.copy()  # einfacher Index, keine MultiIndex-Fehler

stat = network.optimize.optimize_security_constrained(
    snapshots=[snap],
    branch_outages=branch_outages_scopf,  # ALLE Lines in N-1
    solver_name="gurobi",
    assign_all_duals=True,
)
print("SCLOPF-Status:", stat)

# ===== 2) Basis-Ergebnisse (N-0) =====
print("\nGenerator P-Dispatch [MW]:")
print(network.generators_t.p.loc[snap].sort_values(ascending=False).to_string())

mc = network.generators["marginal_cost"]
p  = network.generators_t.p.loc[snap]
cost_snapshot = (p * mc).sum()
print(f"\nKosten Snapshot [€/h]: {cost_snapshot:,.2f}")

lmp = network.buses_t.marginal_price.loc[snap]
print("\nLMPs [€/MWh]:")
print("(leer)" if lmp.empty else lmp.sort_values().to_string())

flows_n0 = network.lines_t.p0.loc[snap]
loading_n0 = 100.0 * flows_n0.abs() / network.lines.s_nom
print("\nLeitungs-Auslastungen N-0 [%] (DC, |P|/S_nom):")
print(loading_n0.sort_values(ascending=False).to_string())

# ===== 3) Ex-post N-1-Check mit eingefrorenem SCLOPF-Dispatch =====
network.model = None  # wichtig vor copy()
n2 = network.copy()

n2._whoami = "N2"


# --- Generator-Dispatch einfrieren. Notwendig für AC-Lastfluss, da er auf andere "Obejekte" zugreift, als der lpf!!
n2.generators_t.p_set = pd.DataFrame(0.0, index=n2.snapshots, columns=n2.generators.index)
n2.generators_t.p_set.loc[snap, :] = network.generators_t.p.loc[snap]

# --- Link-Dispatch einfrieren (auf OPF-Ergebnis) ---
if "p_set" not in getattr(n2.links_t, "_series", {}):
    n2.links_t["p_set"] = pd.DataFrame(0.0, index=n2.snapshots, columns=n2.links.index)

# OPF-Resultat holen: bevorzugt links_t.p, sonst links_t.p0
if "p" in network.links_t and not network.links_t.p.empty:
    link_flow = network.links_t.p.loc[snap]      # bevorzugt
elif "p0" in network.links_t and not network.links_t.p0.empty:
    link_flow = network.links_t.p0.loc[snap]     # fallback
else:
    link_flow = pd.Series(0.0, index=n2.links.index)

# Auf p_set (zeitaufgelöst) spiegeln
n2.links_t.p_set.loc[snap, :] = link_flow

# Auch die statische Spalte anlegen/setzen (für Code, der statisch liest)
if "p_set" not in n2.links.columns:
    n2.links["p_set"] = 0.0
# optional: nur für den relevanten Link setzen
if "Link 7-8" in n2.links.index:
    n2.links.at["Link 7-8", "p_set"] = n2.links_t.p_set.loc[snap, "Link 7-8"]


# --- (falls vorhanden) StorageUnits ebenfalls einfrieren
if not n2.storage_units.empty:
    if "p_set" not in getattr(n2.storage_units_t, "_series", {}):
        n2.storage_units_t["p_set"] = pd.DataFrame(0.0, index=n2.snapshots, columns=n2.storage_units.index)
    n2.storage_units_t.p_set.loc[snap, :] = network.storage_units_t.p.loc[snap]

# --- LPF-Contingency mit derselben Ausfallliste (ALLE Lines inkl. "Line 7-8")
res = n2.lpf_contingency(snap, branch_outages=branch_outages_scopf)

# ===== 4) Prozent-Auslastungen je (Zweig, Fall) – robust ohne MultiIndex-Quirks =====
pb = n2.passive_branches()  # Lines + Trafos
s_max = (pb.s_nom * pb.s_max_pu.fillna(1.0)).astype(float)

# sichere Ausrichtung: s_max an res.index, dann numerisch teilen
s_vec = s_max.reindex(res.index).to_numpy().reshape(-1, 1)
loading_c = pd.DataFrame(
    100.0 * np.abs(res.to_numpy()) / s_vec,
    index=res.index,
    columns=res.columns,
)

print("\nWorst-case Auslastung je Zweig über alle Ausfälle (dc-based) [%]:")
print(loading_c.max(axis=1).sort_values(ascending=False).head(20).to_string())

print("\nWorst-case Auslastung je Ausfall über alle Zweige (dc-based) [%]:")
print(loading_c.max(axis=0).sort_values(ascending=False).head(20).to_string())

top_pairs = loading_c.stack().sort_values(ascending=False).head(20)
print("\nTop (Ausfall, Zweig)-Paare nach Auslastung (dc-based) [%]:")
print("Spalte 1 = ausgelasteter Zweig (Zeile), Spalte 2 = ausgefallener Zweig (Spalte)")
print(top_pairs.to_string())

max_pct = loading_c.max().max()
print(f"\nMaximale Auslastung im N-1-Check (dc-based) [%]: {max_pct:.3f}")

# ===== 5) AC-PF (N-0) & AC-Loadings =====
# Für AC-PF einen Slack definieren (falls nicht vorhanden)
slack_gen = n2.generators.index[0]
n2.generators.at[slack_gen, "control"] = "Slack"

out_ac = n2.pf()
print("\nAC-PF Result (ac-based):", out_ac)

v = n2.buses_t.v_mag_pu.loc[snap]
print("\nBus-Spannungen [p.u.] (ac-based):")
print(v.sort_values().to_string())

P = n2.lines_t.p0.loc[snap]
Q = n2.lines_t.q0.loc[snap]
S = np.hypot(P, Q)
s_line_max = n2.lines.s_nom * n2.lines.s_max_pu.fillna(1.0)
loading_line_ac = 100.0 * S / s_line_max

print("\nLeitungen AC-Loading [%] (|S|/S_max) (ac-based):")
print(loading_line_ac.sort_values(ascending=False).round(2).to_string())

if not n2.transformers.empty:
    PT = n2.transformers_t.p0.loc[snap]
    QT = n2.transformers_t.q0.loc[snap]
    ST = np.hypot(PT, QT)
    s_traf_max = n2.transformers.s_nom * n2.transformers.s_max_pu.fillna(1.0)
    loading_traf_ac = 100.0 * ST / s_traf_max
    print("\nTrafos AC-Loading [%] (|S|/S_max) (ac-based):")
    print(loading_traf_ac.sort_values(ascending=False).round(2).to_string())

#%% N-1 AC-based check
def summarize_setpoint_and_security(network, n2, snap, margin=0.96, outages=None):
    import numpy as np
    import pandas as pd
    
        # Sicherstellen, dass eine bool'sche 'status'-Spalte existiert
    for comp in ("lines", "transformers"):
        df = getattr(n2, comp, None)
        if df is not None and not df.empty:
            if "status" not in df.columns:
                df["status"] = True  # alles aktiv als Default


    if outages is None:
        outages = list(n2.lines.index)

    # --- Setpoints: DC (aus SCLOPF) ---
    gen_p_dc = network.generators_t.p.loc[snap].copy()
    series_map_dc = getattr(network.links_t, "_series", {})
    if "p" in series_map_dc:
        link_p_dc = network.links_t.p.loc[snap].copy()
    elif "p0" in series_map_dc:
        link_p_dc = network.links_t.p0.loc[snap].copy()
    else:
        link_p_dc = pd.Series(dtype=float)

    # --- Setpoints: AC (eingefroren, was pf() nutzt) ---
    gen_p_ac = getattr(n2.generators_t, "p_set", n2.generators_t.p).loc[snap].copy()
    series_map_ac = getattr(n2.links_t, "_series", {})
    if "p_set" in series_map_ac:
        link_p_ac = n2.links_t.p_set.loc[snap].copy()
    elif "p0" in series_map_ac:
        link_p_ac = n2.links_t.p0.loc[snap].copy()
    else:
        link_p_ac = pd.Series(dtype=float)

    # --- DC N-1 Check (lpf_contingency) ---
    res_dc = n2.lpf_contingency(snap, branch_outages=outages)
    pb = n2.passive_branches()
    smax = (pb.s_nom * pb.s_max_pu.fillna(1.0)).astype(float)
    loading_dc = 100.0 * res_dc.abs().div(smax, axis=0)
    for o in loading_dc.columns:
        if o in loading_dc.index:
            loading_dc.loc[o, o] = np.nan  # Diagonale ignorieren

    worst_dc_val = float(loading_dc.max().max())
    worst_dc_pair = loading_dc.stack().idxmax()
    secure_dc = worst_dc_val <= 100.0 * margin + 1e-6

    # --- Helper: robustes Auslesen von 'converged' als Skalar ---
    def _pf_converged(out_obj) -> bool:
        conv = out_obj.get("converged", None)
        if conv is None:
            return True  # konservativ
        arr = np.asarray(conv).ravel()
        if arr.size == 0:
            return False
        # arr[0] ist oft np.bool_ / bool
        return bool(arr[0])

    # --- AC N-1 Check (pro Ausfall AC-pf) ---
    limit_pct = 100.0 * margin
    per_outage_max = {}
    violations = []

    for o in outages:
        n = n2.copy()
    
        # Zweig abschalten
        if o in n.lines.index:
            n.lines.loc[o, "status"] = False
        # elif hasattr(n, "transformers") and o in n.transformers.index:
        #     n.transformers.loc[o, "status"] = False
    
        out = n.pf(snap)
    
        # ... danach alive-Indices über die Spalte 'status' holen:
        alive_lines = n.lines.index[n.lines["status"].astype(bool)]
        print(alive_lines)
        PL = n.lines_t.p0.loc[snap, alive_lines].astype(float)
        QL = (n.lines_t.q0.loc[snap, alive_lines].astype(float)
        if "q0" in getattr(n.lines_t, "_series", {}) else 0.0 * PL)
        SL = np.hypot(PL, QL)
        SLmax = (n.lines.s_nom * n.lines.s_max_pu.fillna(1.0)).loc[alive_lines]
        loadL = 100.0 * SL / SLmax
        print(loadL)
        worst = float(loadL.max())

        # Transformers optional
        if not n.transformers.empty:
            alive_tr = n.transformers.index[n.transformers.status.astype(bool)]
            PT = n.transformers_t.p0.loc[snap, alive_tr].astype(float)
            QT = (n.transformers_t.q0.loc[snap, alive_tr].astype(float)
            if "q0" in getattr(n.transformers_t, "_series", {}) else 0.0 * PT)
            ST = np.hypot(PT, QT)
            STmax = (n.transformers.s_nom * n.transformers.s_max_pu.fillna(1.0)).loc[alive_tr]
            loadT = 100.0 * ST / STmax
            worst = max(worst, float(loadT.max()))

        per_outage_max[o] = worst
        if worst > limit_pct + 1e-6:
            violations.append((o, worst))

    worst_ac_outage = max(per_outage_max, key=per_outage_max.get)
    worst_ac_val = per_outage_max[worst_ac_outage]
    secure_ac = worst_ac_val <= limit_pct + 1e-6

    # --- Ausgabe kompakt ---
    print("\n=== SETPOINT (DC, aus SCLOPF) ===")
    print("Links (MW):")
    print(network.links_t.p0)
    #print("(leer)" if link_p_dc.empty else link_p_dc.round(3).to_string())
    print("\nGeneratoren (MW):")
    print(gen_p_dc.sort_values(ascending=False).round(3).to_string())

    print("\n=== SETPOINT (AC, was pf() nutzt) ===")
    print("Links p_set (MW):")
    print(n2.links_t.p0)
    #print("(leer)" if link_p_ac.empty else link_p_ac.round(3).to_string())
    print("\nGeneratoren p_set (MW):")
    print(gen_p_ac.sort_values(ascending=False).round(3).to_string())
    print(n2.generators["p_set"])

    print(f"\n=== N-1 SICHERHEIT (DC) (Margin {margin:.0%}) ===")
    print(f"Sicher? {secure_dc} | Worst-case: {worst_dc_val:.2f}%  bei Paar {worst_dc_pair}")

    print(f"\n=== N-1 SICHERHEIT (AC) (Margin {margin:.0%}) ===")
    print(f"Sicher? {secure_ac} | Worst-case: {worst_ac_val:.2f}%  bei Ausfall '{worst_ac_outage}'")

    if not secure_ac:
        # Top-3 Verletzungen ausgeben
        print("\nViolations (top-3 loads per outage):")
        for o, _ in sorted(per_outage_max.items(), key=lambda kv: kv[1], reverse=True)[:3]:
            print((o, pd.Series(per_outage_max[o]).squeeze()) if isinstance(per_outage_max[o], pd.Series) else (o, per_outage_max[o]))
        print("\nHinweis: DC kann ok sein, aber AC scheitern (Q/Spannungen, Transformatoren, Winkel).")
        print("→ Guard/Regler anpassen (z.B. Link-P reduzieren, Q-Optimierer), bis AC N-1 passt.")



print("\n----- SUMMARY: Setpoint & N-1 Sicherheit -----")
summarize_setpoint_and_security(network, n2, snap, margin=0.96, outages=list(branch_outages_scopf))

# %% Run the COMBINED Optimization with updated VSCController: using dataclass as central point where parameters can be defined

cfg = ControllerConfig(
    angle_limit_deg=25, 
    max_line_loading=0.7, 
    S_rated=500, 
    enforce_target_tag=True,
    target_tag="N2",
    n1_guard_enable=True,
    n1_guard_margin=0.95,
    n1_guard_max_passes=3,
 )
ctl = VSCController(n2, config=cfg)
print("Controller bound to:", getattr(ctl.n2, "_whoami", "unknown"), id(ctl.network))
assert ctl.network is n2
p_results, q_result = ctl.run_mode(mode="combined")


P = n2.lines_t.p0.loc[snap]
Q = n2.lines_t.q0.loc[snap]
S = np.hypot(P, Q)
#s_line_max = n2.lines.s_nom * n2.lines.s_max_pu.fillna(1.0)
loading_line_ac_final = 100.0 * S / network.lines.s_nom 
print("\nLeitungen AC-Loading after SCLOPF and P/Q_Optimizer [%] (|S|/S_max):")
print(loading_line_ac_final.sort_values(ascending=False).round(2).to_string())

res_neu= n2.lpf_contingency(snap, branch_outages=branch_outages_scopf)

p_by_outage_final = res_neu
pb = n2.passive_branches()
s_max = (pb.s_nom * pb.s_max_pu.fillna(1.0)).astype(float)
loading_c_final = 100.0 * p_by_outage_final.abs().div(s_max, axis=0)

#%% Tests 1

P = n2.lines_t.p0
Q = n2.lines_t.q0
S = np.hypot(P, Q) 
loading_finalo = 100 * S / n2.lines.s_nom 

#loading_change_finalo = (loading - loading_S_default) / loading_S_default * 100
#%% Tests 2

snap = n2.snapshots[0]
link = "Link 7-8"
val  = -98

# Timeseries sauberstellen (falls noch nicht vorhanden)
if "p_set" not in getattr(n2.links_t, "_series", {}):
    n2.links_t["p_set"] = pd.DataFrame(0.0, index=n2.snapshots, columns=n2.links.index)

# 1) Zeitreihe setzen
n2.links_t.p_set.loc[snap, link] = val

# 2) (Optional) statische Spalte spiegeln – hilfreich, falls irgendwo statisch gelesen wird
if "p_set" not in n2.links.columns:
    n2.links["p_set"] = 0.0
n2.links.at[link, "p_set"] = val

# 3) (Optional) DC-LF refresh, ist für lpf_contingency nicht zwingend nötig,
#    aber schadet nicht
n2.lpf(snap)

# 4) N-1-Check
p_by_outage = n2.lpf_contingency(snap, branch_outages=branch_outages_scopf)

# 5) Loading berechnen (Index sauber ausrichten!)
pb    = n2.passive_branches()
s_max = (pb.s_nom * pb.s_max_pu.fillna(1.0)).astype(float)
s_max = s_max.reindex(p_by_outage.index)  # <-- wichtig
loading_c = 100.0 * p_by_outage.abs().div(s_max, axis=0)

print("Worst-case je Zweig [%]:")
print(loading_c.max(axis=1).sort_values(ascending=False).head(15).round(2).to_string())

P = n2.lines_t.p0.loc[snap]
Q = n2.lines_t.q0.loc[snap]
S = np.hypot(P, Q)
s_line_max = n2.lines.s_nom * n2.lines.s_max_pu.fillna(1.0)
loading_line_ac_final_test = 100.0 * S / s_line_max
print("\nLeitungen AC-Loading with set link values (tests) [%] (|S|/S_max):")
print(loading_line_ac_final_test.sort_values(ascending=False).round(2).to_string())

# %% Run the Q Optimization with updated VSCController: using dataclass as central point where parameters can be defined

cfg = ControllerConfig(angle_limit_deg=25, max_line_loading=0.7, S_rated=500, enforce_target_tag=True,
    target_tag="N2",)
ctl = VSCController(n2, config=cfg)
print("Controller bound to:", getattr(ctl.network, "_whoami", "unknown"), id(ctl.network))
assert ctl.network is n2


snap = n2.snapshots[0]
print("VOR: VSC q_set\n", n2.controllable_vscs_t.q_set.loc[snap])

q_result = ctl.run_mode(mode="Q_control")


print("NACH: VSC q_set\n", n2.controllable_vscs_t.q_set.loc[snap])

print("Link-P_SET nach:", n2.links_t.p_set.loc[snap, "Link 7-8"])
print("VSC q_set nach:\n", n2.controllable_vscs_t.q_set.loc[snap])

print("Link p0 nach PF:", n2.links_t.p0.loc[snap, "Link 7-8"])
# print("Spannungen nach PF (Top 5 Δ):\n",
#       (n2.buses_t.v_mag_pu.loc[snap] - n2.buses_t.v_mag_pu.loc[snap]).abs().nlargest(5))


P = n2.lines_t.p0.loc[snap]
Q = n2.lines_t.q0.loc[snap]
S = np.hypot(P, Q)
s_line_max = n2.lines.s_nom * n2.lines.s_max_pu.fillna(1.0)
loading_line_ac_final = 100.0 * S / s_line_max
print("\nLeitungen AC-Loading after SCLOPF and Q_Optimizer [%] (|S|/S_max):")
print(loading_line_ac_final.sort_values(ascending=False).round(2).to_string())


p_by_outage_after_Q = n2.lpf_contingency(snap, branch_outages=branch_outages_scopf)
# %% Sanity Check
snap = ctl.network.snapshots[0]
def dump(lbl, n2):
    print(lbl, 
          "stat=", float(n2.links.loc["Link 7-8","p_set"]),
          "ts=",   float(n2.links_t.p_set.loc[snap,"Link 7-8"]),
          "p0=",   float(n2.links_t.p0.loc[snap,"Link 7-8"]))
dump("BASE", network)
dump("N2  ", n2)
dump("CTL ", ctl.network)
# %% Bus Balance P

snap = network.snapshots[0]  # oder ein anderer Snapshot

# 1) Ein-Port-Komponenten (MW)
p_gen  = network.generators_t.p.loc[snap].groupby(network.generators.bus).sum()       # + Einspeisung
p_load = network.loads_t.p.loc[snap].groupby(network.loads.bus).sum()                 # - Verbrauch

# 2) Mehrport-Flüsse an den Bus-Enden (MW): "aus Bus heraus" -> negativ fürs Bus-Konto
p_line_b0 = -network.lines_t.p0.loc[snap].groupby(network.lines.bus0).sum()
p_line_b1 = -network.lines_t.p1.loc[snap].groupby(network.lines.bus1).sum()

p_trafo_b0 = -network.transformers_t.p0.loc[snap].groupby(network.transformers.bus0).sum()
p_trafo_b1 = -network.transformers_t.p1.loc[snap].groupby(network.transformers.bus1).sum()

p_link_b0 = -network.links_t.p0.loc[snap].groupby(network.links.bus0).sum()
p_link_b1 = -network.links_t.p1.loc[snap].groupby(network.links.bus1).sum()

# 3) Alles auf gemeinsamen Bus-Index bringen und fehlende Werte mit 0 füllen
terms = [p_gen, -p_load, p_line_b0, p_line_b1, p_trafo_b0, p_trafo_b1, p_link_b0, p_link_b1]
bus_balance = sum(t.reindex(network.buses.index).fillna(0.0) for t in terms)

print("\n--- Aktive Leistungsbilanz je Bus [MW] (≈ 0 erwartet) ---")
print(np.round(bus_balance,6))

# Größte Residuen (numerische Toleranzen sollten ~1e-3 MW oder kleiner sein)
print("\nMax |Bilanzfehler| [MW]:", np.round(bus_balance.abs().max(),6))
print("Bus mit max. Fehler:", bus_balance.abs().idxmax())

# %% Bus Balance Q

snap = network.snapshots[0]  # gleicher Snapshot wie oben

# 1) Ein-Port-Komponenten (MVAr)
q_gen  = network.generators_t.q.loc[snap].groupby(network.generators.bus).sum()          # + Einspeisung (PV/Slack regelt Q)
q_load = (network.loads_t.q.loc[snap].groupby(network.loads.bus).sum()
          if hasattr(network, "loads_t") and "q" in network.loads_t else 0)              # - Verbrauch (falls vorhanden)

# VSCs (einportig, Q am jeweiligen Bus einspeisen)
if hasattr(network, "controllable_vscs_t") and "q" in network.controllable_vscs_t:
    q_vsc = network.controllable_vscs_t.q.loc[snap].groupby(network.controllable_vscs.bus).sum()
elif hasattr(network, "controllable_vscs") and "q_set" in network.controllable_vscs:
    # Fallback: statischer q_set (falls keine Zeitreihe existiert)
    q_vsc = network.controllable_vscs.q_set.groupby(network.controllable_vscs.bus).sum()
else:
    q_vsc = 0

# Shunts (einportig; Vorzeichen so übernehmen, PyPSA liefert Einspeisung+ / Aufnahme-)
q_shunt = (network.shunt_impedances_t.q.loc[snap].groupby(network.shunt_impedances.bus).sum()
           if hasattr(network, "shunt_impedances_t") and "q" in network.shunt_impedances_t else 0)

# 2) Mehrport-Flüsse an den Bus-Enden (MVAr): „aus Bus heraus“ -> negativ fürs Bus-Konto
q_line_b0  = -network.lines_t.q0.loc[snap].groupby(network.lines.bus0).sum()
q_line_b1  = -network.lines_t.q1.loc[snap].groupby(network.lines.bus1).sum()

q_trafo_b0 = -network.transformers_t.q0.loc[snap].groupby(network.transformers.bus0).sum()
q_trafo_b1 = -network.transformers_t.q1.loc[snap].groupby(network.transformers.bus1).sum()

# Links (DC) tragen in PyPSA kein Q -> nichts hinzufügen

# 3) Alles auf Bus-Index bringen und summieren
q_terms = [q_gen, -q_load, q_vsc, q_shunt, q_line_b0, q_line_b1, q_trafo_b0, q_trafo_b1]
q_bus_balance = sum(t.reindex(network.buses.index).fillna(0.0) for t in q_terms)

print("\n--- Reaktive Leistungsbilanz je Bus [MVAr] (≈ 0 erwartet) ---")
print(np.round(q_bus_balance, 6))

# Größter Betrag (numerische Toleranz typ. ~1e-3…1e-2 MVAr)
q_max_abs = q_bus_balance.abs().max()
print(f"\nMax |Q-Bilanzfehler| [MVAr]: {q_max_abs:.6f}")
print("Bus mit max. Q-Fehler:", q_bus_balance.abs().idxmax())

# Optional: nur nennenswerte Residuen anzeigen
print("\nNicht-nahe-Null Q-Residuen (>|1e-3| MVAr):")
q_nonzero = q_bus_balance[abs(q_bus_balance) > 1e-3].sort_values(key=np.abs, ascending=False)
print(np.round(q_nonzero, 6) if not q_nonzero.empty else "—")