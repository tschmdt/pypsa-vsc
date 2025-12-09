import pyomo.environ as pyo
import numpy as np
import pandas as pd


def link_optimization(
    netz, angle_limit_deg, master_bus, slave_bus, pf_callback=None, lpf_callback=None
):
    # === 1. Vorbereitung ===
    F = np.sqrt(netz.lines_t.p0**2)
    s_nom = netz.lines.s_nom
    s_nom_matrix = pd.DataFrame(
        np.tile(s_nom.values, (len(F.index), 1)), index=F.index, columns=F.columns
    )
    loading_default = 100 * F / s_nom_matrix
    print("Initiale Line-Auslastungen [%]:")
    print(loading_default)
    print("Initiale DC-Übertragung:")
    print(netz.links_t.p0)

    # === 2. Pyomo-Modell ===
    model = pyo.ConcreteModel()

    # Sets
    buses = list(netz.buses.index)
    lines = list(netz.lines.index)
    trafos = list(netz.transformers.index)
    links = list(netz.links.index)

    model.B = pyo.Set(initialize=buses)
    model.L = pyo.Set(initialize=lines)
    model.T = pyo.Set(initialize=trafos)
    model.V = pyo.Set(initialize=links)

    # Netzdaten
    line_from = {l: netz.lines.at[l, "bus0"] for l in lines}
    line_to = {l: netz.lines.at[l, "bus1"] for l in lines}
    trafo_from = {t: netz.transformers.at[t, "bus0"] for t in trafos}
    trafo_to = {t: netz.transformers.at[t, "bus1"] for t in trafos}
    link_from = {k: netz.links.at[k, "bus0"] for k in links}
    link_to = {k: netz.links.at[k, "bus1"] for k in links}

    # Slack-Busse
    slack_buses = (
        netz.generators[netz.generators.control == "Slack"].bus.unique().tolist()
    )

    # Leitungsreaktanzen in pu
    x_pu = {}
    for l in lines:
        bus = netz.lines.at[l, "bus0"]
        U = netz.buses.at[bus, "v_nom"]
        Z_base = (U**2) / 1 * 1e-3  # MVA = 1
        x_pu[l] = netz.lines.at[l, "x"] / Z_base

    # Trafo-Reaktanzen (in pu angegeben bei PyPSA)
    xt_pu = {t: netz.transformers.at[t, "x"] for t in trafos}

    # Kapazitäten
    s_nom_lines = {l: netz.lines.at[l, "s_nom"] for l in lines}
    s_nom_trafos = {t: netz.transformers.at[t, "s_nom"] for t in trafos}
    p_link_bounds = {k: netz.links.at[k, "p_nom"] for k in links}

    # Lasten und Generatoren
    gen_p = netz.generators.groupby("bus")["p_set"].sum().to_dict()
    load_p = netz.loads.groupby("bus")["p_set"].sum().to_dict()

    # === 3. Variablen ===
    model.theta = pyo.Var(model.B, domain=pyo.Reals)
    model.f_line = pyo.Var(model.L, domain=pyo.Reals)
    model.f_trafo = pyo.Var(model.T, domain=pyo.Reals)
    model.p_link = pyo.Var(
        model.V, bounds=lambda m, k: (-p_link_bounds[k], p_link_bounds[k])
    )
    model.p_slack_gen = pyo.Var(slack_buses, bounds=lambda m, b: (-500, 500))

    # === 4. Nebenbedingungen ===

    # Linienfluss
    def line_flow_rule(m, l):
        return m.f_line[l] == (m.theta[line_from[l]] - m.theta[line_to[l]]) / x_pu[l]

    model.LineFlow = pyo.Constraint(model.L, rule=line_flow_rule)

    # Trafofluss
    def trafo_flow_rule(m, t):
        return (
            m.f_trafo[t] == (m.theta[trafo_from[t]] - m.theta[trafo_to[t]]) / xt_pu[t]
        )

    model.TrafoFlow = pyo.Constraint(model.T, rule=trafo_flow_rule)

    # Knotengleichgewicht
    def node_balance_rule(m, b):
        gen = model.p_slack_gen[b] if b in slack_buses else gen_p.get(b, 0)
        load = load_p.get(b, 0)

        inflow = (
            sum(m.f_line[l] for l in lines if line_to[l] == b)
            + sum(m.f_trafo[t] for t in trafos if trafo_to[t] == b)
            + sum(
                m.p_link[k] * netz.links.at[k, "efficiency"]
                for k in links
                if link_to[k] == b
            )
        )
        outflow = (
            sum(m.f_line[l] for l in lines if line_from[l] == b)
            + sum(m.f_trafo[t] for t in trafos if trafo_from[t] == b)
            + sum(m.p_link[k] for k in links if link_from[k] == b)
        )
        return gen - load + inflow - outflow == 0

    model.NodeBalance = pyo.Constraint(model.B, rule=node_balance_rule)

    # Referenzwinkel
    model.slack_theta = pyo.Constraint(expr=model.theta[slack_buses[0]] == 0)

    # Link-Kopplung Master/Slave
    for k in links:
        if (link_from[k], link_to[k]) == (master_bus, slave_bus):
            eff = netz.links.at[k, "efficiency"]
            model.add_component(
                f"LinkCoupling_{k}",
                pyo.Constraint(expr=model.p_link[k] + eff * model.p_link[k] == 0),
            )
        elif (link_from[k], link_to[k]) == (slave_bus, master_bus):
            eff = netz.links.at[k, "efficiency"]
            model.add_component(
                f"LinkCoupling_{k}",
                pyo.Constraint(expr=eff * model.p_link[k] + model.p_link[k] == 0),
            )

    # Flussgrenzen (optional erweiterbar!)
    model.FlowLimit = pyo.ConstraintList()
    for l in lines:
        model.FlowLimit.add(model.f_line[l] <= 0.75 * s_nom_lines[l])
        model.FlowLimit.add(model.f_line[l] >= -0.75 * s_nom_lines[l])
    for t in trafos:
        model.FlowLimit.add(model.f_trafo[t] <= 0.75 * s_nom_trafos[t])
        model.FlowLimit.add(model.f_trafo[t] >= -0.75 * s_nom_trafos[t])

    # === 5. Zielfunktion ===
    def objective_rule(m):
        return sum((m.f_line[l] / m.s_nom[l]) ** 2 for l in m.L) + sum(
            (m.f_trafo[t] / s_nom_trafos[t]) ** 2 for t in m.T
        )

    model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # === 6. Optimieren ===
    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(model, tee=False)

    print("\n===== Ausgabe P-Optimierung =====\n")
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        print("NICHT erfolgreich:", result.solver.termination_condition)
        return None

    # Ergebnisse eintragen
    for k in model.V:
        netz.links_t.p_set.at["now", k] = pyo.value(model.p_link[k])

    # Power Flow prüfen
    if pf_callback is not None:
        pf_callback()
    else:
        netz.pf()

    return {
        "p_link": {k: pyo.value(model.p_link[k]) for k in model.V},
        "theta": {b: pyo.value(model.theta[b]) for b in model.B},
        "f_line": {l: pyo.value(model.f_line[l]) for l in model.L},
        "f_trafo": {t: pyo.value(model.f_trafo[t]) for t in model.T},
    }
