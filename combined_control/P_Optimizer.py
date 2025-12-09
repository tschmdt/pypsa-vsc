import pyomo.environ as pyo
import numpy as np
import pandas as pd


def link_optimization(
    netz, angle_limit_deg, master_bus, slave_bus, pf_callback=None, lpf_callback=None
):
    F = np.sqrt(netz.lines_t.p0**2)
    s_nom = netz.lines.s_nom
    s_nom_matrix = pd.DataFrame(
        np.tile(s_nom.values, (len(F.index), 1)), index=F.index, columns=F.columns
    )
    loading_default = 100 * F / s_nom_matrix
    print("Laoding_default:", loading_default)
    print("Links:", netz.links_t.p0)

    model = pyo.ConcreteModel()

    buses = list(netz.buses.index)
    lines = list(netz.lines.index)
    links = list(netz.links.index)
    slack_buses = (
        netz.generators[netz.generators.control == "Slack"].bus.unique().tolist()
    )

    line_from = {l: netz.lines.at[l, "bus0"] for l in lines}
    line_to = {l: netz.lines.at[l, "bus1"] for l in lines}
    link_from = {k: netz.links.at[k, "bus0"] for k in links}
    link_to = {k: netz.links.at[k, "bus1"] for k in links}

    x_pu = {}
    for l in lines:
        bus = netz.lines.at[l, "bus0"]
        U_n = netz.buses.at[bus, "v_nom"]
        S_n = 1
        Z_n = (U_n**2 / S_n) * 1e-3
        X = netz.lines.at[l, "x"]
        x_pu[l] = X / Z_n

    s_nom = {l: netz.lines.at[l, "s_nom"] for l in lines}
    p_link_bounds = {k: netz.links.at[k, "p_nom"] for k in links}
    gen_p = netz.generators.groupby("bus")["p_set"].sum().to_dict()
    load_p = netz.loads.groupby("bus")["p_set"].sum().to_dict()

    model.B = pyo.Set(initialize=buses)
    model.L = pyo.Set(initialize=lines)
    model.V = pyo.Set(initialize=links)

    model.x = pyo.Param(model.L, initialize=x_pu)
    model.s_nom = pyo.Param(model.L, initialize=s_nom)

    model.theta = pyo.Var(model.B, domain=pyo.Reals)
    model.p_link = pyo.Var(
        model.V, bounds=lambda m, k: (-p_link_bounds[k], p_link_bounds[k])
    )
    model.f_line = pyo.Var(model.L, domain=pyo.Reals)
    model.p_slack_gen = pyo.Var(slack_buses, bounds=lambda m, b: (-500, 500))

    def line_flow_rule(m, l):
        return m.f_line[l] == (m.theta[line_from[l]] - m.theta[line_to[l]]) / m.x[l]

    model.LineFlow = pyo.Constraint(model.L, rule=line_flow_rule)

    def node_balance_rule(m, b):
        gen = model.p_slack_gen[b] if b in slack_buses else gen_p.get(b, 0)
        load = load_p.get(b, 0)
        inflow = sum(m.f_line[l] for l in lines if line_to[l] == b)
        outflow = sum(m.f_line[l] for l in lines if line_from[l] == b)
        link_in = sum(
            m.p_link[k] * netz.links.at[k, "efficiency"]
            for k in links
            if link_to[k] == b
        )
        link_out = sum(m.p_link[k] for k in links if link_from[k] == b)
        return gen - load + inflow - outflow + link_in - link_out == 0

    model.NodeBalance = pyo.Constraint(model.B, rule=node_balance_rule)

    model.slack_theta = pyo.Constraint(expr=model.theta[slack_buses[0]] == 0)

    model.FlowLimit = pyo.ConstraintList()
    for l in lines:
        model.FlowLimit.add(model.f_line[l] <= 0.75 * s_nom[l])
        model.FlowLimit.add(model.f_line[l] >= -0.75 * s_nom[l])

    def objective_rule(m):
        return sum((m.f_line[l] / m.s_nom[l]) ** 2 for l in m.L)

    model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(model, tee=False)

    print("\n===== Ausgabe P-Optimierung =====\n")

    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        print(
            "\n P-Optimierung NICHT erfolgreich:", result.solver.termination_condition
        )
        return None

    print("\n P-Optimierung erfolgreich.")

    print("=== x_pu je Leitung ===")
    for l in lines:
        print(
            f"{l}: x = {netz.lines.at[l, 'x']:.4f} Ohm, U_n = {netz.buses.at[netz.lines.at[l, 'bus0'], 'v_nom']} kV → x_pu = {x_pu[l]:.6f}"
        )

    for k in model.V:
        netz.links_t.p_set.at["now", k] = pyo.value(model.p_link[k])

    if pf_callback is not None:
        pf_callback()
    else:
        netz.pf()

    F = np.sqrt(netz.lines_t.p0**2)
    s_nom = netz.lines.s_nom
    s_nom_matrix = pd.DataFrame(
        np.tile(s_nom.values, (len(F.index), 1)), index=F.index, columns=F.columns
    )

    loading = 100 * F / s_nom_matrix
    loading_change = (loading - loading_default) / loading_default * 100
    print("Loading:", loading)
    print("Loading Change:", loading_change.round(2))
    print("Links:", netz.links_t.p0)

    print("--- Physikalische Winkelprüfung nach pf() ---")
    theta_limit_rad = np.radians(angle_limit_deg)
    verletzung = any(
        abs(netz.buses_t.v_ang.at["now", b]) > theta_limit_rad for b in netz.buses.index
    )
    if verletzung:
        print(
            f" Winkelgrenze verletzt – mindestens ein Bus überschreitet ±{angle_limit_deg:.1f}°"
        )
    else:
        print(f" Alle Bus-Winkel liegen innerhalb ±{angle_limit_deg:.1f}°")

    print("Lines:")
    print(netz.lines_t.p0)
    print("\nGenerators:")
    print(netz.generators_t.p)
    print("\nAngles [°]:")
    print(np.degrees(netz.buses_t.v_ang))

    return {
        "p_link": {k: pyo.value(model.p_link[k]) for k in model.V},
        "theta": {b: pyo.value(model.theta[b]) for b in model.B},
        "f_line": {l: pyo.value(model.f_line[l]) for l in model.L},
        "loading": loading.copy(),
        "loading_change": loading_change.copy(),
    }
