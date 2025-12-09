import pyomo.environ as pyo
import numpy as np
import pandas as pd


def link_optimization(
    network,
    angle_limit_deg,
    pf_callback=None,
    lpf_callback=None,
    max_line_loading=0.95,
    detail_level=None,
    snapshots="all",
    guard_active: bool = False,
):
    print("[link_opt] target:", getattr(network, "_whoami", "unknown"), id(network))

    # 0. Setup Optimization Snapshots

    results = {}

    for snapshot in network.snapshots:
        network.snapshot = snapshot

        # 1. Default state after initial pf (triggerd in VSCController class)

        P0 = network.lines_t.p0.loc[snapshot]
        P1 = network.lines_t.p1.loc[snapshot]
        Q0 = network.lines_t.q0.loc[snapshot]
        Q1 = network.lines_t.q1.loc[snapshot]

        S0 = (P0**2 + Q0**2) ** 0.5
        S1 = (P1**2 + Q1**2) ** 0.5

        s_nom = network.lines.s_nom
        s_max_pu = network.lines.get(
            "s_max_pu", pd.Series(1.0, index=s_nom.index)
        ).reindex(s_nom.index)
        S_limit = (s_nom * s_max_pu).replace(0, np.nan)

        loading0_ac = 100 * S0.div(S_limit)
        loading1_ac = 100 * S1.div(S_limit)

        loading_S_default = pd.concat([loading0_ac, loading1_ac], axis=1).max(axis=1)

        # Transformers
        P_T0 = network.transformers_t.p0.loc[snapshot]
        P_T1 = network.transformers_t.p1.loc[snapshot]
        Q_T0 = network.transformers_t.q0.loc[snapshot]
        Q_T1 = network.transformers_t.q1.loc[snapshot]

        S_T0 = (P_T0**2 + Q_T0**2) ** 0.5
        S_T1 = (P_T1**2 + Q_T1**2) ** 0.5

        s_T_nom = network.transformers.s_nom
        s_T_max_pu = network.transformers.get(
            "s_max_pu", pd.Series(1.0, index=s_T_nom.index)
        ).reindex(s_T_nom.index)
        S_T_limit = (s_T_nom * s_T_max_pu).replace(0, np.nan)

        loading_T0_ac = 100 * S_T0.div(S_T_limit)
        loading_T1_ac = 100 * S_T1.div(S_T_limit)

        loading_T_S_default = pd.concat([loading_T0_ac, loading_T1_ac], axis=1).max(
            axis=1
        )

        # 2. Set up optimization model

        model = pyo.ConcreteModel()

        # 3. Pre-Definitions for Sets / Values Extraction

        # Network Objects
        theta_limit_rad = np.radians(angle_limit_deg) - np.radians(
            3.0
        )  # Margin fo DC to AC discrepancy
        print("theta limit rad", theta_limit_rad)
        buses = list(network.buses.index)
        lines = list(network.lines.index)
        links = list(network.links.index)
        transformers = list(network.transformers.index)

        # Slack Buses
        slack_buses_list = (
            network.generators[network.generators.control == "Slack"]
            .bus.unique()
            .tolist()
        )  # prevents dupicates; np.array to a list

        # Mappings/Network Data
        line_from = {l: network.lines.at[l, "bus0"] for l in lines}
        line_to = {l: network.lines.at[l, "bus1"] for l in lines}
        link_from = {k: network.links.at[k, "bus0"] for k in links}
        link_to = {k: network.links.at[k, "bus1"] for k in links}
        eff_k = {k: network.links.at[k, "efficiency"] for k in links}
        trafo_from = {t: network.transformers.at[t, "bus0"] for t in transformers}
        trafo_to = {t: network.transformers.at[t, "bus1"] for t in transformers}

        # Read values & convert into p.u. notation
        x_pu = {}
        for l in lines:
            bus = network.lines.at[l, "bus0"]  # beginning of each line
            U_n = network.buses.at[
                bus, "v_nom"
            ]  # voltage most likely highest at line beginning; kV
            S_base = 1  # MVA
            Z_base = (U_n * 10**3) ** 2 / (S_base * 10**6)
            X = network.lines.at[l, "x"]
            x_pu[l] = (
                X / Z_base
            )  # l comes from the loop and is key for each corresponding line x_pu

        # Trafos p.u. on the base of s_nom(trafo). Needs to be converted to same base as Lines (s_nom=1 MVA)
        x_trafo = {
            t: network.transformers.at[t, "x"] / network.transformers.at[t, "s_nom"]
            for t in transformers
        }

        # Capacities (Boundaries)
        s_nom_line = {l: network.lines.at[l, "s_nom"] for l in lines}
        s_nom_trafo = {t: network.transformers.at[t, "s_nom"] for t in transformers}

        # Generators and Loads (Sum at each Bus)
        if "p_set" in getattr(network.generators_t, "_series", {}):
            gen_p = (
                network.generators_t.p_set.loc[snapshot]
                .groupby(network.generators.bus)
                .sum()
                .to_dict()
            )
        else:
            gen_p = network.generators.groupby("bus")["p_set"].sum().to_dict()

        if "p_set" in getattr(network.loads_t, "_series", {}):
            load_p = (
                network.loads_t.p_set.loc[snapshot]
                .groupby(network.loads.bus)
                .sum()
                .to_dict()
            )
        else:
            load_p = network.loads.groupby("bus")["p_set"].sum().to_dict()

        # 4. Sets (Index-Quantities using lists of Network Objects)

        model.B = pyo.Set(initialize=buses)
        model.L = pyo.Set(initialize=lines)
        model.T = pyo.Set(initialize=transformers)
        model.K = pyo.Set(initialize=links)
        model.S = pyo.Set(initialize=slack_buses_list)

        # 5. Parameters (Indexed over the sets)

        model.x = pyo.Param(
            model.L, initialize=x_pu
        )  # initialize defines the (fixed) value
        model.x_trafo = pyo.Param(model.T, initialize=x_trafo)
        model.s_nom_line = pyo.Param(model.L, initialize=s_nom_line)
        model.s_nom_trafo = pyo.Param(model.T, initialize=s_nom_trafo)

        # 6. Variables (Decision Variables - to be optimized)

        # Bus Angles
        model.theta = pyo.Var(model.B, domain=pyo.Reals)

        # Set Slack Bus fix (more robust)
        ref_bus = slack_buses_list[0] if slack_buses_list else buses[0]
        model.theta[ref_bus].fix(0.0)

        # Power Flow over Lines and Transformers
        model.f_line = pyo.Var(model.L, domain=pyo.Reals)
        model.f_trafo = pyo.Var(model.T, domain=pyo.Reals)

        # Load Flow over DC-Links
        model.p_link_pos = pyo.Var(
            model.K, domain=pyo.NonNegativeReals
        )  # bus0 --> bus1
        model.p_link_neg = pyo.Var(
            model.K, domain=pyo.NonNegativeReals
        )  # bus1 --> bus0
        model.p_link = pyo.Var(model.K)

        for k in links:
            p_nom = network.links.at[k, "p_nom"]
            p_min = (
                network.links.at[k, "p_min_pu"] * p_nom
            )  # p_min_pu is negative--> p_min is negative
            p_max = network.links.at[k, "p_max_pu"] * p_nom
            model.p_link_pos[k].setub(max(p_max, 0.0))  # setub=set upper bound
            model.p_link_neg[k].setub(
                max(-p_min, 0.0)
            )  # p_min is negativ; -p_min becomes positive

        for k in model.K:
            model.p_link_pos[k].value = 0.0
            model.p_link_neg[k].value = 0.0
            model.p_link[k].value = 0.0

        model.y_dir = pyo.Var(
            model.K, domain=pyo.Binary
        )  # 1--> pos active, 0--> neg active

        def dir_pos_rule(m, k):
            ub = m.p_link_pos[k].ub or 0.0
            return m.p_link_pos[k] <= ub * m.y_dir[k]

        def dir_neg_rule(m, k):
            ub = m.p_link_neg[k].ub or 0.0
            return m.p_link_neg[k] <= ub * (1 - m.y_dir[k])

        model.DirPos = pyo.Constraint(model.K, rule=dir_pos_rule)
        model.DirNeg = pyo.Constraint(model.K, rule=dir_neg_rule)

        # Slack Generator Power
        model.p_slack_gen = pyo.Var(model.S, bounds=lambda m, b: (-10000, 10000))

        # 7. Constraints using Construction Rules

        # Link Split (for internal differentiation)
        def link_split_rule(m, k):
            return m.p_link[k] == m.p_link_pos[k] - m.p_link_neg[k]

        model.PLinkSplit = pyo.Constraint(model.K, rule=link_split_rule)

        # Line Flow Constraint
        def line_flow_rule(m, l):
            return (
                m.f_line[l] == (m.theta[line_from[l]] - m.theta[line_to[l]]) / m.x[l]
            )  # m.x is in p.u. ()

        model.LineFlow = pyo.Constraint(model.L, rule=line_flow_rule)

        # Transformer Flow (Modelled Simplified as Line)
        def trafo_flow_rule(m, t):
            return (
                m.f_trafo[t]
                == (m.theta[trafo_from[t]] - m.theta[trafo_to[t]]) / m.x_trafo[t]
            )

        model.TrafoFlow = pyo.Constraint(model.T, rule=trafo_flow_rule)

        # Node Balance
        def node_balance_rule(m, b):
            if b in model.S:
                gen = model.p_slack_gen[b] + gen_p.get(b, 0)
            else:
                gen = gen_p.get(b, 0)

            load = load_p.get(b, 0)
            line_in = sum(
                m.f_line[l] for l in lines if line_to[l] == b
            )  # Sum of all flows ending at bus b (into b)
            line_out = sum(
                m.f_line[l] for l in lines if line_from[l] == b
            )  # Sum of all flows ending at bus b (into b)
            trafo_in = sum(m.f_trafo[t] for t in transformers if trafo_to[t] == b)
            trafo_out = sum(m.f_trafo[t] for t in transformers if trafo_from[t] == b)
            # DC-Link In-/Outflow with directionality of losses (per bus)
            link_in = sum(
                m.p_link_pos[k] * eff_k[k] for k in links if link_to[k] == b
            ) + sum(m.p_link_neg[k] * eff_k[k] for k in links if link_from[k] == b)
            link_out = sum(m.p_link_pos[k] for k in links if link_from[k] == b) + sum(
                m.p_link_neg[k] for k in links if link_to[k] == b
            )

            return (
                gen
                - load
                + line_in
                - line_out
                + link_in
                - link_out
                + trafo_in
                - trafo_out
                == 0
            )  # Energy Balance

        model.NodeBalance = pyo.Constraint(model.B, rule=node_balance_rule)

        # Max Line Loading
        def line_limit_rule(m, l):
            return pyo.inequality(
                -max_line_loading * m.s_nom_line[l],
                m.f_line[l],
                max_line_loading * m.s_nom_line[l],
            )

        model.LineLimit = pyo.Constraint(model.L, rule=line_limit_rule)

        # Max Trafo Loading
        def trafo_limit_rule(m, t):
            return pyo.inequality(
                -max_line_loading * m.s_nom_trafo[t],
                m.f_trafo[t],
                max_line_loading * m.s_nom_trafo[t],
            )

        model.TrafoLimit = pyo.Constraint(model.T, rule=trafo_limit_rule)

        # Max Angle Difference Line
        def angle_diff_rule(m, l):
            return pyo.inequality(
                -theta_limit_rad,
                m.theta[line_from[l]] - m.theta[line_to[l]],
                theta_limit_rad,
            )

        model.AngleDiff = pyo.Constraint(model.L, rule=angle_diff_rule)

        # Max Angle Difference Trafo (modelled as a Line)
        def angle_diff_trafo_rule(m, t):
            return pyo.inequality(
                -theta_limit_rad,
                m.theta[trafo_from[t]] - m.theta[trafo_to[t]],
                theta_limit_rad,
            )

        model.AngleDiffTrafo = pyo.Constraint(model.T, rule=angle_diff_trafo_rule)

        # 8. Objective (Least-Squares)

        def objective_rule(m):
            return sum((m.f_line[l] / m.s_nom_line[l]) ** 2 for l in lines) + sum(
                (m.f_trafo[t] / m.s_nom_trafo[t]) ** 2 for t in transformers
            )

        model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # 9. Solver + debug (general)
        solver = pyo.SolverFactory("gurobi")
        result = solver.solve(
            model,
            tee=False,
            options={
                "DualReductions": 0,  # disambiguate infeasible vs unbounded
                "InfUnbdInfo": 1,
            },
        )

        from math import isfinite
        from pyomo.opt import TerminationCondition

        tc = result.solver.termination_condition

        def _seed_vars_for_logging(m):
            """Give None-valued vars finite values so Pyomo's log_* can evaluate expressions."""
            seeded = []
            for v in m.component_data_objects(pyo.Var, active=True):
                if v.value is None:
                    if v.is_binary():
                        v.set_value(0)  # valid binary
                    elif v.is_integer():
                        lb = pyo.value(v.lb) if v.has_lb() else None
                        ub = pyo.value(v.ub) if v.has_ub() else None
                        if (
                            lb is not None
                            and ub is not None
                            and isfinite(lb)
                            and isfinite(ub)
                        ):
                            v.set_value(int(round(0.5 * (lb + ub))))
                        elif lb is not None and isfinite(lb):
                            v.set_value(int(round(lb)))
                        elif ub is not None and isfinite(ub):
                            v.set_value(int(round(ub)))
                        else:
                            v.set_value(0)
                    else:  # real
                        lb = pyo.value(v.lb) if v.has_lb() else None
                        ub = pyo.value(v.ub) if v.has_ub() else None
                        if (
                            lb is not None
                            and ub is not None
                            and isfinite(lb)
                            and isfinite(ub)
                        ):
                            v.set_value(0.5 * (lb + ub))
                        elif lb is not None and isfinite(lb):
                            v.set_value(lb)
                        elif ub is not None and isfinite(ub):
                            v.set_value(ub)
                        else:
                            v.set_value(0.0)
                    seeded.append(v)
            return seeded

        def _unseed_vars(vars_):
            for v in vars_:
                v.set_value(None)

        if tc in (
            TerminationCondition.infeasible,
            TerminationCondition.infeasibleOrUnbounded,
        ):
            # Seed so logs show numeric residuals (no "evaluation error")
            _seeded = _seed_vars_for_logging(model)

            from pyomo.util.infeasible import (
                log_infeasible_bounds,
                log_infeasible_constraints,
            )

            print("\nModel infeasible — violated bounds:")
            log_infeasible_bounds(model, tol=1e-8)
            print("\nModel infeasible — violated constraints:")
            log_infeasible_constraints(model, tol=1e-8, log_expression=True)

            # Also request a minimal conflicting set (IIS) from Gurobi
            try:
                opt_p = pyo.SolverFactory("gurobi_persistent")
                opt_p.set_instance(model, symbolic_solver_labels=True)  # readable names
                g = opt_p._solver_model
                g.computeIIS()

                # Print IIS contents to console
                iis_constr = [c.ConstrName for c in g.getConstrs() if c.IISConstr]
                print("\n=== IIS constraints ===")
                for name in iis_constr:
                    print("  ", name)

                print("\n=== IIS variable bounds ===")
                for v in g.getVars():
                    if getattr(v, "IISLB", False):
                        print(f"  {v.VarName}: lower bound participates (LB={v.LB})")
                    if getattr(v, "IISUB", False):
                        print(f"  {v.VarName}: upper bound participates (UB={v.UB})")

                # Save IIS as a tiny LP containing only the conflict
                g.write("model.ilp")
                print("Wrote IIS mini-model to model.ilp")
            except Exception as e:
                print("IIS not available:", e)

            # Restore guessed values so you don't carry them forward
            _unseed_vars(_seeded)

        elif tc == TerminationCondition.unbounded:
            print("\nModel unbounded — variables with missing or infinite bounds:")
            for v in model.component_data_objects(pyo.Var, active=True):
                lb = pyo.value(v.lb) if v.has_lb() else None
                ub = pyo.value(v.ub) if v.has_ub() else None
                bad_lb = (lb is None) or (not isfinite(lb))
                bad_ub = (ub is None) or (not isfinite(ub))
                if bad_lb or bad_ub:
                    print(f"  {v.name}: lb={lb}, ub={ub}")

        # 11. Status Return

        if result.solver.termination_condition != pyo.TerminationCondition.optimal:
            print(
                f"\n P-optimization NOT successfull for snapshot {snapshot}: {result.solver.termination_condition}"
            )
            results[snapshot] = {"status": "failed"}
            continue
        print("\n P-optimization successfull.")

        # 11. Return Value to PyPSA

        # static
        for k in links:
            network.links.at[k, "p_set"] = pyo.value(model.p_link[k])

        # Get timeseries: case, if link_optimization is used directly
        if "p_set" not in getattr(network.links_t, "_series", {}):
            network.links_t["p_set"] = pd.DataFrame(
                0.0, index=network.snapshots, columns=network.links.index
            )

        # timeseries (more important)
        network.links_t.p_set.loc[snapshot, links] = [
            float(pyo.value(model.p_link[k])) for k in links
        ]

        # Mirror, dor Reporting/PF (optional)
        if "p0" in network.links_t and not network.links_t.p0.empty:
            network.links_t.p0.loc[snapshot, links] = network.links_t.p_set.loc[
                snapshot, links
            ]

        # 12. AC-Load Flow Calculation with new setting

        if not guard_active:
            if pf_callback is not None:
                pf_callback()
            else:
                network.pf()
            # if pf_callback is not None:
            #     pf_callback()
            # else:
            #     network.pf()

        # 13. Analysis

        angles = network.buses_t.v_ang.loc[snapshot]  # rad

        # Lines: deltatheta rad-array
        dtheta_line_rad = []
        for l in network.lines.index:
            i = network.lines.at[l, "bus0"]
            j = network.lines.at[l, "bus1"]
            dtheta_line_rad.append(float(angles[i] - angles[j]))
        dtheta_line_rad = np.array(dtheta_line_rad)

        # Trafos: deltatheta rad-array (trafos modelled as lines)
        dtheta_trafo_rad = []
        for t in network.transformers.index:
            i = network.transformers.at[t, "bus0"]
            j = network.transformers.at[t, "bus1"]
            dtheta_trafo_rad.append(float(angles[i] - angles[j]))
        dtheta_trafo_rad = np.array(dtheta_trafo_rad)

        max_dtheta_line_rad = (
            float(np.max(np.abs(dtheta_line_rad))) if dtheta_line_rad.size else 0.0
        )
        max_dtheta_trafo_rad = (
            float(np.max(np.abs(dtheta_trafo_rad))) if dtheta_trafo_rad.size else 0.0
        )

        # Flag if not valid
        violation_line = (
            bool((np.abs(dtheta_line_rad) > (theta_limit_rad + np.radians(3.0))).any())
            if dtheta_line_rad.size
            else False
        )
        violation_trafo = (
            bool((np.abs(dtheta_trafo_rad) > (theta_limit_rad + np.radians(3.0))).any())
            if dtheta_trafo_rad.size
            else False
        )
        violation = violation_line or violation_trafo

        # In Degree for Report
        max_dtheta_line_deg = float(np.degrees(max_dtheta_line_rad))
        max_dtheta_trafo_deg = float(np.degrees(max_dtheta_trafo_rad))

        # New Loadings (Lines & Trafos) nach der Optimierung

        # ===== Leitungen =====
        P0_new = network.lines_t.p0.loc[snapshot]
        P1_new = network.lines_t.p1.loc[snapshot]
        Q0_new = network.lines_t.q0.loc[snapshot]
        Q1_new = network.lines_t.q1.loc[snapshot]

        S0_new = np.hypot(P0_new, Q0_new)
        S1_new = np.hypot(P1_new, Q1_new)

        s_nom = network.lines.s_nom
        s_max_pu = network.lines.get(
            "s_max_pu", pd.Series(1.0, index=s_nom.index)
        ).reindex(s_nom.index)

        S_limit = (s_nom * s_max_pu).replace(0, np.nan)

        loading0_ac_new = 100 * S0_new / S_limit
        loading1_ac_new = 100 * S1_new / S_limit

        # konservativ wieder: je Leitung das größere Ende
        loading_lines_new = pd.concat([loading0_ac_new, loading1_ac_new], axis=1).max(
            axis=1
        )

        # relative Änderung ggü. Default (%)
        # Schutz gegen Division durch 0 bei Leitungen, die vorher 0 % hatten
        loading_S_default_safe = loading_S_default.replace(0, np.nan)
        loading_change = (
            (loading_lines_new - loading_S_default_safe) / loading_S_default_safe * 100
        )

        # ===== Transformatoren =====
        P_T0_new = network.transformers_t.p0.loc[snapshot]
        P_T1_new = network.transformers_t.p1.loc[snapshot]
        Q_T0_new = network.transformers_t.q0.loc[snapshot]
        Q_T1_new = network.transformers_t.q1.loc[snapshot]

        S_T0_new = np.hypot(P_T0_new, Q_T0_new)
        S_T1_new = np.hypot(P_T1_new, Q_T1_new)

        s_T_nom = network.transformers.s_nom
        s_T_max_pu = network.transformers.get(
            "s_max_pu", pd.Series(1.0, index=s_T_nom.index)
        ).reindex(s_T_nom.index)

        S_T_limit = (s_T_nom * s_T_max_pu).replace(0, np.nan)

        loading_T0_ac_new = 100 * S_T0_new / S_T_limit
        loading_T1_ac_new = 100 * S_T1_new / S_T_limit

        loading_trafo_new = pd.concat(
            [loading_T0_ac_new, loading_T1_ac_new], axis=1
        ).max(axis=1)

        # ===== Aggregierte Kennzahlen =====

        # Default-Gesamtbelastung (Leitungen + Trafos)
        total_loading_default = loading_S_default.sum() + loading_T_S_default.sum()

        # Optimierte Gesamtbelastung (Leitungen + Trafos)
        total_loading_opt = loading_lines_new.sum() + loading_trafo_new.sum()

        # Falls du weiter mit einer Kennzahl arbeitest:
        final_total_loading = total_loading_opt

        results[snapshot] = {
            "model": model,
            "angle_limit_deg": angle_limit_deg,
            "f_lines": {l: pyo.value(model.f_line[l]) for l in model.L},
            "theta": {b: pyo.value(model.theta[b]) for b in model.B},
            # "loading_default": loading_default, # deprecated
            "loading_S_default": loading_S_default.sort_values(ascending=False).round(
                2
            ),
            "loading": loading_lines_new.sort_values(ascending=False).round(2),
            "loading_change": loading_change.sort_values(ascending=False).round(2),
            "trafo_loading_default": loading_T_S_default.sort_values(
                ascending=False
            ).round(2),
            "trafo_loading": loading_trafo_new.sort_values(ascending=False).round(2),
            "total_loading_default": total_loading_default,
            "final_total_loading": final_total_loading,
            "theta_limit_rad": theta_limit_rad,
            "violation": violation,
            "max_abs_dtheta_line_rad": max_dtheta_line_rad,
            "max_abs_dtheta_trafo_rad": max_dtheta_trafo_rad,
            "max_abs_dtheta_line_deg": max_dtheta_line_deg,
            "max_abs_dtheta_trafo_deg": max_dtheta_trafo_deg,
            "angle_violation_line": violation_line,
            "angle_violation_trafo": violation_trafo,
            "max_angle_line_number": max_dtheta_line_rad,
        }

    if detail_level is not None:
        show_snapshot_report(
            results, network, snapshots=snapshots, detail_level=detail_level
        )

    return results


# 14. Report OUTPUT - TO BE ADAPTED


def show_snapshot_report(
    results,
    network,
    snapshots="all",
    detail_level=2,
    vsi_default=None,
):  # Default values. Overwritten from run_p_control
    """
    detail_level:
        0 = only most important parameters
        1 = medium-level detailed output
        2 = full output for analysis
    """

    # No results? --> termination
    if not results:
        print("No results to show")
        return

    # Determine snapshots
    if snapshots is None:
        snapshots_to_show = list(results.keys())
    elif snapshots == "all":
        snapshots_to_show = list(results.keys())
    elif isinstance(snapshots, (list, tuple, pd.Index)):
        snapshots_to_show = list(snapshots)
    else:
        snapshots_to_show = [snapshots]

    # If empty --> take next available
    if not snapshots_to_show:
        snapshots_to_show = [list(results.keys())[0]]

    # Make sure, all keys exist in results
    snapshots_to_show = [snap for snap in snapshots_to_show if snap in results]

    if not snapshots_to_show:
        print("None of the requested snapshots found in the results.")
        return

    for snapshot in snapshots_to_show:
        if snapshot not in results:
            print("Snapshot ", snapshot, "Not in Results")
            continue

        res = results[snapshot]
        print(f"\n ====== Snapshot: {snapshot} ======")

        if res.get("status") == "failed":
            print("Optimizations failed\n")
            continue

        if detail_level >= 0:
            if res["violation"]:
                print(
                    f" Angle limit violated – at least one branch exceeds ±{res['angle_limit_deg']:.1f}°--> Note: Tolerance deviation between model (DC) and report (AC)"
                )
            else:
                print(
                    f" All branch angle differences are within ±{res['angle_limit_deg']:.1f}°"
                )

        if detail_level >= 1:
            # print("\n Line Loading Default:")
            # print(res["loading_default"])

            print("\n Line Loading Default (MVA based):")
            print(res["loading_S_default"])

            print("\n Line Loading (w/o guard):")
            print(res["loading"])

            print("\n Line Loading Change [%]:")
            print(res["loading_change"].round(2))

            # print("\n Line Loading (final, after Guard):")
            # print(loading_fin.sort_values(ascending=False).round(2))

            # print("\n Line Loading Change (Default--> Guard [%]:")
            # loading_change_guard = (loading_fin - res["loading_S_default"]) / res["loading_S_default"] * 100
            # print(loading_change_guard)

            print("\n Trafo Loading Default: ")
            print(res["trafo_loading_default"])

            print("\n Trafo Loading (w/o guard): ")
            print(res["trafo_loading"])

            # print("\n Trafo Loading (final, after Guard): ")
            # if trafo_loading_fin is not None:
            #     print(trafo_loading_fin.reindex(res["trafo_loading_default"].index))
            # else:
            #     print("(no transformer time series)")

        if detail_level >= 2:
            print("\n Line Flows:")
            for l, val in res["f_lines"].items():
                print(f" {l}: {val:.2f} MW")

            if hasattr(network, "links_t") and "p0" in network.links_t:
                print("\n Link Transfers (p0):")
                print(network.links_t.p0.loc[snapshot])

            if hasattr(network, "transformers_t") and "p0" in network.transformers_t:
                print("\n Transformer Flows:")
                print(network.transformers_t.p0.loc[snapshot])

            print("\n Generators P:")
            print(network.generators_t.p.loc[snapshot])

            if hasattr(network.generators_t, "q"):
                print("\n Generators Q:")
                print(network.generators_t.q.loc[snapshot])

            print("\n Bus Angles [°]:")
            print(np.degrees(network.buses_t.v_ang.loc[snapshot]))

            print(
                "\n Max |Δθ| Line : "
                f"{res.get('max_abs_dtheta_line_rad', 0.0):.5f} rad "
                f"({res.get('max_abs_dtheta_line_deg', 0.0):.3f}°)"
            )
            print(
                " Max |Δθ| Trafo: "
                f"{res.get('max_abs_dtheta_trafo_rad', 0.0):.5f} rad "
                f"({res.get('max_abs_dtheta_trafo_deg', 0.0):.3f}°)"
            )
            # print("Line where max. angle occurs: " f"{ res.get('max_angle_line_number')})

            if vsi_default is not None:
                if isinstance(vsi_default, dict):
                    if snapshot in vsi_default:
                        print("\n FVSI Default:")
                        print(vsi_default[snapshot].sort_values(ascending=False))
                else:
                    print("\n FVSI Default:")
                    print(vsi_default.sort_values(ascending=False))


# --- NEW: finaler Report NACH Guard (AC-basiert) ----------------------------
def show_snapshot_report_after_guard(results, network, snapshots="all"):
    """
    Zeigt den finalen AC-Zustand NACH dem N-1-Guard.
    Nutzt network.*_t.* (p0/q0) und druckt S-basierte Auslastungen, Links-p0 und Winkel.
    """
    if not results:
        print("No results to show")
        return

    if snapshots is None:
        snapshots_to_show = list(results.keys())
    elif snapshots == "all":
        snapshots_to_show = list(results.keys())
    elif isinstance(snapshots, (list, tuple, pd.Index)):
        snapshots_to_show = list(snapshots)
    else:
        snapshots_to_show = [snapshots]

    if not snapshots_to_show:
        snapshots_to_show = [list(results.keys())[0]]

    snapshots_to_show = [s for s in snapshots_to_show if s in results]

    if not snapshots_to_show:
        print("None of the requested snapshots found in the results (after guard).")
        return

    # Preparation:

    s_line_max = network.lines.s_nom * network.lines.s_max_pu.fillna(1.0)
    has_trafos = not getattr(network, "transformers", pd.DataFrame()).empty
    if has_trafos:
        s_trafo_max = network.transformers.s_nom * network.transformers.s_max_pu.fillna(
            1.0
        )

    # --- Ausgabe je Snapshot ---
    for snap in snapshots_to_show:
        res = results.get(snap, {})
        print(f"\n ====== Snapshot: {snap} (AFTER GUARD) ======")

        # 0) Angle Check
        angle_limit_deg = res.get("angle_limit_deg", 25.0)
        limit_rad = np.radians(angle_limit_deg)
        # Buswinkel (rad) aus dem *aktuellen* Netz (nach Guard-pf()):
        angles = network.buses_t.v_ang.loc[snap]

        # Lines: deltatheta rad-array
        dtheta_line_rad_after_guard = []
        for l in network.lines.index:
            i = network.lines.at[l, "bus0"]
            j = network.lines.at[l, "bus1"]
            dtheta_line_rad_after_guard.append(float(angles[i] - angles[j]))
        dtheta_line_rad_after_guard = np.array(dtheta_line_rad_after_guard)

        # Trafos: deltatheta rad-array (trafos modelled as lines)
        dtheta_trafo_rad_after_guard = []
        for t in network.transformers.index:
            i = network.transformers.at[t, "bus0"]
            j = network.transformers.at[t, "bus1"]
            dtheta_trafo_rad_after_guard.append(float(angles[i] - angles[j]))
        dtheta_trafo_rad_after_guard = np.array(dtheta_trafo_rad_after_guard)

        max_dtheta_line_rad_after_guard = (
            float(np.max(np.abs(dtheta_line_rad_after_guard)))
            if dtheta_line_rad_after_guard.size
            else 0.0
        )
        max_dtheta_trafo_rad_after_guard = (
            float(np.max(np.abs(dtheta_trafo_rad_after_guard)))
            if dtheta_trafo_rad_after_guard.size
            else 0.0
        )

        # In Degree for Report
        max_dtheta_line_deg = float(np.degrees(max_dtheta_line_rad_after_guard))
        max_dtheta_trafo_deg = float(np.degrees(max_dtheta_trafo_rad_after_guard))

        violated = (max_dtheta_line_rad_after_guard > limit_rad) or (
            max_dtheta_trafo_rad_after_guard > limit_rad
        )
        if violated:
            print(
                f" Angle limit violated – at least one branch exceeds ±{angle_limit_deg:.1f}°"
            )
        else:
            print(f" All branch angle differences are within ±{angle_limit_deg:.1f}°")

        print(
            "\n Max |Δθ| Line (after Guard): "
            f"{max_dtheta_line_rad_after_guard:.5f} rad ({max_dtheta_line_deg:.3f}°)"
        )
        print(
            " Max |Δθ| Trafo (after Guard): "
            f"{max_dtheta_trafo_rad_after_guard:.5f} rad ({max_dtheta_trafo_deg:.3f}°)"
        )

        # 1) Print new optimized Link
        if hasattr(network, "links_t") and "p0" in network.links_t:
            print("\n Link Transfers (p0):")
            print(network.links_t.p0.loc[snap])

        # 2) Lines Default and Now
        default_lines_S = res.get("loading_S_default", None)
        if default_lines_S is not None:
            print("\n Line Loading Default (MVA based):")
            print(default_lines_S.sort_values(ascending=False).round(2))

        P = network.lines_t.p0.loc[snap].astype(float)
        Q = (
            network.lines_t.q0.loc[snap].astype(float)
            if "q0" in getattr(network.lines_t, "_series", {})
            else 0.0 * P
        )
        S = np.hypot(P, Q)
        loading_lines_S = 100.0 * S / s_line_max

        print("\n Line Loading (final, after Guard):")
        print(loading_lines_S.sort_values(ascending=False).round(2))

        # Delta (Default -> Guard), wenn Default vorhanden
        if default_lines_S is not None:
            change_lines = (loading_lines_S - default_lines_S).reindex(
                loading_lines_S.index
            )
            print("\n Line Loading Change (Default --> Guard) [%]:")
            print(change_lines.sort_values(ascending=False).round(2))

        # 3) Trafos Default and Now

        default_trafos_S = res.get("trafo_loading_default", None)
        if default_trafos_S is not None:
            print("\n Trafo Loading Default:")
            print(default_trafos_S.sort_values(ascending=False).round(2))

        if has_trafos and "p0" in getattr(network.transformers_t, "_series", {}):
            PT = network.transformers_t.p0.loc[snap].astype(float)
            QT = (
                network.transformers_t.q0.loc[snap].astype(float)
                if "q0" in getattr(network.transformers_t, "_series", {})
                else 0.0 * PT
            )
            ST = np.hypot(PT, QT)
            loading_trafos_S = 100.0 * ST / s_trafo_max

            print("\n Trafo Loading (final, after Guard):")
            print(loading_trafos_S.sort_values(ascending=False).round(2))

            if default_trafos_S is not None:
                change_trafos = (loading_trafos_S - default_trafos_S).reindex(
                    loading_trafos_S.index
                )
                print("\n Trafo Loading Change (Default --> Guard) [%]:")
                print(change_trafos.sort_values(ascending=False).round(2))

        # 4) Link Transfers (p0)
        # if "p0" in getattr(network.links_t, "_series", {}):
        #     print("\n Link Transfers (p0):")
        #     print(network.links_t.p0.loc[snap])

        # 5) Generators P/Q
        print("\n Generators P:")
        print(network.generators_t.p.loc[snap].round(6))
        if hasattr(network.generators_t, "q"):
            print("\n Generators Q:")
            print(network.generators_t.q.loc[snap].round(6))

        # 6) Buswinkel
        print("\n Bus Angles [°]:")
        print(np.degrees(network.buses_t.v_ang.loc[snap]).round(6))
