import linopy as lp
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# MIQP in linopy: Link-Optimierung (Portierung deines Pyomo-Codes)
# ------------------------------------------------------------


def link_optimization_linopy(
    network,
    angle_limit_deg,
    pf_callback=None,
    lpf_callback=None,  # wird hier nicht verwendet; belassen zur API-Kompatibilität
    max_line_loading=0.95,
    detail_level=None,
    snapshots="all",
    solver_name="gurobi",  # "gurobi" empfohlen für MIQP; alternativ "highs" (falls Version MIQP kann)
    solver_io_api="direct",  # "direct" für Gurobi; bei HiGHS kann "highs" genügen
    solver_options=None,  # dict mit Solver-Optionen (optional)
):
    """
    Linopy-Version deiner P-Optimierung als MIQP.
    Alle Logiken aus dem Pyomo-Original sind abgebildet.
    """
    results = {}
    theta_limit_rad = np.radians(angle_limit_deg)

    # Wähle Snapshots
    if snapshots is None or snapshots == "all":
        snapshots_iter = list(network.snapshots)
    elif isinstance(snapshots, (list, tuple, pd.Index)):
        snapshots_iter = list(snapshots)
    else:
        snapshots_iter = [snapshots]

    for snapshot in snapshots_iter:
        network.snapshot = snapshot

        # --- 0) Default-Zustand, Line-Loading vor Optimierung (Report) ---
        S = np.sqrt(network.lines_t.p0.loc[snapshot] ** 2)
        s_nom = network.lines.s_nom
        s_nom_matrix = pd.DataFrame(
            np.tile(s_nom.values, (1, 1)), index=[snapshot], columns=S.index
        )
        loading_default = S / s_nom_matrix.loc[snapshot] * 100

        # --- 1) Sets / networkdaten ---
        # Koordinaten-Indizes mit Namen (verhindert dim_0-Warnungen)
        buses = pd.Index(network.buses.index, name="bus")
        lines = pd.Index(network.lines.index, name="line")
        links = pd.Index(network.links.index, name="link")
        transformers = pd.Index(network.transformers.index, name="trafo")

        # Slack-Busse
        slack_buses_list = (
            network.generators[network.generators.control == "Slack"]
            .bus.unique()
            .tolist()
        )
        ref_bus = slack_buses_list[0] if slack_buses_list else buses[0]

        # Mappings
        line_from = {l: network.lines.at[l, "bus0"] for l in lines}
        line_to = {l: network.lines.at[l, "bus1"] for l in lines}
        link_from = {k: network.links.at[k, "bus0"] for k in links}
        link_to = {k: network.links.at[k, "bus1"] for k in links}
        eff_k = {k: float(network.links.at[k, "efficiency"]) for k in links}
        trafo_from = {t: network.transformers.at[t, "bus0"] for t in transformers}
        trafo_to = {t: network.transformers.at[t, "bus1"] for t in transformers}

        # Reaktanzen der Leitungen auf 1 MVA-Basis (p.u.)
        x_pu = {}
        for l in lines:
            bus0 = network.lines.at[l, "bus0"]
            U_n = network.buses.at[bus0, "v_nom"]  # kV
            S_base = 1.0  # MVA
            Z_base = (U_n * 1e3) ** 2 / (S_base * 1e6)
            X = network.lines.at[l, "x"]
            x_pu[l] = float(X / Z_base)

        # Trafos (x pro s_nom) → ebenfalls auf 1 MVA-Basis
        x_trafo = {
            t: float(
                network.transformers.at[t, "x"] / network.transformers.at[t, "s_nom"]
            )
            for t in transformers
        }

        # Capacities [MVA]. Typed CIGRE lines store s_nom on line_types;
        # apply_line_types copies r/x/b but not s_nom, so the column can be 0.
        s_nom_line = {}
        for l in lines:
            sn = float(network.lines.at[l, "s_nom"])
            if sn <= 0.0:
                typ = network.lines.at[l, "type"]
                npar = float(network.lines.at[l, "num_parallel"])
                sn = float(network.line_types.at[typ, "s_nom"]) * npar
                network.lines.at[l, "s_nom"] = sn
            s_nom_line[l] = sn
        s_nom_trafo = {
            t: float(network.transformers.at[t, "s_nom"]) for t in transformers
        }

        # Link-Bounds (Richtungssplitting)
        p_nom = {k: float(network.links.at[k, "p_nom"]) for k in links}
        p_min = {
            k: float(network.links.at[k, "p_min_pu"]) * p_nom[k] for k in links
        }  # negativ möglich
        p_max = {k: float(network.links.at[k, "p_max_pu"]) * p_nom[k] for k in links}

        ub_pos = {k: max(p_max[k], 0.0) for k in links}
        ub_neg = {k: max(-p_min[k], 0.0) for k in links}  # -p_min ist positiv

        # Generatoren & Lasten je Bus
        gen_p = network.generators.groupby("bus")["p_set"].sum().to_dict()
        load_p = network.loads.groupby("bus")["p_set"].sum().to_dict()

        # --- 2) Modell aufsetzen ---
        m = lp.Model()
        # Koordinaten als Liste von (name, values)-Tupeln angeben
        coords_bus = [("bus", buses)]
        coords_line = [("line", lines)]
        coords_link = [("link", links)]
        coords_trafo = [("trafo", transformers)]

        # Variablen mit den neuen coords
        theta = m.add_variables(
            name="theta", lower=-np.inf, upper=np.inf, coords=coords_bus
        )
        f_line = m.add_variables(
            name="f_line", lower=-np.inf, upper=np.inf, coords=coords_line
        )
        f_trafo = m.add_variables(
            name="f_trafo", lower=-np.inf, upper=np.inf, coords=coords_trafo
        )

        p_link_pos = m.add_variables(
            name="p_link_pos",
            lower=0.0,
            upper=pd.Series(ub_pos).reindex(links),
            coords=coords_link,
        )
        p_link_neg = m.add_variables(
            name="p_link_neg",
            lower=0.0,
            upper=pd.Series(ub_neg).reindex(links),
            coords=coords_link,
        )
        p_link = m.add_variables(
            name="p_link", lower=-np.inf, upper=np.inf, coords=coords_link
        )
        y_dir = m.add_variables(name="y_dir", binary=True, coords=coords_link)

        # Referenzbus fixieren (theta = 0)
        m.add_constraints(theta.sel(bus=ref_bus) == 0.0)

        # --- 3) Nebenbedingungen ---

        # Link-Splitting
        for k in links:
            m.add_constraints(
                p_link.sel(link=k) == p_link_pos.sel(link=k) - p_link_neg.sel(link=k)
            )

        # Richtungslogik (Big-M mit den UBs)
        for k in links:
            if ub_pos[k] > 0:
                m.add_constraints(
                    p_link_pos.sel(link=k) <= ub_pos[k] * y_dir.sel(link=k)
                )
            else:
                # Falls UB==0, ist die Variable ohnehin durch Upper-Bound 0 fixiert
                pass
            if ub_neg[k] > 0:
                m.add_constraints(
                    p_link_neg.sel(link=k) <= ub_neg[k] * (1 - y_dir.sel(link=k))
                )

        # Flussgleichungen Leitungen/Trafos
        for l in lines:
            i = line_from[l]
            j = line_to[l]
            m.add_constraints(
                f_line.sel(line=l) == (theta.sel(bus=i) - theta.sel(bus=j)) / x_pu[l]
            )
        for t in transformers:
            i = trafo_from[t]
            j = trafo_to[t]
            m.add_constraints(
                f_trafo.sel(trafo=t)
                == (theta.sel(bus=i) - theta.sel(bus=j)) / x_trafo[t]
            )

        # Winkelgrenzen
        for l in lines:
            i = line_from[l]
            j = line_to[l]
            dij = theta.sel(bus=i) - theta.sel(bus=j)
            m.add_constraints(dij <= theta_limit_rad)
            m.add_constraints(dij >= -theta_limit_rad)
        for t in transformers:
            i = trafo_from[t]
            j = trafo_to[t]
            dij = theta.sel(bus=i) - theta.sel(bus=j)
            m.add_constraints(dij <= theta_limit_rad)
            m.add_constraints(dij >= -theta_limit_rad)

        # Lastgrenzen Leitungen/Trafos
        for l in lines:
            cap = max_line_loading * s_nom_line[l]
            m.add_constraints(f_line.sel(line=l) <= cap)
            m.add_constraints(f_line.sel(line=l) >= -cap)
        for t in transformers:
            cap = max_line_loading * s_nom_trafo[t]
            m.add_constraints(f_trafo.sel(trafo=t) <= cap)
            m.add_constraints(f_trafo.sel(trafo=t) >= -cap)

        # Knotenbilanz inkl. Richtungsverluste auf Links
        # Slack-Generator nur auf Slack-Bussen
        slack_set = set(slack_buses_list)
        p_slack_var = None
        if slack_buses_list:
            p_slack_var = m.add_variables(
                name="p_slack_gen",
                lower=-1e4,
                upper=1e4,
                coords=[("bus", slack_buses_list)],
            )

        for b in buses:
            # Konstanten (als floats)
            gen_const = gen_p.get(b, 0.0)
            load_const = load_p.get(b, 0.0)

            # Starte mit einer leeren linopy-Expression (0 * Variable)
            expr = 0 * theta.sel(bus=b)

            # Slack-Generator (falls Bus in Slack-Menge)
            if p_slack_var is not None and b in slack_set:
                expr = expr + p_slack_var.sel(bus=b)

            # --- Sammle Beiträge je Bus ---

            # Leitungen
            line_in = [f_line.sel(line=l) for l in lines if line_to[l] == b]
            line_out = [f_line.sel(line=l) for l in lines if line_from[l] == b]

            # Trafos
            trafo_in = [f_trafo.sel(trafo=t) for t in transformers if trafo_to[t] == b]
            trafo_out = [
                f_trafo.sel(trafo=t) for t in transformers if trafo_from[t] == b
            ]

            # DC-Links mit richtungsabhängigen Verlusten
            link_in_terms = []
            link_out_terms = []
            for k in links:
                eff = eff_k[k]
                if link_to[k] == b:
                    # pos: bus0->bus1 kommt an bus1 mit Wirkungsgrad an
                    link_in_terms.append(p_link_pos.sel(link=k) * eff)
                    # neg: bus1->bus0 verlässt bus1 ohne Wirkungsgrad
                    link_out_terms.append(p_link_neg.sel(link=k))
                if link_from[k] == b:
                    # neg: bus1->bus0 kommt an bus0 mit Wirkungsgrad an
                    link_in_terms.append(p_link_neg.sel(link=k) * eff)
                    # pos: bus0->bus1 verlässt bus0 ohne Wirkungsgrad
                    link_out_terms.append(p_link_pos.sel(link=k))

            # --- Bilanz zusammenbauen ---
            if line_in:
                expr = expr + sum(line_in)
            if line_out:
                expr = expr - sum(line_out)
            if trafo_in:
                expr = expr + sum(trafo_in)
            if trafo_out:
                expr = expr - sum(trafo_out)
            if link_in_terms:
                expr = expr + sum(link_in_terms)
            if link_out_terms:
                expr = expr - sum(link_out_terms)

            # Konstanten ganz zum Schluss
            expr = expr + gen_const - load_const

            m.add_constraints(expr == 0)

        # --- 4) Objective (convex quadratic): sum of (f/s_nom)^2 ---
        # Do not start from the Python int 0: linopy 0.9 rejects a nonzero .const.
        obj = None
        for l in lines:
            term = (f_line.sel(line=l) / s_nom_line[l]) ** 2
            obj = term if obj is None else obj + term
        for t in transformers:
            term = (f_trafo.sel(trafo=t) / s_nom_trafo[t]) ** 2
            obj = term if obj is None else obj + term
        m.add_objective(obj)

        # --- 5) Solve ---

        m.solve(
            solver_name="gurobi",
            io_api="direct",
            MIPGap=1e-3,
            TimeLimit=600,
        )

        # --- 6) Debug: Richtungskonsistenz ---
        eps = 1e-6
        try:
            ppos_series = p_link_pos.solution.to_series()
            pneg_series = p_link_neg.solution.to_series()
            for k in links:
                ppos = float(ppos_series.loc[k])
                pneg = float(pneg_series.loc[k])
                if ppos > eps and pneg > eps:
                    print(
                        f"Warnung: Beide Richtungen aktiv auf {k}: p+={ppos:.3f}, p-={pneg:.3f}"
                    )
                else:
                    print(f" {k}: Link-Fluss nur in eine Richtung aktiv. Korrekt!")
        except Exception:
            pass

        if m.status not in ("ok", "optimal"):
            print(
                f"\n P-Optimierung NICHT erfolgreich für Snapshot {snapshot}: {m.status}"
            )
            results[snapshot] = {"status": "failed"}
            continue
        print("\n P-Optimierung erfolgreich.")

        # --- 7) Rückschreiben der Link-Setpoints und AC-PF ---
        # p_link.solution ist ein xarray.DataArray; .sel(...).item() ist robust skalar
        for k in links:
            network.links.at[k, "p_set"] = float(p_link.solution.sel(link=k).item())

        if pf_callback is not None:
            pf_callback()
        else:
            network.pf()

        # --- 8) Analyse/Report (wie in deinem Code) ---
        angles = network.buses_t.v_ang.loc[snapshot]  # rad

        dtheta_line_rad = []
        for l in network.lines.index:
            i = network.lines.at[l, "bus0"]
            j = network.lines.at[l, "bus1"]
            dtheta_line_rad.append(float(angles[i] - angles[j]))
        dtheta_line_rad = np.array(dtheta_line_rad)

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

        verletzung_line = (
            bool((np.abs(dtheta_line_rad) > theta_limit_rad).any())
            if dtheta_line_rad.size
            else False
        )
        verletzung_trafo = (
            bool((np.abs(dtheta_trafo_rad) > theta_limit_rad).any())
            if dtheta_trafo_rad.size
            else False
        )
        verletzung = verletzung_line or verletzung_trafo

        max_dtheta_line_deg = float(np.degrees(max_dtheta_line_rad))
        max_dtheta_trafo_deg = float(np.degrees(max_dtheta_trafo_rad))

        S = np.sqrt(network.lines_t.p0.loc[snapshot] ** 2)
        s_nom = network.lines.s_nom
        s_nom_matrix = pd.DataFrame(
            np.tile(s_nom.values, (1, 1)), index=[snapshot], columns=S.index
        )
        loading = 100 * S / s_nom_matrix
        loading_change = (loading - loading_default) / loading_default * 100
        total_loading_default = loading_default.sum().sum()
        total_loading_opt = loading.loc[snapshot].sum()
        trafo_loading = (
            np.abs(network.transformers_t.p0.loc[snapshot])
            / network.transformers.s_nom.values
        ) * 100
        final_total_loading = loading.sum().sum() + trafo_loading.sum().sum()

        # Lösungen für Rückgabe
        f_lines_sol = {}
        theta_sol = {}
        try:
            f_lines_sol = {
                l: float(m.solution["f_line"].to_series().loc[l]) for l in lines
            }
            theta_sol = {
                b: float(m.solution["theta"].to_series().loc[b]) for b in buses
            }
        except Exception:
            pass

        results[snapshot] = {
            "model": m,
            "angle_limit_deg": angle_limit_deg,
            "f_lines": f_lines_sol,
            "theta": theta_sol,
            "loading_default": loading_default,
            "loading": loading,
            "loading_change": loading_change,
            "trafo_loading": trafo_loading,
            "total_loading_default": total_loading_default,
            "final_total_loading": final_total_loading,
            "theta_limit_rad": theta_limit_rad,
            "verletzung": verletzung,
            "max_abs_dtheta_line_rad": max_dtheta_line_rad,
            "max_abs_dtheta_trafo_rad": max_dtheta_trafo_rad,
            "max_abs_dtheta_line_deg": max_dtheta_line_deg,
            "max_abs_dtheta_trafo_deg": max_dtheta_trafo_deg,
            "angle_violation_line": verletzung_line,
            "angle_violation_trafo": verletzung_trafo,
            "max_angle_line_number": max_dtheta_line_rad,  # wie im Original
            "status": "optimal",
            "objective": float(m.objective.value)
            if hasattr(m, "objective") and hasattr(m.objective, "value")
            else None,
        }

    if detail_level is not None:
        show_snapshot_report(
            results, network, snapshots=snapshots, detail_level=detail_level
        )

    return results


# ------------------------------------------------------------
# Report-Funktion aus deinem Code (unverändert, nur übernommen)
# ------------------------------------------------------------
def show_snapshot_report(results, network, snapshots="all", detail_level=2):
    """
    detail_level:
        0 = nur wichtigste Parameter
        1 = mittlere Detailtiefe
        2 = volle Ausgabe
    """
    if not results:
        print(" Keine Ergebnisse zum Anzeigen.")
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

    snapshots_to_show = [snap for snap in snapshots_to_show if snap in results]
    if not snapshots_to_show:
        print(" Keine der angeforderten Snapshots in den Ergebnissen gefunden.")
        return

    for snapshot in snapshots_to_show:
        if snapshot not in results:
            print("Snapshot ", snapshot, "not in Results")
            continue

        res = results[snapshot]
        print(f"\n ====== Snapshot: {snapshot} ======")

        if res.get("status") == "failed":
            print("Optimierung fehlgeschlagen\n")
            continue

        if detail_level >= 0:
            if res["verletzung"]:
                print(
                    f" Winkelgrenze verletzt – mindestens ein Zweig überschreitet ±{res['angle_limit_deg']:.1f}°--> Beachte: Toleranzabweichung von Modell (DC) und Report (AC)"
                )
            else:
                print(
                    f" Alle Zweig-Winkeldifferenzen liegen innerhalb ±{res['angle_limit_deg']:.1f}°"
                )

        if detail_level >= 1:
            print("\n Line Loading Default:")
            print(res["loading_default"])

            print("\n Line Loading:")
            print(res["loading"])

            print("\n Line Loading Change [%]:")
            print(res["loading_change"].round(2))

            print("\n Trafo laodings: ")
            print(res["trafo_loading"])

        if detail_level >= 2:
            print("\n Default Total Loading:")
            print(res["total_loading_default"])

            print("\n Final Total Loading:")
            print(res["final_total_loading"])

            print("\n Line Flows:")
            for l, val in res["f_lines"].items():
                print(f" {l}: {val:.2f} MW")

            if hasattr(network, "links_t") and "p0" in network.links_t:
                print("\n Link Transfers (p0):")
                print(network.links_t.p0.loc[snapshot])

            if hasattr(network, "transformers_t") and "p0" in network.transformers_t:
                print("\n🔌 Transformer Flows:")
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
