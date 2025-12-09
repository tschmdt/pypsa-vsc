import numpy as np
from scipy.optimize import lsq_linear
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import matplotlib.colors as mcolors


# --- Helpers -------------------------------------------------


def check_S_column(network, snapshot, S_df, vsc_name, dq=1.0):
    """
    Prüft eine Spalte der Sensitivitätsmatrix S_df gegen einen numerischen dV/dQ-Versuch.

    network : pypsa.Network
    snapshot : z.B. network.snapshots[0] oder "now"
    S_df : DataFrame mit Index = PQ-Busse, Columns = VSC-Namen
    vsc_name : Name eines ControllableVSC
    dq : kleine Q-Änderung (in MVAr)
    """
    network.snapshot = snapshot

    # nur PQ-Busse (wie in deiner S-Berechnung)
    pq_buses = network.buses.query('control == "PQ"').index

    # 1) Basiszustand
    network.pf()
    v_base = network.buses_t.v_mag_pu.loc[snapshot, pq_buses].copy()

    # 2) numerische Perturbation: Q am gewählten VSC ändern
    q_old = network.controllable_vscs_t.q_set.loc[snapshot, vsc_name]
    network.controllable_vscs_t.q_set.loc[snapshot, vsc_name] = q_old + dq
    network.pf()

    v_new = network.buses_t.v_mag_pu.loc[snapshot, pq_buses]
    dv_num = v_new - v_base  # numerisches dV

    # Zustand wieder zurücksetzen
    network.controllable_vscs_t.q_set.loc[snapshot, vsc_name] = q_old
    network.pf()

    # 3) analytisches dV aus S-Matrix
    dv_ana = S_df.loc[pq_buses, vsc_name] * dq  # Sdf Spalte aus Schur Ansatz

    diff = dv_num.values - dv_ana.values  # Vergleich numerisch vs Schur
    max_err = np.max(np.abs(diff))
    corr = np.corrcoef(dv_num.values, dv_ana.values)[0, 1]

    print(f"\n=== Check für {vsc_name} (ΔQ = {dq} MVAr) ===")
    print("max |dv_num - dv_ana| =", max_err)
    print("Korrelationskoeffizient:", corr)
    return max_err, corr


def _subnet_bus_index(subnet) -> pd.Index:
    buses_attr = getattr(subnet, "buses", None)
    if buses_attr is None:
        raise RuntimeError("SubNetwork has no 'buses' attribute/method.")
    buses = buses_attr() if callable(buses_attr) else buses_attr
    if isinstance(buses, pd.DataFrame):
        return pd.Index(buses.index)
    if isinstance(buses, (pd.Index, list, tuple, set, np.ndarray)):
        return pd.Index(list(buses))
    if hasattr(buses, "index"):
        return pd.Index(buses.index)
    raise TypeError(f"Unsupported type for subnet.buses: {type(buses)}")


def _subnet_slack_bus(subnet) -> str:
    slack_attr = getattr(subnet, "slack_bus", None)
    if slack_attr is None:
        raise RuntimeError("SubNetwork has no 'slack_bus' attribute/method.")
    slack = slack_attr() if callable(slack_attr) else slack_attr
    if isinstance(slack, (pd.Series, pd.DataFrame)) and hasattr(slack, "name"):
        return slack.name
    return str(slack)


def build_B_mapping(network, vsc_names, bus_indexer):
    # sn_mva=1 in deinem Netz → keine Skalierung nötig.
    n_b = len(bus_indexer)
    n_v = len(vsc_names)
    B = np.zeros((n_b, n_v))
    for j, vsc in enumerate(vsc_names):
        bus = network.controllable_vscs.at[vsc, "bus"]
        if bus in bus_indexer:
            i = bus_indexer.get_loc(bus)
            B[i, j] = 1.0  # pu/MVAr bei sn_mva=1
    return B


# --- Korrekte Jacobian-Blöcke (exakt wie in deiner Formelgrafik) -----


def compute_jacobian_blocks_from_Ybus(Ybus, V, theta):
    G = Ybus.real
    B = Ybus.imag

    Vk = V.reshape(-1, 1)
    Vm = V.reshape(1, -1)

    th = theta.reshape(-1, 1) - theta.reshape(1, -1)  # θ_km
    cos_ = np.cos(th)
    sin_ = np.sin(th)

    # P_k, Q_k in pu
    P = (Vk * (G * cos_ + B * sin_) * Vm).sum(axis=1)
    Q = (Vk * (G * sin_ - B * cos_) * Vm).sum(axis=1)

    # Offdiag
    H = Vk * Vm * (G * sin_ - B * cos_)  # dP/dθ
    N = Vk * (G * cos_ + B * sin_)  # dP/d|V|
    M = -Vk * Vm * (G * cos_ + B * sin_)  # dQ/dθ
    L = Vk * (G * sin_ - B * cos_)  # dQ/d|V|

    # Diagonalen
    Bdiag = np.diag(B)
    Gdiag = np.diag(G)
    Vsafe = np.maximum(V, 1e-12)

    np.fill_diagonal(H, -Q - Bdiag * (V**2))
    np.fill_diagonal(N, P / Vsafe + Gdiag * V)
    np.fill_diagonal(M, P - Gdiag * (V**2))
    np.fill_diagonal(L, Q / Vsafe - Bdiag * V)

    return H, N, M, L


# --- Schur-S für ein Subnetz -----------------------------------------


def schur_S_matrix_for_subnetwork(subnet, snapshot, network, pq_buses_mask=None):
    buses = _subnet_bus_index(subnet)
    buses = buses.intersection(network.buses_t.v_mag_pu.columns)
    if len(buses) == 0:
        return pd.DataFrame(), [], []

    V = network.buses_t.v_mag_pu.loc[snapshot, buses].to_numpy()
    theta = network.buses_t.v_ang.loc[snapshot, buses].to_numpy()

    if not hasattr(subnet, "Y") or subnet.Y is None:
        subnet.calculate_Y(skip_pre=False, active_branches_only=True)
    Ybus = subnet.Y.toarray() if hasattr(subnet.Y, "toarray") else np.asarray(subnet.Y)

    if pq_buses_mask is None:
        pq_all = network.buses.index[network.buses.control.eq("PQ")]
        pq_buses = buses.intersection(pq_all)
    else:
        pq_buses = buses[pq_buses_mask]
    if len(pq_buses) == 0:
        return (
            pd.DataFrame(columns=list(network.controllable_vscs.index)),
            list(network.controllable_vscs.index),
            [],
        )

    slack_bus = _subnet_slack_bus(subnet)
    non_slack = buses.difference([slack_bus])

    idx = {b: i for i, b in enumerate(buses)}
    i_th = np.array([idx[b] for b in non_slack], dtype=int)  # Winkel-Variablen
    i_v = np.array([idx[b] for b in pq_buses], dtype=int)  # |V|-Variablen

    H, N, M, L = compute_jacobian_blocks_from_Ybus(Ybus, V, theta)

    Hred = H[np.ix_(i_th, i_th)]
    Nred = N[np.ix_(i_th, i_v)]
    Mred = M[np.ix_(i_v, i_th)]
    Lred = L[np.ix_(i_v, i_v)]

    H_lu = splu(csc_matrix(Hred))
    X = H_lu.solve(Nred)  # H^{-1}N
    K = csc_matrix(Lred) - csc_matrix(Mred) @ csc_matrix(X)

    vsc_list = list(network.controllable_vscs.index)
    Bfull = build_B_mapping(network, vsc_list, buses)  # nb x n_vsc
    Bpq = Bfull[i_v, :]

    K_lu = splu(K.tocsc())
    S_mat = K_lu.solve(Bpq)  # nPQ x nVSC

    return (
        pd.DataFrame(S_mat, index=pq_buses, columns=vsc_list),
        vsc_list,
        list(pq_buses),
    )


# --- Aggregation über alle Subnetze ----------------------------------


def compute_S_matrix_all_subnets(network, snapshot):
    blocks = []
    for sn in network.sub_networks.obj:
        if not hasattr(sn, "Y") or sn.Y is None:
            sn.calculate_Y(skip_pre=False, active_branches_only=True)
        S_df, _, _ = schur_S_matrix_for_subnetwork(sn, snapshot, network)
        if not S_df.empty:
            blocks.append(S_df)
    if not blocks:
        return pd.DataFrame()
    cols = blocks[0].columns
    blocks = [b.reindex(columns=cols, fill_value=0.0) for b in blocks]
    return pd.concat(blocks, axis=0)


def q_opt_step_without_pf_loop(
    network, snapshot, angle_limit_deg, v_target=1.0, plot_heatmap=True
):
    """
    Ersatz für deinen bisherigen 'epsilon/PF-Schleifen'-Schritt:
    - baut S via Schur,
    - löst lsq_linear wie zuvor,
    - schreibt q_set zurück und macht genau EINE PF (für Analyse).
    """
    from scipy.optimize import lsq_linear

    # 0) Betriebspunkt muss stehen
    network.snapshot = snapshot
    if not np.isfinite(network.buses_t.v_mag_pu.loc[snapshot]).all():
        network.pf()

    # 1) Zielvektor Δv für PQ-Busse
    pq_buses = network.buses.query('control == "PQ"').index
    v_base = network.buses_t.v_mag_pu.loc[snapshot, pq_buses].copy()
    delta_v = (v_target - v_base).to_numpy()

    # 2) S-Matrix via Schur
    S_df = compute_S_matrix_all_subnets(network, snapshot)
    # Sicherstellen, dass Zeilen zu pq_buses passen (ggf. reindex)
    S_df = S_df.reindex(index=pq_buses)
    S = S_df.to_numpy()

    # 3) Bounds
    # 2) S-Matrix via Schur
    S_df = compute_S_matrix_all_subnets(network, snapshot)
    S_df = S_df.reindex(index=pq_buses, columns=network.controllable_vscs.index).fillna(
        0.0
    )

    S = S_df.to_numpy()
    delta_v_vec = (v_target - v_base).to_numpy()

    # --- Sanity-Checks
    assert S.shape == (len(pq_buses), len(S_df.columns)), f"S shape mismatch: {S.shape}"
    assert delta_v_vec.shape[0] == S.shape[0], "delta_v length mismatch"

    # 3) Bounds in Spaltenreihenfolge von S_df
    q_min = network.controllable_vscs.loc[S_df.columns, "q_min"].to_numpy()
    q_max = network.controllable_vscs.loc[S_df.columns, "q_max"].to_numpy()

    # 4) Least Squares (wie bei dir)
    res = lsq_linear(S, delta_v, bounds=(q_min, q_max))
    q_opt = pd.Series(res.x, index=S_df.columns)

    # 5) Übergabe & eine PF
    network.controllable_vscs_t.q_set.loc[snapshot, q_opt.index] = q_opt.values
    network.controllable_vscs.loc[q_opt.index, "q_set"] = q_opt.values
    network.pf()

    # 6) (optional) Heatmap
    if plot_heatmap:
        import matplotlib.pyplot as plt
        import seaborn as sns

        vmax_val = S_df.values.max()
        plt.figure(figsize=(14, 6))
        sns.heatmap(
            S_df,
            cmap="GnBu",
            annot=False,
            cbar=True,
            linewidths=0.5,
            linecolor="lightgray",
            vmin=0,
            vmax=vmax_val,
        )
        plt.tight_layout()
        plt.show()

    return {"S_df": S_df, "result": res, "q_opt": q_opt}


def q_optimization(
    network,
    angle_limit_deg,
    epsilon=1,
    v_target=1.0,
    pf_callback=None,
    lpf_callback=None,
    plot_heatmap=True,
):
    print("[link_opt] target:", getattr(network, "_whoami", "unknown"), id(network))

    # 0. Setup Optimization Snapshots

    results = {}

    for snapshot in network.snapshots:
        network.snapshot = snapshot

        # if pf_callback is not None:
        #     pf_callback()
        # else:
        #     network.pf()

        # Default state After Initial pf (pf triggered by P_Optimizer AFTER link_optimization)
        # base_q = network.controllable_vscs[
        #     "q_set"
        # ].copy()  # Important with base_q.copy(): this sate can be re-generated
        base_q_t = network.controllable_vscs_t.q_set.loc[snapshot].copy()
        network.controllable_vscs_t.q_set.loc[snapshot] = base_q_t.values
        # 1. Initial Voltage Magnitudes

        v_mag_default = network.buses_t.v_mag_pu.loc[snapshot]

        # 2. Find Relevant Buses (PQ) & Save Voltages

        pq_buses = network.buses.query('control == "PQ"').index
        print("PQ buses List:", pq_buses)
        v_base = network.buses_t.v_mag_pu.loc[snapshot, pq_buses].copy()
        delta_v = v_target - v_base
        m = len(pq_buses)
        n = len(network.controllable_vscs)

        # 3. Sensitivity Matrix S (m x n). Check influence of each controllable_vsc (new component) at each bus. #VSCs == #PFs

        S_df = compute_S_matrix_all_subnets(network, snapshot)
        S_df = S_df.reindex(
            index=pq_buses, columns=network.controllable_vscs.index
        ).fillna(0.0)
        S_mat = S_df.to_numpy()
        delta_v_vec = (v_target - v_base).to_numpy()

        # Checks
        assert S_mat.shape == (len(pq_buses), len(S_df.columns)), (
            f"S shape mismatch: {S_mat.shape}"
        )
        assert delta_v_vec.shape[0] == S_mat.shape[0], "delta_v length mismatch"

        # WICHTIG: Zeilen und Spalten passend sortieren
        S_df = S_df.reindex(
            index=pq_buses, columns=network.controllable_vscs.index
        ).fillna(0.0)

        S_mat = S_df.to_numpy()

        # 4. Reactive Power Limits ("Constraints")

        q_min = network.controllable_vscs.loc[S_df.columns, "q_min"].to_numpy()
        q_max = network.controllable_vscs.loc[S_df.columns, "q_max"].to_numpy()

        # 5. Optimization (least-squares problem; https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html)

        result = lsq_linear(S_mat, delta_v.to_numpy(), bounds=(q_min, q_max))
        q_opt = result.x
        q_opt_series = pd.Series(
            q_opt, index=S_df.columns
        )  # convert to pd.Series (from array)

        if not result.success:
            print(
                f"\n Q-optimization NOT successful for snapshot {snapshot}: {result.message}"
            )
            results[snapshot] = {"status": "failed"}
            continue  # process next snapshot
        print(f"\n Q-optimization successful for snapshot {snapshot}.")

        # 6. Handover Optimized Q-Value(s) to the network.  q_opt (output) serves as input for the real VSC Controller

        network.controllable_vscs_t.q_set.loc[snapshot, q_opt_series.index] = (
            q_opt_series.values
        )
        network.controllable_vscs.loc[q_opt_series.index, "q_set"] = q_opt_series.values

        # 7. Final pf

        pf_callback() if pf_callback is not None else network.pf()

        # 8. Voltage Analysis AFTER DC-based optimization, returning optimization variable to network and final AC-Load Flow claculation

        v_mag_optimized = network.buses_t.v_mag_pu.loc[snapshot]
        v_diff = ((v_mag_optimized - v_mag_default) / v_mag_default) * 100

        angles = network.buses_t.v_ang.loc[snapshot]  # rad

        # Analog to P-Optimizer: Derive the Branch Angles
        # Lines: deltatheta rad-array
        theta_limit_rad = np.radians(angle_limit_deg) - np.radians(
            3.0
        )  # Margin fo DC to AC discrepancy

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

        # 9. Final loading analysis: takes into account the small influence of Q on the Laodings

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

        # Deprecated
        # P = network.lines_t.p0.loc[snapshot]
        # Q = network.lines_t.q0.loc[snapshot]
        # S = np.hypot(P, Q)
        # loading_line_final = 100 * S / network.lines.s_nom

        # P_T = network.transformers_t.p0.loc[snapshot]
        # Q_T = network.transformers_t.q0.loc[snapshot]
        # S_T = np.hypot(P_T, Q_T)
        # loading_trafo_final = 100 * S_T / network.transformers.s_nom

        # Heatmap
        if plot_heatmap:
            S_plot = S_df.copy()

            vmin_val = 0.0
            vmax_val = np.quantile(S_plot.values, 0.95)

            norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin_val, vmax=vmax_val)

            fig, ax = plt.subplots(figsize=(5, 3))
            hm = sns.heatmap(
                S_plot,
                cmap="GnBu",
                norm=norm,
                annot=False,
                cbar=True,
                linewidths=0.5,
                linecolor="lightgray",
                cbar_kws={"shrink": 0.8},
                ax=ax,
            )

            ax.set_xlabel("Controllable VSC")
            ax.set_ylabel("Bus")

            # ---------- Colorbar skalieren: Anzeige * 1e3 ----------
            cbar = hm.collections[0].colorbar
            factor = 1e3  # wir zeigen Werte in 10^-3 an

            ticks = cbar.get_ticks()
            cbar.set_ticklabels([f"{t * factor:.2f}" for t in ticks])

            # Titel über die Legende schreiben, z.B. ×10⁻³
            cbar.ax.set_title(r"$\times 10^{-3}$", fontsize=10, pad=6)
            # --------------------------------------------------------

            plt.tight_layout()
            plt.show()

        # 10. Return Values for easy use outside the modul/inside the class. I.e. controller.q_results["q_opt"]

        results[snapshot] = {
            "S_df": S_df,
            "result": result,
            "q_opt": q_opt_series,
            "v_mag_default": v_mag_default,
            "v_mag_optimized": v_mag_optimized,
            "v_diff": v_diff,
            "angle_limit_deg": angle_limit_deg,
            "violation": violation,
            "max_abs_dtheta_line_rad": max_dtheta_line_rad,
            "max_abs_dtheta_trafo_rad": max_dtheta_trafo_rad,
            "max_abs_dtheta_line_deg": max_dtheta_line_deg,
            "max_abs_dtheta_trafo_deg": max_dtheta_trafo_deg,
            "angle_violation_line": violation_line,
            "angle_violation_trafo": violation_trafo,
            "max_angle_line_number": max_dtheta_line_rad,
            "loading_line_final": loading_lines_new,
            "loading_trafo_final": loading_trafo_new,
        }

    return results


def show_snapshot_q_report(
    results,
    network,
    snapshots="all",
    detail_level=2,
    vsi_after_P=None,
    vsi_opt=None,
    vsi_default=None,
):
    """
    detail_level:
        0 = only most important parameters
        1 = medium-level detailed output
        2 = full output for analysis

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

    for snapshot in snapshots_to_show:
        if snapshot not in results:
            print("Snapshot ", snapshot, "Not in Results")
            continue

        res = results[snapshot]
        print(f"\n ====== Snapshot: {snapshot} ======")

        if detail_level >= 0:
            print("\n VSC Q Optimized [MVAr]:")
            for vsc in network.controllable_vscs.index:
                if vsc in res["q_opt"].index:  # in case q_opt is a series with names
                    print(f"{vsc}: Q = {res['q_opt'][vsc]:.3f} MVAr")
            if res["violation"]:
                print(
                    f" Angle limit violated – at least one branch exceeds ±{res['angle_limit_deg']:.1f}°--> Note: Tolerance deviation between model (DC) and report (AC)"
                )
            else:
                print(
                    f" All branch angle differences are within ±{res['angle_limit_deg']:.1f}°"
                )

        if detail_level >= 1:
            print("\n Final Line Loadings (incl the small influence of Q) ")
            print(res["loading_line_final"].head(3))

            print("\n Voltages (Default): ")
            print(res["v_mag_default"].head(3))

            print("\n Voltages (Optimized): ")
            print(res["v_mag_optimized"].head(3))

            print("\n Voltage Difference [%]: ")
            print(res["v_diff"].head(3))

        if detail_level >= 2:
            print("\n Final Bus Angles [°]:")
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

            print("\n Sensitivity Matrix: ")
            print(res["S_df"])

            if vsi_default is not None:
                if isinstance(vsi_default, dict):
                    if snapshot in vsi_default:
                        print("\n FVSI Default:")
                        print(
                            vsi_default[snapshot].sort_values(ascending=False).head(3)
                        )
                else:
                    print("\n FVSI Default:")
                    print(vsi_default.sort_values(ascending=False).head(3))

            if vsi_after_P is not None:
                if isinstance(vsi_after_P, dict):
                    # if vsi_after_P is a dic
                    if snapshot in vsi_after_P:
                        print("\n FVSI before Q-optimization (after P-optimization):")
                        print(
                            vsi_after_P[snapshot].sort_values(ascending=False).head(3)
                        )
                else:
                    # vsi_after_P is "normal" object (series, df,...)
                    print("\n FVSI before Q-optimization (after P-optimization):")
                    print(vsi_after_P.sort_values(ascending=False).head(3))

            if vsi_opt is not None:
                if isinstance(vsi_opt, dict):
                    if snapshot in vsi_opt:
                        print("\n FVSI after Q-optimization:")
                        print(vsi_opt[snapshot].sort_values(ascending=False).head(3))
                else:
                    print("\n FVSI after Q-optimization:")
                    print(vsi_opt.sort_values(ascending=False).head(3))
