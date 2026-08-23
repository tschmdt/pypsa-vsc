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
    snapshot : z.B. network.snapshots[0]
    S_df : DataFrame mit Index = PQ-Busse, Columns = VSC-Namen
    vsc_name : Name eines ControllableVSC
    dq : kleine Q-Änderung [MVAr]
    """
    network.snapshot = snapshot

    pq_buses = network.buses.query('control == "PQ"').index

    # 1) Basiszustand
    network.pf()
    v_base = network.buses_t.v_mag_pu.loc[snapshot, pq_buses].copy()

    # 2) numerische Perturbation
    q_old = network.controllable_vscs_t.q_set.loc[snapshot, vsc_name]
    network.controllable_vscs_t.q_set.loc[snapshot, vsc_name] = q_old + dq
    network.pf()

    v_new = network.buses_t.v_mag_pu.loc[snapshot, pq_buses]
    dv_num = v_new - v_base

    # Zustand zurücksetzen
    network.controllable_vscs_t.q_set.loc[snapshot, vsc_name] = q_old
    network.pf()

    # 3) analytisches dV aus S-Matrix
    dv_ana = S_df.loc[pq_buses, vsc_name] * dq

    diff = dv_num.values - dv_ana.values
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
    # sn_mva = 1 -> no further scaling required
    n_b = len(bus_indexer)
    n_v = len(vsc_names)

    B = np.zeros((n_b, n_v))

    for j, vsc in enumerate(vsc_names):
        bus = network.controllable_vscs.at[vsc, "bus"]

        if bus in bus_indexer:
            i = bus_indexer.get_loc(bus)
            B[i, j] = 1.0

    return B


# --- Jacobian blocks -----------------------------------------


def compute_jacobian_blocks_from_Ybus(Ybus, V, theta):
    G = Ybus.real
    B = Ybus.imag

    Vk = V.reshape(-1, 1)
    Vm = V.reshape(1, -1)

    th = theta.reshape(-1, 1) - theta.reshape(1, -1)

    cos_ = np.cos(th)
    sin_ = np.sin(th)

    # P_k, Q_k
    P = (Vk * (G * cos_ + B * sin_) * Vm).sum(axis=1)
    Q = (Vk * (G * sin_ - B * cos_) * Vm).sum(axis=1)

    # Off-diagonal terms
    H = Vk * Vm * (G * sin_ - B * cos_)       # dP/dtheta
    N = Vk * (G * cos_ + B * sin_)            # dP/d|V|
    M = -Vk * Vm * (G * cos_ + B * sin_)      # dQ/dtheta
    L = Vk * (G * sin_ - B * cos_)            # dQ/d|V|

    # Diagonal terms
    Bdiag = np.diag(B)
    Gdiag = np.diag(G)

    Vsafe = np.maximum(V, 1e-12)

    np.fill_diagonal(H, -Q - Bdiag * (V**2))
    np.fill_diagonal(N, P / Vsafe + Gdiag * V)
    np.fill_diagonal(M, P - Gdiag * (V**2))
    np.fill_diagonal(L, Q / Vsafe - Bdiag * V)

    return H, N, M, L


# --- Schur sensitivity matrix for one subnetwork ------------


def schur_S_matrix_for_subnetwork(
    subnet,
    snapshot,
    network,
    pq_buses_mask=None,
):
    # Ybus rows follow buses_o (Slack/PV first). Keep that order for V, θ, B.
    buses = subnet.buses_o
    buses = buses[buses.isin(network.buses_t.v_mag_pu.columns)]

    if len(buses) == 0:
        return pd.DataFrame(), [], []

    V = network.buses_t.v_mag_pu.loc[snapshot, buses].to_numpy()
    theta = network.buses_t.v_ang.loc[snapshot, buses].to_numpy()

    if not hasattr(subnet, "Y") or subnet.Y is None:
        subnet.calculate_Y(skip_pre=False, active_branches_only=True)

    Ybus = subnet.Y.toarray() if hasattr(subnet.Y, "toarray") else np.asarray(subnet.Y)

    if pq_buses_mask is None:
        pq_all = network.buses.index[network.buses.control.eq("PQ")]
        pq_buses = buses[buses.isin(pq_all)]
    else:
        pq_buses = buses[pq_buses_mask]

    if len(pq_buses) == 0:
        return (
            pd.DataFrame(columns=list(network.controllable_vscs.index)),
            list(network.controllable_vscs.index),
            [],
        )

    slack_bus = _subnet_slack_bus(subnet)
    non_slack = buses[buses != slack_bus]

    idx = {b: i for i, b in enumerate(buses)}

    i_th = np.array([idx[b] for b in non_slack], dtype=int)
    i_v = np.array([idx[b] for b in pq_buses], dtype=int)

    H, N, M, L = compute_jacobian_blocks_from_Ybus(
        Ybus,
        V,
        theta,
    )

    Hred = H[np.ix_(i_th, i_th)]
    Nred = N[np.ix_(i_th, i_v)]
    Mred = M[np.ix_(i_v, i_th)]
    Lred = L[np.ix_(i_v, i_v)]

    H_lu = splu(csc_matrix(Hred))

    X = H_lu.solve(Nred)

    K = csc_matrix(Lred) - csc_matrix(Mred) @ csc_matrix(X)

    vsc_list = list(network.controllable_vscs.index)

    Bfull = build_B_mapping(
        network,
        vsc_list,
        buses,
    )

    Bpq = Bfull[i_v, :]

    K_lu = splu(K.tocsc())

    S_mat = K_lu.solve(Bpq)

    return (
        pd.DataFrame(
            S_mat,
            index=pq_buses,
            columns=vsc_list,
        ),
        vsc_list,
        list(pq_buses),
    )


# --- Aggregation over subnetworks ----------------------------


def compute_S_matrix_all_subnets(network, snapshot):
    blocks = []

    for sn in network.sub_networks.obj:
        if not hasattr(sn, "Y") or sn.Y is None:
            sn.calculate_Y(
                skip_pre=False,
                active_branches_only=True,
            )

        S_df, _, _ = schur_S_matrix_for_subnetwork(
            sn,
            snapshot,
            network,
        )

        if not S_df.empty:
            blocks.append(S_df)

    if not blocks:
        return pd.DataFrame()

    cols = blocks[0].columns

    blocks = [
        b.reindex(
            columns=cols,
            fill_value=0.0,
        )
        for b in blocks
    ]

    return pd.concat(
        blocks,
        axis=0,
    )


def q_opt_step_without_pf_loop(
    network,
    snapshot,
    angle_limit_deg,
    v_target=1.0,
    plot_heatmap=True,
):
    """
    Single Q-optimization step:
    - build S at the current AC operating point
    - solve for Delta Q
    - set q_new = q_now + Delta Q
    - perform one AC power flow
    """
    network.snapshot = snapshot

    if not np.isfinite(
        network.buses_t.v_mag_pu.loc[snapshot]
    ).all():
        network.pf()

    pq_buses = network.buses.query(
        'control == "PQ"'
    ).index

    v_base = network.buses_t.v_mag_pu.loc[
        snapshot,
        pq_buses,
    ].copy()

    delta_v = (
        v_target - v_base
    ).to_numpy()

    S_df = compute_S_matrix_all_subnets(
        network,
        snapshot,
    )

    S_df = S_df.reindex(
        index=pq_buses,
        columns=network.controllable_vscs.index,
    ).fillna(0.0)

    S = S_df.to_numpy()

    assert S.shape == (
        len(pq_buses),
        len(S_df.columns),
    )

    q_min = network.controllable_vscs.loc[
        S_df.columns,
        "q_min",
    ].to_numpy()

    q_max = network.controllable_vscs.loc[
        S_df.columns,
        "q_max",
    ].to_numpy()

    q_now = network.controllable_vscs_t.q_set.loc[
        snapshot,
        S_df.columns,
    ].to_numpy()

    result = lsq_linear(
        S,
        delta_v,
        bounds=(
            q_min - q_now,
            q_max - q_now,
        ),
    )

    delta_q = result.x

    q_opt = pd.Series(
        np.clip(
            q_now + delta_q,
            q_min,
            q_max,
        ),
        index=S_df.columns,
    )

    v_pred = v_base + pd.Series(
        S @ delta_q,
        index=pq_buses,
    )

    print("\nQ adjustment:")
    print(pd.DataFrame({
        "Q before": q_now,
        "Delta Q": delta_q,
        "Q after": q_opt.values,
    }, index=S_df.columns))

    print("\nLinear voltage prediction:")
    print(pd.DataFrame({
        "V before": v_base,
        "V predicted": v_pred,
    }))

    network.controllable_vscs_t.q_set.loc[
        snapshot,
        q_opt.index,
    ] = q_opt.values

    network.controllable_vscs.loc[
        q_opt.index,
        "q_set",
    ] = q_opt.values

    network.pf()

    if plot_heatmap:
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

    return {
        "S_df": S_df,
        "result": result,
        "q_opt": q_opt,
    }


# --- Main Q optimization ------------------------------------


def q_optimization(
    network,
    angle_limit_deg,
    v_target=1.0,
    pf_callback=None,
    lpf_callback=None,
    plot_heatmap=True,
    q_limit_callback=None,
    v_base0: pd.DataFrame | None = None,
    loading_base0: pd.DataFrame | None = None,
):
    print(
        "[link_opt] target:",
        getattr(network, "_whoami", "unknown"),
        id(network),
    )

    results = {}

    for snapshot in network.snapshots:
        network.snapshot = snapshot

        if q_limit_callback is not None:
            q_limit_callback(snapshot)

        # Ensure current AC operating point
        if pf_callback is not None:
            pf_callback()
        else:
            network.pf()

        # 1. Initial voltage state
        v_mag_default = network.buses_t.v_mag_pu.loc[snapshot].copy()

        # 2. PQ buses and voltage target
        pq_buses = network.buses.query('control == "PQ"').index

        print("PQ buses List:", pq_buses)

        v_band = 0.03
        max_q_iter = 3
        v_tol = 0.01

        vsc_cols = network.controllable_vscs.index
        q_min = network.controllable_vscs.loc[vsc_cols, "q_min"].to_numpy()
        q_max = network.controllable_vscs.loc[vsc_cols, "q_max"].to_numpy()
        q_now = network.controllable_vscs_t.q_set.loc[snapshot, vsc_cols].to_numpy()
        q_snap0 = q_now.copy()

        S_df = None
        S_mat = None
        result = None
        delta_q = None
        v_base = None
        q_opt = q_now

        def apply_q(q_vec: np.ndarray) -> None:
            network.controllable_vscs_t.q_set.loc[snapshot, vsc_cols] = q_vec
            network.controllable_vscs.loc[vsc_cols, "q_set"] = q_vec
            if pf_callback is not None:
                pf_callback()
            else:
                network.pf()

        for _it in range(max_q_iter):
            v_base = network.buses_t.v_mag_pu.loc[snapshot, pq_buses].copy()
            e = v_base.to_numpy() - v_target
            err_before = float(np.max(np.abs(e)))
            if err_before <= v_tol:
                break

            # e = V - v_target [p.u.]; deadband: no ΔV request if |e| <= v_band
            delta_v = np.where(np.abs(e) <= v_band, 0.0, v_target - v_base.to_numpy())

            S_df = compute_S_matrix_all_subnets(network, snapshot)
            S_df = S_df.reindex(index=pq_buses, columns=vsc_cols).fillna(0.0)
            S_mat = S_df.to_numpy()

            result = lsq_linear(
                S_mat,
                delta_v,
                bounds=(q_min - q_now, q_max - q_now),
            )
            if not result.success:
                print(
                    f"\nQ-optimization NOT successful for snapshot "
                    f"{snapshot}: {result.message}"
                )
                results[snapshot] = {"status": "failed"}
                break

            delta_q = result.x
            q_opt = np.clip(q_now + delta_q, q_min, q_max)
            apply_q(q_opt)

            e_after = (
                network.buses_t.v_mag_pu.loc[snapshot, pq_buses].to_numpy() - v_target
            )
            if float(np.max(np.abs(e_after))) > err_before:
                delta_q = 0.5 * delta_q
                q_opt = np.clip(q_now + delta_q, q_min, q_max)
                apply_q(q_opt)

            q_now = q_opt

        if result is not None and not result.success:
            continue

        if S_df is None:
            S_df = compute_S_matrix_all_subnets(network, snapshot)
            S_df = S_df.reindex(index=pq_buses, columns=vsc_cols).fillna(0.0)
            S_mat = S_df.to_numpy()
            v_base = network.buses_t.v_mag_pu.loc[snapshot, pq_buses].copy()
            delta_q = np.zeros(len(vsc_cols))

        print(f"\nQ-optimization successful for snapshot {snapshot}.")

        q_opt_series = pd.Series(q_now, index=vsc_cols)
        v_pred = v_base + pd.Series(S_mat @ delta_q, index=pq_buses)

        print("\nQ adjustment:")
        print(
            pd.DataFrame(
                {
                    "Q before": q_snap0,
                    "Delta Q": q_now - q_snap0,
                    "Q after": q_now,
                },
                index=vsc_cols,
            )
        )

        print("\nLinear voltage prediction:")
        print(
            pd.DataFrame(
                {
                    "V before": v_base,
                    "V predicted": v_pred,
                }
            )
        )

        # Loading bar vs initial PF: max |S|/s_nom [%]
        P0 = network.lines_t.p0.loc[snapshot]
        Q0 = network.lines_t.q0.loc[snapshot]
        load_pct = 100.0 * np.hypot(P0, Q0) / network.lines.s_nom
        if loading_base0 is not None:
            load0_max = float(loading_base0.loc[snapshot].max())
            if float(load_pct.max()) > load0_max:
                q_acc = q_now.copy()
                for alpha in (0.5, 0.0):
                    q_try = np.clip(q_snap0 + alpha * (q_acc - q_snap0), q_min, q_max)
                    apply_q(q_try)
                    P0 = network.lines_t.p0.loc[snapshot]
                    Q0 = network.lines_t.q0.loc[snapshot]
                    load_pct = 100.0 * np.hypot(P0, Q0) / network.lines.s_nom
                    q_now = q_try
                    if float(load_pct.max()) <= load0_max:
                        break
                q_opt_series = pd.Series(q_now, index=vsc_cols)

        # Diagnostic: predicted vs actual AC voltage
        print(
            "\nVoltage prediction vs actual AC result:"
        )

        print(
            pd.DataFrame(
                {
                    "V before": v_base,
                    "V predicted": v_pred,
                    "V AC after": network.buses_t.v_mag_pu.loc[
                        snapshot,
                        pq_buses,
                    ],
                }
            )
        )

        # 8. Voltage analysis
        v_mag_optimized = network.buses_t.v_mag_pu.loc[
            snapshot
        ].copy()

        v_diff = (
            (v_mag_optimized - v_mag_default)
            / v_mag_default
        ) * 100

        angles = network.buses_t.v_ang.loc[
            snapshot
        ]

        # 9. Branch angle differences
        theta_limit_rad = np.radians(
            angle_limit_deg
        ) - np.radians(3.0)

        dtheta_line_rad = []

        for line in network.lines.index:
            i = network.lines.at[line, "bus0"]
            j = network.lines.at[line, "bus1"]

            dtheta_line_rad.append(
                float(
                    angles[i]
                    - angles[j]
                )
            )

        dtheta_line_rad = np.array(
            dtheta_line_rad
        )

        dtheta_trafo_rad = []

        for trafo in network.transformers.index:
            i = network.transformers.at[
                trafo,
                "bus0",
            ]

            j = network.transformers.at[
                trafo,
                "bus1",
            ]

            dtheta_trafo_rad.append(
                float(
                    angles[i]
                    - angles[j]
                )
            )

        dtheta_trafo_rad = np.array(
            dtheta_trafo_rad
        )

        max_dtheta_line_rad = (
            float(
                np.max(
                    np.abs(
                        dtheta_line_rad
                    )
                )
            )
            if dtheta_line_rad.size
            else 0.0
        )

        max_dtheta_trafo_rad = (
            float(
                np.max(
                    np.abs(
                        dtheta_trafo_rad
                    )
                )
            )
            if dtheta_trafo_rad.size
            else 0.0
        )

        violation_line = (
            bool(
                (
                    np.abs(
                        dtheta_line_rad
                    )
                    > (
                        theta_limit_rad
                        + np.radians(3.0)
                    )
                ).any()
            )
            if dtheta_line_rad.size
            else False
        )

        violation_trafo = (
            bool(
                (
                    np.abs(
                        dtheta_trafo_rad
                    )
                    > (
                        theta_limit_rad
                        + np.radians(3.0)
                    )
                ).any()
            )
            if dtheta_trafo_rad.size
            else False
        )

        violation = (
            violation_line
            or violation_trafo
        )

        max_dtheta_line_deg = float(
            np.degrees(
                max_dtheta_line_rad
            )
        )

        max_dtheta_trafo_deg = float(
            np.degrees(
                max_dtheta_trafo_rad
            )
        )

        # 10. Final line loading
        P0_new = network.lines_t.p0.loc[
            snapshot
        ]

        P1_new = network.lines_t.p1.loc[
            snapshot
        ]

        Q0_new = network.lines_t.q0.loc[
            snapshot
        ]

        Q1_new = network.lines_t.q1.loc[
            snapshot
        ]

        S0_new = np.hypot(
            P0_new,
            Q0_new,
        )

        S1_new = np.hypot(
            P1_new,
            Q1_new,
        )

        s_nom = network.lines.s_nom

        s_max_pu = network.lines.get(
            "s_max_pu",
            pd.Series(
                1.0,
                index=s_nom.index,
            ),
        ).reindex(
            s_nom.index
        )

        S_limit = (
            s_nom
            * s_max_pu
        ).replace(
            0,
            np.nan,
        )

        loading0_ac_new = (
            100
            * S0_new
            / S_limit
        )

        loading1_ac_new = (
            100
            * S1_new
            / S_limit
        )

        loading_lines_new = pd.concat(
            [
                loading0_ac_new,
                loading1_ac_new,
            ],
            axis=1,
        ).max(
            axis=1
        )

        # Transformers
        P_T0_new = network.transformers_t.p0.loc[
            snapshot
        ]

        P_T1_new = network.transformers_t.p1.loc[
            snapshot
        ]

        Q_T0_new = network.transformers_t.q0.loc[
            snapshot
        ]

        Q_T1_new = network.transformers_t.q1.loc[
            snapshot
        ]

        S_T0_new = np.hypot(
            P_T0_new,
            Q_T0_new,
        )

        S_T1_new = np.hypot(
            P_T1_new,
            Q_T1_new,
        )

        s_T_nom = network.transformers.s_nom

        s_T_max_pu = network.transformers.get(
            "s_max_pu",
            pd.Series(
                1.0,
                index=s_T_nom.index,
            ),
        ).reindex(
            s_T_nom.index
        )

        S_T_limit = (
            s_T_nom
            * s_T_max_pu
        ).replace(
            0,
            np.nan,
        )

        loading_T0_ac_new = (
            100
            * S_T0_new
            / S_T_limit
        )

        loading_T1_ac_new = (
            100
            * S_T1_new
            / S_T_limit
        )

        loading_trafo_new = pd.concat(
            [
                loading_T0_ac_new,
                loading_T1_ac_new,
            ],
            axis=1,
        ).max(
            axis=1
        )

        # Heatmap
        if plot_heatmap:
            S_plot = S_df.copy()

            vmin_val = 0.0

            vmax_val = np.quantile(
                S_plot.values,
                0.95,
            )

            norm = mcolors.PowerNorm(
                gamma=0.5,
                vmin=vmin_val,
                vmax=vmax_val,
            )

            fig, ax = plt.subplots(
                figsize=(5, 3)
            )

            hm = sns.heatmap(
                S_plot,
                cmap="GnBu",
                norm=norm,
                annot=False,
                cbar=True,
                linewidths=0.5,
                linecolor="lightgray",
                cbar_kws={
                    "shrink": 0.8
                },
                ax=ax,
            )

            ax.set_xlabel(
                "Controllable VSC"
            )

            ax.set_ylabel(
                "Bus"
            )

            cbar = hm.collections[0].colorbar

            factor = 1e3

            ticks = cbar.get_ticks()

            cbar.set_ticks(ticks)

            cbar.set_ticklabels(
                [
                    f"{t * factor:.2f}"
                    for t in ticks
                ]
            )

            cbar.ax.set_title(
                r"$\times 10^{-3}$",
                fontsize=10,
                pad=6,
            )

            plt.tight_layout()
            plt.show()

        # 11. Results
        results[snapshot] = {
            "S_df": S_df,
            "result": result,
            "q_opt": q_opt_series,
            "delta_q": pd.Series(
                delta_q,
                index=S_df.columns,
            ),
            "v_pred": v_pred,
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
    if not results:
        print("No results to show")
        return

    if snapshots is None:
        snapshots_to_show = list(
            results.keys()
        )

    elif snapshots == "all":
        snapshots_to_show = list(
            results.keys()
        )

    elif isinstance(
        snapshots,
        (
            list,
            tuple,
            pd.Index,
        ),
    ):
        snapshots_to_show = list(
            snapshots
        )

    else:
        snapshots_to_show = [
            snapshots
        ]

    if not snapshots_to_show:
        snapshots_to_show = [
            list(
                results.keys()
            )[0]
        ]

    for snapshot in snapshots_to_show:
        if snapshot not in results:
            print(
                "Snapshot ",
                snapshot,
                "Not in Results",
            )
            continue

        res = results[
            snapshot
        ]

        print(
            f"\n ====== Snapshot: {snapshot} ======"
        )

        if detail_level >= 0:
            print(
                "\n VSC Q Optimized [MVAr]:"
            )

            for vsc in network.controllable_vscs.index:
                if vsc in res[
                    "q_opt"
                ].index:
                    print(
                        f"{vsc}: "
                        f"Q = {res['q_opt'][vsc]:.3f} MVAr"
                    )

            if res[
                "violation"
            ]:
                print(
                    f" Angle limit violated – at least one branch "
                    f"exceeds ±{res['angle_limit_deg']:.1f}°"
                )

            else:
                print(
                    f" All branch angle differences are within "
                    f"±{res['angle_limit_deg']:.1f}°"
                )

        if detail_level >= 1:
            print(
                "\n Final Line Loadings "
            )

            print(
                res[
                    "loading_line_final"
                ].head(5)
            )

            print(
                "\n Voltages (Default): "
            )

            print(
                res[
                    "v_mag_default"
                ].head(5)
            )

            print(
                "\n Voltages (Optimized): "
            )

            print(
                res[
                    "v_mag_optimized"
                ].head(5)
            )

            print(
                "\n Voltage Difference [%]: "
            )

            print(
                res[
                    "v_diff"
                ].head(5)
            )

        if detail_level >= 2:
            print(
                "\n Final Bus Angles [°]:"
            )

            print(
                np.degrees(
                    network.buses_t.v_ang.loc[
                        snapshot
                    ]
                )
            )

            print(
                "\n Max |dtheta| Line : "
                f"{res.get('max_abs_dtheta_line_rad', 0.0):.5f} rad "
                f"({res.get('max_abs_dtheta_line_deg', 0.0):.3f} deg)"
            )

            print(
                " Max |dtheta| Trafo: "
                f"{res.get('max_abs_dtheta_trafo_rad', 0.0):.5f} rad "
                f"({res.get('max_abs_dtheta_trafo_deg', 0.0):.3f} deg)"
            )

            print(
                "\n Sensitivity Matrix: "
            )

            print(
                res[
                    "S_df"
                ]
            )

            if vsi_default is not None:
                if isinstance(
                    vsi_default,
                    dict,
                ):
                    if snapshot in vsi_default:
                        print(
                            "\n FVSI Default:"
                        )

                        print(
                            vsi_default[
                                snapshot
                            ].sort_values(
                                ascending=False
                            ).head(3)
                        )

                else:
                    print(
                        "\n FVSI Default:"
                    )

                    print(
                        vsi_default.sort_values(
                            ascending=False
                        ).head(3)
                    )

            if vsi_after_P is not None:
                if isinstance(
                    vsi_after_P,
                    dict,
                ):
                    if snapshot in vsi_after_P:
                        print(
                            "\n FVSI before Q-optimization "
                            "(after P-optimization):"
                        )

                        print(
                            vsi_after_P[
                                snapshot
                            ].sort_values(
                                ascending=False
                            ).head(3)
                        )

                else:
                    print(
                        "\n FVSI before Q-optimization "
                        "(after P-optimization):"
                    )

                    print(
                        vsi_after_P.sort_values(
                            ascending=False
                        ).head(3)
                    )

            if vsi_opt is not None:
                if isinstance(
                    vsi_opt,
                    dict,
                ):
                    if snapshot in vsi_opt:
                        print(
                            "\n FVSI after Q-optimization:"
                        )

                        print(
                            vsi_opt[
                                snapshot
                            ].sort_values(
                                ascending=False
                            ).head(3)
                        )

                else:
                    print(
                        "\n FVSI after Q-optimization:"
                    )

                    print(
                        vsi_opt.sort_values(
                            ascending=False
                        ).head(3)
                    )
