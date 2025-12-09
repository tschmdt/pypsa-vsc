import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from linopy import Model
from types import SimpleNamespace


def q_optimization_linopy(
    network,
    epsilon=20,
    v_target=1.0,
    pf_callback=None,
    lpf_callback=None,
    plot_heatmap=True,
    solver_name="gurobi",  # wähle hier deinen QP-fähigen Solver (z.B. "gurobi"); je nach Setup auch "highs"
    solver_options=None,  # dict mit solver-spezifischen Optionen (optional)
):
    """
    Drop-in-Ersatz für q_optimization – nur Schritt 5 wird mit linopy formuliert/gelöst.
    Rückgabestruktur (results) bleibt identisch, inkl. eines result-Objekts mit .x/.success/.message/.status.
    """
    results = {}

    for snapshot in network.snapshots:
        network.snapshot = snapshot

        # 0. Default-Zustand nach initialem PF
        base_q = network.controllable_vscs["q_set"].copy()

        # 1. Initiale Spannungen (per Unit)
        v_mag_default = network.buses_t.v_mag_pu.loc[snapshot]

        # 2. Relevante Busse (PQ) + Zielabweichung
        pq_buses = network.buses.query('control == "PQ"').index
        print("PQ buses List:", pq_buses)
        v_base = network.buses_t.v_mag_pu.loc[snapshot, pq_buses]
        delta_v = v_target - v_base
        m = len(pq_buses)
        n = len(network.controllable_vscs)

        # 3. Sensitivitätsmatrix S (m x n): dV/dQ via kleinem Schritt epsilon
        S = np.zeros((m, n))
        for j, vsc in enumerate(network.controllable_vscs.index):
            network.controllable_vscs["q_set"] = base_q.copy()
            network.controllable_vscs.at[vsc, "q_set"] += epsilon  # [MVAr]
            if pf_callback is not None:
                pf_callback()
            else:
                network.pf()
            v_eps = network.buses_t.v_mag_pu.loc[snapshot, pq_buses]
            S[:, j] = (v_eps - v_base) / epsilon

        S_df = pd.DataFrame(S, index=pq_buses, columns=network.controllable_vscs.index)
        vmax_val = S_df.values.max()
        if plot_heatmap:
            plt.figure(figsize=(14, 6))
            sns.heatmap(
                S_df,
                cmap="GnBu",
                annot=True,
                cbar=True,
                linewidths=0.5,
                linecolor="lightgray",
                vmin=0,
                vmax=vmax_val,
            )
            plt.show()

        # Q auf Ausgangszustand zurücksetzen
        network.controllable_vscs["q_set"] = base_q.copy()

        # 4. Blindleistungsgrenzen (Box-Bounds)
        q_min = network.controllable_vscs["q_min"].to_numpy()
        q_max = network.controllable_vscs["q_max"].to_numpy()

        # 5. Optimierung (Least-Squares als QP in linopy)
        #    min 0.5 * || S q - delta_v ||_2^2   s.t.  q_min <= q <= q_max
        vscs = list(network.controllable_vscs.index)
        buses = list(pq_buses)

        S_da = xr.DataArray(S, coords=[("bus", buses), ("vsc", vscs)])
        dv_da = xr.DataArray(delta_v.to_numpy(), coords=[("bus", buses)])
        qmin_da = xr.DataArray(q_min, coords=[("vsc", vscs)])
        qmax_da = xr.DataArray(q_max, coords=[("vsc", vscs)])

        mlin = Model()

        q = mlin.add_variables(lower=qmin_da, upper=qmax_da, name="q")  # dim='vsc'

        residual = (S_da * q).sum("vsc") - dv_da  # dim='bus', linear
        objective = 0.5 * (residual * residual).sum()  # quadratisch, skalar

        mlin.add_objective(objective, sense="min")

        # optional: Solver-Optionen setzen
        solve_kwargs = {}
        if solver_options:
            solve_kwargs.update(solver_options)

        solve_res = mlin.solve(solver_name="gurobi", **solve_kwargs)

        # Ergebnis in SciPy-ähnliches Objekt gießen (für 1:1-Kompatibilität)
        # q.value -> xr.DataArray über 'vsc'
        q_series = q.value.to_pandas().reindex(vscs)
        q_opt = q_series.to_numpy()

        success = np.isfinite(q_opt).all()
        message = "optimal" if success else "no feasible solution or solver failed"
        status = 0 if success else 1

        result = SimpleNamespace(
            x=q_opt,
            success=success,
            message=message,
            status=status,
            solver_result=solve_res,
        )

        q_opt_series = pd.Series(q_opt, index=network.controllable_vscs.index)

        if not result.success:
            print(
                f"\n Q-optimization NOT successful for snapshot {snapshot}: {result.message}"
            )
            results[snapshot] = {"status": "failed"}
            continue
        print(f"\n Q-optimization successful for snapshot {snapshot}.")

        # 6. Übergabe: Optimiertes Q zurück ins Netzwerk
        network.controllable_vscs["q_set"] = q_opt

        # 7. Finaler PF (wie im Original ohne Schutz – hier wird derselbe Call beibehalten)
        pf_callback()

        # 8. Spannungsanalyse nach Optimierung
        v_mag_optimized = network.buses_t.v_mag_pu.loc[snapshot]
        v_diff = ((v_mag_optimized - v_mag_default) / v_mag_default) * 100

        # 9. Rückgabe-Struktur beibehalten
        results[snapshot] = {
            "S_df": S_df,
            "result": result,
            "q_opt": q_opt_series,
            "v_mag_default": v_mag_default,
            "v_mag_optimized": v_mag_optimized,
            "v_diff": v_diff,
        }

    return results
