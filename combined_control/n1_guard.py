import math
from typing import Any

import pandas as pd

from pypsa import Network


def ensure_link_pset_timeseries(network: Network) -> None:
    """
    The P-Optimizer uses timeseries values (*_t.*). If this is empty e.g. no SCLOPF beforehan d,
    static values p_set needs to be "copied" to the timeseries values.
    Ensures that links_t.p_set exists and is aligned to (snapshots x VSCs).
    If only static p_set values exist (or none at all), it is initialized cleanly.
    """
    n = network
    if n.links.empty:
        return

    if len(n.snapshots) == 0:
        n.set_snapshots(pd.Index([pd.Timestamp("2000-01-01")]))

    cols = n.links.index

    # If timeseries available--> use it
    if not n.links_t.p_set.empty:
        n.links_t.p_set = n.links_t.p_set.reindex(
            index=n.snapshots, columns=cols, fill_value=0.0
        ).astype(float)
        return

    # If timesries missing--> build from static values (or 0.0)
    if "p_set" in n.links.columns:
        base = n.links["p_set"].reindex(cols).fillna(0.0)
    else:
        base = pd.Series(0.0, index=cols)

    df = pd.DataFrame(0.0, index=n.snapshots, columns=cols, dtype=float)
    for snap in n.snapshots:
        df.loc[snap, :] = base.values
    n.links_t["p_set"] = df


class N1Guard:
    """Preventive N-1 guard: ISF/BODF interval heuristic on link p_set [MW]."""

    def __init__(self, network: Network, cfg: Any) -> None:
        self.network = network
        self.cfg = cfg

    def _get_sn_single(self) -> Any:
        """
        Liefert das (erste) AC-SubNetwork-Objekt robust über PyPSA-Versionen hinweg.
        """
        n = self.network
        n.determine_network_topology()

        # Neuere PyPSA: n.sub_networks mit Accessor .obj (Series/array/etc.)
        if hasattr(n, "sub_networks"):
            df = n.sub_networks
            if hasattr(df, "obj"):
                obj = df.obj
                # obj kann Series / numpy array / list / dict-like sein
                try:
                    return obj.iloc[0]
                except Exception:
                    try:
                        return obj[0]
                    except Exception:
                        return next(iter(obj))
            if isinstance(df, pd.DataFrame) and "obj" in df.columns:
                return df["obj"].iloc[0]

        # Älter: n.sub_networks_obj
        if hasattr(n, "sub_networks_obj") and n.sub_networks_obj is not None:
            obj = n.sub_networks_obj
            try:
                return obj[0]
            except Exception:
                return next(iter(obj))

        raise RuntimeError("Kein SubNetwork gefunden. Bitte PyPSA-Version prüfen.")

    def _get_ptdf_single(self) -> pd.DataFrame:
        """PTDF [p.u.]: rows = `_branch_index` (Line+Transformer), columns = buses."""
        if hasattr(self, "_PTDF_single"):
            return self._PTDF_single

        sn = self._get_sn_single()
        sn.calculate_PTDF()

        arr = getattr(sn, "PTDF", None)
        if arr is None:
            raise RuntimeError("sn.calculate_PTDF() hat kein sn.PTDF befüllt.")

        ptdf = pd.DataFrame(arr, index=sn.branches_i(), columns=sn.buses_i())
        ptdf = ptdf.reindex(self._branch_index()).astype(float)

        self._PTDF_single = ptdf
        return ptdf

    def _get_bodf_single(self) -> pd.DataFrame:
        """BODF/LODF [-]: rows/cols = `_branch_index` (Line+Transformer)."""
        if hasattr(self, "_BODF_single"):
            return self._BODF_single

        sn = self._get_sn_single()

        # Je nach Version steht Ergebnis in sn.BODF oder kommt als Rückgabewert
        bodf_calc = sn.calculate_BODF()
        if bodf_calc is None and hasattr(sn, "BODF"):
            bodf_calc = sn.BODF

        if bodf_calc is None:
            raise RuntimeError("sn.calculate_BODF() hat kein Ergebnis geliefert.")

        branches = self._branch_index()
        bodf = pd.DataFrame(bodf_calc, index=sn.branches_i(), columns=sn.branches_i())
        bodf = bodf.reindex(index=branches, columns=branches).fillna(0.0).astype(float)

        self._BODF_single = bodf
        return bodf

    def _isf_for_links(self) -> dict[str, pd.Series]:
        """
        ISF of a link setpoint step on every AC branch [MW/MW].

        p_set > 0 withdraws ΔP at bus0 and injects ΔP at bus1 (η=1), so
        ISF = PTDF[:, bus1] − PTDF[:, bus0]. Index = `_branch_index`.
        """
        n = self.network
        PTDF = self._get_ptdf_single()
        isf: dict[str, pd.Series] = {}
        for k in n.links.index:
            b0 = n.links.at[k, "bus0"]
            b1 = n.links.at[k, "bus1"]
            col0 = PTDF[b0] if b0 in PTDF.columns else pd.Series(0.0, index=PTDF.index)
            col1 = PTDF[b1] if b1 in PTDF.columns else pd.Series(0.0, index=PTDF.index)
            isf[k] = (col1 - col0).astype(float)
        return isf

    def _lines_s_max(self) -> pd.Series:
        """Line-only thermal ratings [MVA]; prefer `_branch_s_max` in guard paths."""
        s = self._branch_s_max()
        if s.empty or "Line" not in s.index.get_level_values(0):
            return pd.Series(dtype=float)
        return s.xs("Line", level="type")

    def _branch_index(self) -> pd.MultiIndex:
        """Active AC branches as MultiIndex (type, name): Line then Transformer."""
        n = self.network
        keys = [("Line", name) for name in n.lines.index] + [
            ("Transformer", name) for name in n.transformers.index
        ]
        return pd.MultiIndex.from_tuples(keys, names=["type", "name"])

    def _branch_p0(self, snapshot: object) -> pd.Series:
        """DC branch flow p0 at snapshot [MW], indexed by `_branch_index`."""
        n = self.network
        parts: list[pd.Series] = []
        if not n.lines.empty:
            s = n.lines_t.p0.loc[snapshot].astype(float)
            s.index = pd.MultiIndex.from_product(
                [["Line"], s.index], names=["type", "name"]
            )
            parts.append(s)
        if not n.transformers.empty:
            s = n.transformers_t.p0.loc[snapshot].astype(float)
            s.index = pd.MultiIndex.from_product(
                [["Transformer"], s.index], names=["type", "name"]
            )
            parts.append(s)
        if not parts:
            return pd.Series(dtype=float)
        return pd.concat(parts).reindex(self._branch_index()).astype(float)

    def _branch_s_max(self) -> pd.Series:
        """Thermal rating s_nom * s_max_pu [MVA≈MW in DC], indexed by `_branch_index`."""
        n = self.network
        parts: list[pd.Series] = []
        if not n.lines.empty:
            s = (n.lines["s_nom"] * n.lines["s_max_pu"].fillna(1.0)).astype(float)
            s.index = pd.MultiIndex.from_product(
                [["Line"], s.index], names=["type", "name"]
            )
            parts.append(s)
        if not n.transformers.empty:
            s = (
                n.transformers["s_nom"] * n.transformers["s_max_pu"].fillna(1.0)
            ).astype(float)
            s.index = pd.MultiIndex.from_product(
                [["Transformer"], s.index], names=["type", "name"]
            )
            parts.append(s)
        if not parts:
            return pd.Series(dtype=float)
        return pd.concat(parts).reindex(self._branch_index()).astype(float)

    def _resolve_outages(self) -> pd.MultiIndex:
        """
        Contingency set as MultiIndex (type, name).

        Default: all lines and transformers. Configured strings map to the unique
        branch with that name; tuples `(type, name)` are used directly. Ambiguous
        or missing entries are skipped.
        """
        branches = self._branch_index()
        cfg_outages = self.cfg.n1_guard_outages
        if not cfg_outages:
            return branches

        resolved: list[tuple[str, str]] = []
        for o in cfg_outages:
            if isinstance(o, tuple) and len(o) == 2:
                key = (str(o[0]), str(o[1]))
                if key in branches:
                    resolved.append(key)
                continue
            matches = [b for b in branches if b[1] == o]
            if len(matches) == 1:
                resolved.append(matches[0])
        if not resolved:
            return pd.MultiIndex.from_tuples([], names=branches.names)
        return pd.MultiIndex.from_tuples(resolved, names=branches.names)

    def _lpf_refresh(self, snapshot: object) -> None:
        """Ensures that p0 is up to date (DC flows at the armature/intermediate point)"""
        self.network.lpf(snapshot)

    @staticmethod
    def _interval_from_abs_linear(
        a: float, b: float, limit: float
    ) -> tuple[float, float] | None:
        # |b + a*x| <= limit  → Intervall für x
        eps = 1e-12
        if abs(a) < eps:
            return (-float("inf"), float("inf")) if abs(b) <= limit else None
        lo = (-limit - b) / a
        hi = (limit - b) / a
        return (min(lo, hi), max(lo, hi))

    def _link_box(self) -> dict[str, tuple[float, float]]:
        """ΔP bounds [MW] from the current p_set into p_nom · p_min_pu / p_max_pu."""
        n = self.network
        box: dict[str, tuple[float, float]] = {}
        for k in n.links.index:
            p_nom = float(n.links.at[k, "p_nom"]) if "p_nom" in n.links.columns else 0.0
            if p_nom <= 0.0:
                continue
            p_now = float(n.links.at[k, "p_set"]) if "p_set" in n.links.columns else 0.0
            pmin_pu = (
                float(n.links.at[k, "p_min_pu"])
                if "p_min_pu" in n.links.columns
                else -1.0
            )
            pmax_pu = (
                float(n.links.at[k, "p_max_pu"])
                if "p_max_pu" in n.links.columns
                else 1.0
            )
            box[k] = (p_nom * pmin_pu - p_now, p_nom * pmax_pu - p_now)
        return box

    def _store_joint_rows(
        self,
        outages: pd.MultiIndex,
        branches: pd.Index,
        f_base: pd.Series,
        pmax: pd.Series,
        bodf: pd.DataFrame,
        isf_map: dict[str, pd.Series],
    ) -> None:
        """
        Record each finite (outage, branch) row and classify it.

        b [MW] is the post-contingency DC flow. limit [MW] is
        s_nom · s_max_pu · margin. a_k [MW/MW] is ISF_k + L[:,o] · ISF_k[o],
        with p_set > 0 withdrawing at bus0 and injecting at bus1 (η = 1).
        A pair is influenceable when the converter box can reduce |b| by at
        least 0.05 · limit. The outaged branch is omitted.
        """
        box = self._link_box()
        room_down = {k: max(-lo, 0.0) for k, (lo, _hi) in box.items()}
        room_up = {k: max(hi, 0.0) for k, (_lo, hi) in box.items()}
        for k in isf_map:
            room_down.setdefault(k, 0.0)
            room_up.setdefault(k, 0.0)

        rows: dict[tuple[object, object], tuple[float, float, dict[str, float]]] = {}
        influenceable: set[tuple[object, object]] = set()
        protected: set[tuple[object, object]] = set()
        for o in outages:
            if o in bodf.columns:
                lcol = bodf[o].reindex(branches).fillna(0.0)
                f_o = (f_base + lcol * float(f_base.get(o, 0.0))).astype(float)
            else:
                lcol = pd.Series(0.0, index=branches)
                f_o = f_base
            a_by_link = {
                k: (isf + lcol * float(isf.get(o, 0.0))).astype(float)
                for k, isf in isf_map.items()
            }
            for ell in branches:
                if ell == o:
                    continue
                b = float(f_o.get(ell, 0.0))
                lim = float(pmax.get(ell, 0.0))
                if lim <= 0.0 or not math.isfinite(lim) or not math.isfinite(b):
                    continue
                a_row = {
                    k: float(a_vec.get(ell, 0.0)) for k, a_vec in a_by_link.items()
                }
                sign_b = 1.0 if b > 0.0 else (-1.0 if b < 0.0 else 0.0)
                relief = 0.0
                for k, a in a_row.items():
                    if sign_b == 0.0 or a == 0.0:
                        continue
                    room = room_down[k] if sign_b * a > 0.0 else room_up[k]
                    relief += abs(a) * room
                key = (o, ell)
                rows[key] = (b, lim, a_row)
                if relief + 1e-9 >= 0.05 * lim:
                    influenceable.add(key)
                else:
                    protected.add(key)
        self._joint_rows = rows
        self._influenceable_pairs = influenceable
        self._protected_pairs = protected

    def _flow_ends(
        self, b: float, a_row: dict[str, float], box: dict[str, tuple[float, float]]
    ) -> tuple[float, float]:
        """Endpoint flows [MW] of b + Σ a_k ΔP_k over the converter box."""
        flow_lo = b
        flow_hi = b
        for k, a in a_row.items():
            if k not in box or a == 0.0:
                continue
            lo, hi = box[k]
            if a > 0.0:
                flow_lo += a * lo
                flow_hi += a * hi
            else:
                flow_lo += a * hi
                flow_hi += a * lo
        return flow_lo, flow_hi

    def _joint_link_dp(self, links: list[str]) -> dict[str, float]:
        """
        Minimize t [MW], with |ΔP_k| ≤ t, inside the converter box.

        First: |b + Σ a_k ΔP_k| ≤ limit on influenceable pairs that are over
        the limit or that the box can push over it. Protected pairs that the
        box can raise to the current worst protected loading stay at or below
        that loading. If that program is infeasible: minimize the worst
        predicted influenceable loading, then t. Either min-t solution is then
        replaced by the smallest Σ ΔP_k² inside t ≤ t* + 1e-6. ΔP and t are MW
        from p_set. Σ ΔP_k² is MW² and is not added to t.
        """
        import pyomo.environ as pyo
        from pyomo.opt import TerminationCondition

        from combined_control.P_Optimizer_V2 import _solve_scip_direct

        zero = dict.fromkeys(links, 0.0)
        rows = self._joint_rows
        box = self._link_box()
        influenceable = [key for key in self._influenceable_pairs if key in rows]
        if not influenceable:
            return zero

        def worsen(key: tuple[object, object]) -> float:
            b, lim, a_row = rows[key]
            flow_lo, flow_hi = self._flow_ends(b, a_row, box)
            return max(abs(flow_lo), abs(flow_hi)) - abs(b)

        worst_pro = 0.0
        for key in self._protected_pairs:
            b, lim, _a = rows[key]
            worst_pro = max(worst_pro, abs(b) / lim)
        worst_inf = max(abs(rows[key][0]) / rows[key][1] for key in influenceable)

        security = []
        for key in influenceable:
            b, lim, _a = rows[key]
            flow_lo, flow_hi = self._flow_ends(b, rows[key][2], box)
            over = abs(b) > lim + 1e-6
            can_cross = max(abs(flow_lo), abs(flow_hi)) > lim + 1e-6
            if over or can_cross:
                security.append(key)
        if not security:
            return zero

        def can_reach(key: tuple[object, object], worst: float) -> bool:
            b, lim, _a = rows[key]
            now = abs(b) / lim
            rise = max(worsen(key), 0.0) / lim
            if worst <= 1e-9:
                return rise > 1e-9
            return now >= worst - 1e-9 or now + rise >= worst - 1e-9

        protected = [key for key in self._protected_pairs if can_reach(key, worst_pro)]
        reach = [key for key in influenceable if can_reach(key, worst_inf)]

        def add_abs(
            clist: pyo.ConstraintList, model: pyo.ConcreteModel, key: tuple, rhs: float
        ) -> None:
            b, _lim, a_row = rows[key]
            expr = b + sum(a_row.get(k, 0.0) * model.dP[k] for k in links)
            clist.add(expr <= rhs)
            clist.add(expr >= -rhs)

        def build(
            min_t: bool, security_on: bool, u_cap: float | None
        ) -> pyo.ConcreteModel:
            model = pyo.ConcreteModel()
            model.K = pyo.Set(initialize=links)
            model.dP = pyo.Var(model.K)
            model.box = pyo.ConstraintList()
            for k in links:
                lo, hi = box[k]
                model.box.add(model.dP[k] >= lo)
                model.box.add(model.dP[k] <= hi)
            if min_t:
                model.t = pyo.Var(domain=pyo.NonNegativeReals)
                for k in links:
                    model.box.add(model.dP[k] <= model.t)
                    model.box.add(model.dP[k] >= -model.t)
            model.rows = pyo.ConstraintList()
            if security_on:
                for key in security:
                    add_abs(model.rows, model, key, rows[key][1])
            else:
                model.u = pyo.Var(domain=pyo.NonNegativeReals)
                for key in reach:
                    add_abs(model.rows, model, key, model.u * rows[key][1])
                if u_cap is not None:
                    model.rows.add(model.u <= u_cap)
            for key in protected:
                add_abs(model.rows, model, key, worst_pro * rows[key][1])
            if min_t:
                model.obj = pyo.Objective(expr=model.t, sense=pyo.minimize)
            else:
                model.obj = pyo.Objective(expr=model.u, sense=pyo.minimize)
            return model

        def solved(model: pyo.ConcreteModel) -> bool:
            result = _solve_scip_direct(model, tee=False, timelimit=30.0)
            tc = result.solver.termination_condition
            if tc in (
                TerminationCondition.infeasible,
                TerminationCondition.infeasibleOrUnbounded,
            ):
                return False
            if tc not in (
                TerminationCondition.optimal,
                TerminationCondition.locallyOptimal,
            ):
                raise RuntimeError(f"Guard coordination solve ended {tc}.")
            return True

        def read(model: pyo.ConcreteModel) -> dict[str, float]:
            return {k: float(pyo.value(model.dP[k])) for k in links}

        def polish(model: pyo.ConcreteModel) -> dict[str, float]:
            t_star = float(pyo.value(model.t))
            model.t_freeze = pyo.Constraint(expr=model.t <= t_star + 1e-6)
            model.del_component(model.obj)
            model.obj = pyo.Objective(
                expr=sum(model.dP[k] ** 2 for k in links),
                sense=pyo.minimize,
            )
            if not solved(model):
                raise RuntimeError("Guard coordination could not minimize Σ ΔP².")
            return read(model)

        first = build(min_t=True, security_on=True, u_cap=None)
        if solved(first):
            return polish(first)
        relief = build(min_t=False, security_on=False, u_cap=None)
        if not solved(relief):
            raise RuntimeError("Guard coordination has no feasible partial step.")
        u_star = float(pyo.value(relief.u))
        second = build(min_t=True, security_on=False, u_cap=u_star + 1e-6)
        if not solved(second):
            raise RuntimeError("Guard coordination could not minimize t.")
        return polish(second)

    def _enforce_n1_guard_once(self, snapshot: object) -> bool:
        """
        Prüft N-1 (per BODF) und verschiebt Link-p_set minimal (via ISF-Intervalle).
        NEU: Wenn der globale Intervallschnitt leer ist, wähle den 'best-effort' ΔP
        aus den einzelnen verletzten Nebenbedingungen (nächstliegende Projektion).
        """
        n = self.network
        if n.links.empty:
            return True

        margin = float(self.cfg.n1_guard_margin)
        eps_isf = 1e-8
        branches = self._branch_index()

        # Base DC flows [MW] and limits on all AC branches (lines + transformers)
        self._lpf_refresh(snapshot)
        F_base = self._branch_p0(snapshot)
        Pmax = self._branch_s_max() * margin
        outages = self._resolve_outages()

        BODF = self._get_bodf_single()
        isf_map = self._isf_for_links()
        self._store_joint_rows(outages, branches, F_base, Pmax, BODF, isf_map)

        movable = list(self._link_box())
        if len(movable) >= 2:
            overloaded = any(
                abs(b) > lim + 1e-9 for b, lim, _a in self._joint_rows.values()
            )
            if not overloaded:
                return True
            p_now = {
                k: float(n.links.at[k, "p_set"]) if "p_set" in n.links.columns else 0.0
                for k in movable
            }
            dP = self._cap_link_vector(snapshot, p_now, self._joint_link_dp(movable))
            changed = any(abs(step) > 1e-9 for step in dP.values())
            for k, step in dP.items():
                if abs(step) <= 1e-9:
                    continue
                self._write_link_p(k, snapshot, p_now[k] + step)
            if changed:
                self._lpf_refresh(snapshot)
            return changed

        any_violation = False
        link_intervals: dict[str, tuple[float, float]] = {
            k: (-float("inf"), float("inf")) for k in n.links.index
        }
        # Best-effort: Kandidaten und Nebenbedingungen je Link sammeln.
        # Falls kein global zulässiges Intervall existiert, wird später das ΔP
        # gewählt, das die verbleibende maximale relative Überlast minimiert.
        link_candidates: dict[str, list[float]] = {k: [] for k in n.links.index}
        link_constraints: dict[str, list[tuple[float, float, float]]] = {
            k: [] for k in n.links.index
        }

        for o in outages:
            # LODF column for outage o; F_o = F_base + L[:,o] * F_base[o]
            if o in BODF.columns:
                Lcol = BODF[o].reindex(branches).fillna(0.0)
                F_o = (F_base + Lcol * float(F_base.get(o, 0.0))).astype(float)
            else:
                Lcol = pd.Series(0.0, index=branches)
                F_o = F_base

            # Nur weiter, wenn dieser Ausfall überhaupt verletzt
            if not (F_o.abs() > (Pmax + 1e-9)).any():
                continue

            any_violation = True

            for k, ISF in isf_map.items():
                # a = ISF + L[:,o] * ISF[o]  (link sensitivity under outage o)
                a_vec = (ISF + Lcol * float(ISF.get(o, 0.0))).astype(float)

                lo_k, hi_k = link_intervals[k]

                for ell in branches:
                    a = float(a_vec.get(ell, 0.0))
                    b = float(F_o.get(ell, 0.0))
                    lim = float(Pmax.get(ell, 0.0))

                    # Kann der Link diese Leitung praktisch nicht beeinflussen,
                    # ist eine bestehende Überlast für diesen Link nicht lösbar.
                    if abs(a) <= eps_isf:
                        if abs(b) > lim + 1e-9:
                            lo_k, hi_k = 1.0, 0.0
                        continue

                    link_constraints[k].append((a, b, lim))
                    iv = self._interval_from_abs_linear(a, b, lim)

                    if iv is None:
                        lo_k, hi_k = 1.0, 0.0
                        continue

                    # Für verletzte Einzelbedingungen einen best-effort Kandidaten
                    # sammeln: kleinste Änderung, die diese Bedingung erfüllen würde.
                    if abs(b) > lim + 1e-9:
                        candidate = iv[0] if abs(iv[0]) < abs(iv[1]) else iv[1]
                        link_candidates[k].append(candidate)

                    if lo_k <= hi_k:
                        lo_k = max(lo_k, iv[0])
                        hi_k = min(hi_k, iv[1])

                link_intervals[k] = (lo_k, hi_k)

        if not any_violation:
            return True

        changed = False

        # Anwenden: erst Hardware-Grenzen schneiden, dann ΔP wählen.
        for k, (lo, hi) in link_intervals.items():
            p_nom = float(n.links.at[k, "p_nom"]) if "p_nom" in n.links.columns else 0.0
            p_now = float(n.links.at[k, "p_set"]) if "p_set" in n.links.columns else 0.0
            pmin_pu = (
                float(n.links.at[k, "p_min_pu"])
                if "p_min_pu" in n.links.columns
                else -1.0
            )
            pmax_pu = (
                float(n.links.at[k, "p_max_pu"])
                if "p_max_pu" in n.links.columns
                else 1.0
            )
            if p_nom <= 0.0:
                continue

            # Hardware-ΔP
            dP_lo_hw = p_nom * pmin_pu - p_now
            dP_hi_hw = p_nom * pmax_pu - p_now

            # 1) Globaler Schnitt nicht leer und 0 nicht enthalten: kleinstes |ΔP|.
            dP = 0.0
            took_interval = False
            if lo <= hi and not (lo <= 0.0 <= hi):
                lo_clip = max(lo, dP_lo_hw)
                hi_clip = min(hi, dP_hi_hw)
                if lo_clip <= hi_clip:
                    dP = lo_clip if abs(lo_clip) < abs(hi_clip) else hi_clip
                    took_interval = True

            # 2) Fallback, nur wenn der Schnitt kein ΔP geliefert hat.
            constraints = link_constraints.get(k, [])
            if not took_interval:
                cands = link_candidates.get(k, [])
                if cands and constraints:
                    candidates = [0.0]
                    candidates += [
                        min(max(dP_raw, dP_lo_hw), dP_hi_hw) for dP_raw in cands
                    ]
                    dP = min(
                        candidates,
                        key=lambda x: (
                            self._worst_rel_overload(x, constraints),
                            abs(x),
                        ),
                    )

            if abs(dP) <= 1e-9:
                continue
            dP = self._cap_link_dp(k, snapshot, p_now, dP)
            if abs(dP) <= 1e-9:
                continue
            p_new = p_now + dP
            n.links.loc[k, "p_set"] = p_new
            if n.links_t.p_set.empty:
                ensure_link_pset_timeseries(n)
            n.links_t.p_set.loc[snapshot, k] = p_new
            changed = True

        if changed:
            self._lpf_refresh(snapshot)
        return changed

    def _branch_s_nom(self) -> pd.Series:
        """
        s_nom [MVA] for the same terms as the P-opt objective: every line and
        every transformer. P-opt has no other component in J.

        A line with s_nom <= 0 uses line_types.s_nom * num_parallel [MVA], as
        P-opt does. The network column is left unchanged.
        """
        n = self.network
        parts: list[pd.Series] = []
        if not n.lines.empty:
            s = n.lines["s_nom"].astype(float).copy()
            for name in s.index[s <= 0.0]:
                typ = n.lines.at[name, "type"]
                npar = float(n.lines.at[name, "num_parallel"])
                s.at[name] = float(n.line_types.at[typ, "s_nom"]) * npar
            s.index = pd.MultiIndex.from_product(
                [["Line"], s.index], names=["type", "name"]
            )
            parts.append(s)
        if not n.transformers.empty:
            s = n.transformers["s_nom"].astype(float)
            s.index = pd.MultiIndex.from_product(
                [["Transformer"], s.index], names=["type", "name"]
            )
            parts.append(s)
        if not parts:
            return pd.Series(dtype=float)
        return pd.concat(parts).reindex(self._branch_index()).astype(float)

    def _j_loading(self, flow_mw: pd.Series, s_nom: pd.Series) -> float:
        """
        J = Σ (p0/s_nom)² over lines and transformers, same sum as P-opt.
        p0 [MW], s_nom [MVA]. A non-positive rating is left out.
        """
        s = s_nom.reindex(flow_mw.index).astype(float)
        use = s > 0.0
        ratio = flow_mw.loc[use] / s.loc[use]
        return float((ratio**2).sum())

    def _worst_rel_overload(
        self, dP: float, constraints: list[tuple[float, float, float]]
    ) -> float:
        """Worst max(|b + a·ΔP| − lim, 0) / lim. ΔP, b, lim in MW."""
        if not constraints:
            return 0.0
        return max(
            max(abs(b + a * dP) - lim, 0.0) / max(lim, 1e-12)
            for a, b, lim in constraints
        )

    def _write_link_p(self, link: str, snapshot: object, p_mw: float) -> None:
        """Set link p_set [MW], bus0 → bus1. Update the timeseries only if it exists."""
        n = self.network
        n.links.at[link, "p_set"] = p_mw
        if not n.links_t.p_set.empty:
            n.links_t.p_set.loc[snapshot, link] = p_mw

    def _worst_contingency_loading(
        self,
        snapshot: object,
        pairs: set[tuple[object, object]] | None = None,
    ) -> float:
        """
        Worst |F| / limit over resolved outages, from the current DC p0.

        Limit is s_nom · s_max_pu · n1_guard_margin [MW]. 1.36 means 136% of
        that limit. The outaged branch and non-finite rows are skipped.
        `pairs` restricts the score to those (outage, branch) keys.
        """
        branches = self._branch_index()
        F = self._branch_p0(snapshot)
        limit = self._branch_s_max() * float(self.cfg.n1_guard_margin)
        bodf = self._get_bodf_single()
        worst = 0.0
        for outage in self._resolve_outages():
            if outage in bodf.columns:
                flowed = (F + bodf[outage] * float(F.get(outage, 0.0))).reindex(
                    branches
                )
            else:
                flowed = F
            for branch in branches:
                if branch == outage:
                    continue
                if pairs is not None and (outage, branch) not in pairs:
                    continue
                lim = float(limit.get(branch, 0.0))
                flow = float(flowed.get(branch, 0.0))
                if lim <= 0.0 or not math.isfinite(lim) or not math.isfinite(flow):
                    continue
                worst = max(worst, abs(flow) / lim)
        return worst

    def _worst_ac_loading(self, snapshot: object) -> float:
        """Worst max(|S0|, |S1|) / s_nom after an AC pf. S and s_nom are MVA."""
        n = self.network
        s_nom = self._branch_s_nom()
        worst = 0.0
        tables = []
        if not n.lines.empty:
            tables.append(("Line", n.lines_t))
        if not n.transformers.empty:
            tables.append(("Transformer", n.transformers_t))
        for kind, flows in tables:
            for name in flows.p0.columns:
                rating = float(s_nom.get((kind, name), 0.0))
                if rating <= 0.0 or not math.isfinite(rating):
                    continue
                s0 = math.hypot(
                    float(flows.p0.at[snapshot, name]),
                    float(flows.q0.at[snapshot, name]),
                )
                s1 = math.hypot(
                    float(flows.p1.at[snapshot, name]),
                    float(flows.q1.at[snapshot, name]),
                )
                worst = max(worst, s0 / rating, s1 / rating)
        return worst

    def _ac_pf(self) -> None:
        """AC pf with the controller slack. Positive link p_set is bus0 → bus1."""
        self.network.pf(
            distribute_slack=self.cfg.distributed_slack,
            slack_weights="p_set",
        )

    def _border_accepts(
        self,
        snapshot: object,
        p_before: pd.Series,
        base_inf: float,
        base_pro: float,
    ) -> bool:
        """
        True when the written trial passes the load-flow borders.

        J = Σ (p0/s_nom)² stays within 10% of the pre-guard point. The worst
        influenceable |F|/limit falls by at least 0.05. The worst protected
        |F|/limit does not rise. A sign change of p_set also needs the AC
        worst max(|S|/s_nom) strictly below the initial pf. |S| and s_nom are MVA.
        """
        self._lpf_refresh(snapshot)
        j = self._j_loading(self._branch_p0(snapshot), self._branch_s_nom())
        inf = self._worst_contingency_loading(snapshot, self._influenceable_pairs)
        pro = self._worst_contingency_loading(snapshot, self._protected_pairs)
        if not (
            j <= 1.10 * self._j_opt_frozen + 1e-9
            and base_inf - inf >= 0.05
            and pro <= base_pro + 1e-6
        ):
            return False
        p_now = self.network.links["p_set"].astype(float)
        reverses = any(
            float(p_before[k]) * float(p_now[k]) < 0.0 for k in p_before.index
        )
        if not reverses:
            return True
        self._ac_pf()
        return (
            self._worst_ac_loading(snapshot) < self._initial_worst_ac_loading[snapshot]
        )

    def _cap_link_dp(
        self, link: str, snapshot: object, p_now: float, dP: float
    ) -> float:
        """
        Largest fraction of ΔP [MW], same sign, that a DC lpf accepts.

        Off: return ΔP. Otherwise the borders in `_border_accepts`. The trial
        is re-solved, not taken from a·ΔP.
        """
        if not self.cfg.n1_guard_j_cap or abs(dP) <= 1e-9:
            return dP
        if not self._influenceable_pairs:
            return 0.0
        self._lpf_refresh(snapshot)
        base_inf = self._worst_contingency_loading(snapshot, self._influenceable_pairs)
        if base_inf < 0.05:
            return 0.0
        base_pro = self._worst_contingency_loading(snapshot, self._protected_pairs)
        p_before = self.network.links["p_set"].astype(float).copy()
        accepted = 0.0
        for i in range(20, 0, -1):
            trial = dP * (i / 20.0)
            self._write_link_p(link, snapshot, p_now + trial)
            if self._border_accepts(snapshot, p_before, base_inf, base_pro):
                accepted = trial
                break
        self._write_link_p(link, snapshot, p_now)
        self._lpf_refresh(snapshot)
        return accepted

    def _cap_link_vector(
        self,
        snapshot: object,
        p_now: dict[str, float],
        dP: dict[str, float],
    ) -> dict[str, float]:
        """
        Largest common fraction of a joint ΔP [MW] that one DC lpf accepts.

        Off: return ΔP. Empty influenceable set: every component is 0.
        """
        zero = dict.fromkeys(dP, 0.0)
        if not self.cfg.n1_guard_j_cap:
            return dP
        if not self._influenceable_pairs or all(
            abs(step) <= 1e-9 for step in dP.values()
        ):
            return zero
        self._lpf_refresh(snapshot)
        base_inf = self._worst_contingency_loading(snapshot, self._influenceable_pairs)
        if base_inf < 0.05:
            return zero
        base_pro = self._worst_contingency_loading(snapshot, self._protected_pairs)
        p_before = self.network.links["p_set"].astype(float).copy()
        accepted = zero
        for i in range(20, 0, -1):
            scale = i / 20.0
            for k, step in dP.items():
                self._write_link_p(k, snapshot, p_now[k] + scale * step)
            if self._border_accepts(snapshot, p_before, base_inf, base_pro):
                accepted = {k: scale * step for k, step in dP.items()}
                break
        for k, p_mw in p_now.items():
            self._write_link_p(k, snapshot, p_mw)
        self._lpf_refresh(snapshot)
        return accepted

    def enforce_n1_guard(self, snapshot: object) -> bool:
        if not self.cfg.n1_guard_enable:
            return True
        self._lpf_refresh(snapshot)
        self._j_opt_frozen = self._j_loading(
            self._branch_p0(snapshot), self._branch_s_nom()
        )
        for _ in range(int(self.cfg.n1_guard_max_passes)):
            changed = self._enforce_n1_guard_once(snapshot)
            if self._is_safe_bodf(snapshot):
                return True
            if not changed:
                return False
        return False

    def _is_safe_bodf(self, snapshot: object) -> bool:
        """True if no branch exceeds Pmax under any resolved outage (DC, MW)."""
        margin = float(self.cfg.n1_guard_margin)
        branches = self._branch_index()
        self._lpf_refresh(snapshot)
        F = self._branch_p0(snapshot)
        Pmax = self._branch_s_max() * margin
        outages = self._resolve_outages()
        BODF = self._get_bodf_single()

        for o in outages:
            if o in BODF.columns:
                F_o = (F + BODF[o] * float(F.get(o, 0.0))).reindex(
                    branches, fill_value=0.0
                )
            else:
                F_o = F
            if (F_o.abs() > Pmax + 1e-6).any():
                return False
        return True
