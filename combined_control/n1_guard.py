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
        ISF of bipolar link injection on every AC branch [MW/MW].

        Default: +ΔP at bus0, −ΔP at bus1 (η=1). Index = `_branch_index`.
        """
        n = self.network
        PTDF = self._get_ptdf_single()
        isf: dict[str, pd.Series] = {}
        for k in n.links.index:
            b0 = n.links.at[k, "bus0"]
            b1 = n.links.at[k, "bus1"]
            col0 = PTDF[b0] if b0 in PTDF.columns else pd.Series(0.0, index=PTDF.index)
            col1 = PTDF[b1] if b1 in PTDF.columns else pd.Series(0.0, index=PTDF.index)
            isf[k] = (col0 - col1).astype(float)
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

            # 1) Falls globaler Schnitt NICHT leer und 0 NICHT enthalten:
            if lo <= hi and not (lo <= 0.0 <= hi):
                lo_clip = max(lo, dP_lo_hw)
                hi_clip = min(hi, dP_hi_hw)
                if lo_clip <= hi_clip:
                    dP = lo_clip if abs(lo_clip) < abs(hi_clip) else hi_clip
                    # anwenden
                    p_new = p_now + dP
                    n.links.loc[k, "p_set"] = p_new
                    if n.links_t.p_set.empty:
                        ensure_link_pset_timeseries(n)
                    n.links_t.p_set.loc[snapshot, k] = p_new
                    changed = True
                    continue  # zum nächsten Link

            # 2) Fallback: kein vollständig zulässiger Schnitt.
            #    Wähle innerhalb der Hardwaregrenzen das Kandidaten-ΔP, das die
            #    größte verbleibende relative Überlast möglichst klein macht.
            cands = link_candidates.get(k, [])
            constraints = link_constraints.get(k, [])
            if cands and constraints:
                candidates = [0.0]
                candidates += [min(max(dP_raw, dP_lo_hw), dP_hi_hw) for dP_raw in cands]

                def worst_overload(dP: float) -> float:
                    return max(
                        max(abs(b + a * dP) - lim, 0.0) / max(lim, 1e-12)
                        for a, b, lim in constraints
                    )

                dP = min(candidates, key=lambda x: (worst_overload(x), abs(x)))

                if abs(dP) > 0.0:
                    p_new = p_now + dP
                    n.links.loc[k, "p_set"] = p_new
                    if n.links_t.p_set.empty:
                        ensure_link_pset_timeseries(n)
                    n.links_t.p_set.loc[snapshot, k] = p_new
                    changed = True

        if changed:
            self._lpf_refresh(snapshot)
        return changed

    def enforce_n1_guard(self, snapshot: object) -> bool:
        if not self.cfg.n1_guard_enable:
            return True
        for _ in range(int(self.cfg.n1_guard_max_passes)):
            self._enforce_n1_guard_once(snapshot)
            if self._is_safe_bodf(snapshot):
                return True
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
