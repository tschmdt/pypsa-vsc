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

    def _get_ptdf_single(self, slack_bus: str = "Bus 9") -> pd.DataFrame:
        """
        PTDF als DataFrame (Lines x Buses), nur Leitungen, Index=Leitungsnamen.
        """
        if hasattr(self, "_PTDF_single"):
            return self._PTDF_single

        sn = self._get_sn_single()

        # In deiner Version: Ergebnis steckt in sn.PTDF; Rückgabewert ist None.
        try:
            sn.calculate_PTDF(slack_bus=slack_bus)
        except TypeError:
            # ganz alte Signatur: kein slack_bus-Argument
            sn.calculate_PTDF()

        arr = getattr(sn, "PTDF", None)
        if arr is None:
            raise RuntimeError("sn.calculate_PTDF() hat kein sn.PTDF befüllt.")

        ptdf = pd.DataFrame(arr, index=sn.branches_i(), columns=sn.buses_i())

        # Nur Lines behalten und MultiIndex -> reine Namen
        if isinstance(ptdf.index, pd.MultiIndex):
            mask_line = ptdf.index.get_level_values(0) == "Line"
            ptdf = ptdf.loc[mask_line, :].copy()
            ptdf.index = ptdf.index.get_level_values(1)

        # Auf aktuelle Leitungsreihenfolge bringen
        ptdf = ptdf.reindex(self.network.lines.index).astype(float)

        self._PTDF_single = ptdf
        return ptdf

    def _get_bodf_single(self) -> pd.DataFrame:
        """
        BODF als DataFrame (Lines x Lines), nur Leitungen, Index/Spalten=Leitungsnamen.
        """
        if hasattr(self, "_BODF_single"):
            return self._BODF_single

        sn = self._get_sn_single()

        # Je nach Version steht Ergebnis in sn.BODF oder kommt als Rückgabewert
        bodf_calc = sn.calculate_BODF()
        if bodf_calc is None and hasattr(sn, "BODF"):
            bodf_calc = sn.BODF

        if bodf_calc is None:
            raise RuntimeError("sn.calculate_BODF() hat kein Ergebnis geliefert.")

        bodf = pd.DataFrame(bodf_calc, index=sn.branches_i(), columns=sn.branches_i())

        # Nur Lines x Lines und MultiIndex -> reine Namen
        if isinstance(bodf.index, pd.MultiIndex):
            idx_line = bodf.index.get_level_values(0) == "Line"
            col_line = bodf.columns.get_level_values(0) == "Line"
            bodf = bodf.loc[idx_line, col_line].copy()
            bodf.index = bodf.index.get_level_values(1)
            bodf.columns = bodf.columns.get_level_values(1)

        # Auf aktuelle Leitungsreihenfolge/-menge bringen
        li = self.network.lines.index
        bodf = bodf.reindex(index=li, columns=li).fillna(0.0).astype(float)

        self._BODF_single = bodf
        return bodf

    def _isf_for_links(self) -> dict[str, pd.Series]:
        """
        Provides an ISF vector across all AC branches for each link k.
        Default: balanced injection (+Δp at bus0, −Δp at bus1).
        Optional: consider efficiency (+Δp at bus0, −ηΔp at bus1, slack compensates for mismatch).

        """
        n = self.network
        PTDF = self._get_ptdf_single()  # Lines x Buses
        isf: dict[str, pd.Series] = {}
        for k in n.links.index:
            b0 = n.links.at[k, "bus0"]
            b1 = n.links.at[k, "bus1"]
            col0 = PTDF[b0] if b0 in PTDF.columns else pd.Series(0.0, index=PTDF.index)
            col1 = PTDF[b1] if b1 in PTDF.columns else pd.Series(0.0, index=PTDF.index)
            s = (col0 - col1).astype(
                float
            )  # Index = reine Liniennamen (matcht n.lines.index)
            isf[k] = s
        return isf

    def _lines_s_max(self) -> pd.Series:
        n = self.network
        return (n.lines["s_nom"] * n.lines["s_max_pu"].fillna(1.0)).astype(float)

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

        # Basis-DC-Flüsse
        self._lpf_refresh(snapshot)
        F_base = n.lines_t.p0.loc[snapshot].astype(float)
        Pmax = self._lines_s_max() * margin

        outages = (
            list(self.cfg.n1_guard_outages)
            if self.cfg.n1_guard_outages
            else list(n.lines.index)
        )

        BODF = self._get_bodf_single()  # Lines x Lines
        isf_map = self._isf_for_links()  # dict[link] -> Series(Lines)

        any_violation = False
        link_intervals: dict[str, tuple[float, float]] = {
            k: (-float("inf"), float("inf")) for k in n.links.index
        }
        # NEU: sammle ΔP-Kandidaten aus einzelnen verletzten Nebenbedingungen
        link_candidates: dict[str, list[float]] = {k: [] for k in n.links.index}

        for o in outages:
            # LODF-Spalte (BODF) für den Ausfall o
            if o in BODF.columns:
                Lcol = BODF[o].reindex(n.lines.index).fillna(0.0)
                # b-Vektor: Post-Contingency-Grundfluss ohne Link-Änderung
                F_o = (F_base + Lcol * F_base.get(o, 0.0)).astype(float)
            else:
                Lcol = pd.Series(0.0, index=n.lines.index)
                F_o = F_base

            # Nur weiter, wenn dieser Ausfall überhaupt verletzt
            if not (F_o.abs() > (Pmax + 1e-9)).any():
                continue

            any_violation = True

            for k, ISF in isf_map.items():
                # a-Vektor: Einfluss des Links im Ausfall o  => ISF(ℓ) + L_{ℓ,o}*ISF(o)
                a_vec = (ISF + Lcol * float(ISF.get(o, 0.0))).astype(float)

                # winzige Koeffizienten ignorieren (stabiler)
                sel = a_vec.abs() > eps_isf
                lo_k, hi_k = link_intervals[k]

                for ell, a in a_vec[sel].items():
                    b = float(F_o.get(ell, 0.0))
                    lim = float(Pmax.get(ell, 0.0))
                    iv = self._interval_from_abs_linear(a, b, lim)
                    if iv is None:
                        lo_k, hi_k = 1.0, 0.0  # leeres Intervall
                        break
                    lo_k = max(lo_k, iv[0])
                    hi_k = min(hi_k, iv[1])
                    if lo_k > hi_k:
                        break

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

            # 2) Fallback: globaler Schnitt leer (oder 0 drin -> keine Notwendigkeit).
            #    Nimm best-effort Kandidaten (nächstliegende Projektion) aus EINZEL-Bedingungen.
            cands = link_candidates.get(k, [])
            if len(cands) > 0:
                # wähle minimalen |ΔP| und schneide HW
                dP_raw = min(cands, key=lambda x: abs(x))
                dP = min(max(dP_raw, dP_lo_hw), dP_hi_hw)
                if abs(dP) > 0.0:  # Bewegung vorhanden
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
        n = self.network
        margin = float(self.cfg.n1_guard_margin)
        self._lpf_refresh(snapshot)
        F = n.lines_t.p0.loc[snapshot].astype(float)
        Pmax = self._lines_s_max() * margin

        outages = (
            list(self.cfg.n1_guard_outages)
            if self.cfg.n1_guard_outages
            else list(n.lines.index)
        )
        BODF = self._get_bodf_single()  # Lines x Lines

        for o in outages:
            if o in BODF.columns:
                F_o = (F + BODF[o] * F.get(o, 0.0)).reindex(
                    n.lines.index, fill_value=0.0
                )
            else:
                F_o = F
            if (F_o.abs() > Pmax + 1e-6).any():
                return False
        return True
