from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from combined_control.n1_guard import N1Guard, ensure_link_pset_timeseries
from combined_control.P_Optimizer_V2 import (
    link_optimization,
    show_snapshot_report,
    show_snapshot_report_after_guard,
)
from combined_control.Q_Optimizer import q_optimization, show_snapshot_q_report
from combined_control.stability_indices import calc_vsi


@dataclass
class ControllerConfig:
    """
    Central configuartaion for the VSC Controller.
    Here, defaults are defined ("single source of truth")")
        -angle_limit_deg: Limit angle (degrees) for P optimization
        -v_target: Target voltages (p.u.) for Q optimization
        -epsilon: Step size (MVAr) for sensitivity calculation in Q optimization
        -run_vsi: Activate FVSI (fast voltage stability index) calculation before/after Q optimization
        -max_line_loading: (therm.) line loading limit (upper boundary) in P optimization
        -S_rated: Rated apparent power of the VSC(s) in MVA
        -Q_reserve: Amount of reactive power to be retained for Q optimization AFTER P optimization. Constraints the P optimization
        -q_reserve_ratio: Default distribution of P and Q capacity. Following S^2=P^2+Q^2
        -enforce_target_tag: Necessary to deal with analysis protocolls with several network objects (e.g. copies)
        -target_tag: Necessary to deal with analysis protocolls with several network objects (e.g. copies)
        -n1_guard_enable:
        -n1_guard_tol:
        -n1_guard_max_iter:
        -n1_guard_outages:
            

    """

    angle_limit_deg: float = 25.0
    v_target: float = 1.0
    epsilon: float = 20.0
    run_vsi: bool = True
    max_line_loading: float = 0.95  # 1.0 = 100%
    S_rated: float = 400.0  # [MVA]
    Q_reserve: float | None = None  # [MVAr] absolute reserve
    q_reserve_ratio: float = 1 / np.sqrt(2)
    enforce_target_tag: bool = False
    target_tag: str | None = None
    
    n1_guard_enable: bool = False 
    n1_guard_margin: float=0.98
    n1_guard_tol: float=0.0
    n1_guard_max_passes: int = 3
    n1_guard_outages: list[str] | None=None


class VSCController:
    """
    Main Class for control logic and procedure. Contains several methods:
        -__init__:
        -pf_callback:
        -lpf_callback:
        -calculate_vsi:
        -run_p_control:
        -run_q_control:
        -run_combined_control:
        -show_report:
        -_update_vsc_limit:
        -run_mode:

    """

    def __init__(self, network, config: ControllerConfig | None = None):
        self.network = network
        self.cfg = config or ControllerConfig()
        self._S_rated = float(self.cfg.S_rated)
        self.p_result = None
        self.q_result = None
        self.pf_counter = 0
        self.lpf_counter = 0
        self._ptdf_cache: dict[int, pd.DataFrame] = {}
        self._bodf_cache: dict[int, pd.DataFrame] = {}
        self.n1_guard = N1Guard(self.network, self.cfg)

        
        # Use the tags to ensure working with the correct network object
        if self.cfg.enforce_target_tag and self.cfg.target_tag is not None:
            assert getattr(self.network, "_whoami", None) == self.cfg.target_tag, \
                f"Wrong target network: {getattr(self.network,'_whoami','?')}, id={id(self.network)}"

        # Automated mapping of VSCs to their corresponding Link. Used later in _update_vsc_limits(). First part can be deleted.
        if not {"link", "side"}.issubset(self.network.controllable_vscs.columns):
            raise ValueError(
                "controllable_vscs requires columns ‘link’ and ‘side’ (bus0/bus1)"
            )
        # Creates a dict self.vsc_to_link, linking each VSC name on a tupel (link_name,side)
        self.vsc_to_link = {
            vsc: (row["link"], row["side"])
            for vsc, row in self.network.controllable_vscs.iterrows()
        }

    # Callback method for power flow counter
    def pf_callback(self):
        self.network.pf()
        self.pf_counter += 1

    def lpf_callback(self):
        self.network.lpf()
        self.lpf_counter += 1

    # Method call to run calc_vsi function from imported, separate script
    def calculate_vsi(self, snapshot):
        return calc_vsi(self.network, snapshot)

    # Method call to run P-optimizer
    def run_p_control(
        self,
        angle_limit_deg: float | None = None,
        report_snapshots=None,
        show_report: bool = True,
        pf_first: bool = False,
        max_line_loading: float | None = None,
        run_vsi: bool | None = None,
    ):
        """
        -runs the P-Optimization script optionalli with vsi calulation
        -pf_first false (default)
        -pf_first ist automatically set True, if P_control mode is activated
        -angle_limit_degree: optional override of the config
        -max_line_loading: optional override of the config

        """
        if pf_first:
            print("\n === Initial pf() ===")
            self.pf_callback()
            
            

        if len(self.network.snapshots) == 0:
            self.network.set_snapshots(
                pd.Index([pd.Timestamp("2000-01-01")])
            )  # Dummy timestamp
            
        self._ensure_link_pset_timeseries()


        if (
            report_snapshots is None
        ):  # standard if nothing is handed over- use all snaps
            if len(self.network.snapshots) == 1:
                report_snapshots = list(self.network.snapshots)
            else:
                report_snapshots = "all"
                
        # if None, then config values (default) are used. Otherwise the value set within the method call.
        run_vsi = self.cfg.run_vsi if run_vsi is None else bool(run_vsi)

        vsi_default = {}

        if run_vsi:
            for snap in self.network.snapshots:
                vsi_default[snap] = self.calculate_vsi(snap)


        print("\n ======= Link Optimization =======")

        # If None, then config values (default) are used. Otherwise the values set within the method call.
        effective_angle = (
            self.cfg.angle_limit_deg if angle_limit_deg is None else angle_limit_deg
        )
        effective_mll = (
            self.cfg.max_line_loading
            if max_line_loading is None
            else float(max_line_loading)
        )

        self.p_result = link_optimization(
            self.network,
            angle_limit_deg=effective_angle,
            pf_callback=self.pf_callback,
            lpf_callback=self.lpf_callback,
            max_line_loading=effective_mll, # <-- directly to pyomo-model
            guard_active=self.cfg.n1_guard_enable
        )
        
        # Debug
        PTDF = self.n1_guard._get_ptdf_single()
        BODF = self.n1_guard._get_bodf_single()
        print("PTDF sample:\n", PTDF.loc[PTDF.index[:5], PTDF.columns[:5]])
        print("BODF sample:\n", BODF.iloc[:5, :5])
        

        # Use N-1 Guard
        if self.cfg.n1_guard_enable:
            for snap in self.network.snapshots:
                self.enforce_n1_guard(snap)
            self.pf_callback()
            if show_report:
                show_snapshot_report_after_guard(self.p_result, self.network, report_snapshots)
                    
        else:
            show_snapshot_report(self.p_result, 
                                 self.network, 
                                 snapshots=report_snapshots,
                                 vsi_default=vsi_default if run_vsi else None,
                                 )

        return self.p_result

    # Method call to run Q-optimizer
    def run_q_control(
        self,
        angle_limit_deg: float | None = None,
        report_snapshots=None,
        show_report: bool = True,
        run_vsi: bool | None = None,
        pf_first: bool = False,
    ):
        """
        -runs the Q-Optimization script, optionally with FVSI evaluation
        -epsilon: optional override of the config
        -v_target: optional override of the config
        -(run_vsi: optional override of the config)

        """
        
        vsi_default = None
        
        if pf_first:
            print("\n === Initial pf() ===")
            self.pf_callback()
            
            # if None, then config values (default) are used. Otherwise the value set within the method call.
            #run_vsi = self.cfg.run_vsi if run_vsi is None else bool(run_vsi)

            if run_vsi:
                vsi_default = {}
                for snap in self.network.snapshots:
                    vsi_default[snap] = self.calculate_vsi(snap)
               

        if len(self.network.snapshots) == 0:
            self.network.set_snapshots(pd.Index([pd.Timestamp("2000-01-01")]))
            
        self._ensure_vsc_qset_timeseries()

        if report_snapshots is None:
            if len(self.network.snapshots) == 1:
                report_snapshots = list(self.network.snapshots)
            else:
                report_snapshots = "all"

         # if None, then config values (default) are used. Otherwise the values set within the method call.
        effective_angle = (
             self.cfg.angle_limit_deg if angle_limit_deg is None else angle_limit_deg
         )
         
        # if None, then config values (default) are used. Otherwise the value set within the method call.
        run_vsi = self.cfg.run_vsi if run_vsi is None else bool(run_vsi)

        vsi_after_P = {}
        vsi_opt = {}

        if run_vsi:
            for snap in self.network.snapshots:
                vsi_after_P[snap] = self.calculate_vsi(snap)

        print("\n ======= Converter Reactive Power Output Optimization =======")

        self.q_result = q_optimization(
            self.network,
            angle_limit_deg=effective_angle,
            epsilon=self.cfg.epsilon,
            v_target=self.cfg.v_target,
            pf_callback=self.pf_callback,
            lpf_callback=self.lpf_callback,
            q_limit_callback=self._apply_q_limits_for_snapshot,
        )

        if run_vsi:
            for snap in self.network.snapshots:
                vsi_opt[snap] = self.calculate_vsi(snap)

        if show_report:
            show_snapshot_q_report(
                self.q_result,
                self.network,
                snapshots=report_snapshots,
                vsi_default=vsi_default if run_vsi else None,
                vsi_after_P=vsi_after_P if run_vsi else None,
                vsi_opt=vsi_opt if run_vsi else None,
                )

        return (
            self.q_result,
            (vsi_after_P if run_vsi else None),
            (vsi_opt if run_vsi else None),
        )

    # DEPRECATED. Method call for combined control. Instead use run_mode(mode="combined")
    def run_combined_control(self, pf_first=True, angle_limit_deg=25):
        if pf_first:
            print("\n === Initial pf() ===")
            self.pf_callback()

        self.p_result = self.run_p_control(angle_limit_deg=angle_limit_deg)
        self.q_result, vsi_after_P, vsi_opt = self.run_q_control()

    # Method call to control and dictate the converter behaviour
    def run_mode(
        self,
        mode: str = "combined",
        angle_limit_deg: float | None = None,
        S_rated: float | None = None,
        Q_reserve: float | None = None,
        max_line_loading: float | None = None,
    ):
        """
        run_mode allows selection of:
          -'combined'  : P then Q
          -'P_control' : only P
          -'Q_control' : only Q
          Optional: angle_limit_deg, S_rated, Q_reserve, max_line_loading as Overrides.

        """
        print(f"\n === MODE: {mode.upper()} ===")

        # if None, then config values (default) are used. Otherwise the values set within the method call.
        effective_angle = (
            self.cfg.angle_limit_deg if angle_limit_deg is None else angle_limit_deg
        )
        effective_mll = (
            self.cfg.max_line_loading
            if max_line_loading is None
            else float(max_line_loading)
        )

        if mode == "combined":
            self._update_vsc_limits(
                S_rated=S_rated, Q_reserve=Q_reserve, set_q_limits=False
            )
            self.run_p_control(
                angle_limit_deg=effective_angle,
                pf_first=True,
                max_line_loading=effective_mll,
            )
            self._update_vsc_limits(
                S_rated=S_rated, Q_reserve=Q_reserve, set_q_limits=True
            )
            print(self.network.controllable_vscs[["q_min","q_max","q_set"]])

            self.run_q_control()
            result = (self.p_result, self.q_result)

        elif mode == "P_control":
            self._update_vsc_limits(
                S_rated=self.cfg.S_rated if S_rated is None else S_rated,
                Q_reserve=0,
                set_q_limits=False,
            )
            self.run_p_control(
                pf_first=True,
                angle_limit_deg=effective_angle,
                max_line_loading=effective_mll,
            )
            result = self.p_result

        elif mode == "Q_control":
            S_eff = self.cfg.S_rated if S_rated is None else S_rated
            self._update_vsc_limits(S_rated=S_eff, Q_reserve=S_eff, set_q_limits=True)
            self.q_result, vsi_after_P, vsi_opt = self.run_q_control(pf_first=True)
            result = (self.q_result, vsi_after_P, vsi_opt)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"--- MODE {mode.upper()} DONE ---")
        print(f"Total pf() calls: {self.pf_counter}")
        print(f"Total lpf() calls: {self.lpf_counter}")

        return result

    # Method call to update the vsc limits during the optimization workflow
    def _update_vsc_limits(
        self,
        S_rated: float | None = None,
        Q_reserve: float | None = None,
        set_q_limits: bool = True,
    ):
        """
        Calculates P-/Q-limits depending on the selected mode.
            -to be deleted, when implemented in new pypsa version?

        """
        Sr = self.cfg.S_rated if S_rated is None else float(S_rated)
        self._S_rated = Sr

        if Q_reserve is not None:
            Qr = float(Q_reserve)
        else:
            Qr = (
                self.cfg.Q_reserve
                if self.cfg.Q_reserve is not None
                else self.cfg.q_reserve_ratio * Sr
            )
            

        for link_name in self.network.links.index:
            Pmax_from_Q = np.sqrt(max(Sr**2 - Qr**2, 0.0))
            Pnom_link = self.network.links.at[link_name, "p_nom"]
            Pmax_final = min(Pmax_from_Q, Pnom_link)
            # =1 or <1. If =1, no headroom left possibly (if Pmax_final=Pnom_link and =P_optimal)
            self.network.links.at[link_name, "p_min_pu"] = -Pmax_final / Pnom_link
            self.network.links.at[link_name, "p_max_pu"] = Pmax_final / Pnom_link

        if not set_q_limits:
            return

        self._apply_q_limits_from_p(self.network.links["p_set"].astype(float))

    def _apply_q_limits_from_p(self, p_mw: pd.Series) -> None:
        """Set q_min/q_max [MVAr] from S_rated^2 = P^2 + Q^2. p_mw is link P [MW]."""
        Sr = float(self._S_rated)
        for vsc, (link_name, _side) in self.vsc_to_link.items():
            if link_name not in self.network.links.index:
                continue
            if link_name in p_mw.index:
                P_now = float(p_mw[link_name])
            else:
                P_now = float(self.network.links.at[link_name, "p_set"])
            Qmax_dyn = np.sqrt(max(Sr**2 - abs(P_now) ** 2, 0.0))
            self.network.controllable_vscs.at[vsc, "q_min"] = -Qmax_dyn
            self.network.controllable_vscs.at[vsc, "q_max"] = Qmax_dyn

    def _apply_q_limits_for_snapshot(self, snapshot: object) -> None:
        """Q box from this snapshot's post-P (post-guard) p_set [MW]."""
        n = self.network
        p_ts = n.links_t.p_set
        if not p_ts.empty and snapshot in p_ts.index:
            p_mw = p_ts.loc[snapshot].astype(float)
        else:
            p_mw = n.links["p_set"].astype(float)
        self._apply_q_limits_from_p(p_mw)

    # Method to ensure VSC q_set timeseries
    def _ensure_vsc_qset_timeseries(self):
        """
        The Q-Optimizer uses timeseries values (*_t.*). If this is empty e.g. no SCLOPF beforehand,
        static values q_set needs to be "copied" to the timeseries values.
        Ensures that controllable_vscs_t.q_set exists and is aligned to (snapshots x VSCs). 
        If only static q_set values exist (or none at all), it is initialized cleanly.
        """
        n= self.network
        if n.controllable_vscs.empty:
            return
        
        # If no snapshot exits, set a dummy snap
        if len(n.snapshots) == 0:
            n.set_snapshots(pd.Index([pd.Timestamp("2000-01-01")]))
        
        cols=n.controllable_vscs.index
        
        # If timeseries available--> use it
        if not n.controllable_vscs_t.q_set.empty:
            n.controllable_vscs_t.q_set = (
                n.controllable_vscs_t.q_set
                .reindex(index=n.snapshots, columns=cols, fill_value=0.0)
                .astype(float)
            )
            return
    
        # If timesries missing--> build from static values (or 0.0)
        if "q_set" in n.controllable_vscs.columns:
            base = n.controllable_vscs["q_set"].reindex(cols).fillna(0.0)
        else:
            base = pd.Series(0.0, index=cols)
                
        df = pd.DataFrame(0.0, index=n.snapshots, columns=cols, dtype=float)
        for snap in n.snapshots:
            df.loc[snap, :] = base.values  # gleiche Startwerte für alle Snapshots
    
        n.controllable_vscs_t["q_set"] = df
              
        
    def _ensure_link_pset_timeseries(self) -> None:
        """Copy static links.p_set into links_t.p_set when the time series is empty."""
        ensure_link_pset_timeseries(self.network)

    def _get_subnetwork_for_line(self, line: str) -> Any:
        # Alias für alte Aufrufe – wir arbeiten bewusst mit einem EINZIGEN AC-Subnetz
        return self.n1_guard._get_sn_single()

    def enforce_n1_guard(self, snapshot: object) -> bool:
        return self.n1_guard.enforce_n1_guard(snapshot)

    def get_sensitivity_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Gib PTDF (Lines×Buses) und BODF (Lines×Lines) als DataFrames zurück
        und lege sie auch als self.debug_PTDF/self.debug_BODF ab."""
        PTDF = self.n1_guard._get_ptdf_single().copy()
        BODF = self.n1_guard._get_bodf_single().copy()
        self.debug_PTDF = PTDF
        self.debug_BODF = BODF
        return PTDF, BODF

    def dump_sensitivity_tables(self, ptdf_path="ptdf.csv", bodf_path="bodf.csv"):
        PTDF, BODF = self.get_sensitivity_tables()
        PTDF.to_csv(ptdf_path, float_format="%.6f")
        BODF.to_csv(bodf_path, float_format="%.6f")
        print(f"PTDF -> {ptdf_path}, BODF -> {bodf_path}")
        
        
    def report_if_guard_active(self):
        # Maximalwerte für S-basierte Auslastung
        s_line_max  = self.network.lines.s_nom * self.network.lines.s_max_pu.fillna(1.0)
        has_trafos  = not getattr(self.network, "transformers", pd.DataFrame()).empty
        if has_trafos:
            s_trafo_max = (self.network.transformers.s_nom *
                           self.network.transformers.s_max_pu.fillna(1.0))
    
        for snap in self.network.snapshots:
            res = self.p_result.setdefault(snap, {})
            res["guard_active"] = True
    
            # Linien – finale AC-Ströme (S) & Auslastungen
            P = self.network.lines_t.p0.loc[snap]
            Q = self.network.lines_t.q0.loc[snap]
            S = np.hypot(P, Q)
            loading_lines_S = 100.0 * S / s_line_max
    
            # Trafos – finale AC-Ströme (S) & Auslastungen (falls vorhanden)
            if has_trafos and not self.network.transformers_t.p0.empty:
                PT = self.network.transformers_t.p0.loc[snap]
                QT = (self.network.transformers_t.q0.loc[snap]
                      if not self.network.transformers_t.q0.empty
                      else PT*0.0)
                ST = np.hypot(PT, QT)
                loading_trafos_S = 100.0 * ST / s_trafo_max
            else:
                loading_trafos_S = None
    
            # Finale Link-Leistung (DC-Modell: p0)
            links_p0_final = (self.network.links_t.p0.loc[snap].copy()
                              if not self.network.links_t.p0.empty
                              else None)
    
            # Buswinkel (nur Info)
            theta_deg = np.degrees(self.network.buses_t.v_ang.loc[snap].copy())
    
            # Alles in den Result-Container packen
            res["after_guard"] = {
                "loading_S":        loading_lines_S,     # Serien (in %)
                "trafo_loading_S":  loading_trafos_S,    # Serien (in %) oder None
                "links_p0":         links_p0_final,      # Serie
                "angles_deg":       theta_deg,           # Serie
            }
