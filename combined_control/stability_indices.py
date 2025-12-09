import numpy as np
import pandas as pd


# claculates the FVSI index acc. to Musirin & Rahman, 2002
# alternatively MVSI can be calculated (Mokred, Wang, Chen, 2023)-> results in negative values-> tbc


# FVSI nach Musirin & Rahman (2002), jetzt konsistent in p.u.
def calc_vsi(netz, snapshot, eps_mw=1e-6):
    vsi_values = {}

    for line in netz.lines_t.q1.columns:
        # Richtung wie bei dir: über Vorzeichen von P0
        # P0 = float(netz.lines_t.p0.loc[snapshot, line])  # MW
        v0 = float(netz.buses_t.v_mag_pu.loc[snapshot, netz.lines.at[line, "bus0"]])
        v1 = float(netz.buses_t.v_mag_pu.loc[snapshot, netz.lines.at[line, "bus1"]])

        if v0 >= v1:
            sending_bus = netz.lines.at[line, "bus0"]
            receiving_bus = netz.lines.at[line, "bus1"]
            Q_R_MVAr = float(netz.lines_t.q1.loc[snapshot, line])
        else:
            sending_bus = netz.lines.at[line, "bus1"]
            receiving_bus = netz.lines.at[line, "bus0"]
            Q_R_MVAr = float(netz.lines_t.q0.loc[snapshot, line])

        # Vs in p.u. (KEIN kV mehr!)
        Vs_pu = float(netz.buses_t.v_mag_pu.loc[snapshot, sending_bus])

        # R, X in p.u. auf Basis des SENDESBUSSES; S_base = 1 MVA
        R_ohm = float(netz.lines.at[line, "r"])
        X_ohm = float(netz.lines.at[line, "x"])
        U_nom_kV = float(netz.buses.at[sending_bus, "v_nom"])  # kV

        # Z_base = U_base^2 / S_base  (U in V, S in VA). Hier S_base = 1 MVA.
        Z_base_ohm = (U_nom_kV * 1e3) ** 2 / 1e6
        # Schutz gegen exotische Fälle
        if Z_base_ohm == 0 or X_ohm == 0 or Vs_pu == 0:
            vsi_values[line] = np.nan
            continue

        R_pu = R_ohm / Z_base_ohm
        X_pu = X_ohm / Z_base_ohm
        Z_pu = np.hypot(R_pu, X_pu)

        # Q_R in p.u. (bei S_base=1 MVA entspricht 1 MVAr = 1 p.u.)
        Q_R_pu = max(-Q_R_MVAr, 0.0)
        # FVSI = (4 * Z^2 * Q_R) / (Vs^2 * X), alles in p.u.
        fvsi_num = 4.0 * (Z_pu**2) * Q_R_pu
        fvsi_den = (Vs_pu**2) * X_pu

        vsi_values[line] = fvsi_num / fvsi_den if abs(fvsi_den) > eps_mw else np.nan

    return pd.Series(vsi_values, name="FVSI")


def _line_end_selection(netz, snapshot, line, by_voltage=True):
    b0 = netz.lines.at[line, "bus0"]
    b1 = netz.lines.at[line, "bus1"]
    if by_voltage:
        v0 = float(netz.buses_t.v_mag_pu.loc[snapshot, b0])
        v1 = float(netz.buses_t.v_mag_pu.loc[snapshot, b1])
        sending_bus = b0 if v0 >= v1 else b1
        receiving_bus = b1 if v0 >= v1 else b0
        q_end = (
            float(netz.lines_t.q1.loc[snapshot, line])
            if sending_bus == b0
            else float(netz.lines_t.q0.loc[snapshot, line])
        )
    else:
        P0 = float(netz.lines_t.p0.loc[snapshot, line])
        sending_bus = b0 if P0 >= 0 else b1
        receiving_bus = b1 if P0 >= 0 else b0
        q_end = (
            float(netz.lines_t.q1.loc[snapshot, line])
            if P0 >= 0
            else float(netz.lines_t.q0.loc[snapshot, line])
        )
    return sending_bus, receiving_bus, q_end


def fvsi_components(netz, snapshot, by_voltage=True, eps=1e-12):
    rows = []
    for line in netz.lines_t.q1.columns:
        sending_bus, receiving_bus, q_end = _line_end_selection(
            netz, snapshot, line, by_voltage
        )
        Vs = float(netz.buses_t.v_mag_pu.loc[snapshot, sending_bus])

        # R, X sind bei dir Gesamtwerte (Ohm)
        R_ohm = float(netz.lines.at[line, "r"])
        X_ohm = float(netz.lines.at[line, "x"])
        U_nom_kV = float(netz.buses.at[sending_bus, "v_nom"])
        Z_base = (U_nom_kV * 1e3) ** 2 / 1e6  # S_base = 1 MVA

        if Z_base <= 0 or X_ohm == 0 or Vs <= 0:
            FVSI = np.nan
            Zpu = np.nan
            Xpu = np.nan
            QR = np.nan
        else:
            Rpu = R_ohm / Z_base
            Xpu = X_ohm / Z_base
            Zpu = np.hypot(Rpu, Xpu)
            QR = max(-q_end, 0.0)  # nur *aufgenommene* MVAr am Empfangsende
            k = 4.0 * (Zpu**2) / max(Xpu, eps)
            FVSI = k * QR / (Vs**2)

        rows.append(
            dict(
                line=line,
                sending_bus=sending_bus,
                receiving_bus=receiving_bus,
                Vs=Vs,
                q_end=q_end,
                Q_R=QR,
                R_pu=Rpu if Z_base > 0 else np.nan,
                X_pu=Xpu if Z_base > 0 else np.nan,
                Z_pu=Zpu,
                FVSI=FVSI,
            )
        )
    df = pd.DataFrame(rows).set_index("line")
    return df


def compare_fvsi(netz_before, netz_after, snapshot, by_voltage=True, eps=1e-12):
    a = fvsi_components(netz_before, snapshot, by_voltage=by_voltage)
    b = fvsi_components(netz_after, snapshot, by_voltage=by_voltage)
    df = a.join(b, lsuffix="_bef", rsuffix="_aft")

    # gleiche k = 4*Z^2/X je Leitung (Z & X ändern sich nicht)
    k = 4.0 * (df["Z_pu_bef"] ** 2) / df["X_pu_bef"].replace(0, np.nan)

    # Counterfactuals:
    # nur Vs ändert sich (Q_R bleibt wie vorher)
    fvsi_vs_only = k * df["Q_R_bef"] / (df["Vs_aft"] ** 2)
    # nur Q_R ändert sich (Vs bleibt wie vorher)
    fvsi_q_only = k * df["Q_R_aft"] / (df["Vs_bef"] ** 2)

    out = pd.DataFrame(
        {
            "FVSI_bef": df["FVSI_bef"],
            "FVSI_aft": df["FVSI_aft"],
            "ΔFVSI": df["FVSI_aft"] - df["FVSI_bef"],
            "Vs_bef": df["Vs_bef"],
            "Vs_aft": df["Vs_aft"],
            "ΔVs_%": 100.0
            * (df["Vs_aft"] - df["Vs_bef"])
            / df["Vs_bef"].replace(0, np.nan),
            "Q_R_bef": df["Q_R_bef"],
            "Q_R_aft": df["Q_R_aft"],
            "ΔQ_R": df["Q_R_aft"] - df["Q_R_bef"],
            "X_pu": df["X_pu_bef"],
            "Z_pu": df["Z_pu_bef"],
            "FVSI_if_only_Vs": fvsi_vs_only,
            "FVSI_if_only_Q": fvsi_q_only,
        }
    )
    # Anteilsmäßige Attribution (log-diff), vorsichtig bei Nullen:
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog_vs = -2.0 * np.log(df["Vs_aft"] / df["Vs_bef"])
        dlog_q = np.log((df["Q_R_aft"] + eps) / (df["Q_R_bef"] + eps))
        out["approx_ΔFVSI_from_Vs"] = df["FVSI_bef"] * dlog_vs
        out["approx_ΔFVSI_from_Q"] = df["FVSI_bef"] * dlog_q
    return out.sort_values("FVSI_aft", ascending=False)


# Bonus: schnelle Einzelauswertung einer Leitung (z.B. "Line 1-2")
def explain_one_line(netz_bef, netz_aft, snapshot, line, by_voltage=True):
    A = fvsi_components(netz_bef, snapshot, by_voltage=by_voltage).loc[line]
    B = fvsi_components(netz_aft, snapshot, by_voltage=by_voltage).loc[line]
    k = 4.0 * (A["Z_pu"] ** 2) / A["X_pu"]
    fvsi_vs_only = k * A["Q_R"] / (B["Vs"] ** 2)
    fvsi_q_only = k * B["Q_R"] / (A["Vs"] ** 2)
    print(
        f"[{line}] sending {A.name}: Vs {A['Vs']:.4f}->{B['Vs']:.4f}, Q_R {A['Q_R']:.4f}->{B['Q_R']:.4f}"
    )
    print(f" FVSI: {A['FVSI']:.6f} -> {B['FVSI']:.6f} (Δ={B['FVSI'] - A['FVSI']:+.6f})")
    print(
        f" Counterfactuals: only Vs -> {fvsi_vs_only:.6f}, only Q -> {fvsi_q_only:.6f}"
    )
