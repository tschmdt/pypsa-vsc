import pypsa
import numpy as np
import pandas as pd
from pypsa import descriptors

# Neues leeres Netz
netz = pypsa.Network()

netz.set_snapshots(pd.Index([0]))



# Busse
netz.add("Bus", "Bus 1", v_nom=110, carrier="AC")
netz.add("Bus", "Bus 2", v_nom=110, carrier="AC")
netz.add("Bus", "Bus 3", v_nom=110, carrier="AC")
netz.add("Bus", "Bus 4", v_nom=110, carrier="AC")
netz.add("Bus", "Bus 5", v_nom=220, carrier="AC")
netz.add("Bus", "Bus 6", v_nom=220, carrier="AC")


# Leitung (0.005+0.1j)*100km
netz.add("Line", "1-2", bus0="Bus 1", bus1="Bus 2", x=10, r=0.5, s_nom=250)
netz.add("Line", "1-4", bus0="Bus 1", bus1="Bus 4", x=15, r=0.25, s_nom=250)
netz.add("Line", "2-3", bus0="Bus 2", bus1="Bus 3", x=12, r=0.2, s_nom=300)
netz.add("Line", "3-4", bus0="Bus 3", bus1="Bus 4", x=10, r=0.2, s_nom=250)
netz.add("Line", "5-6", bus0="Bus 5", bus1="Bus 6", x=10, r=0.2, s_nom=250)
# netz.add("Line", "4-5", bus0="Bus 4", bus1="Bus 5", x=10, r=0.2,s_nom=250)

# Trafo
netz.add("Transformer", "Trafo 1", bus0="Bus 5", bus1="Bus 4", x=0.1, r=0.01, s_nom=500)

# Shunt
netz.add("ShuntImpedance", "Shunt 1", bus="Bus 4", g=0.0, b=-50.0)

# Generator 
netz.add("Generator", "Gen 1", bus="Bus 1", p_set=200, control="Slack", carrier="gas")
netz.add("Generator", "Gen 3", bus="Bus 3", p_set=100, control="PQ", carrier="coal")
netz.add("Generator", "Gen 6", bus="Bus 6", p_set=200, control="PV", carrier="solar")

netz.generators.loc["Gen 1", ["p_nom","marginal_cost","p_min_pu","p_max_pu"]] = [400, 30.0, 0.0, 1.0]
netz.generators.loc["Gen 3", ["p_nom","marginal_cost","p_min_pu","p_max_pu"]] = [300, 35.0, 0.0, 1.0]
netz.generators.loc["Gen 6", ["p_nom","marginal_cost","p_min_pu","p_max_pu"]] = [350, 29, 0.0, 1.0]


# Last
netz.add("Load", "Load 2", bus="Bus 2", p_set=150, q_set=50)
netz.add("Load", "Load 3", bus="Bus 3", p_set=100, q_set=50)
netz.add("Load", "Load 4", bus="Bus 4", p_set=300, q_set=20)
netz.add("Load", "Load 6", bus="Bus 6", p_set=125, q_set=10)

# Link
netz.add(
    "Link", "Link 1", bus0="Bus 3", bus1="Bus 4", p_set=5.3, efficiency=0.9, p_nom=150,carrier="DC"
)
netz.add(
    "Link", "Link 2", bus0="Bus 5", bus1="Bus 6", p_set=5.3, efficiency=0.9, p_nom=150,carrier="DC"
)

# Voltage Source Converter (VSC)
netz.add("ControllableVSC", "VSC 1", bus="Bus 3", q_min=-100, q_max=100)
netz.add("ControllableVSC", "VSC 2", bus="Bus 4", q_min=-100, q_max=100)
netz.add("ControllableVSC", "VSC 3", bus="Bus 5", q_min=-50, q_max=50)
netz.add("ControllableVSC", "VSC 4", bus="Bus 6", q_min=-50, q_max=50)

# VSC 1 sitzt an bus0 von Link 1
netz.controllable_vscs.loc["VSC 1", ["link", "side"]] = ["Link 1", "bus0"]
# VSC 2 sitzt an bus1 von Link 1
netz.controllable_vscs.loc["VSC 2", ["link", "side"]] = ["Link 1", "bus1"]

netz.controllable_vscs.loc["VSC 3", ["link", "side"]] = ["Link 2", "bus0"]
netz.controllable_vscs.loc["VSC 4", ["link", "side"]] = ["Link 2", "bus1"]


# "behandele ControllableVSC wie eine nominale Komponente mit Nominalwert p_nom "


# Modul aufrufen
# optimize_vsc_q_voltage_support(netz)

# Power Flow
netz.lpf()

# Ergebnisse anzeigen
# print(netz.buses_t.v_mag_pu)

F = np.sqrt(netz.lines_t.p0**2)  # ==p0
s_nom = netz.lines.s_nom
s_nom_matrix = pd.DataFrame(
    np.tile(s_nom.values, (len(F.index), 1)), index=F.index, columns=F.columns
)

loading_default = 100 * F / s_nom_matrix
print("Laoding_default:", loading_default)
print("Links:", netz.links_t.p0)


netz.optimize()



import IPython; IPython.embed()  # startet interaktive Python-Shell für einfache Analyse