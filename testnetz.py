import pypsa

# Neues leeres Netz
netz = pypsa.Network()

# Busse
netz.add("Bus", "Bus 1", v_nom=110)
netz.add("Bus", "Bus 2", v_nom=110)

# Leitung (0.005+0.1j)*100km
netz.add("Line", "1-2", bus0="Bus 1", bus1="Bus 2", x=10, r=0.5)

# Generator (Slack)
netz.add("Generator", "Gen 1", bus="Bus 1", p_set=200, control="Slack")

# Last 
netz.add("Load", "Load 2", bus="Bus 2", p_set=100)

# Link
#netz.add("Link", "Link 1", bus0="Bus 1", bus1="Bus 2", p_set=30, efficiency=0.9)

# Voltage Source Converter (VSC)
netz.add("ControllableVSC", "VSC 1", bus="Bus 2", q_set=20)

# Power Flow
netz.pf()

# Ergebnisse anzeigen
print(netz.buses_t.v_mag_pu)

netz.export_to_netcdf("netz.nc")

import IPython; IPython.embed()  # startet interaktive Python-Shell für einfache Analyse