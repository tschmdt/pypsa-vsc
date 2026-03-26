[![Documentation Status](https://readthedocs.org/projects/pypsa/badge/?version=latest)](https://pypsa.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/pypsa.svg)](LICENSE.txt)
**![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FPyPSA%2FPyPSA%2Fmaster%2Fpyproject.toml)**


# PyPSA-VSC - Python for Power System Analysis with Voltage-Source Converter extension

### Functionality
This PyPSA version extends the original implementation with additional functionalities for direct-current power transmission. The converter technology of choice is the Voltage Source Converter (VSC). The VSC has unique features, most notably its ability to control active and reactive power almost independently.
The VSC is modeled as an emulated component using the existing Link component and the newly created ControllableVSC component. Together, they allow to model an HVDC-VSC-Link, referred to as VHL.
To effectively model a network with a VHL, several key considerations must be taken into account. As a user, you have multiple degrees of freedom to configure the VHL optimization. The complete set of attributes and functionalities can be found in the main document. The most critical attributes include:

- Power angle limit in degrees, with a default of 25.
- Maximum line loading in p.u., with a default of 0.95.
- Rated apparent power of the VSC in MVA, with a default of 400.

The PyPSA-VSC Version is an extension of PyPSA, therefore all original functionalities remain unchanged.

PyPSA-VSC can be interpreted as an additional framework for modeling and analyzing VSC-based HVDC links. Although simplifications are applied, the VHL representation captures all relevant physical properties and operational constraints. In this formulation, bus0 is always defined as the master bus, providing two degrees of freedom: active power (P) and reactive power (Q).

Building on this physical representation, PyPSA-VSC enables users to run an optimized operation mode of the VHL. The selection of bus0 (the sending or starting bus) is critical, as it is designated as the master bus; consequently, the optimization variables are determined at this bus. Bus1 is treated as the slave bus.

The objective of the VHL operation optimization is to homogenize AC line loadings and smooth voltage profiles across the network. The optimizer takes the system-wide network state as input and determines the optimal active and reactive power setpoints of the link to achieve these goals.

Active and reactive power control can be executed separately. By default (combined run mode), active power optimization is performed first. During this step, the available apparent power limit is restricted to $\frac{1}{\sqrt{2}} * S_{vsc}$, ensuring sufficient headroom for subsequent reactive power optimization.

After determining the optimal active power setpoint, an AC power flow calculation is performed to accurately represent the network state. Based on these results, the remaining reactive power headroom is recalculated and imposed as a constraint for the subsequent reactive power optimization step. The process concludes with a final AC power flow.

Internally, the optimization is implemented using linearized approximations.

### Updates
By march 2026 the version has migrated to uv! Hence, the main branch has beeing updated! Old instructions using conda are deprecated. Use uv instead. All dependencies are updated in pyproject.toml.

### Installation using uv
Using Bash:
git clone https://github.com/tschmdt/pypsa-vsc.git
cd pypsa-vsc
uv sync
uv sync --extra gurobipy (Note: A valid Gurobi license is required for optimization.)
uv run python <path/to/script.py>
optional: uv sync --extra dev
Note: All dependencies as requirements are set and can be found in pyproject.toml. Using uv (sync) automatically creates a .venv and installs PyPSA-VSC in editable mode using the rquired settings.

### Usage 
To set up a VHL in our model, we utilize the *ControllerConfig* class and the *VSCController* class, both found in the combined_control folder within *VSCController.py*.

A VHL is implemented by:
```py

from combined_control.VSCController import ControllerConfig,VSCController

n.add("Link", "Link 3-4", bus0= "Bus 3", bus1 = "Bus 4", p_set = -100 , efficiency = 0.95, p_nom = 500)

n.add("ControllableVSC", "VSC 1", bus="Bus 3", link = "Link 3-4", side = "bus0")

n.add("ControllableVSC", "VSC 2", bus="Bus 4", link = "Link 3-4", side = "bus1")
```
The main objective of the VHL is to operate system supportive. Therefore two optimization problems are solved, with the aim to
- reduce system-wide ac line loadings (highly loaded lines are taken into account more heavily in the objective function), by controlling the active power transfer via the Link.
- stabilize system-wide bus voltages (1 p.u.), by adjusting reactive power injection/consumption.

This optimization run can be started with:

```py
cfg = ControllerConfig(angle_limit_deg = 25, max_line_loading = 0.9, S_rated = 300, n1_guard_enable = False)

ctl = VSCController(n, config= cfg)

p_results, q_results = ctl.run_mode(mode="combined")
```

### Example 

### PyPSA is published under MIT license:
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.

### Features of PyPSA

- **Economic Dispatch (ED):** Models short-term market-based dispatch including
unit commitment, renewable availability, short-duration and seasonal storage
including hydro reservoirs with inflow and spillage dynamics, elastic demands,
load shedding and conversion between energy carriers, using either perfect
operational foresight or rolling horizon time resolution.

- **Linear Optimal Power Flow (LOPF):** Extends economic dispatch to determine
the least-cost dispatch while respecting network constraints in meshed AC-DC
networks, using a linearised representation of power flow (KVL, KCL) with
optional loss approximations.

- **Security-Constrained LOPF (SCLOPF):** Extends LOPF by accounting for line
outage contingencies to ensure system reliability under $N-1$ conditions.

- **Capacity Expansion Planning (CEP):** Supports least-cost
long-term system planning with investment decisions for generation, storage,
conversion, and transmission infrastructure. Handles both single and multiple
investment periods. Continuous and discrete investments are supported.

- **Pathway Planning:** Supports co-optimisation of multiple investment periods to
plan energy system transitions over time with perfect planning foresight.

- **Stochastic Optimisation:** Implements two-stage stochastic programming
framework with scenario-weighted uncertain inputs, with investments as
first-stage decisions and dispatch as recourse decisions.

- **Modelling-to-Generate-Alternatives (MGA):** Explores near-optimal decision
spaces to provide insight into the range of feasible system configurations with
similar costs.

- **Sector-Coupling:** Modelling integrated energy systems with multiple energy
  carriers (electricity, heat, hydrogen, etc.) and conversion between them.
  Flexible representation of technologies such as heat pumps, electrolysers,
  battery electric vehicles (BEVs), direct air capture (DAC), and synthetic
  fuels production.

- **Static Power Flow Analysis:** Computes both full non-linear and linearised
  load flows for meshed AC and DC grids using Newton-Raphson method.

### Documentation

PyPSA has extensive [documentation](https://docs.pypsa.org) with tutorials, user guides, examples and an API reference.


### Citing PyPSA

If you use PyPSA for your research, please cite the following paper:

-   T. Brown, J. Hörsch, D. Schlachtberger, [PyPSA: Python for Power
    System Analysis](https://arxiv.org/abs/1707.09913), 2018, [Journal
    of Open Research
    Software](https://openresearchsoftware.metajnl.com/), 6(1),
    [arXiv:1707.09913](https://arxiv.org/abs/1707.09913),
    [DOI:10.5334/jors.188](https://doi.org/10.5334/jors.188)

Please use the following BibTeX:

    @article{PyPSA,
       author = {T. Brown and J. H\"orsch and D. Schlachtberger},
       title = {{PyPSA: Python for Power System Analysis}},
       journal = {Journal of Open Research Software},
       volume = {6},
       issue = {1},
       number = {4},
       year = {2018},
       eprint = {1707.09913},
       url = {https://doi.org/10.5334/jors.188},
       doi = {10.5334/jors.188}
    }

### Citing PyPSA-VSC
Please cite PyPSA additionally, when using PyPSA-VSC.

https://doi.org/10.34726/hss.2026.134500

    @article{schmidt2026conceptual,
      title={Conceptual Design and Optimization of Hybrid AC/DC Power Systems},
      author={Schmidt, Timo},
      year={2026},
      institution={Technische Universit{\"a}t Wien},
      year={2026},
      url={http://hdl.handle.net/20.500.12708/226579}
    }

PyPSA is licensed under the open source [MIT
License](https://github.com/PyPSA/PyPSA/blob/master/LICENSE.txt).
Copyright 2015-2025 [PyPSA
Developers](https://pypsa.readthedocs.io/en/latest/developers.html)

