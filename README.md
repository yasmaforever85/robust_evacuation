# Hurricane Impact Scenario Generator & Stochastic Evacuation Optimizer

**Geospatial scenario generation via GMM with temporal Bayesian updating, and two-stage stochastic programming for multicommodity evacuation under uncertainty**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.XXXXXXX-blue)](https://doi.org/10.6084/m9.figshare.XXXXXXX)

## Overview

This repository contains two complementary components for hurricane emergency management:

1. **Scenario Generation Engine** — A geospatial framework that produces discrete hurricane impact scenarios with calibrated, time-evolving probabilities using Gaussian Mixture Models and Bayesian updating.

2. **Stochastic Evacuation Validation** — A computational validation suite for a two-stage stochastic programming model that optimizes multicommodity evacuation flows under the generated scenarios.

## Associated Papers

> **Paper 1 (Scenario Generation):**
> Fernández-Fernández, Y., Allende Alonso, S.M., Miranda Pérez, R., Bouza Allende, G. & Cabrera Álvarez, E.N. (2026). Geospatial Scenario Generation for Hurricane Impact Assessment: A Gaussian Mixture Framework with Temporal Bayesian Updating in a Geographically Amplified Feature Space. *International Journal of Geographical Information Science* (under review).

> **Paper 2 (Stochastic Evacuation):**
> Fernández-Fernández, Y., Allende Alonso, S.M., Miranda Pérez, R., Bouza Allende, G. & Cabrera Álvarez, E.N. (2026). Data-driven scenario generation and two-stage stochastic programming for multicommodity evacuation under hurricane uncertainty. *Modern Stochastics: Theory and Applications* (under review).

## Repository Structure

```
robust_evacuation/
├── GeneradorEscenariosHuracan.py   # Core scenario engine (Papers 1 & 2)
├── PI_Plan_Flujo.py                # Deterministic flow MIP PI^D (Paper 2)
├── PI_Estoc_Esc.py                 # Two-stage stochastic MIP PI^S (Paper 2)
├── validacion_integrada_MIP.py     # 100-experiment validation with real solver (Paper 2)
├── ModuloGeneradorEscenarios.py    # PyQt6 graphical interface (Paper 1)
├── hurricane_db_cuba.json          # Historical hurricane database (21 storms)
├── data/
│   └── VMSTA_validation_results.xlsx  # Validation results (Paper 2)
├── requirements.txt                # Python dependencies
├── CITATION.cff                    # Citation metadata
├── LICENSE                         # MIT License
└── README.md
```

## Core Engine (`GeneradorEscenariosHuracan.py`)

The computational engine implements:

* **HURDAT2 parsing** with geographic Cuba-impact filtering
* **9-dimensional geospatial feature space** with geographic amplification (κ = 2.0)
* **Gaussian Mixture Models** (EM algorithm) with BIC model selection
* **Bayesian temporal updating** across forecast horizons (72 → 12 h)
* **Demand factor** (Eq. 9 in Paper 2): η_s = α₀ + α_cat · c_s + α_wind · v̂_s − α_dist · d̂_s + α_land · ι_s, calibrated against Huang, Lindell & Prater (2016) meta-analytic evacuation rates
* **NHC real-time advisory parsing** (CurrentStorms.json + RSS feeds)
* **Haversine geolocation** mapping centroids to 40 Cuban population centres

### Demand Factor Coefficients (Eq. 9)

| Coefficient | Value | Predictor |
|---|---|---|
| α₀ | 0.36 | Base compliance rate |
| α_cat | 0.22 | Saffir-Simpson category |
| α_wind | 0.08 | Normalized max sustained wind (v/185) |
| α_dist | 0.10 | Normalized coastal distance (d/500), attenuates |
| α_land | 0.12 | Landfall indicator (binary) |

## Validation Suite (`validacion_integrada.py`)

Runs 100 experiments: 5 network instances × 4 hurricanes × 5 forecast horizons.

### Network Instances

| Instance | Description | Nodes | P_nom | Σπ | Cap ratio |
|---|---|---|---|---|---|
| G1 | Havana (base) | 5 | 60 | 75 | 1.25 |
| G2 | Havana extended | 9 | 150 | 180 | 1.20 |
| G3 | Western Cuba | 14 | 350 | 413 | 1.18 |
| G4 | Multi-province | 23 | 800 | 920 | 1.15 |
| G5 | National-scale | 31 | 1500 | 1680 | 1.12 |

### Key Results

| Instance | Nodes | Arcs | Infeasibility (%) | Mean Coverage (%) | Mean VSS (%) | Max Deficit |
|---|---|---|---|---|---|---|
| G1 | 5 | 4 | 70 | 87.4 | 35.3 | 26 |
| G2 | 9 | 14 | 70 | 92.2 | 35.6 | 60 |
| G3 | 14 | 27 | 70 | 88.0 | 10.7 | 126 |
| G4 | 23 | 60 | 75 | 87.5 | 7.0 | 150 |
| G5 | 31 | 90 | 85 | 79.8 | — | 450 |

Overall: 74/100 deterministic problems infeasible (74%), confirming the value of the stochastic formulation. The MIP solver uses OR-Tools SCIP via `PI_Estoc_Esc.py` (two-stage stochastic) and `PI_Plan_Flujo.py` (deterministic flow).

## Installation

### Requirements

* Python 3.11 or later
* OR-Tools (`pip install ortools`) for MIP solver
* No proprietary software or data dependencies

### Setup

```bash
git clone https://github.com/yasmaforever85/robust_evacuation.git
cd robust_evacuation
pip install -r requirements.txt
```

### Running the validation (Paper 2)

```bash
python validacion_integrada_MIP.py
```

Produces: `VMSTA_validation_results.xlsx` (3 sheets: All_Experiments, Instance_Summary, Metadata) and console summary. Requires OR-Tools (`pip install ortools`).

### Running the graphical interface (Paper 1)

```bash
python ModuloGeneradorEscenarios.py
```

### Using the engine programmatically

```python
from GeneradorEscenariosHuracan import ScenarioGenerator

gen = ScenarioGenerator()
scenarios = gen.generate_scenarios(
    target_name="IAN",
    horizon_hours=72,
    method="gmm",
    seed=42
)

for s in scenarios.scenarios:
    print(f"P={s.probability:.3f}  Cat{s.category}  "
          f"η={s.demand_factor:.3f}  "
          f"({s.impact_lat:.2f}, {s.impact_lon:.2f})  "
          f"{s.impact_province}")
```

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| κ (alpha_geo) | 2.0 | Geographic amplification weight |
| σ²(h) | {1.0, 0.6, 0.4, 0.25, 0.15} | Bayesian precision schedule |
| k | 2–7 | GMM components (BIC-selected) |
| θ⁺ | 10,000 | Unmet demand penalty (per person) |
| θ⁻ | 100 | Excess evacuation penalty (per person) |

## Citation

If you use this software in your research, please cite:

```bibtex
@article{FernandezFernandez2026scenario,
  author  = {Fern{\'a}ndez-Fern{\'a}ndez, Yasmany and Allende Alonso, Sira Mar{\'i}a
             and Miranda P{\'e}rez, Ridelio and Bouza Allende, Gemayqzel
             and Cabrera {\'A}lvarez, Elia Natividad},
  title   = {Geospatial Scenario Generation for Hurricane Impact Assessment:
             A {G}aussian Mixture Framework with Temporal {B}ayesian Updating
             in a Geographically Amplified Feature Space},
  journal = {International Journal of Geographical Information Science},
  year    = {2026},
  note    = {Under review}
}

@article{FernandezFernandez2026stochastic,
  author  = {Fern{\'a}ndez-Fern{\'a}ndez, Yasmany and Allende Alonso, Sira Mar{\'i}a
             and Miranda P{\'e}rez, Ridelio and Bouza Allende, Gemayqzel
             and Cabrera {\'A}lvarez, Elia Natividad},
  title   = {Data-driven scenario generation and two-stage stochastic programming
             for multicommodity evacuation under hurricane uncertainty},
  journal = {Modern Stochastics: Theory and Applications},
  year    = {2026},
  note    = {Under review}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Authors

* **Yasmany Fernández-Fernández** — Universidad Politécnica Estatal del Carchi (UPEC), Ecuador / Universidad de La Habana, Cuba
* **Sira María Allende Alonso** — Universidad de La Habana, Cuba
* **Ridelio Miranda Pérez** — Universidad de Cienfuegos, Cuba
* **Gemayqzel Bouza Allende** — Universidad de La Habana, Cuba
* **Elia Natividad Cabrera Álvarez** — Universidad de Cienfuegos, Cuba

## Acknowledgements

This work is part of the doctoral thesis *"Modelos de Optimización Robusta para Gestión de Emergencias Meteorológicas bajo Incertidumbre"* at the Faculty of Mathematics and Computing (MATCOM), Universidad de La Habana, Cuba.
