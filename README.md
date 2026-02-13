# Hurricane Impact Scenario Generator

**Geospatial scenario generation for hurricane emergency management via Gaussian Mixture Models with temporal Bayesian updating**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.XXXXXXX-blue)](10.6084/m9.figshare.31337560)

## Overview

This repository contains the implementation of a two-stage geospatial framework for generating discrete hurricane impact scenarios with calibrated, time-evolving probabilities. The framework addresses the **Scenario Generation Problem under Temporal Uncertainty (SGPTU)**: given a hurricane approaching a territory, produce a finite set of geolocated impact scenarios whose probabilities sharpen from near-uniform at 72 hours to quasi-deterministic at 12 hours before potential landfall.

The method embeds historical hurricane analogs in a 9-dimensional feature space that blends meteorological attributes with geographically amplified impact coordinates (operationalising Tobler's First Law), clusters them via Gaussian Mixture Models, and refines the resulting probabilities through Bayesian updating calibrated to NHC forecast cone errors.

**Associated paper:**

> Fernández-Fernández, Y., Allende Alonso, S.M., Miranda Pérez, R., Bouza Allende, G. & Cabrera Álvarez, E.N. (2026). Geospatial Scenario Generation for Hurricane Impact Assessment: A Gaussian Mixture Framework with Temporal Bayesian Updating in a Geographically Amplified Feature Space. *International Journal of Geographical Information Science* (under review).

## Repository Structure

```
scenario_generation/
├── GeneradorEscenariosHuracan.py   # Core computational engine (2,465 lines)
├── ModuloGeneradorEscenarios.py    # PyQt6 graphical interface (1,452 lines)
├── hurricane_db_cuba.json          # Historical hurricane database (211 tracks)
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## Core Engine (`GeneradorEscenariosHuracan.py`)

The computational engine implements:

- **HURDAT2 parsing** with geographic Cuba-impact filtering
- **9-dimensional geospatial feature space** with geographic amplification parameter α_geo
- **Gaussian Mixture Models** (EM algorithm) for hurricane impact clustering
- **K-Means++** with Silhouette validation as baseline comparison
- **Bayesian temporal updating** across forecast horizons (72 → 48 → 36 → 24 → 12 h)
- **Monte Carlo ensemble generation** for robustness analysis
- **NHC real-time advisory parsing** (CurrentStorms.json + RSS feeds)
- **Haversine geolocation** mapping centroids to 40 Cuban population centres

### Output Format

Each scenario is a complete spatial object:

```
ω_k = (P(ω_k|h), Cat_k, V_k, D_k, lat_k, lon_k, landfall_k, province_k)
```

directly ingestible by downstream optimisation models (stochastic programming, robust optimisation).

## Graphical Interface (`ModuloGeneradorEscenarios.py`)

A PyQt6 application with five tabs:

1. **Historical Analysis** — scenario generation and impact coordinate mapping
2. **Temporal Cascade** — Bayesian convergence visualisation (72 → 12 h)
3. **Analysis & Reports** — charts, statistics, and Excel export
4. **NHC Real-Time** — live advisory monitoring and scenario updating
5. **Catalogue** — historical storm browser with filtering

## Data

### `hurricane_db_cuba.json`

Historical hurricane database containing 211 Atlantic tropical cyclone tracks, extracted and processed from the NHC Best Track archive (HURDAT2). The dataset covers storms that affected or passed near Cuba from 1851 to 2024. Of these, 21 major hurricanes (Category 1–5) that directly impacted the island between 1926 and 2024 form the primary analysis set.

**Source:** National Hurricane Center, NOAA — [HURDAT2 archive](https://www.nhc.noaa.gov/data/)

## Installation

### Requirements

- Python 3.11 or later
- No proprietary software or data dependencies

### Setup

```bash
git clone https://github.com/yasmaforever85/scenario_generation.git
cd scenario_generation
pip install -r requirements.txt
```

### Running the graphical interface

```bash
python ModuloGeneradorEscenarios.py
```

### Using the engine programmatically

```python
from GeneradorEscenariosHuracan import ScenarioEngine

engine = ScenarioEngine("hurricane_db_cuba.json")
scenarios = engine.generate_scenarios(
    target_storm="IAN",
    horizon=12,
    alpha_geo=2.0,
    method="gmm"
)

for s in scenarios:
    print(f"P={s.probability:.3f}  Cat{s.category}  "
          f"({s.lat:.2f}, {s.lon:.2f})  {s.province}")
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha_geo` | 2.0 | Geographic amplification weight (validated via ablation) |
| `sigma²(h)` | {1.0, 0.6, 0.4, 0.25, 0.15} | Precision schedule for h ∈ {72, 48, 36, 24, 12} h |
| `k` | 2–7 | Number of GMM components (selected by BIC) |
| `ε` | 10⁻⁶ | Covariance regularisation constant |

## Validation Results

Tested on 21 historical hurricanes affecting Cuba (1926–2024):

- **82%** Shannon entropy reduction (72 h → 12 h)
- **43%** geographic coherence within 200 km at 12 h
- **p < 0.01** superiority over all baselines (Wilcoxon signed-rank test)
- **12.3 s** total computation time on standard hardware

## Citation

If you use this software in your research, please cite:

```bibtex
@article{FernandezFernandez2026,
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
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Authors

- **Yasmany Fernández-Fernández** — Universidad Politécnica Estatal del Carchi (UPEC), Ecuador / Universidad de La Habana, Cuba
- **Sira María Allende Alonso** — Universidad de La Habana, Cuba
- **Ridelio Miranda Pérez** — Universidad de Cienfuegos, Cuba
- **Gemayqzel Bouza Allende** — Universidad de La Habana, Cuba
- **Elia Natividad Cabrera Álvarez** — Universidad de Cienfuegos, Cuba

## Acknowledgements

This work is part of the doctoral thesis *"Modelos de Optimización Robusta para Gestión de Emergencias Meteorológicas bajo Incertidumbre"* at the Faculty of Mathematics and Computing (MATCOM), Universidad de La Habana, Cuba.
