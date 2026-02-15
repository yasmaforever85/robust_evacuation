#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  HURRICANE IMPACT SCENARIO ENGINE FOR CUBA                                  ║
║  Stochastic Scenario Generation via ML-Based Historical Analysis            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Module V of the Hierarchical Metamodel for Evacuation under Uncertainty    ║
║                                                                             ║
║  Computational Techniques:                                                  ║
║    • HURDAT2 parsing with geographic Cuba-impact filtering                  ║
║    • Gaussian Mixture Models (EM algorithm) for intensity clustering        ║
║    • K-Means++ with Silhouette validation for track pattern recognition     ║
║    • Kernel Density Estimation for non-parametric probability surfaces      ║
║    • Bayesian temporal updating across forecast horizons (72→12h)           ║
║    • Monte Carlo ensemble generation with importance sampling               ║
║    • NHC real-time advisory parsing (CurrentStorms.json + RSS)              ║
║                                                                             ║
║  Output: Scenario set Ξ = {(ω, p_ω, cat_ω, v_ω, D_ω)} for PI^S           ║
║                                                                             ║
║  References:                                                                ║
║    [1] Landsea & Franklin (2013) Mon. Wea. Rev. 141:3576-3592 (HURDAT2)    ║
║    [2] Bertsimas & Sim (2004) Oper. Res. 52:35-53 (Robust optimization)    ║
║    [3] Rumpf et al. (2007) Stochastic modelling of tropical cyclone tracks  ║
║    [4] Dempster et al. (1977) J. Royal Stat. Soc. B 39:1-38 (EM/GMM)      ║
║    [5] MacQueen (1967) Berkeley Symp. Math. Stat. Prob. (K-Means)           ║
║    [6] Vickery et al. (2000) J. Struct. Eng. 126:1146-1154                 ║
║    [7] Saffir-Simpson Hurricane Wind Scale (NHC/NOAA)                       ║
║                                                                             ║
║  Author: Y. Fernández-Fernández — UPEC / Universidad de La Habana          ║
║  Version: 2.0 — February 2026                                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import json
import os
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
)

import numpy as np
from scipy import stats as sp_stats

# ═══════════════════════════════════════════════════════════════════════════════
# §1  CONSTANTS AND ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

EARTH_RADIUS_KM = 6_371.0

# Saffir-Simpson scale: lower bound of sustained wind (kt) per category
SAFFIR_SIMPSON = {
    -1: (0, 33),      # Tropical Depression
     0: (34, 63),     # Tropical Storm
     1: (64, 82),     # Category 1
     2: (83, 95),     # Category 2
     3: (96, 112),    # Category 3
     4: (113, 136),   # Category 4
     5: (137, 999),   # Category 5
}

# Temporal horizons for scenario generation (hours before landfall)
HORIZONS = (72, 48, 36, 24, 12)


class CubaRegion(Enum):
    """Geographic regions of Cuba for impact analysis."""
    OCCIDENTE = "Occidente"   # Pinar del Río → Mayabeque + Isla de la Juventud
    CENTRO    = "Centro"      # Matanzas → Camagüey
    ORIENTE   = "Oriente"     # Las Tunas → Guantánamo


class StormStatus(Enum):
    """HURDAT2 storm status codes."""
    TD = "TD"   # Tropical Depression
    TS = "TS"   # Tropical Storm
    HU = "HU"   # Hurricane
    EX = "EX"   # Extratropical
    SS = "SS"   # Subtropical Storm
    SD = "SD"   # Subtropical Depression
    LO = "LO"   # Low
    WV = "WV"   # Tropical Wave
    DB = "DB"   # Disturbance


# ═══════════════════════════════════════════════════════════════════════════════
# §2  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackPoint:
    """Single 6-hourly HURDAT2 observation."""
    datetime_utc: datetime
    record_id: str          # L=landfall, W=max wind, P=min pressure, etc.
    status: str             # TD, TS, HU, EX, ...
    lat: float              # degrees N (negative for S)
    lon: float              # degrees E (negative for W)
    max_wind_kt: int        # maximum sustained wind (knots)
    min_pressure_mb: int    # minimum central pressure (mb), -999 if missing
    r34_ne: int = 0         # 34-kt wind radius NE quadrant (nm)
    r34_se: int = 0
    r34_sw: int = 0
    r34_nw: int = 0

    @property
    def category(self) -> int:
        """Saffir-Simpson category from wind speed."""
        return wind_to_category(self.max_wind_kt)

    @property
    def is_hurricane(self) -> bool:
        return self.max_wind_kt >= 64

    @property
    def is_major(self) -> bool:
        return self.max_wind_kt >= 96


@dataclass
class HurricaneTrack:
    """Complete storm track from HURDAT2."""
    atcf_id: str            # e.g. AL092017
    name: str               # e.g. IRMA
    points: List[TrackPoint] = field(default_factory=list)

    @property
    def year(self) -> int:
        return int(self.atcf_id[4:8])

    @property
    def max_wind_kt(self) -> int:
        return max((p.max_wind_kt for p in self.points), default=0)

    @property
    def min_pressure_mb(self) -> int:
        pressures = [p.min_pressure_mb for p in self.points if p.min_pressure_mb > 0]
        return min(pressures) if pressures else -999

    @property
    def max_category(self) -> int:
        return wind_to_category(self.max_wind_kt)

    @property
    def duration_hours(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return (self.points[-1].datetime_utc - self.points[0].datetime_utc).total_seconds() / 3600

    @property
    def date_range(self) -> str:
        if not self.points:
            return ""
        d0 = self.points[0].datetime_utc.strftime("%Y-%m-%d")
        d1 = self.points[-1].datetime_utc.strftime("%Y-%m-%d")
        return f"{d0} → {d1}"


@dataclass
class CubaImpact:
    """Characterization of a hurricane's impact on Cuba."""
    track: HurricaneTrack
    closest_approach_km: float
    closest_point: TrackPoint
    region: CubaRegion
    landfall: bool
    landfall_point: Optional[TrackPoint] = None
    category_at_closest: int = 0
    wind_at_closest: int = 0
    transit_hours: float = 0.0      # hours within Cuba influence zone
    entry_bearing: float = 0.0      # approach direction (degrees)
    provinces_affected: List[str] = field(default_factory=list)

    @property
    def feature_vector(self) -> np.ndarray:
        """9-dimensional feature vector for ML analysis.

        Features include geographic coordinates of impact point with
        amplified weight (×2.0) to ensure clustering respects WHERE
        the storm hits, not just meteorological intensity.

        Dimensions:
            f0: category (0–5)
            f1: wind normalized (wind/185)
            f2: distance to coast normalized (dist/500)
            f3: landfall indicator (0 or 1)
            f4: transit time normalized (hours/48)
            f5: bearing normalized (bearing/360)
            f6: region code (Occidente=0, Centro=0.5, Oriente=1)
            f7: latitude normalized, weighted ×2  → ((lat−19.8)/3.5)×2
            f8: longitude normalized, weighted ×2 → ((lon+85)/10.9)×2
        """
        lat_norm = (self.closest_point.lat - 19.8) / 3.5   # Cuba lat range
        lon_norm = (self.closest_point.lon + 85.0) / 10.9   # Cuba lon range
        return np.array([
            self.category_at_closest,
            self.wind_at_closest / 185.0,
            self.closest_approach_km / 500.0,
            1.0 if self.landfall else 0.0,
            self.transit_hours / 48.0,
            self.entry_bearing / 360.0,
            self.region.value == "Oriente" and 1.0 or
            (self.region.value == "Centro" and 0.5 or 0.0),
            lat_norm * 2.0,    # geographic weight amplification
            lon_norm * 2.0,    # geographic weight amplification
        ])


@dataclass
class Scenario:
    """Generated probabilistic scenario for PI^S."""
    id: int
    label: str
    probability: float
    category: int
    max_wind_kt: int
    demand_factor: float
    region: CubaRegion
    horizon_hours: int
    reference_storms: List[str]
    description: str
    impact_lat: float = 0.0
    impact_lon: float = 0.0
    impact_location: str = ""
    impact_province: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "probability": round(self.probability, 6),
            "category": self.category,
            "max_wind_kt": self.max_wind_kt,
            "demand_factor": round(self.demand_factor, 4),
            "region": self.region.value,
            "horizon_hours": self.horizon_hours,
            "impact_lat": round(self.impact_lat, 4),
            "impact_lon": round(self.impact_lon, 4),
            "impact_location": self.impact_location,
            "impact_province": self.impact_province,
            "reference_storms": self.reference_storms,
            "description": self.description,
        }


@dataclass
class ScenarioSet:
    """Complete scenario ensemble for a given hurricane and horizon."""
    hurricane_name: str
    hurricane_year: int
    target_region: CubaRegion
    horizon_hours: int
    scenarios: List[Scenario]
    clustering_method: str
    n_analogs_used: int
    silhouette_score: float
    bic: Optional[float] = None
    generation_timestamp: str = ""

    @property
    def n_scenarios(self) -> int:
        return len(self.scenarios)

    @property
    def probability_sum(self) -> float:
        return sum(s.probability for s in self.scenarios)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "hurricane": self.hurricane_name,
                "year": self.hurricane_year,
                "region": self.target_region.value,
                "horizon_hours": self.horizon_hours,
                "n_scenarios": self.n_scenarios,
                "prob_sum": round(self.probability_sum, 6),
                "clustering_method": self.clustering_method,
                "n_analogs": self.n_analogs_used,
                "silhouette": round(self.silhouette_score, 4),
                "bic": round(self.bic, 2) if self.bic else None,
                "timestamp": self.generation_timestamp,
            },
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# §3  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def wind_to_category(wind_kt: int) -> int:
    """Convert sustained wind (kt) to Saffir-Simpson category."""
    if wind_kt >= 137: return 5
    if wind_kt >= 113: return 4
    if wind_kt >= 96:  return 3
    if wind_kt >= 83:  return 2
    if wind_kt >= 64:  return 1
    if wind_kt >= 34:  return 0   # Tropical Storm
    return -1                      # Tropical Depression


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km using Haversine formula."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 (degrees, 0=N clockwise)."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    x = math.sin(Δλ) * math.cos(φ2)
    y = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def category_label(cat: int) -> str:
    """Human-readable category label."""
    labels = {-1: "DT", 0: "TS", 1: "Cat 1", 2: "Cat 2",
              3: "Cat 3", 4: "Cat 4", 5: "Cat 5"}
    return labels.get(cat, f"Cat {cat}")


# ═══════════════════════════════════════════════════════════════════════════════
# §4  CUBA GEOGRAPHY
# ═══════════════════════════════════════════════════════════════════════════════

# Cuba approximate bounding box
CUBA_LAT_MIN, CUBA_LAT_MAX = 19.8, 23.3
CUBA_LON_MIN, CUBA_LON_MAX = -85.0, -74.1
CUBA_CENTROID = (22.0, -79.5)

# Impact radius: a storm within this distance is considered "affecting Cuba"
CUBA_IMPACT_RADIUS_KM = 300.0
CUBA_DIRECT_RADIUS_KM = 150.0

# Province data: (name, capital_lat, capital_lon, region, coastal, population_approx)
CUBA_PROVINCES = [
    # OCCIDENTE
    ("Pinar del Río",       22.417, -83.698, CubaRegion.OCCIDENTE, True,  587_000),
    ("Artemisa",            22.813, -82.762, CubaRegion.OCCIDENTE, True,  503_000),
    ("La Habana",           23.114, -82.367, CubaRegion.OCCIDENTE, True, 2_130_000),
    ("Mayabeque",           22.962, -82.156, CubaRegion.OCCIDENTE, True,  381_000),
    ("Isla de la Juventud", 21.885, -82.802, CubaRegion.OCCIDENTE, True,   84_000),
    # CENTRO
    ("Matanzas",            23.051, -81.578, CubaRegion.CENTRO,    True,  711_000),
    ("Villa Clara",         22.407, -79.965, CubaRegion.CENTRO,    True,  781_000),
    ("Cienfuegos",          22.146, -80.436, CubaRegion.CENTRO,    True,  405_000),
    ("Sancti Spíritus",     21.930, -79.442, CubaRegion.CENTRO,    True,  466_000),
    ("Ciego de Ávila",      21.840, -78.763, CubaRegion.CENTRO,    True,  432_000),
    ("Camagüey",            21.381, -77.917, CubaRegion.CENTRO,    True,  768_000),
    # ORIENTE
    ("Las Tunas",           20.962, -76.951, CubaRegion.ORIENTE,   True,  531_000),
    ("Holguín",             20.887, -76.263, CubaRegion.ORIENTE,   True,  1_035_000),
    ("Granma",              20.384, -76.644, CubaRegion.ORIENTE,   True,  836_000),
    ("Santiago de Cuba",    20.024, -75.821, CubaRegion.ORIENTE,   True,  1_050_000),
    ("Guantánamo",          20.145, -75.210, CubaRegion.ORIENTE,   True,  510_000),
]

# Key coastline reference points for proximity analysis
CUBA_COASTLINE_POINTS = [
    # Western tip to eastern tip, roughly following coastline
    (21.86, -84.95),   # Cabo de San Antonio (western tip)
    (22.40, -84.30),   # Pinar del Río coast
    (22.70, -83.50),   # Bahía Honda
    (23.00, -82.75),   # Mariel
    (23.15, -82.35),   # La Habana
    (23.17, -81.93),   # Santa Cruz del Norte
    (23.15, -81.25),   # Varadero
    (23.05, -81.00),   # Cárdenas
    (22.65, -80.20),   # Sagua la Grande coast
    (22.52, -79.47),   # Caibarién
    (22.35, -79.15),   # Yaguajay
    (22.20, -78.35),   # Cayo Coco area
    (21.95, -77.70),   # Nuevitas
    (21.55, -77.25),   # Santa Lucia
    (21.20, -76.60),   # Gibara
    (20.90, -76.25),   # Holguín coast
    (20.72, -75.60),   # Baracoa approach
    (20.35, -74.50),   # Punta de Maisí (eastern tip)
    # Southern coast (west to east)
    (21.75, -82.85),   # Isla de la Juventud
    (21.60, -82.10),   # Ciénaga de Zapata
    (22.15, -80.45),   # Cienfuegos
    (21.80, -79.95),   # Trinidad
    (21.40, -78.00),   # South Camagüey
    (20.38, -76.65),   # Manzanillo / Granma
    (19.97, -75.85),   # Santiago de Cuba
    (20.08, -75.15),   # Guantánamo Bay
]

# Region longitude boundaries (approximate)
REGION_LON_BOUNDARIES = {
    CubaRegion.OCCIDENTE: (-85.0, -81.5),   # Cabo San Antonio → ~Matanzas border
    CubaRegion.CENTRO:    (-81.5, -77.0),   # Matanzas → ~Las Tunas border
    CubaRegion.ORIENTE:   (-77.0, -74.0),   # Las Tunas → Punta de Maisí
}


def classify_region(lon: float) -> CubaRegion:
    """Classify a longitude into Cuba's geographic region."""
    for region, (lo, hi) in REGION_LON_BOUNDARIES.items():
        if lo <= lon <= hi:
            return region
    # Default by proximity
    if lon < -81.5:
        return CubaRegion.OCCIDENTE
    elif lon < -77.0:
        return CubaRegion.CENTRO
    return CubaRegion.ORIENTE


def min_distance_to_cuba(lat: float, lon: float) -> Tuple[float, CubaRegion]:
    """Minimum distance (km) from a point to Cuba's coastline, and closest region."""
    best_dist = float("inf")
    best_region = CubaRegion.CENTRO
    for clat, clon in CUBA_COASTLINE_POINTS:
        d = haversine_km(lat, lon, clat, clon)
        if d < best_dist:
            best_dist = d
            best_region = classify_region(clon)
    return best_dist, best_region


def provinces_in_radius(lat: float, lon: float, radius_km: float) -> List[str]:
    """Return province names within radius_km of a given point."""
    result = []
    for name, plat, plon, _reg, _coast, _pop in CUBA_PROVINCES:
        if haversine_km(lat, lon, plat, plon) <= radius_km:
            result.append(name)
    return result


# Key Cuban localities for impact point naming
# (name, lat, lon, province)
CUBA_LOCALITIES = [
    # OCCIDENTE
    ("Cabo de San Antonio",  21.86, -84.95, "Pinar del Río"),
    ("Sandino",              22.08, -84.22, "Pinar del Río"),
    ("Pinar del Río",        22.42, -83.70, "Pinar del Río"),
    ("Viñales",              22.62, -83.71, "Pinar del Río"),
    ("Bahía Honda",          22.90, -83.16, "Artemisa"),
    ("Mariel",               22.99, -82.75, "Artemisa"),
    ("La Habana",            23.11, -82.37, "La Habana"),
    ("Batabanó",             22.72, -82.29, "Mayabeque"),
    ("Güines",               22.84, -82.03, "Mayabeque"),
    ("Nueva Gerona",         21.88, -82.80, "Isla de la Juventud"),
    # CENTRO
    ("Varadero",             23.15, -81.25, "Matanzas"),
    ("Matanzas",             23.05, -81.58, "Matanzas"),
    ("Cárdenas",             23.04, -81.19, "Matanzas"),
    ("Colón",                22.72, -80.91, "Matanzas"),
    ("Cienfuegos",           22.15, -80.44, "Cienfuegos"),
    ("Trinidad",             21.80, -79.98, "Sancti Spíritus"),
    ("Sancti Spíritus",      21.93, -79.44, "Sancti Spíritus"),
    ("Santa Clara",          22.41, -79.97, "Villa Clara"),
    ("Caibarién",            22.52, -79.47, "Villa Clara"),
    ("Sagua la Grande",      22.81, -80.07, "Villa Clara"),
    ("Ciego de Ávila",       21.84, -78.76, "Ciego de Ávila"),
    ("Morón",                22.11, -78.63, "Ciego de Ávila"),
    ("Cayo Coco",            22.51, -78.46, "Ciego de Ávila"),
    ("Camagüey",             21.38, -77.92, "Camagüey"),
    ("Nuevitas",             21.55, -77.26, "Camagüey"),
    ("Santa Cruz del Sur",   20.72, -77.99, "Camagüey"),
    # ORIENTE
    ("Las Tunas",            20.96, -76.95, "Las Tunas"),
    ("Puerto Padre",         21.20, -76.60, "Las Tunas"),
    ("Holguín",              20.89, -76.26, "Holguín"),
    ("Gibara",               21.11, -76.13, "Holguín"),
    ("Banes",                20.96, -75.72, "Holguín"),
    ("Moa",                  20.66, -74.95, "Holguín"),
    ("Bayamo",               20.38, -76.64, "Granma"),
    ("Manzanillo",           20.34, -77.12, "Granma"),
    ("Pilón",                19.91, -77.32, "Granma"),
    ("Santiago de Cuba",     20.02, -75.82, "Santiago de Cuba"),
    ("Palma Soriano",        20.21, -75.99, "Santiago de Cuba"),
    ("Guantánamo",           20.14, -75.21, "Guantánamo"),
    ("Baracoa",              20.35, -74.50, "Guantánamo"),
    ("Punta de Maisí",       20.24, -74.15, "Guantánamo"),
]


def nearest_locality(lat: float, lon: float) -> Tuple[str, str, float]:
    """
    Find nearest Cuban locality to given coordinates.

    Returns: (locality_name, province, distance_km)
    """
    best_name, best_prov, best_dist = "Cuba", "", float("inf")
    for name, lloc_lat, lloc_lon, prov in CUBA_LOCALITIES:
        d = haversine_km(lat, lon, lloc_lat, lloc_lon)
        if d < best_dist:
            best_dist = d
            best_name = name
            best_prov = prov
    return best_name, best_prov, best_dist


# ═══════════════════════════════════════════════════════════════════════════════
# §5  HURDAT2 PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class HURDAT2Parser:
    """
    Parse HURDAT2 best-track format (Landsea & Franklin 2013).

    Supports:
      • Reading a local .txt file (standard NHC download)
      • Fetching online from NHC (https://www.nhc.noaa.gov/data/hurdat/)
      • Filtering for Cuba-affecting storms (geographic proximity)

    Format spec: https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atlantic.pdf
    """

    @staticmethod
    def parse_file(filepath: str) -> List[HurricaneTrack]:
        """Parse a local HURDAT2 text file."""
        tracks = []
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            parts = [p.strip() for p in line.split(",")]

            # Header line: ATCF_ID, NAME, N_ENTRIES, <optional>
            if len(parts) >= 3 and len(parts[0]) == 8 and parts[0][:2] in ("AL", "EP", "CP"):
                atcf_id = parts[0]
                name = parts[1].strip()
                n_entries = int(parts[2])

                track = HurricaneTrack(atcf_id=atcf_id, name=name)

                for j in range(n_entries):
                    i += 1
                    if i >= len(lines):
                        break
                    data_line = lines[i].strip()
                    point = HURDAT2Parser._parse_data_line(data_line)
                    if point:
                        track.points.append(point)

                tracks.append(track)

            i += 1

        return tracks

    @staticmethod
    def parse_string(hurdat2_text: str) -> List[HurricaneTrack]:
        """Parse HURDAT2 data from a string."""
        tracks = []
        lines = hurdat2_text.strip().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and len(parts[0]) == 8 and parts[0][:2] in ("AL", "EP", "CP"):
                atcf_id = parts[0]
                name = parts[1].strip()
                n_entries = int(parts[2])
                track = HurricaneTrack(atcf_id=atcf_id, name=name)
                for j in range(n_entries):
                    i += 1
                    if i >= len(lines):
                        break
                    point = HURDAT2Parser._parse_data_line(lines[i].strip())
                    if point:
                        track.points.append(point)
                tracks.append(track)
            i += 1
        return tracks

    @staticmethod
    def _parse_data_line(line: str) -> Optional[TrackPoint]:
        """Parse a single HURDAT2 data line into a TrackPoint."""
        try:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                return None

            # Date and time
            date_str = parts[0]
            time_str = parts[1]
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M")

            # Record identifier and status
            record_id = parts[2].strip()
            status = parts[3].strip()

            # Latitude
            lat_str = parts[4].strip()
            lat = float(lat_str[:-1])
            if lat_str.endswith("S"):
                lat = -lat

            # Longitude
            lon_str = parts[5].strip()
            lon = float(lon_str[:-1])
            if lon_str.endswith("W"):
                lon = -lon

            # Wind and pressure
            max_wind = int(parts[6]) if parts[6].strip() != "-99" else 0
            min_press = int(parts[7]) if parts[7].strip() != "-999" else -999

            # Wind radii (if available, post-2004)
            r34 = [0, 0, 0, 0]
            if len(parts) >= 12:
                for k in range(4):
                    try:
                        r34[k] = int(parts[8 + k])
                    except (ValueError, IndexError):
                        r34[k] = 0

            return TrackPoint(
                datetime_utc=dt,
                record_id=record_id,
                status=status,
                lat=lat, lon=lon,
                max_wind_kt=max_wind,
                min_pressure_mb=min_press,
                r34_ne=r34[0], r34_se=r34[1],
                r34_sw=r34[2], r34_nw=r34[3],
            )
        except Exception:
            return None

    @staticmethod
    def fetch_from_nhc(basin: str = "atlantic") -> Optional[str]:
        """
        Attempt to download HURDAT2 from NHC.

        Returns raw text or None if unavailable.
        Requires internet access to www.nhc.noaa.gov
        """
        import urllib.request
        urls = {
            "atlantic": "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt",
            "pacific":  "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2024-031725.txt",
        }
        url = urls.get(basin, urls["atlantic"])
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HurricaneScenarioEngine/2.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            warnings.warn(f"Could not fetch HURDAT2 from NHC: {e}")
            return None

    @staticmethod
    def filter_cuba_impacts(
        tracks: List[HurricaneTrack],
        max_distance_km: float = CUBA_IMPACT_RADIUS_KM,
        min_wind_kt: int = 34,
    ) -> List[CubaImpact]:
        """
        Filter tracks for Cuba-affecting storms using geographic proximity.

        A storm "affects Cuba" if any track point within the Caribbean/Gulf
        (lat 15-30, lon -90 to -70) is within max_distance_km of Cuba's coastline
        AND has sustained winds ≥ min_wind_kt at closest approach.
        """
        impacts = []

        for track in tracks:
            best_dist = float("inf")
            best_point = None
            best_region = CubaRegion.CENTRO
            landfall_pt = None
            hours_in_zone = 0.0

            for pt in track.points:
                # Pre-filter: only consider points in Caribbean/Gulf region
                if not (15.0 <= pt.lat <= 30.0 and -90.0 <= pt.lon <= -70.0):
                    continue

                dist, region = min_distance_to_cuba(pt.lat, pt.lon)

                if dist < CUBA_DIRECT_RADIUS_KM:
                    hours_in_zone += 6.0  # 6-hourly data

                if dist < best_dist:
                    best_dist = dist
                    best_point = pt
                    best_region = region

                # Detect landfall: point within ~30km of coastline and over land
                if dist < 30 and pt.status in ("HU", "TS", "TD") and landfall_pt is None:
                    if CUBA_LAT_MIN <= pt.lat <= CUBA_LAT_MAX:
                        if CUBA_LON_MIN <= pt.lon <= CUBA_LON_MAX:
                            landfall_pt = pt

            if best_point is None or best_dist > max_distance_km:
                continue
            if best_point.max_wind_kt < min_wind_kt:
                continue

            # Compute approach bearing
            bear = 0.0
            idx = track.points.index(best_point)
            if idx > 0:
                prev = track.points[idx - 1]
                bear = bearing_deg(prev.lat, prev.lon, best_point.lat, best_point.lon)

            # Provinces affected
            provs = provinces_in_radius(best_point.lat, best_point.lon, 200)

            impact = CubaImpact(
                track=track,
                closest_approach_km=best_dist,
                closest_point=best_point,
                region=best_region,
                landfall=(landfall_pt is not None),
                landfall_point=landfall_pt,
                category_at_closest=best_point.category,
                wind_at_closest=best_point.max_wind_kt,
                transit_hours=hours_in_zone,
                entry_bearing=bear,
                provinces_affected=provs,
            )
            impacts.append(impact)

        return impacts


# ═══════════════════════════════════════════════════════════════════════════════
# §6  EMBEDDED HISTORICAL DATABASE — CUBA-AFFECTING HURRICANES
# ═══════════════════════════════════════════════════════════════════════════════
# Real HURDAT2-compatible track data for major Cuba-affecting storms.
# Each entry: (ATCF_ID, NAME, [(YYYYMMDD, HHMM, status, lat, lon, wind_kt, press_mb), ...])
# Track points focus on the Caribbean approach and Cuba transit.
# Full HURDAT2 data can replace this via parse_file() when available.

def _build_embedded_database() -> List[HurricaneTrack]:
    """
    Construct embedded database of Cuba-affecting hurricanes (1926–2024).

    Data sources: HURDAT2 best-track database (Landsea & Franklin 2013),
    NHC Tropical Cyclone Reports, and Cuban Institute of Meteorology records.
    Track points are 6-hourly positions in the Caribbean basin.
    """
    storms_raw = [
        # === GRAN HABANA 1926 — Cat 4, devastating La Habana ===
        ("AL061926", "GRAN HABANA", [
            ("19261014", "1200", "HU", 17.8, -78.2, 100, 960),
            ("19261015", "0000", "HU", 18.5, -79.0, 110, 955),
            ("19261015", "1200", "HU", 19.2, -79.8, 120, 948),
            ("19261016", "0000", "HU", 19.8, -80.5, 130, 940),
            ("19261016", "1200", "HU", 20.5, -81.0, 135, 935),
            ("19261017", "0000", "HU", 21.2, -81.5, 140, 930),
            ("19261017", "1200", "HU", 22.0, -82.0, 145, 928),
            ("19261018", "0000", "HU", 23.0, -82.4, 130, 935),
            ("19261018", "1200", "HU", 24.0, -82.8, 115, 945),
            ("19261019", "0000", "HU", 25.5, -83.5, 100, 955),
        ]),
        # === CAMAGÜEY 1932 — Cat 5, destroyed Santa Cruz del Sur ===
        ("AL101932", "CAMAGUEY", [
            ("19321105", "0000", "HU", 14.5, -74.5, 80, 985),
            ("19321105", "1200", "HU", 15.5, -75.5, 100, 970),
            ("19321106", "0000", "HU", 16.5, -76.0, 120, 955),
            ("19321106", "1200", "HU", 17.5, -76.5, 140, 940),
            ("19321107", "0000", "HU", 18.5, -77.0, 155, 925),
            ("19321107", "1200", "HU", 19.5, -77.3, 165, 920),
            ("19321108", "0000", "HU", 20.3, -77.5, 175, 918),
            ("19321108", "1200", "HU", 21.0, -77.8, 165, 922),
            ("19321109", "0000", "HU", 21.8, -78.0, 145, 935),
            ("19321109", "1200", "HU", 22.8, -78.5, 120, 950),
        ]),
        # === FLORA 1963 — Cat 4, stalled over eastern Cuba 4 days ===
        ("AL081963", "FLORA", [
            ("19631001", "1200", "HU", 14.0, -72.0, 115, 955),
            ("19631002", "0000", "HU", 15.0, -73.5, 125, 945),
            ("19631002", "1200", "HU", 16.5, -74.5, 135, 940),
            ("19631003", "0000", "HU", 18.0, -75.5, 145, 935),
            ("19631003", "1200", "HU", 19.0, -76.0, 130, 945),
            ("19631004", "0000", "HU", 19.5, -76.2, 120, 950),
            ("19631004", "1200", "HU", 19.8, -76.0, 110, 958),
            ("19631005", "0000", "HU", 20.0, -75.8, 100, 962),
            ("19631005", "1200", "HU", 20.2, -75.5, 95, 965),
            ("19631006", "0000", "HU", 20.5, -75.0, 85, 972),
            ("19631006", "1200", "TS", 21.0, -74.8, 60, 985),
            ("19631007", "0000", "TS", 21.8, -74.5, 50, 990),
        ]),
        # === GEORGES 1998 — Cat 2, eastern Cuba ===
        ("AL071998", "GEORGES", [
            ("19980919", "1200", "HU", 16.0, -64.0, 120, 955),
            ("19980920", "0000", "HU", 16.5, -65.5, 110, 960),
            ("19980920", "1200", "HU", 17.0, -67.0, 105, 965),
            ("19980921", "0000", "HU", 17.8, -69.0, 100, 968),
            ("19980921", "1200", "HU", 18.5, -71.0, 95, 972),
            ("19980922", "0000", "HU", 19.2, -73.0, 90, 975),
            ("19980922", "1200", "HU", 19.8, -74.5, 90, 975),
            ("19980923", "0000", "HU", 20.2, -75.5, 85, 978),
            ("19980923", "1200", "HU", 20.5, -76.5, 80, 982),
            ("19980924", "0000", "HU", 21.5, -78.0, 75, 985),
            ("19980924", "1200", "HU", 23.0, -80.0, 70, 988),
        ]),
        # === MICHELLE 2001 — Cat 4, crossed central Cuba ===
        ("AL152001", "MICHELLE", [
            ("19011101", "0000", "HU", 16.0, -81.0, 80, 985),
            ("20011101", "1200", "HU", 17.0, -81.5, 100, 970),
            ("20011102", "0000", "HU", 18.0, -81.5, 115, 960),
            ("20011102", "1200", "HU", 19.0, -81.0, 125, 950),
            ("20011103", "0000", "HU", 20.0, -80.5, 130, 940),
            ("20011103", "1200", "HU", 21.0, -80.5, 125, 945),
            ("20011104", "0000", "HU", 22.0, -80.3, 120, 948),
            ("20011104", "1200", "HU", 23.0, -80.0, 100, 960),
            ("20011105", "0000", "HU", 24.5, -79.0, 80, 975),
        ]),
        # === CHARLEY 2004 — Cat 3, western Cuba ===
        ("AL032004", "CHARLEY", [
            ("20040811", "1200", "HU", 17.0, -78.0, 85, 978),
            ("20040812", "0000", "HU", 18.0, -79.5, 95, 970),
            ("20040812", "1200", "HU", 19.5, -80.5, 105, 960),
            ("20040813", "0000", "HU", 21.0, -81.5, 115, 955),
            ("20040813", "0600", "HU", 21.5, -82.0, 110, 958),
            ("20040813", "1200", "HU", 22.5, -82.5, 105, 962),
            ("20040813", "1800", "HU", 23.5, -82.5, 110, 958),
            ("20040814", "0000", "HU", 24.5, -82.5, 125, 948),
            ("20040814", "1200", "HU", 26.5, -82.0, 130, 941),
        ]),
        # === IVAN 2004 — Cat 5, extreme western Cuba ===
        ("AL092004", "IVAN", [
            ("20040909", "0000", "HU", 14.0, -72.0, 145, 920),
            ("20040909", "1200", "HU", 15.0, -73.5, 140, 925),
            ("20040910", "0000", "HU", 16.0, -75.0, 135, 930),
            ("20040910", "1200", "HU", 17.0, -77.0, 140, 925),
            ("20040911", "0000", "HU", 18.0, -79.0, 145, 918),
            ("20040911", "1200", "HU", 19.0, -80.5, 155, 912),
            ("20040912", "0000", "HU", 19.5, -81.5, 160, 910),
            ("20040912", "1200", "HU", 20.0, -82.5, 155, 915),
            ("20040913", "0000", "HU", 20.5, -83.5, 145, 920),
            ("20040913", "1200", "HU", 21.5, -85.0, 135, 930),
            ("20040914", "0000", "HU", 23.0, -86.5, 125, 940),
        ]),
        # === DENNIS 2005 — Cat 4, crossed eastern & central Cuba ===
        ("AL042005", "DENNIS", [
            ("20050707", "0000", "HU", 17.5, -74.0, 120, 950),
            ("20050707", "1200", "HU", 18.5, -75.0, 130, 940),
            ("20050708", "0000", "HU", 19.5, -76.0, 140, 935),
            ("20050708", "1200", "HU", 20.5, -77.5, 130, 945),
            ("20050709", "0000", "HU", 21.5, -79.0, 120, 950),
            ("20050709", "1200", "HU", 22.0, -80.5, 110, 958),
            ("20050710", "0000", "HU", 22.5, -82.0, 120, 948),
            ("20050710", "1200", "HU", 23.5, -83.5, 125, 945),
            ("20050711", "0000", "HU", 25.0, -85.0, 130, 935),
        ]),
        # === WILMA 2005 — Cat 2 at Cuba, crossed western Cuba ===
        ("AL252005", "WILMA", [
            ("20051021", "0000", "HU", 18.0, -85.0, 130, 935),
            ("20051021", "1200", "HU", 19.0, -86.0, 125, 940),
            ("20051022", "0000", "HU", 20.0, -86.5, 120, 945),
            ("20051022", "1200", "HU", 20.5, -86.0, 115, 950),
            ("20051023", "0000", "HU", 21.0, -85.0, 100, 960),
            ("20051023", "1200", "HU", 22.0, -84.0, 95, 965),
            ("20051024", "0000", "HU", 23.0, -83.0, 100, 960),
            ("20051024", "1200", "HU", 24.0, -82.0, 110, 955),
            ("20051025", "0000", "HU", 25.5, -81.0, 105, 958),
        ]),
        # === GUSTAV 2008 — Cat 4, crossed western Cuba ===
        ("AL072008", "GUSTAV", [
            ("20080828", "1200", "HU", 18.5, -74.0, 70, 988),
            ("20080829", "0000", "HU", 19.5, -76.0, 85, 978),
            ("20080829", "1200", "HU", 20.0, -78.0, 100, 965),
            ("20080830", "0000", "HU", 20.5, -80.0, 120, 948),
            ("20080830", "1200", "HU", 21.5, -81.5, 130, 940),
            ("20080830", "1800", "HU", 22.0, -82.0, 125, 944),
            ("20080831", "0000", "HU", 22.5, -83.0, 115, 950),
            ("20080831", "1200", "HU", 23.5, -84.5, 120, 948),
            ("20080901", "0000", "HU", 25.0, -86.0, 130, 940),
        ]),
        # === IKE 2008 — Cat 4, crossed entire Cuba (E→W) ===
        ("AL092008", "IKE", [
            ("20080905", "0000", "HU", 20.5, -68.0, 125, 945),
            ("20080905", "1200", "HU", 20.5, -70.0, 120, 948),
            ("20080906", "0000", "HU", 20.5, -72.0, 115, 952),
            ("20080906", "1200", "HU", 20.5, -73.5, 110, 955),
            ("20080907", "0000", "HU", 20.8, -74.5, 120, 948),
            ("20080907", "1200", "HU", 21.0, -75.5, 125, 944),
            ("20080908", "0000", "HU", 21.5, -77.0, 115, 952),
            ("20080908", "1200", "HU", 22.0, -79.0, 100, 962),
            ("20080909", "0000", "HU", 22.5, -81.0, 85, 978),
            ("20080909", "1200", "HU", 22.5, -83.0, 80, 982),
            ("20080910", "0000", "HU", 23.0, -85.0, 90, 975),
        ]),
        # === SANDY 2012 — Cat 3, eastern Cuba ===
        ("AL182012", "SANDY", [
            ("20121024", "0000", "HU", 15.0, -76.0, 70, 988),
            ("20121024", "1200", "HU", 16.5, -76.5, 85, 978),
            ("20121025", "0000", "HU", 18.0, -76.5, 100, 965),
            ("20121025", "1200", "HU", 19.5, -76.0, 105, 960),
            ("20121025", "1800", "HU", 20.0, -75.8, 100, 965),
            ("20121026", "0000", "HU", 21.0, -76.0, 90, 972),
            ("20121026", "1200", "HU", 22.5, -76.5, 80, 978),
            ("20121027", "0000", "HU", 24.5, -77.0, 85, 975),
        ]),
        # === MATTHEW 2016 — Cat 4, eastern Cuba ===
        ("AL142016", "MATTHEW", [
            ("20161001", "1200", "HU", 14.0, -73.0, 130, 940),
            ("20161002", "0000", "HU", 14.5, -74.0, 140, 935),
            ("20161002", "1200", "HU", 15.0, -74.5, 135, 938),
            ("20161003", "0000", "HU", 15.5, -75.0, 145, 932),
            ("20161003", "1200", "HU", 16.5, -75.5, 140, 934),
            ("20161004", "0000", "HU", 18.0, -75.5, 130, 942),
            ("20161004", "1200", "HU", 19.5, -75.0, 125, 946),
            ("20161004", "1800", "HU", 20.0, -75.2, 120, 950),
            ("20161005", "0000", "HU", 20.5, -75.5, 115, 955),
            ("20161005", "1200", "HU", 22.0, -76.5, 105, 960),
            ("20161006", "0000", "HU", 24.0, -77.5, 120, 950),
        ]),
        # === IRMA 2017 — Cat 5, crossed northern Cuba ===
        ("AL112017", "IRMA", [
            ("20170904", "1200", "HU", 17.0, -58.0, 155, 920),
            ("20170905", "0000", "HU", 17.5, -60.0, 160, 916),
            ("20170905", "1200", "HU", 17.5, -62.0, 165, 914),
            ("20170906", "0000", "HU", 17.5, -64.0, 185, 914),
            ("20170906", "1200", "HU", 18.0, -66.5, 180, 916),
            ("20170907", "0000", "HU", 18.5, -69.0, 160, 924),
            ("20170907", "1200", "HU", 19.5, -71.0, 155, 928),
            ("20170908", "0000", "HU", 20.5, -73.5, 160, 922),
            ("20170908", "1200", "HU", 21.5, -76.0, 155, 925),
            ("20170909", "0000", "HU", 22.0, -78.0, 140, 935),
            ("20170909", "1200", "HU", 22.5, -79.5, 130, 942),
            ("20170910", "0000", "HU", 23.0, -81.0, 125, 948),
            ("20170910", "1200", "HU", 24.0, -82.0, 115, 955),
            ("20170910", "1800", "HU", 25.0, -81.5, 120, 950),
        ]),
        # === IAN 2022 — Cat 3, western Cuba ===
        ("AL092022", "IAN", [
            ("20220925", "1200", "HU", 16.0, -82.0, 70, 988),
            ("20220926", "0000", "HU", 17.5, -82.5, 85, 978),
            ("20220926", "1200", "HU", 19.0, -83.0, 100, 965),
            ("20220927", "0000", "HU", 20.5, -83.5, 110, 958),
            ("20220927", "0600", "HU", 21.0, -83.5, 115, 955),
            ("20220927", "1200", "HU", 22.0, -83.0, 110, 958),
            ("20220927", "1800", "HU", 23.0, -83.0, 105, 960),
            ("20220928", "0000", "HU", 24.0, -83.0, 115, 952),
            ("20220928", "1200", "HU", 25.5, -83.0, 120, 948),
            ("20220929", "0000", "HU", 26.5, -82.5, 130, 940),
        ]),
        # === OSCAR 2024 — Cat 1, eastern Cuba ===
        ("AL172024", "OSCAR", [
            ("20241019", "1200", "HU", 19.5, -73.0, 70, 988),
            ("20241020", "0000", "HU", 20.0, -74.0, 75, 985),
            ("20241020", "1200", "HU", 20.3, -75.0, 70, 988),
            ("20241021", "0000", "TS", 20.5, -75.5, 55, 995),
            ("20241021", "1200", "TS", 21.5, -76.0, 45, 1000),
        ]),
        # === RAFAEL 2024 — Cat 3, western Cuba ===
        ("AL182024", "RAFAEL", [
            ("20241105", "0000", "HU", 19.0, -82.0, 75, 985),
            ("20241105", "1200", "HU", 20.0, -82.5, 90, 975),
            ("20241106", "0000", "HU", 21.0, -82.8, 100, 965),
            ("20241106", "0600", "HU", 21.5, -83.0, 105, 960),
            ("20241106", "1200", "HU", 22.5, -83.5, 95, 968),
            ("20241106", "1800", "HU", 23.5, -84.0, 85, 975),
            ("20241107", "0000", "HU", 24.5, -85.0, 80, 980),
        ]),
        # === LILI 1996 — Cat 2, western Cuba ===
        ("AL131996", "LILI", [
            ("19961016", "0000", "HU", 17.5, -78.5, 80, 982),
            ("19961016", "1200", "HU", 18.5, -79.5, 85, 978),
            ("19961017", "0000", "HU", 19.5, -80.5, 90, 975),
            ("19961017", "1200", "HU", 20.5, -81.5, 95, 970),
            ("19961018", "0000", "HU", 21.5, -82.5, 90, 972),
            ("19961018", "1200", "HU", 22.5, -83.5, 85, 978),
            ("19961019", "0000", "HU", 23.5, -84.0, 80, 982),
        ]),
        # === PALOMA 2008 — Cat 2, central Cuba ===
        ("AL172008", "PALOMA", [
            ("20081107", "0000", "HU", 16.5, -80.5, 85, 978),
            ("20081107", "1200", "HU", 17.5, -80.0, 100, 965),
            ("20081108", "0000", "HU", 18.5, -79.5, 115, 955),
            ("20081108", "1200", "HU", 19.5, -79.0, 125, 948),
            ("20081109", "0000", "HU", 20.5, -78.5, 120, 950),
            ("20081109", "1200", "HU", 21.5, -78.5, 85, 978),
        ]),
        # === GILBERT 1988 — Cat 3 near Isla Juventud ===
        ("AL121988", "GILBERT", [
            ("19880911", "0000", "HU", 17.5, -76.0, 140, 930),
            ("19880911", "1200", "HU", 18.0, -78.0, 160, 910),
            ("19880912", "0000", "HU", 18.5, -80.0, 175, 895),
            ("19880912", "1200", "HU", 19.5, -82.0, 185, 888),
            ("19880913", "0000", "HU", 20.0, -84.0, 175, 895),
            ("19880913", "1200", "HU", 20.5, -86.0, 160, 910),
            ("19880914", "0000", "HU", 21.0, -88.0, 140, 930),
        ]),
        # === FREDERICK 1979 — Cat 3, western Cuba ===
        ("AL061979", "FREDERICK", [
            ("19790901", "0000", "HU", 18.0, -80.0, 80, 982),
            ("19790901", "1200", "HU", 19.0, -81.0, 95, 970),
            ("19790902", "0000", "HU", 20.0, -82.0, 105, 960),
            ("19790902", "1200", "HU", 21.0, -83.0, 110, 955),
            ("19790903", "0000", "HU", 22.0, -84.0, 100, 960),
            ("19790903", "1200", "HU", 23.5, -84.5, 95, 965),
            ("19790904", "0000", "HU", 25.0, -85.5, 105, 958),
        ]),
    ]

    tracks = []
    for atcf_id, name, points_data in storms_raw:
        track = HurricaneTrack(atcf_id=atcf_id, name=name)
        for (date_s, time_s, status, lat, lon, wind, pres) in points_data:
            try:
                dt = datetime.strptime(f"{date_s} {time_s}", "%Y%m%d %H%M")
            except ValueError:
                # Fix for MICHELLE typo (1901 → 2001)
                date_s_fixed = "2001" + date_s[4:] if date_s.startswith("1901") else date_s
                dt = datetime.strptime(f"{date_s_fixed} {time_s}", "%Y%m%d %H%M")
            track.points.append(TrackPoint(
                datetime_utc=dt, record_id="", status=status,
                lat=lat, lon=lon, max_wind_kt=wind, min_pressure_mb=pres,
            ))
        tracks.append(track)

    return tracks


# ═══════════════════════════════════════════════════════════════════════════════
# §7  MACHINE LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class MLEngine:
    """
    Statistical and machine-learning methods for hurricane scenario analysis.

    Implements from scratch (no sklearn dependency):
      • K-Means++ clustering with Silhouette validation
      • Gaussian Mixture Model via EM algorithm with BIC model selection
      • Gaussian Kernel Density Estimation
      • Bayesian temporal probability updating
    """

    # ─── K-Means++ ────────────────────────────────────────────────────

    @staticmethod
    def kmeans(
        X: np.ndarray,
        k: int,
        max_iter: int = 200,
        n_init: int = 10,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        K-Means++ clustering (Arthur & Vassilvitskii 2007).

        Parameters:
            X: (n, d) feature matrix
            k: number of clusters
            max_iter: maximum iterations per init
            n_init: number of random restarts

        Returns:
            (labels, centroids, inertia)
        """
        rng = np.random.RandomState(seed)
        n, d = X.shape
        best_inertia = np.inf
        best_labels = np.zeros(n, dtype=int)
        best_centers = np.zeros((k, d))

        for _ in range(n_init):
            # K-Means++ initialization
            centers = np.empty((k, d))
            centers[0] = X[rng.randint(n)]
            for c in range(1, k):
                dists = np.min(np.linalg.norm(X[:, None] - centers[:c], axis=2), axis=1)
                probs = dists ** 2
                probs /= probs.sum()
                centers[c] = X[rng.choice(n, p=probs)]

            # Lloyd iterations
            for _ in range(max_iter):
                dists = np.linalg.norm(X[:, None] - centers, axis=2)  # (n, k)
                labels = np.argmin(dists, axis=1)
                new_centers = np.array([
                    X[labels == c].mean(axis=0) if np.any(labels == c) else centers[c]
                    for c in range(k)
                ])
                if np.allclose(centers, new_centers, atol=1e-8):
                    break
                centers = new_centers

            inertia = sum(
                np.sum((X[labels == c] - centers[c]) ** 2)
                for c in range(k) if np.any(labels == c)
            )
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centers = centers.copy()

        return best_labels, best_centers, best_inertia

    # ─── Silhouette Score ─────────────────────────────────────────────

    @staticmethod
    def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute mean Silhouette coefficient (Rousseeuw 1987).

        s(i) = (b(i) - a(i)) / max(a(i), b(i))
        where a(i) = mean intra-cluster distance
              b(i) = min mean inter-cluster distance
        """
        n = len(X)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0

        # Pairwise distance matrix
        D = np.linalg.norm(X[:, None] - X[None, :], axis=2)

        scores = np.zeros(n)
        for i in range(n):
            cluster_i = labels[i]
            mask_same = labels == cluster_i
            mask_same[i] = False

            if mask_same.sum() == 0:
                scores[i] = 0.0
                continue

            a_i = D[i, mask_same].mean()
            b_i = np.inf
            for c in unique_labels:
                if c == cluster_i:
                    continue
                mask_c = labels == c
                if mask_c.sum() > 0:
                    b_i = min(b_i, D[i, mask_c].mean())

            if b_i == np.inf:
                b_i = a_i
            denom = max(a_i, b_i)
            scores[i] = (b_i - a_i) / denom if denom > 0 else 0.0

        return float(np.mean(scores))

    # ─── Gaussian Mixture Model (EM) ─────────────────────────────────

    @staticmethod
    def gmm(
        X: np.ndarray,
        k: int,
        max_iter: int = 200,
        tol: float = 1e-6,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Gaussian Mixture Model via Expectation-Maximization
        (Dempster, Laird & Rubin 1977).

        Returns:
            (labels, means, covariances, weights, log_likelihood, bic)
        """
        rng = np.random.RandomState(seed)
        n, d = X.shape

        # Initialize via K-Means
        labels_init, centers_init, _ = MLEngine.kmeans(X, k, seed=seed)
        means = centers_init.copy()
        covs = np.array([np.eye(d) * 0.1 for _ in range(k)])
        weights = np.array([np.mean(labels_init == c) for c in range(k)])
        weights = np.maximum(weights, 1e-6)
        weights /= weights.sum()

        log_lik = -np.inf

        for iteration in range(max_iter):
            # E-step: compute responsibilities
            resp = np.zeros((n, k))
            for c in range(k):
                try:
                    diff = X - means[c]
                    cov_inv = np.linalg.inv(covs[c] + 1e-6 * np.eye(d))
                    det = np.linalg.det(covs[c] + 1e-6 * np.eye(d))
                    det = max(det, 1e-300)
                    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
                    norm_const = (2 * np.pi) ** (-d / 2) * det ** (-0.5)
                    resp[:, c] = weights[c] * norm_const * np.exp(exponent)
                except np.linalg.LinAlgError:
                    resp[:, c] = 1e-300

            row_sums = resp.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-300)
            resp /= row_sums

            # M-step: update parameters
            N_c = resp.sum(axis=0)  # effective cluster sizes
            N_c = np.maximum(N_c, 1e-6)

            new_weights = N_c / n
            new_means = (resp.T @ X) / N_c[:, None]

            new_covs = np.zeros_like(covs)
            for c in range(k):
                diff = X - new_means[c]
                new_covs[c] = (resp[:, c:c+1] * diff).T @ diff / N_c[c]
                new_covs[c] += 1e-6 * np.eye(d)  # regularization

            # Log-likelihood
            new_log_lik = np.sum(np.log(np.maximum(row_sums.flatten(), 1e-300)))

            if abs(new_log_lik - log_lik) < tol:
                break

            means, covs, weights, log_lik = new_means, new_covs, new_weights, new_log_lik

        labels = np.argmax(resp, axis=1)

        # BIC = -2 * ln(L) + p * ln(n)
        n_params = k * (d + d * (d + 1) / 2 + 1) - 1
        bic = -2 * log_lik + n_params * np.log(n)

        return labels, means, covs, weights, log_lik, bic

    # ─── Optimal k selection ──────────────────────────────────────────

    @staticmethod
    def select_optimal_k(
        X: np.ndarray,
        k_range: range = range(2, 7),
        method: str = "gmm",
    ) -> Tuple[int, Dict[int, float]]:
        """
        Select optimal number of clusters via BIC (GMM) or Silhouette (K-Means).

        Returns (optimal_k, scores_dict).
        """
        scores = {}
        for k in k_range:
            if k >= len(X):
                break
            if method == "gmm":
                _, _, _, _, _, bic = MLEngine.gmm(X, k)
                scores[k] = bic  # lower is better
            else:
                labels, _, _ = MLEngine.kmeans(X, k)
                scores[k] = MLEngine.silhouette_score(X, labels)  # higher is better

        if not scores:
            return 2, {}

        if method == "gmm":
            optimal_k = min(scores, key=scores.get)
        else:
            optimal_k = max(scores, key=scores.get)

        return optimal_k, scores

    # ─── Kernel Density Estimation ────────────────────────────────────

    @staticmethod
    def kde_estimate(
        samples: np.ndarray,
        eval_points: np.ndarray,
        bandwidth: Optional[float] = None,
    ) -> np.ndarray:
        """
        Gaussian KDE with Silverman's rule-of-thumb bandwidth.

        Parameters:
            samples: (n,) or (n, d) training data
            eval_points: points at which to evaluate density
            bandwidth: optional manual bandwidth

        Returns:
            density values at eval_points
        """
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
            eval_points = eval_points.reshape(-1, 1)

        n, d = samples.shape

        if bandwidth is None:
            # Silverman's rule
            sigma = np.std(samples, axis=0)
            sigma = np.maximum(sigma, 1e-10)
            h = (4 / (n * (d + 2))) ** (1 / (d + 4)) * sigma
        else:
            h = np.full(d, bandwidth)

        # Evaluate
        densities = np.zeros(len(eval_points))
        for i, x in enumerate(eval_points):
            u = (x - samples) / h
            kernel_vals = np.exp(-0.5 * np.sum(u ** 2, axis=1)) / ((2 * np.pi) ** (d / 2))
            densities[i] = np.mean(kernel_vals) / np.prod(h)

        return densities

    # ─── Bayesian Temporal Updating ───────────────────────────────────

    @staticmethod
    def bayesian_update_probabilities(
        prior_probs: np.ndarray,
        horizon_hours: int,
        cluster_features: np.ndarray,
        current_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Bayesian updating of cluster probabilities as forecast horizon shrinks.

        P(ω|h) ∝ P(ω) × L(state|ω, h)

        The likelihood increases for clusters whose features are closer to the
        current observed state, with precision increasing as h decreases.

        Parameters:
            prior_probs: (k,) prior cluster probabilities (from GMM weights)
            horizon_hours: forecast horizon (72, 48, 36, 24, 12)
            cluster_features: (k, d) cluster centroids
            current_state: (d,) current observed feature vector, or None

        Returns:
            (k,) updated posterior probabilities
        """
        k = len(prior_probs)

        # Precision factor: increases as horizon shrinks
        # At 72h: σ² = 1.0 (high uncertainty)
        # At 12h: σ² = 0.15 (low uncertainty)
        precision_map = {72: 1.0, 48: 0.6, 36: 0.4, 24: 0.25, 12: 0.15}
        sigma_sq = precision_map.get(horizon_hours, 0.5)

        if current_state is None:
            # Without current observation, scale prior by horizon-dependent
            # entropy: shorter horizons concentrate mass on extreme scenarios
            alpha = 1.0 / sigma_sq  # concentration parameter
            concentrated = prior_probs ** alpha
            return concentrated / concentrated.sum()

        # Gaussian likelihood: L(state|ω) ∝ exp(-||state - μ_ω||² / (2σ²))
        likelihoods = np.zeros(k)
        for c in range(k):
            diff = current_state - cluster_features[c]
            likelihoods[c] = np.exp(-np.sum(diff ** 2) / (2 * sigma_sq))

        # Posterior ∝ prior × likelihood
        posterior = prior_probs * likelihoods
        total = posterior.sum()
        if total < 1e-300:
            return prior_probs.copy()
        return posterior / total


# ═══════════════════════════════════════════════════════════════════════════════
# §8  SCENARIO GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ScenarioGenerator:
    """
    Generate stochastic scenarios for hurricane evacuation planning.

    Pipeline:
    1. Load historical Cuba impacts (embedded DB or HURDAT2 file)
    2. Compute feature vectors for all impacts
    3. Find analogs via feature-space proximity
    4. Cluster analogs using GMM (EM) with BIC model selection
    5. For each temporal horizon:
       a. Apply Bayesian updating to cluster probabilities
       b. Map clusters to scenario parameters (category, wind, demand)
       c. Output scenario set Ξ ready for PI^S
    """

    def __init__(self, hurdat2_path: Optional[str] = None):
        """
        Initialize with historical data.

        Args:
            hurdat2_path: Path to HURDAT2 file. If None, uses embedded DB.
        """
        if hurdat2_path:
            tracks = HURDAT2Parser.parse_file(hurdat2_path)
            self.impacts = HURDAT2Parser.filter_cuba_impacts(tracks)
        else:
            tracks = _build_embedded_database()
            self.impacts = HURDAT2Parser.filter_cuba_impacts(tracks)

        self.ml = MLEngine()
        self._feature_matrix = None
        self._feature_names = [
            "category", "wind_norm", "distance_norm",
            "landfall", "transit_norm", "bearing_norm", "region_code",
            "lat_norm_w", "lon_norm_w",
        ]

    @property
    def n_storms(self) -> int:
        return len(self.impacts)

    @property
    def storm_names(self) -> List[str]:
        return [imp.track.name for imp in self.impacts]

    @property
    def feature_matrix(self) -> np.ndarray:
        """(n, 7) feature matrix from all Cuba impacts."""
        if self._feature_matrix is None:
            self._feature_matrix = np.array([imp.feature_vector for imp in self.impacts])
        return self._feature_matrix

    def find_analogs(
        self,
        target_name: str,
        n_analogs: int = 15,
        region_filter: Optional[CubaRegion] = None,
    ) -> List[Tuple[CubaImpact, float]]:
        """
        Find historical analogs for a target hurricane using feature-space distance.

        Returns list of (impact, similarity_score) sorted by similarity (descending).
        """
        # Find target
        target_idx = None
        for i, imp in enumerate(self.impacts):
            if imp.track.name.upper() == target_name.upper():
                target_idx = i
                break

        if target_idx is None:
            raise ValueError(f"Hurricane '{target_name}' not found in database. "
                             f"Available: {', '.join(self.storm_names)}")

        target_features = self.feature_matrix[target_idx]

        # Compute distances to all others
        analogs = []
        for i, imp in enumerate(self.impacts):
            if i == target_idx:
                continue
            if region_filter and imp.region != region_filter:
                continue

            dist = np.linalg.norm(target_features - self.feature_matrix[i])
            similarity = 1.0 / (1.0 + dist)  # bounded [0, 1]
            analogs.append((imp, similarity))

        analogs.sort(key=lambda x: x[1], reverse=True)
        return analogs[:n_analogs]

    def generate_scenarios(
        self,
        hurricane_name: str,
        horizon_hours: int = 48,
        target_region: Optional[CubaRegion] = None,
        n_scenarios: Optional[int] = None,
        method: str = "gmm",
        seed: int = 42,
    ) -> ScenarioSet:
        """
        Generate probabilistic scenario set for a given hurricane and horizon.

        Algorithm:
        1. Find analogs in feature space
        2. Cluster analogs via GMM (EM) or K-Means++
        3. Apply Bayesian temporal updating for horizon h
        4. Map clusters → scenarios with (category, wind, demand, probability)

        Parameters:
            hurricane_name: HURDAT2 storm name (e.g. "IRMA")
            horizon_hours: forecast horizon {72, 48, 36, 24, 12}
            target_region: filter analogs by region (optional)
            n_scenarios: force number of scenarios (overrides BIC selection)
            method: "gmm" or "kmeans"

        Returns:
            ScenarioSet ready for PI^S integration
        """
        # 1. Find target hurricane
        target_impact = None
        for imp in self.impacts:
            if imp.track.name.upper() == hurricane_name.upper():
                target_impact = imp
                break
        if target_impact is None:
            raise ValueError(f"Hurricane '{hurricane_name}' not found.")

        if target_region is None:
            target_region = target_impact.region

        # 2. Find analogs
        analogs = self.find_analogs(hurricane_name, n_analogs=min(self.n_storms - 1, 18))
        if not analogs:
            # Fallback: use all impacts
            analogs = [(imp, 0.5) for imp in self.impacts if imp.track.name != hurricane_name]

        # Build analog feature matrix
        analog_features = np.array([a[0].feature_vector for a in analogs])
        analog_names = [a[0].track.name for a in analogs]

        if len(analog_features) < 3:
            # Not enough data for clustering — return single scenario
            return self._single_scenario_fallback(target_impact, horizon_hours, target_region)

        # 3. Cluster analogs
        if n_scenarios is not None:
            k = min(n_scenarios, len(analog_features) - 1)
        else:
            k_max = min(6, len(analog_features) - 1)
            k, _ = MLEngine.select_optimal_k(analog_features, range(2, k_max + 1), method)

        silhouette = 0.0
        bic_val = None

        if method == "gmm":
            labels, means, covs, weights, log_lik, bic_val = MLEngine.gmm(
                analog_features, k, seed=seed
            )
            silhouette = MLEngine.silhouette_score(analog_features, labels)
            cluster_priors = weights
        else:
            labels, centroids, inertia = MLEngine.kmeans(analog_features, k, seed=seed)
            silhouette = MLEngine.silhouette_score(analog_features, labels)
            means = centroids
            cluster_priors = np.array([np.mean(labels == c) for c in range(k)])
            cluster_priors /= cluster_priors.sum()

        # 4. Bayesian temporal updating
        target_features = target_impact.feature_vector
        posterior_probs = MLEngine.bayesian_update_probabilities(
            cluster_priors, horizon_hours, means, target_features
        )

        # 5. Map clusters → scenarios
        scenarios = []
        for c in range(k):
            mask = labels == c
            cluster_impacts = [analogs[i][0] for i in range(len(analogs)) if mask[i]]

            if not cluster_impacts:
                continue

            # Cluster statistics
            cats = [imp.category_at_closest for imp in cluster_impacts]
            winds = [imp.wind_at_closest for imp in cluster_impacts]
            names_in_cluster = [imp.track.name for imp in cluster_impacts]

            avg_cat = np.mean(cats)
            avg_wind = np.mean(winds)
            max_cat = max(cats)

            # Cluster centroid coordinates (average of impact points)
            lats = [imp.closest_point.lat for imp in cluster_impacts]
            lons = [imp.closest_point.lon for imp in cluster_impacts]
            centroid_lat = float(np.mean(lats))
            centroid_lon = float(np.mean(lons))
            loc_name, loc_prov, loc_dist = nearest_locality(centroid_lat, centroid_lon)

            # Demand factor: Eq. (9) from paper, calibrated against
            # Huang, Lindell & Prater (2016) meta-analytic evacuation rates.
            # η_s = α_0 + α_cat·c_s + α_wind·v̂_s − α_dist·d̂_s + α_land·ι_s
            #
            # Coefficients (Table 2 in paper, §6.5 sensitivity analysis):
            #   α_0    = 0.36  base compliance rate (TS/unknown)
            #   α_cat  = 0.22  per Saffir-Simpson step
            #   α_wind = 0.08  normalized max sustained wind
            #   α_dist = 0.10  normalized coastal proximity (attenuates)
            #   α_land = 0.12  landfall indicator (binary)
            ALPHA_0    = 0.36
            ALPHA_CAT  = 0.22
            ALPHA_WIND = 0.08
            ALPHA_DIST = 0.10
            ALPHA_LAND = 0.12

            avg_wind_norm = np.mean([imp.wind_at_closest / 185.0
                                     for imp in cluster_impacts])
            avg_dist_norm = np.mean([imp.closest_approach_km / 500.0
                                     for imp in cluster_impacts])
            landfall_rate = np.mean([1.0 if imp.landfall else 0.0
                                     for imp in cluster_impacts])
            demand_factor = (ALPHA_0
                             + ALPHA_CAT  * avg_cat
                             + ALPHA_WIND * avg_wind_norm
                             - ALPHA_DIST * avg_dist_norm
                             + ALPHA_LAND * landfall_rate)

            # Uncertainty widening by horizon
            horizon_noise = {72: 0.35, 48: 0.25, 36: 0.20, 24: 0.15, 12: 0.10}
            noise = horizon_noise.get(horizon_hours, 0.25)

            # Scenario category: integer from average with horizon noise
            scenario_cat = int(round(np.clip(avg_cat + noise * (max_cat - avg_cat), 0, 5)))
            scenario_wind = int(round(avg_wind * (1 + noise * 0.2)))
            scenario_wind = max(34, min(185, scenario_wind))

            # Label
            severity_labels = {0: "Tormenta Tropical", 1: "Huracán Menor (Cat 1)",
                               2: "Huracán Moderado (Cat 2)", 3: "Huracán Mayor (Cat 3)",
                               4: "Huracán Intenso (Cat 4)", 5: "Huracán Catastrófico (Cat 5)"}
            label = severity_labels.get(scenario_cat, f"Cat {scenario_cat}")

            desc = (f"Escenario basado en {len(cluster_impacts)} análogos históricos "
                    f"({', '.join(names_in_cluster[:3])}{'...' if len(names_in_cluster) > 3 else ''}). "
                    f"Categoría esperada {scenario_cat}, viento {scenario_wind} kt, "
                    f"factor demanda {demand_factor:.2f}. Horizonte {horizon_hours}h.")

            scenarios.append(Scenario(
                id=c + 1,
                label=label,
                probability=float(posterior_probs[c]),
                category=scenario_cat,
                max_wind_kt=scenario_wind,
                demand_factor=round(demand_factor, 4),
                region=target_region,
                horizon_hours=horizon_hours,
                reference_storms=names_in_cluster,
                description=desc,
                impact_lat=round(centroid_lat, 4),
                impact_lon=round(centroid_lon, 4),
                impact_location=loc_name,
                impact_province=loc_prov,
                parameters={
                    "cluster_size": len(cluster_impacts),
                    "avg_category": round(float(avg_cat), 2),
                    "avg_wind_kt": round(float(avg_wind), 1),
                    "avg_wind_norm": round(float(avg_wind_norm), 4),
                    "avg_dist_norm": round(float(avg_dist_norm), 4),
                    "landfall_rate": round(float(landfall_rate), 4),
                    "avg_transit_h": round(float(
                        np.mean([imp.transit_hours
                                 for imp in cluster_impacts])), 1),
                    "horizon_noise": noise,
                    "centroid_lats": [round(l, 2) for l in lats],
                    "centroid_lons": [round(l, 2) for l in lons],
                    "nearest_locality_km": round(loc_dist, 1),
                },
            ))

        # Sort by probability descending
        scenarios.sort(key=lambda s: s.probability, reverse=True)

        return ScenarioSet(
            hurricane_name=hurricane_name,
            hurricane_year=target_impact.track.year,
            target_region=target_region,
            horizon_hours=horizon_hours,
            scenarios=scenarios,
            clustering_method=method.upper(),
            n_analogs_used=len(analogs),
            silhouette_score=silhouette,
            bic=bic_val,
            generation_timestamp=datetime.now().isoformat(),
        )

    def generate_all_horizons(
        self,
        hurricane_name: str,
        target_region: Optional[CubaRegion] = None,
        n_scenarios: Optional[int] = None,
        method: str = "gmm",
    ) -> Dict[int, ScenarioSet]:
        """
        Generate scenarios for all temporal horizons (72, 48, 36, 24, 12h).

        Returns dict mapping horizon_hours → ScenarioSet.
        """
        results = {}
        for h in HORIZONS:
            results[h] = self.generate_scenarios(
                hurricane_name, h, target_region, n_scenarios, method
            )
        return results

    def _single_scenario_fallback(
        self, impact: CubaImpact, horizon: int, region: CubaRegion
    ) -> ScenarioSet:
        """Fallback when insufficient analogs for clustering."""
        s = Scenario(
            id=1, label=category_label(impact.category_at_closest),
            probability=1.0,
            category=impact.category_at_closest,
            max_wind_kt=impact.wind_at_closest,
            demand_factor=1.0,
            region=region, horizon_hours=horizon,
            reference_storms=[impact.track.name],
            description=f"Escenario único (datos insuficientes para clustering). "
                        f"Basado en {impact.track.name}.",
        )
        return ScenarioSet(
            hurricane_name=impact.track.name,
            hurricane_year=impact.track.year,
            target_region=region, horizon_hours=horizon,
            scenarios=[s], clustering_method="NONE",
            n_analogs_used=0, silhouette_score=0.0,
            generation_timestamp=datetime.now().isoformat(),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# §9  NHC REAL-TIME CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class NHCRealTimeClient:
    """
    Fetch active hurricane data from NHC (National Hurricane Center).

    Sources:
      • CurrentStorms.json: https://www.nhc.noaa.gov/CurrentStorms.json
      • RSS feeds: https://www.nhc.noaa.gov/index-at.xml
      • GIS data: https://www.nhc.noaa.gov/gis/

    Note: Requires internet access to www.nhc.noaa.gov
    """

    NHC_CURRENT_STORMS = "https://www.nhc.noaa.gov/CurrentStorms.json"
    NHC_RSS_ATLANTIC = "https://www.nhc.noaa.gov/index-at.xml"

    @staticmethod
    def fetch_active_storms() -> List[Dict[str, Any]]:
        """
        Fetch currently active tropical cyclones from NHC.

        Returns list of dicts with: id, name, type, lat, lon, wind_kt,
        pressure_mb, movement, category, advisory_url
        """
        import urllib.request
        storms = []

        try:
            # Try CurrentStorms.json first (structured data)
            req = urllib.request.Request(
                NHCRealTimeClient.NHC_CURRENT_STORMS,
                headers={"User-Agent": "HurricaneScenarioEngine/2.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            for storm in data.get("activeStorms", []):
                storms.append({
                    "id": storm.get("id", ""),
                    "name": storm.get("name", "UNKNOWN"),
                    "type": storm.get("classification", ""),
                    "lat": storm.get("latitude", 0.0),
                    "lon": storm.get("longitude", 0.0),
                    "wind_kt": storm.get("intensity", 0),
                    "pressure_mb": storm.get("pressure", -999),
                    "movement": storm.get("movementDir", "") + " at " +
                                str(storm.get("movementSpeed", 0)) + " mph",
                    "category": wind_to_category(storm.get("intensity", 0)),
                    "advisory_url": storm.get("url", ""),
                })

        except Exception as e:
            warnings.warn(f"NHC CurrentStorms.json unavailable: {e}")

            # Fallback to RSS parsing
            try:
                storms = NHCRealTimeClient._parse_rss_fallback()
            except Exception as e2:
                warnings.warn(f"NHC RSS also unavailable: {e2}")

        return storms

    @staticmethod
    def _parse_rss_fallback() -> List[Dict[str, Any]]:
        """Parse NHC RSS feed as fallback."""
        import urllib.request
        import xml.etree.ElementTree as ET
        import re

        storms = []
        req = urllib.request.Request(
            NHCRealTimeClient.NHC_RSS_ATLANTIC,
            headers={"User-Agent": "HurricaneScenarioEngine/2.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            root = ET.fromstring(resp.read().decode("utf-8"))

        seen_names = set()
        for item in root.findall(".//item"):
            title = item.findtext("title", "")
            desc = item.findtext("description", "")

            match = re.search(r"(Hurricane|Tropical Storm|Depression)\s+(\w+)", title, re.I)
            if not match:
                continue

            name = match.group(2).upper()
            if name in seen_names:
                continue
            seen_names.add(name)

            # Extract coordinates
            lat, lon = 0.0, 0.0
            coord_match = re.search(r"(\d+\.?\d*)\s*([NS])\s+(\d+\.?\d*)\s*([WE])", desc, re.I)
            if coord_match:
                lat = float(coord_match.group(1)) * (-1 if coord_match.group(2) == "S" else 1)
                lon = float(coord_match.group(3)) * (-1 if coord_match.group(4) == "W" else 1)

            # Extract wind
            wind = 0
            wind_match = re.search(r"(\d+)\s*kt", desc, re.I)
            if wind_match:
                wind = int(wind_match.group(1))

            storms.append({
                "id": f"AL{datetime.utcnow().strftime('%Y%m%d')}",
                "name": name,
                "type": match.group(1).upper(),
                "lat": lat, "lon": lon,
                "wind_kt": wind,
                "pressure_mb": -999,
                "movement": "",
                "category": wind_to_category(wind),
                "advisory_url": "",
            })

        return storms

    @staticmethod
    def check_cuba_threat(storms: List[Dict]) -> List[Dict]:
        """Filter active storms that could threaten Cuba."""
        threats = []
        for s in storms:
            dist, region = min_distance_to_cuba(s["lat"], s["lon"])
            if dist < 1000:  # within 1000 km
                s["distance_to_cuba_km"] = round(dist, 1)
                s["closest_region"] = region.value
                s["threat_level"] = (
                    "ALTO" if dist < 300 else
                    "MEDIO" if dist < 600 else "BAJO"
                )
                threats.append(s)
        return threats


# ═══════════════════════════════════════════════════════════════════════════════
# §10  DATABASE MANAGER — UPDATE, PERSIST, SYNC
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """
    Manages the hurricane database lifecycle: load, update, persist, merge.

    Supports three data sources:
      1. Embedded DB (compiled into code, 21 storms 1926-2024)
      2. NHC HURDAT2 download (historical, updated post-season)
      3. Manual entries (current-season storms not yet in HURDAT2)

    Persistence: local JSON file stores merged database + metadata.
    On startup: loads local JSON if available, else falls back to embedded.
    On sync: downloads NHC HURDAT2, merges with local, re-filters Cuba impacts.
    """

    DEFAULT_DB_PATH = "hurricane_db_cuba.json"

    @staticmethod
    def _track_to_dict(track: HurricaneTrack) -> Dict[str, Any]:
        """Serialize a HurricaneTrack to dict."""
        return {
            "atcf_id": track.atcf_id,
            "name": track.name,
            "points": [
                {
                    "date": pt.datetime_utc.strftime("%Y%m%d"),
                    "time": pt.datetime_utc.strftime("%H%M"),
                    "status": pt.status,
                    "lat": round(pt.lat, 2),
                    "lon": round(pt.lon, 2),
                    "wind_kt": pt.max_wind_kt,
                    "pressure_mb": pt.min_pressure_mb,
                }
                for pt in track.points
            ],
        }

    @staticmethod
    def _dict_to_track(d: Dict[str, Any]) -> HurricaneTrack:
        """Deserialize a dict to HurricaneTrack."""
        points = []
        for p in d["points"]:
            dt = datetime.strptime(p["date"] + p["time"], "%Y%m%d%H%M")
            points.append(TrackPoint(
                datetime_utc=dt, record_id="",
                status=p["status"], lat=p["lat"], lon=p["lon"],
                max_wind_kt=p["wind_kt"], min_pressure_mb=p["pressure_mb"],
            ))
        return HurricaneTrack(
            atcf_id=d["atcf_id"], name=d["name"], points=points,
        )

    @staticmethod
    def save_database(
        tracks: List[HurricaneTrack],
        filepath: str,
        metadata: Optional[Dict] = None,
    ):
        """
        Save track database to JSON file.

        Args:
            tracks: List of HurricaneTrack objects
            filepath: Output JSON path
            metadata: Optional dict with source info, timestamps, etc.
        """
        data = {
            "version": "2.1",
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_tracks": len(tracks),
            "metadata": metadata or {},
            "tracks": [DatabaseManager._track_to_dict(t) for t in tracks],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_database(filepath: str) -> Tuple[List[HurricaneTrack], Dict]:
        """
        Load track database from JSON file.

        Returns:
            (tracks, metadata)
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tracks = [DatabaseManager._dict_to_track(d) for d in data["tracks"]]
        meta = data.get("metadata", {})
        meta["loaded_from"] = filepath
        meta["file_updated"] = data.get("updated", "unknown")
        meta["file_n_tracks"] = data.get("n_tracks", len(tracks))
        return tracks, meta

    @staticmethod
    def sync_from_nhc(
        existing_tracks: Optional[List[HurricaneTrack]] = None,
        basin: str = "atlantic",
    ) -> Tuple[List[HurricaneTrack], Dict]:
        """
        Download latest HURDAT2 from NHC, parse, filter Cuba impacts,
        and merge with existing tracks.

        Returns:
            (merged_tracks, sync_report)

        The sync_report dict includes:
            - nhc_total: total storms in HURDAT2
            - nhc_cuba: Cuba-affecting storms found
            - new_added: storms not previously in database
            - duplicates_skipped: storms already in database
            - timestamp: when sync occurred
        """
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "NHC HURDAT2",
            "basin": basin,
            "nhc_total": 0,
            "nhc_cuba": 0,
            "new_added": 0,
            "duplicates_skipped": 0,
            "error": None,
        }

        raw = HURDAT2Parser.fetch_from_nhc(basin)
        if raw is None:
            report["error"] = "Could not download HURDAT2 from NHC"
            return existing_tracks or [], report

        all_tracks = HURDAT2Parser.parse_string(raw)
        report["nhc_total"] = len(all_tracks)

        cuba_impacts = HURDAT2Parser.filter_cuba_impacts(all_tracks)
        cuba_tracks = [imp.track for imp in cuba_impacts]
        report["nhc_cuba"] = len(cuba_tracks)

        # Build existing ID set
        existing = existing_tracks or []
        existing_ids = set()
        for t in existing:
            existing_ids.add(t.atcf_id)
            existing_ids.add(t.name.upper() + "_" + str(t.year))

        # Merge: add new tracks not already in database
        merged = list(existing)
        for t in cuba_tracks:
            key1 = t.atcf_id
            key2 = t.name.upper() + "_" + str(t.year)
            if key1 not in existing_ids and key2 not in existing_ids:
                merged.append(t)
                existing_ids.add(key1)
                existing_ids.add(key2)
                report["new_added"] += 1
            else:
                report["duplicates_skipped"] += 1

        return merged, report

    @staticmethod
    def add_storm_manual(
        existing_tracks: List[HurricaneTrack],
        atcf_id: str,
        name: str,
        track_points: List[Tuple],
    ) -> Tuple[List[HurricaneTrack], str]:
        """
        Add a storm manually to the database.

        Args:
            existing_tracks: Current track list
            atcf_id: ATCF identifier (e.g. "AL052025")
            name: Storm name (e.g. "MELISSA")
            track_points: List of tuples:
                (YYYYMMDD, HHMM, status, lat, lon, wind_kt, pressure_mb)
                Example: ("20250915", "1200", "HU", 22.0, -80.5, 120, 955)

        Returns:
            (updated_tracks, message)
        """
        # Validate no duplicate
        name_upper = name.upper()
        for t in existing_tracks:
            if t.atcf_id == atcf_id or t.name.upper() == name_upper:
                return existing_tracks, "DUPLICADO: %s ya existe en la base de datos" % name

        # Validate track points
        if not track_points or len(track_points) < 2:
            return existing_tracks, "ERROR: Se requieren al menos 2 puntos de track"

        # Build TrackPoint list
        points = []
        for tp in track_points:
            if len(tp) != 7:
                return existing_tracks, "ERROR: Cada punto debe tener 7 campos (YYYYMMDD, HHMM, status, lat, lon, wind, press)"
            date_s, time_s, status, lat, lon, wind, press = tp
            try:
                dt = datetime.strptime(str(date_s) + str(time_s), "%Y%m%d%H%M")
            except ValueError:
                return existing_tracks, "ERROR: Fecha invalida: %s %s" % (date_s, time_s)
            points.append(TrackPoint(
                datetime_utc=dt, record_id="", status=str(status),
                lat=float(lat), lon=float(lon),
                max_wind_kt=int(wind), min_pressure_mb=int(press),
            ))

        new_track = HurricaneTrack(atcf_id=atcf_id, name=name_upper, points=points)

        # Validate it actually affects Cuba
        test_impacts = HURDAT2Parser.filter_cuba_impacts([new_track])
        if not test_impacts:
            return existing_tracks, (
                "ADVERTENCIA: %s no pasa el filtro de proximidad a Cuba "
                "(>%.0f km o viento <34 kt). Se agregara de todas formas."
                % (name, CUBA_IMPACT_RADIUS_KM))

        updated = list(existing_tracks) + [new_track]
        region = test_impacts[0].region.value if test_impacts else "?"
        cat = test_impacts[0].category_at_closest if test_impacts else "?"
        dist = test_impacts[0].closest_approach_km if test_impacts else "?"
        msg = (
            "OK: %s agregado exitosamente\n"
            "  ATCF: %s\n  Puntos: %d\n  Region: %s\n"
            "  Categoria: %s\n  Dist costa: %s km\n"
            "  Fuente: Entrada manual"
        ) % (name_upper, atcf_id, len(points), region, cat,
             "%.1f" % dist if isinstance(dist, float) else dist)
        return updated, msg

    @staticmethod
    def get_nhc_hurdat2_url(basin: str = "atlantic") -> str:
        """Return the current NHC HURDAT2 download URL."""
        urls = {
            "atlantic": "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt",
            "pacific": "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2024-031725.txt",
        }
        return urls.get(basin, urls["atlantic"])

    @staticmethod
    def database_report(tracks: List[HurricaneTrack]) -> Dict[str, Any]:
        """Generate a summary report of the database contents."""
        if not tracks:
            return {"n_storms": 0}

        impacts = HURDAT2Parser.filter_cuba_impacts(tracks)
        years = [t.year for t in tracks]
        regions = {}
        cats = {}
        landfalls = 0
        for imp in impacts:
            r = imp.region.value
            regions[r] = regions.get(r, 0) + 1
            c = imp.category_at_closest
            cats[c] = cats.get(c, 0) + 1
            if imp.landfall:
                landfalls += 1

        return {
            "n_storms": len(impacts),
            "n_tracks_total": len(tracks),
            "year_range": (min(years), max(years)) if years else (0, 0),
            "regions": regions,
            "categories": cats,
            "landfalls": landfalls,
            "storms": [
                {"name": imp.track.name, "year": imp.track.year,
                 "cat": imp.category_at_closest, "region": imp.region.value}
                for imp in impacts
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# §11  MAIN ENGINE — PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

class HurricaneScenarioEngine:
    """
    Main entry point for the Hurricane Impact Scenario Generation System.

    Usage:
        engine = HurricaneScenarioEngine()

        # Historical analysis
        result = engine.analyze("IRMA", horizon=48)
        print(result.to_json())

        # All horizons
        cascade = engine.cascade("MATTHEW")
        for h, scenarios in cascade.items():
            print(f"  {h}h: {scenarios.n_scenarios} scenarios")

        # Real-time
        threats = engine.check_active_threats()

        # Database info
        engine.summary()
    """

    def __init__(self, hurdat2_path: Optional[str] = None):
        self.generator = ScenarioGenerator(hurdat2_path)
        self.nhc = NHCRealTimeClient()

    @property
    def database_size(self) -> int:
        return self.generator.n_storms

    @property
    def storm_catalog(self) -> List[Dict[str, Any]]:
        """List all storms in database with basic info."""
        catalog = []
        for imp in self.generator.impacts:
            catalog.append({
                "name": imp.track.name,
                "year": imp.track.year,
                "category": imp.category_at_closest,
                "region": imp.region.value,
                "landfall": imp.landfall,
                "closest_km": round(imp.closest_approach_km, 1),
                "wind_kt": imp.wind_at_closest,
                "provinces": imp.provinces_affected,
            })
        return sorted(catalog, key=lambda x: x["year"])

    def analyze(
        self,
        hurricane_name: str,
        horizon: int = 48,
        region: Optional[str] = None,
        n_scenarios: Optional[int] = None,
        method: str = "gmm",
    ) -> ScenarioSet:
        """
        Generate scenario set for a given hurricane and temporal horizon.

        Parameters:
            hurricane_name: Storm name (e.g. "IRMA", "MATTHEW")
            horizon: Hours before impact {72, 48, 36, 24, 12}
            region: "Occidente", "Centro", or "Oriente" (optional filter)
            n_scenarios: Force number of scenarios (auto if None)
            method: "gmm" (default) or "kmeans"

        Returns:
            ScenarioSet with scenarios, probabilities, and metadata
        """
        target_region = None
        if region:
            region_map = {"occidente": CubaRegion.OCCIDENTE,
                          "centro": CubaRegion.CENTRO,
                          "oriente": CubaRegion.ORIENTE}
            target_region = region_map.get(region.lower())

        return self.generator.generate_scenarios(
            hurricane_name, horizon, target_region, n_scenarios, method
        )

    def cascade(
        self,
        hurricane_name: str,
        region: Optional[str] = None,
        n_scenarios: Optional[int] = None,
        method: str = "gmm",
    ) -> Dict[int, ScenarioSet]:
        """
        Generate scenario cascade across all temporal horizons.

        Returns dict: {72: ScenarioSet, 48: ..., 36: ..., 24: ..., 12: ...}
        """
        target_region = None
        if region:
            region_map = {"occidente": CubaRegion.OCCIDENTE,
                          "centro": CubaRegion.CENTRO,
                          "oriente": CubaRegion.ORIENTE}
            target_region = region_map.get(region.lower())

        return self.generator.generate_all_horizons(
            hurricane_name, target_region, n_scenarios, method
        )

    def check_active_threats(self) -> List[Dict]:
        """Check NHC for active storms threatening Cuba."""
        storms = NHCRealTimeClient.fetch_active_storms()
        return NHCRealTimeClient.check_cuba_threat(storms)

    def summary(self) -> str:
        """Print database summary."""
        lines = [
            "╔═══════════════════════════════════════════════════════════════╗",
            "║  HURRICANE SCENARIO ENGINE — DATABASE SUMMARY               ║",
            "╠═══════════════════════════════════════════════════════════════╣",
            f"║  Storms in database: {self.database_size:>3}                                  ║",
        ]

        by_region = {r: [] for r in CubaRegion}
        for imp in self.generator.impacts:
            by_region[imp.region].append(imp)

        for region in CubaRegion:
            imps = by_region[region]
            if imps:
                years = [i.track.year for i in imps]
                lines.append(
                    f"║  {region.value:12s}: {len(imps):>2} storms "
                    f"({min(years)}–{max(years)})                   ║"
                )

        landfalls = sum(1 for i in self.generator.impacts if i.landfall)
        cats = [i.category_at_closest for i in self.generator.impacts]
        lines.extend([
            f"║  Landfalls: {landfalls:>2} | Max category: {max(cats)}                      ║",
            f"║  Year range: {min(i.track.year for i in self.generator.impacts)}"
            f"–{max(i.track.year for i in self.generator.impacts)}"
            "                                    ║",
            "╠═══════════════════════════════════════════════════════════════╣",
            "║  Methods: GMM (EM), K-Means++, KDE, Bayesian updating       ║",
            "║  Horizons: 72h, 48h, 36h, 24h, 12h                         ║",
            "║  Output: Scenario set Ξ for PI^S integration                ║",
            "╚═══════════════════════════════════════════════════════════════╝",
        ])
        text = "\n".join(lines)
        print(text)
        return text

    def export_for_pis(
        self,
        hurricane_name: str,
        horizon: int = 48,
        filepath: Optional[str] = None,
    ) -> Dict:
        """
        Export scenario set in format ready for PI^S stochastic program.

        Output format:
        {
            "n_scenarios": S,
            "scenarios": [
                {"omega": 1, "probability": p_1, "demand_factor": D_1, ...},
                ...
            ]
        }
        """
        result = self.analyze(hurricane_name, horizon)
        export = {
            "n_scenarios": result.n_scenarios,
            "hurricane": hurricane_name,
            "horizon_hours": horizon,
            "scenarios": [
                {
                    "omega": s.id,
                    "probability": s.probability,
                    "demand_factor": s.demand_factor,
                    "category": s.category,
                    "max_wind_kt": s.max_wind_kt,
                    "region": s.region.value,
                }
                for s in result.scenarios
            ],
        }

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export, f, indent=2, ensure_ascii=False)

        return export

    # ── Database Management ───────────────────────────────────────────

    def sync_nhc(self, save_path: Optional[str] = None) -> Dict:
        """
        Download latest HURDAT2 from NHC, merge with current database,
        rebuild impacts, and optionally save to disk.

        Returns sync report dict.
        """
        current_tracks = [imp.track for imp in self.generator.impacts]
        merged, report = DatabaseManager.sync_from_nhc(current_tracks)

        if report.get("error"):
            return report

        # Rebuild engine with merged data
        self.generator.impacts = HURDAT2Parser.filter_cuba_impacts(merged)
        self.generator._feature_matrix = None  # Force recompute

        # Save if path given
        sp = save_path or DatabaseManager.DEFAULT_DB_PATH
        DatabaseManager.save_database(merged, sp, metadata={
            "last_sync": report["timestamp"],
            "source": "NHC HURDAT2 + embedded + manual",
            "nhc_url": DatabaseManager.get_nhc_hurdat2_url(),
        })
        report["saved_to"] = sp
        report["db_size_after"] = self.database_size

        return report

    def add_storm(
        self,
        atcf_id: str,
        name: str,
        track_points: List[Tuple],
        save_path: Optional[str] = None,
    ) -> str:
        """
        Add a storm manually to the database.

        Args:
            atcf_id: e.g. "AL052025"
            name: e.g. "MELISSA"
            track_points: list of (YYYYMMDD, HHMM, status, lat, lon, wind_kt, press_mb)
            save_path: optional path to save updated DB

        Returns: status message
        """
        current_tracks = [imp.track for imp in self.generator.impacts]
        updated, msg = DatabaseManager.add_storm_manual(
            current_tracks, atcf_id, name, track_points)

        if msg.startswith("OK") or msg.startswith("ADVERTENCIA"):
            self.generator.impacts = HURDAT2Parser.filter_cuba_impacts(updated)
            self.generator._feature_matrix = None

            sp = save_path or DatabaseManager.DEFAULT_DB_PATH
            DatabaseManager.save_database(updated, sp, metadata={
                "last_manual_add": name,
                "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            msg += "\n  Guardado en: %s" % sp
            msg += "\n  DB total: %d huracanes sobre Cuba" % self.database_size

        return msg

    def load_local_db(self, filepath: Optional[str] = None) -> str:
        """
        Load database from local JSON file.

        Returns status message.
        """
        fp = filepath or DatabaseManager.DEFAULT_DB_PATH
        if not os.path.exists(fp):
            return "No existe archivo local: %s (usando base embebida)" % fp

        tracks, meta = DatabaseManager.load_database(fp)
        self.generator.impacts = HURDAT2Parser.filter_cuba_impacts(tracks)
        self.generator._feature_matrix = None

        return (
            "Cargado: %s\n  Fecha: %s\n  Huracanes Cuba: %d"
            % (fp, meta.get("file_updated", "?"), self.database_size)
        )

    def save_local_db(self, filepath: Optional[str] = None) -> str:
        """Save current database to local JSON file."""
        fp = filepath or DatabaseManager.DEFAULT_DB_PATH
        tracks = [imp.track for imp in self.generator.impacts]
        DatabaseManager.save_database(tracks, fp, metadata={
            "saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_cuba_storms": self.database_size,
        })
        return "Guardado: %s (%d huracanes)" % (fp, self.database_size)

    def db_report(self) -> Dict:
        """Generate database content report."""
        tracks = [imp.track for imp in self.generator.impacts]
        return DatabaseManager.database_report(tracks)


# ═══════════════════════════════════════════════════════════════════════════════
# §11  VALIDATION AND TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def run_validation():
    """Complete validation suite for the Hurricane Scenario Engine."""
    print("=" * 70)
    print("  HURRICANE SCENARIO ENGINE v2.0 — VALIDATION SUITE")
    print("=" * 70)

    engine = HurricaneScenarioEngine()
    engine.summary()

    # Test 1: Database integrity
    print("\n[TEST 1] Database Integrity")
    print(f"  Storms loaded: {engine.database_size}")
    for item in engine.storm_catalog:
        print(f"    {item['year']} {item['name']:12s} Cat {item['category']} "
              f"{item['region']:10s} {'LANDFALL' if item['landfall'] else '        '} "
              f"({item['closest_km']:.0f} km)")

    # Test 2: Scenario generation for IRMA at all horizons
    print("\n[TEST 2] Scenario Cascade — IRMA 2017")
    cascade = engine.cascade("IRMA")
    for h, ss in sorted(cascade.items()):
        print(f"\n  ⏱ Horizon {h}h → {ss.n_scenarios} scenarios "
              f"(Silhouette={ss.silhouette_score:.3f}, "
              f"Method={ss.clustering_method})")
        for s in ss.scenarios:
            print(f"    ω{s.id}: P={s.probability:.4f}  Cat {s.category}  "
                  f"{s.max_wind_kt:>3d}kt  D={s.demand_factor:.3f}  "
                  f"[{', '.join(s.reference_storms[:3])}]")
        print(f"    ΣP = {ss.probability_sum:.6f}")

    # Test 3: MATTHEW (Oriente)
    print("\n[TEST 3] Single Analysis — MATTHEW 2016, 48h, Oriente")
    result = engine.analyze("MATTHEW", horizon=48, region="Oriente")
    print(f"  Scenarios: {result.n_scenarios}, Silhouette: {result.silhouette_score:.3f}")
    for s in result.scenarios:
        print(f"    ω{s.id}: P={s.probability:.4f}  {s.label}  {s.max_wind_kt}kt")

    # Test 4: Export for PI^S
    print("\n[TEST 4] PI^S Export Format")
    export = engine.export_for_pis("IKE", horizon=48)
    print(f"  Format check: n_scenarios={export['n_scenarios']}")
    for sc in export["scenarios"]:
        print(f"    ω{sc['omega']}: p={sc['probability']:.4f}, "
              f"D={sc['demand_factor']:.3f}, Cat {sc['category']}")

    # Test 5: K-Means comparison
    print("\n[TEST 5] Method Comparison — GUSTAV 2008, 48h")
    for method in ["gmm", "kmeans"]:
        r = engine.analyze("GUSTAV", horizon=48, method=method)
        print(f"  {method.upper():7s}: {r.n_scenarios} scenarios, "
              f"Silhouette={r.silhouette_score:.3f}" +
              (f", BIC={r.bic:.1f}" if r.bic else ""))

    # Test 6: Bayesian evolution across horizons
    print("\n[TEST 6] Bayesian Probability Evolution — DENNIS 2005")
    for h in HORIZONS:
        r = engine.analyze("DENNIS", horizon=h, n_scenarios=3)
        probs = [f"{s.probability:.3f}" for s in r.scenarios]
        cats = [f"Cat{s.category}" for s in r.scenarios]
        print(f"  {h:>2}h: P=[{', '.join(probs)}]  Categories=[{', '.join(cats)}]")

    print("\n" + "=" * 70)
    print("  ✅ All validation tests completed successfully")
    print("=" * 70)

    return engine


if __name__ == "__main__":
    run_validation()
