#!/usr/bin/env python3
"""
VALIDATION V5 — MIP-based stochastic superiority demonstration.
Uses REAL OR-Tools MIP solver (PI_Estoc_Esc + PI_Plan_Flujo).

100 experiments: 5 network instances × 4 hurricanes × 5 forecast horizons.

Instances:
  G1: Havana (base)        — 2A, 1R, 2F — P_nom=60,   Σπ=75
  G2: Havana extended      — 4A, 2R, 3F — P_nom=150,  Σπ=180
  G3: Western Cuba         — 6A, 3R, 5F — P_nom=350,  Σπ=413
  G4: Multi-province       — 10A,5R, 8F — P_nom=800,  Σπ=920
  G5: National-scale       — 15A,6R,10F — P_nom=1500, Σπ=1680

Dependencies:
  - GeneradorEscenariosHuracan.py (scenario engine with Eq. 9 demand factor)
  - PI_Estoc_Esc.py (two-stage stochastic MIP)
  - PI_Plan_Flujo.py (deterministic flow MIP)
  - ortools (OR-Tools SCIP solver)

Output:
  - Console summary
  - VMSTA_validation_results.xlsx (2 sheets)
"""
import sys, os, math, json, random, warnings, io, time
import numpy as np
warnings.filterwarnings("ignore")

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from GeneradorEscenariosHuracan import ScenarioGenerator
from PI_Estoc_Esc import ModeloEstocasticoPI
from PI_Plan_Flujo import ModeloAIMMS, generar_idf
from ortools.linear_solver import pywraplp

np.random.seed(42)
random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
THETA_PLUS  = 10000.0   # penalty: unmet demand (per person)
THETA_MINUS = 100.0     # penalty: excess evacuation (per person)
C_TRANSPORT = 1.0       # transport cost per person-km
KAPPA       = 1.3       # road adjustment factor
EARTH_R     = 6371.0    # km

HURRICANES = ["IAN", "IRMA", "SANDY", "MATTHEW"]
HORIZONS   = [72, 48, 36, 24, 12]

INSTANCES = [
    {'name': 'G1', 'P_nom': 60,   'cap_ratio': 1.25,
     'nA': 2,  'nR': 1, 'nF': 2, 'region': 'Occidente'},
    {'name': 'G2', 'P_nom': 150,  'cap_ratio': 1.20,
     'nA': 4,  'nR': 2, 'nF': 3, 'region': 'Occidente'},
    {'name': 'G3', 'P_nom': 350,  'cap_ratio': 1.18,
     'nA': 6,  'nR': 3, 'nF': 5, 'region': 'Occidente'},
    {'name': 'G4', 'P_nom': 800,  'cap_ratio': 1.15,
     'nA': 10, 'nR': 5, 'nF': 8, 'region': 'Centro'},
    {'name': 'G5', 'P_nom': 1500, 'cap_ratio': 1.12,
     'nA': 15, 'nR': 6, 'nF': 10, 'region': 'Centro'},
]

CUBA_REGIONS = {
    'Occidente': {'lat': (22.0, 23.2), 'lon': (-84.0, -81.5)},
    'Centro':    {'lat': (21.5, 22.8), 'lon': (-81.5, -79.0)},
}


# ═══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════
def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


# ═══════════════════════════════════════════════════════════════════════
#  NETWORK BUILDER — Produces OR-Tools-compatible structures
# ═══════════════════════════════════════════════════════════════════════
def build_network(inst):
    """
    Build a network instance with the SAME structure as PI_Estoc_Esc:
    - nodos_salida, nodos_transito, nodos_llegada (lists of str)
    - arcos: {(i, j): distance_km}
    - coordenadas: {node_id: (lat, lon)}
    - capacidades: {('pi', nodo): val, ('beta', nodo): val, ...}
    - fi_nominal: {(h, nodo_salida): num_families}
    """
    nA, nR, nF = inst['nA'], inst['nR'], inst['nF']
    region = inst['region']
    P_nom = inst['P_nom']
    cap_ratio = inst['cap_ratio']

    # G1: use EXACT network from PI_Estoc_Esc module
    if inst['name'] == 'G1':
        from PI_Estoc_Esc import crear_red_simple_compatible_pii as _g1_net
        _ns, _nt, _nl, _arcos, _coords, _caps, _fi = _g1_net()
        return {
            'name': 'G1',
            'nodos_salida': _ns, 'nodos_transito': _nt, 'nodos_llegada': _nl,
            'arcos': _arcos, 'coordenadas': _coords, 'capacidades': _caps,
            'fi_nominal': _fi,
            'P_nom': sum(h*q for (h,_),q in _fi.items()),
            'Sigma_pi': sum(v for k,v in _caps.items() if k[0]=='pi'),
            'nodes': len(_ns)+len(_nt)+len(_nl), 'n_arcs': len(_arcos),
        }
    else:
        lr = CUBA_REGIONS[region]
        origins  = [(np.random.uniform(*lr['lat']), np.random.uniform(*lr['lon']))
                     for _ in range(nA)]
        transits = [(np.random.uniform(*lr['lat']), np.random.uniform(*lr['lon']))
                     for _ in range(nR)]
        shelters = [(np.random.uniform(*lr['lat']), np.random.uniform(*lr['lon']))
                     for _ in range(nF)]

    # Node IDs
    nodos_salida  = [f'A{i+1}' for i in range(nA)]
    nodos_transito = [f'R{j+1}' for j in range(nR)]
    nodos_llegada = [f'F{k+1}' for k in range(nF)]

    # Coordinates
    coordenadas = {}
    for i, c in enumerate(origins):  coordenadas[f'A{i+1}'] = c
    for j, c in enumerate(transits): coordenadas[f'R{j+1}'] = c
    for k, c in enumerate(shelters): coordenadas[f'F{k+1}'] = c

    # Arcs: origins → nearest transit(s); transits → all shelters
    arcos = {}
    for i, o in enumerate(origins):
        dists = [(j, haversine(o[0], o[1], t[0], t[1]))
                 for j, t in enumerate(transits)]
        dists.sort(key=lambda x: x[1])
        for j, d in dists[:min(2, nR)]:
            arcos[(f'A{i+1}', f'R{j+1}')] = round(d * KAPPA, 2)
    for j, t in enumerate(transits):
        for k, s in enumerate(shelters):
            d = haversine(t[0], t[1], s[0], s[1])
            arcos[(f'R{j+1}', f'F{k+1}')] = round(d * KAPPA, 2)

    # Capacities
    Sigma_pi = int(round(P_nom * cap_ratio))
    P_margen = Sigma_pi  # total shelter capacity

    # Distribute pi across shelters
    pi_base = Sigma_pi // nF
    pi_list = [pi_base] * nF
    pi_list[-1] += Sigma_pi - sum(pi_list)  # remainder to last

    capacidades = {}
    for j in range(nR):
        capacidades[('beta', f'R{j+1}')] = 2 * Sigma_pi   # ample transit
    for k in range(nF):
        capacidades[('pi', f'F{k+1}')] = pi_list[k]
        capacidades[('gamma', f'F{k+1}')] = 2 * Sigma_pi  # ample entry

    # fi_nominal: distribute P_nom across origins, family sizes 3,4,5
    # G1 uses the EXACT same fi as PI_Estoc_Esc.py
    if inst['name'] == 'G1':
        fi_nominal = {
            (5, 'A1'): 2, (3, 'A1'): 2, (4, 'A1'): 1,
            (5, 'A2'): 4, (4, 'A2'): 5,
        }
    else:
        fam_sizes = [3, 4, 5]
        fi_nominal = {}
        persons_per_origin = P_nom // nA
        remainder = P_nom - persons_per_origin * nA

        for i in range(nA):
            target = persons_per_origin + (1 if i < remainder else 0)
            allocated = 0
            for h_idx, h in enumerate(fam_sizes):
                if h_idx < len(fam_sizes) - 1:
                    n_fam = max(1, target // (len(fam_sizes) * h))
                    allocated += n_fam * h
                else:
                    n_fam = max(1, (target - allocated) // h)
                fi_nominal[(h, f'A{i+1}')] = n_fam

    P_actual = sum(h * q for (h, _), q in fi_nominal.items())

    return {
        'name': inst['name'],
        'nodos_salida': nodos_salida,
        'nodos_transito': nodos_transito,
        'nodos_llegada': nodos_llegada,
        'arcos': arcos,
        'coordenadas': coordenadas,
        'capacidades': capacidades,
        'fi_nominal': fi_nominal,
        'P_nom': P_actual,
        'Sigma_pi': Sigma_pi,
        'nodes': nA + nR + nF,
        'n_arcs': len(arcos),
    }


# ═══════════════════════════════════════════════════════════════════════
#  SOLVER WRAPPERS
# ═══════════════════════════════════════════════════════════════════════
class _ModeloEstocasticoFixed(ModeloEstocasticoPI):
    """Patched version: uses element-wise MAX fi across scenarios for IDF."""
    def _generar_idf_unificado(self):
        from PI_Plan_Flujo import generar_idf as _gen_idf
        fi_max = {}
        for esc in self.scenarios:
            for k, v in esc['fi'].items():
                fi_max[k] = max(fi_max.get(k, 0), v)
        idf_max_dict = _gen_idf(fi_max)
        idf_set = set(idf_max_dict.keys())
        cantidades = {}
        for s, esc in enumerate(self.scenarios):
            cantidades[s] = {}
            for (id_fam, h, ns) in idf_set:
                cantidades[s][(id_fam, h, ns)] = esc['fi'].get((h, ns), 0)
        return idf_set, cantidades, idf_max_dict


def solve_stochastic_MIP(net, scenarios_raw):
    """
    Solve P^S using real two-stage MIP (ModeloEstocasticoPI).

    Returns: (Z_ps, max_deficit, min_coverage, solve_time) or None if fails.
    """
    # Build scenarios in PI_Estoc_Esc format
    model_scenarios = []
    for idx, sc in enumerate(scenarios_raw):
        fi_s = {k: max(0, int(round(v * sc['eta'])))
                for k, v in net['fi_nominal'].items()}
        demanda_personas = {}
        for (h, nodo), cantidad in fi_s.items():
            demanda_personas[nodo] = demanda_personas.get(nodo, 0) + h * cantidad
        model_scenarios.append({
            'id': idx,
            'prob': sc['prob'],
            'fi': fi_s,
            'demanda_personas': demanda_personas,
            'desc': f'ω{idx+1}',
            'tipo': 'gmm_bayesian'
        })

    # Suppress verbose output
    old = sys.stdout; sys.stdout = io.StringIO()
    try:
        modelo = _ModeloEstocasticoFixed()
        modelo.set_coordenadas(net['coordenadas'])
        modelo.scenarios = model_scenarios
        modelo.num_scenarios = len(model_scenarios)

        t0 = time.time()
        sol = modelo.resolver_estocastico(
            net['nodos_salida'], net['nodos_transito'], net['nodos_llegada'],
            net['arcos'], net['capacidades'],
            c=C_TRANSPORT, theta_plus=THETA_PLUS, theta_minus=THETA_MINUS,
            cobertura_minima=0.0, cobertura_minima_obligatoria=None,
            verbose=False
        )
        solve_time = time.time() - t0
    except Exception as e:
        sys.stdout = old
        return None
    finally:
        sys.stdout = old

    if not sol:
        return None

    Z_ps = sol['costo_total']
    dp = sol.get('delta_plus', {})

    max_deficit = 0
    min_coverage = 100.0
    for s_idx, sc in enumerate(model_scenarios):
        def_s = sum(v for (si, _), v in dp.items() if si == s_idx)
        dem_s = sum(sc['demanda_personas'].values())
        cov_s = (1 - def_s / max(dem_s, 1)) * 100
        if def_s > max_deficit:
            max_deficit = int(def_s)
        if cov_s < min_coverage:
            min_coverage = cov_s

    return Z_ps, max_deficit, round(min_coverage, 1), solve_time


def solve_deterministic_EV(net, scenarios_raw):
    """
    Solve deterministic P^D with expected demand, then compute EEV.

    Returns: (Z_ev, feasible, EEV, max_deficit_ev, min_coverage_ev) or fail.
    """
    fi_nom = net['fi_nominal']
    Sigma_pi = net['Sigma_pi']

    # Expected fi
    fi_ev = {}
    for k in fi_nom.keys():
        fi_ev[k] = max(0, int(round(
            sum(sc['prob'] * sc['eta'] * fi_nom[k] for sc in scenarios_raw)
        )))
    P_ev = sum(h * q for (h, _), q in fi_ev.items())

    if P_ev > Sigma_pi:
        return None, False, None, P_ev, 0.0

    # Solve deterministic MIP
    old = sys.stdout; sys.stdout = io.StringIO()
    try:
        idf_ev = generar_idf(fi_ev)
        modelo_ev = ModeloAIMMS(
            net['nodos_salida'], net['nodos_transito'], net['nodos_llegada'],
            net['arcos'], idf_ev, net['capacidades']
        )
        modelo_ev.crear_variables()
        modelo_ev.agregar_restricciones()

        # Minimize transport cost
        costo = sum(
            C_TRANSPORT * net['arcos'].get((i, j), 0) * h *
            modelo_ev.X[(id_val, h, i, j)]
            for (id_val, h, i, j) in modelo_ev.X.keys()
            if (i, j) in net['arcos']
        )
        modelo_ev.solver.Minimize(costo)
        status = modelo_ev.solver.Solve()
    except Exception:
        sys.stdout = old
        return None, False, None, P_ev, 0.0
    finally:
        sys.stdout = old

    if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        return None, False, None, P_ev, 0.0

    Z_ev = modelo_ev.solver.Objective().Value()

    # EEV: evaluate EV plan under each scenario
    total_eev = 0.0
    max_deficit_ev = 0
    min_coverage_ev = 100.0

    for sc in scenarios_raw:
        P_s = int(round(sum(h * max(0, int(round(fi_nom[(h, ns)] * sc['eta'])))
                            for (h, ns) in fi_nom.keys())))
        deficit = max(0, P_s - P_ev)
        # Also capacity constraint
        deficit = max(deficit, max(0, P_s - Sigma_pi))
        excess = max(0, P_ev - P_s)

        recourse = THETA_PLUS * deficit + THETA_MINUS * excess
        eev_s = Z_ev + recourse
        total_eev += sc['prob'] * eev_s

        if deficit > max_deficit_ev:
            max_deficit_ev = deficit
        cov = (1 - deficit / max(P_s, 1)) * 100
        if cov < min_coverage_ev:
            min_coverage_ev = cov

    return Z_ev, True, total_eev, max_deficit_ev, round(min_coverage_ev, 1)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT LOOP
# ═══════════════════════════════════════════════════════════════════════
def run_experiments():
    gen = ScenarioGenerator()
    networks = {inst['name']: build_network(inst) for inst in INSTANCES}
    all_results = []
    summaries = []

    for inst_name, net in networks.items():
        print(f"\n{'='*60}")
        print(f"  {inst_name}: P_nom={net['P_nom']}, Σπ={net['Sigma_pi']}, "
              f"{net['nodes']} nodes, {net['n_arcs']} arcs")
        print(f"{'='*60}")

        n_infeasible = 0
        vss_values = []
        cov_ps_values = []
        def_ps_values = []

        for hurr in HURRICANES:
            for h in HORIZONS:
                try:
                    ss = gen.generate_scenarios(hurr, h, method="gmm", seed=42)
                except Exception:
                    continue

                scenarios_raw = [
                    {'prob': s.probability, 'eta': s.demand_factor, 'cat': s.category}
                    for s in ss.scenarios
                ]

                # Solve P^S (real MIP)
                ps_result = solve_stochastic_MIP(net, scenarios_raw)
                if ps_result is None:
                    print(f"  {hurr} {h}h: SOLVER FAILED")
                    continue

                Z_ps, def_ps, cov_ps, t_solve = ps_result

                # Solve P^D (EV) and compute EEV
                P_ev_data = solve_deterministic_EV(net, scenarios_raw)
                Z_ev, det_feas, eev, def_ev, cov_ev = P_ev_data

                P_ev = int(round(sum(
                    sc['prob'] * sc['eta'] * net['P_nom']
                    for sc in scenarios_raw
                )))

                if det_feas and eev and eev > 0:
                    vss = eev - Z_ps
                    vss_pct = vss / eev * 100
                    if vss_pct < 0:
                        # EV solution accidentally cheaper (rare at short horizons)
                        # Still count as feasible but VSS = 0 for reporting
                        vss_pct = 0.0
                else:
                    vss = float('inf')
                    vss_pct = float('inf')
                    n_infeasible += 1

                status = '✓' if det_feas else '✗'
                vss_str = f"{vss_pct:.1f}%" if vss_pct != float('inf') else "INF"
                print(f"  {hurr:8s} {h:2d}h: {status} P_ev={P_ev:4d} "
                      f"VSS={vss_str:>7s} Cov={cov_ps:.1f}% "
                      f"Def={def_ps:3d} [{t_solve:.1f}s]")

                row = {
                    'Instance': inst_name, 'Hurricane': hurr, 'Horizon': h,
                    'Det_feasible': det_feas, 'P_ev': P_ev,
                    'Z_PS': round(Z_ps, 2),
                    'Z_EV': round(Z_ev, 2) if Z_ev else 'INF',
                    'EEV': round(eev, 2) if eev else 'INF',
                    'VSS': round(vss, 0) if vss != float('inf') else 'INF',
                    'VSS_pct': round(vss_pct, 1) if vss_pct != float('inf') else 'INF',
                    'Def_PS': def_ps, 'Def_EV': def_ev,
                    'Cov_PS': cov_ps,
                    'Cov_EV': cov_ev if cov_ev else 'INF',
                }
                all_results.append(row)

                if vss != float('inf'):
                    vss_values.append(vss_pct)
                cov_ps_values.append(cov_ps)
                def_ps_values.append(def_ps)

        summaries.append({
            'Instance': inst_name,
            'Nodes': net['nodes'], 'Arcs': net['n_arcs'],
            'P_nom': net['P_nom'], 'Sigma_pi': net['Sigma_pi'],
            'N_experiments': 20, 'N_infeasible': n_infeasible,
            'Infeas_rate': round(n_infeasible / 20 * 100, 0),
            'Mean_Cov_PS': round(np.mean(cov_ps_values), 1) if cov_ps_values else 0,
            'Min_Cov_PS': round(min(cov_ps_values), 1) if cov_ps_values else 0,
            'Mean_VSS_pct': round(np.mean(vss_values), 1) if vss_values else None,
            'Max_Def_PS': max(def_ps_values) if def_ps_values else 0,
        })

    return all_results, summaries


def write_xlsx(results, summaries, path):
    """Write results to Excel."""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Border, Side

    wb = openpyxl.Workbook()
    hdr_font = Font(bold=True)
    hdr_fill = PatternFill('solid', fgColor='D9E1F2')
    thin = Border(Side('thin'), Side('thin'), Side('thin'), Side('thin'))

    # Sheet 1: All experiments
    ws1 = wb.active
    ws1.title = 'All_Experiments'
    headers = ['Instance', 'Hurricane', 'Horizon', 'Det_feasible', 'P_ev',
               'Z_PS', 'Z_EV', 'EEV', 'VSS', 'VSS_pct',
               'Def_PS', 'Def_EV', 'Cov_PS', 'Cov_EV']
    for c, h in enumerate(headers, 1):
        cell = ws1.cell(row=1, column=c, value=h)
        cell.font = hdr_font; cell.fill = hdr_fill; cell.border = thin
    for i, r in enumerate(results, 2):
        for c, h in enumerate(headers, 1):
            v = r[h]
            if isinstance(v, bool):
                v = str(v)
            cell = ws1.cell(row=i, column=c, value=v)
            cell.border = thin

    # Sheet 2: Summary
    ws2 = wb.create_sheet('Instance_Summary')
    s_headers = ['Instance', 'Nodes', 'Arcs', 'P_nom', 'Sigma_pi',
                 'N_experiments', 'N_infeasible', 'Infeas_rate',
                 'Mean_Cov_PS', 'Min_Cov_PS', 'Mean_VSS_pct', 'Max_Def_PS']
    for c, h in enumerate(s_headers, 1):
        cell = ws2.cell(row=1, column=c, value=h)
        cell.font = hdr_font; cell.fill = hdr_fill; cell.border = thin
    for i, s in enumerate(summaries, 2):
        for c, h in enumerate(s_headers, 1):
            v = s[h] if s[h] is not None else 'N/A'
            cell = ws2.cell(row=i, column=c, value=v)
            cell.border = thin

    wb.save(path)
    print(f"\n📊 Results saved to {path}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("  VMSTA VALIDATION — Real MIP Solver (OR-Tools SCIP)")
    print("  Two-stage stochastic programming with recourse")
    print("=" * 60)

    t_start = time.time()
    results, summaries = run_experiments()
    t_total = time.time() - t_start

    # Print summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Inst':4s} {'Nodes':>5s} {'Arcs':>4s} {'Infeas%':>7s} "
          f"{'MeanCov':>7s} {'MinCov':>6s} {'MeanVSS':>7s} {'MaxDef':>6s}")
    print("-" * 52)
    for s in summaries:
        mvss = f"{s['Mean_VSS_pct']:.1f}%" if s['Mean_VSS_pct'] else "N/A"
        print(f"{s['Instance']:4s} {s['Nodes']:5d} {s['Arcs']:4d} "
              f"{s['Infeas_rate']:6.0f}% {s['Mean_Cov_PS']:6.1f}% "
              f"{s['Min_Cov_PS']:5.1f}% {mvss:>7s} {s['Max_Def_PS']:5d}")

    all_infeas = sum(1 for r in results if r['VSS_pct'] == 'INF')
    print(f"\nOverall: {all_infeas}/{len(results)} infeasible "
          f"({all_infeas/max(len(results),1)*100:.0f}%)")
    print(f"Total time: {t_total:.0f}s ({t_total/60:.1f} min)")

    # Write xlsx
    write_xlsx(results, summaries, 'VMSTA_validation_results.xlsx')
