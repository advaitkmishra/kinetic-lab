import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

try:
    import cantera as ct
    CANTERA_AVAILABLE = True
except ImportError:
    CANTERA_AVAILABLE = False

_GAS = None

def _get_gas():
    global _GAS
    if not CANTERA_AVAILABLE:
        return None
    if _GAS is None:
        try:
            _GAS = ct.Solution('gri30.yaml')
        except Exception as e:
            print(f"Cantera gri30 load failed: {e}")
            return None
    return _GAS

R_GAS = 8.314  # J/mol/K

# =============================================================================
# SECTION 1: REACTION DATABASE
# =============================================================================

REACTION_DB = {

    'CO_oxidation': {
        'family':   'CO oxidation',
        'equation': 'CO + 1/2 O2 -> CO2',
        'stoich':   {'CO': -1, 'O2': -0.5, 'CO2': +1},
        'A':        1e8, 'b': 0.0, 'Ea': 80000,
        'catalyst': 'Pt', 'T_range': (400, 800),
        'note':     'Standard CO oxidation over Pt.',
        'source':   '[1] Deutschmann et al., Catal. Today, 2000'
    },
    'CO_oxidation_Pd': {
        'family':   'CO oxidation',
        'equation': 'CO + 1/2 O2 -> CO2',
        'stoich':   {'CO': -1, 'O2': -0.5, 'CO2': +1},
        'A':        7.07e8, 'b': 0.0, 'Ea': 92000,
        'catalyst': 'Pd', 'T_range': (400, 900),
        'note':     'Pd variant — higher Ea reflects stronger CO binding.',
        'source':   '[9] Oh & Carpenter, J. Catal., 1986'
    },
    'CO_oxidation_inhibited': {
        'family':   'CO oxidation',
        'equation': 'CO + 1/2 O2 -> CO2 (Langmuir-Hinshelwood)',
        'stoich':   {'CO': -1, 'O2': -0.5, 'CO2': +1},
        'A':        6.37e16, 'b': 0.0, 'Ea': 125600,
        'catalyst': 'Pt/Rh', 'T_range': (400, 800),
        'note':     'Use when CO concentration is high (>5%). Includes inhibition.',
        'source':   '[8] Voltz et al., Ind. Eng. Chem., 1973'
    },
    'WGS_forward': {
        'family':   'Water-gas shift',
        'equation': 'CO + H2O -> CO2 + H2',
        'stoich':   {'CO': -1, 'H2O': -1, 'CO2': +1, 'H2': +1},
        'A':        7.40e11, 'b': 0.0, 'Ea': 67130,
        'catalyst': 'Ni', 'T_range': (500, 900),
        'note':     'Forward WGS. Dominant below ~500C.',
        'source':   '[2] Xu & Froment, AIChE J., 1989'
    },
    'WGS_reverse': {
        'family':   'Water-gas shift',
        'equation': 'CO2 + H2 -> CO + H2O',
        'stoich':   {'CO2': -1, 'H2': -1, 'CO': +1, 'H2O': +1},
        'A':        5.43e10, 'b': 0.0, 'Ea': 94800,
        'catalyst': 'Ni', 'T_range': (600, 1000),
        'note':     'Reverse WGS dominates at high temperature.',
        'source':   '[6] NIST Chemical Kinetics Database, 2024'
    },
    'SMR_R1': {
        'family':   'Steam methane reforming',
        'equation': 'CH4 + H2O -> CO + 3H2',
        'stoich':   {'CH4': -1, 'H2O': -1, 'CO': +1, 'H2': +3},
        'A':        4.225e15, 'b': 0.0, 'Ea': 240100,
        'catalyst': 'Ni', 'T_range': (700, 1000),
        'note':     'Primary reforming. Strongly endothermic.',
        'source':   '[2] Xu & Froment, AIChE J., 1989'
    },
    'SMR_R2': {
        'family':   'Steam methane reforming',
        'equation': 'CH4 + 2H2O -> CO2 + 4H2',
        'stoich':   {'CH4': -1, 'H2O': -2, 'CO2': +1, 'H2': +4},
        'A':        1.020e15, 'b': 0.0, 'Ea': 243900,
        'catalyst': 'Ni', 'T_range': (700, 1000),
        'note':     'Overall reforming to CO2. Combination of SMR_R1 + WGS.',
        'source':   '[2] Xu & Froment, AIChE J., 1989'
    },
    'SMR_WGS': {
        'family':   'Steam methane reforming',
        'equation': 'CO + H2O -> CO2 + H2',
        'stoich':   {'CO': -1, 'H2O': -1, 'CO2': +1, 'H2': +1},
        'A':        1.955e6, 'b': 0.0, 'Ea': 67130,
        'catalyst': 'Ni', 'T_range': (700, 1000),
        'note':     'WGS step within SMR context. Use with SMR_R1.',
        'source':   '[2] Xu & Froment, AIChE J., 1989'
    },
    'CH4_oxidation_Pt': {
        'family':   'Methane oxidation',
        'equation': 'CH4 + 2O2 -> CO2 + 2H2O',
        'stoich':   {'CH4': -1, 'O2': -2, 'CO2': +1, 'H2O': +2},
        'A':        9.00e13, 'b': 0.0, 'Ea': 86000,
        'catalyst': 'Pt', 'T_range': (600, 1000),
        'note':     'Complete methane combustion over Pt.',
        'source':   '[1] Deutschmann et al., Catal. Today, 2000'
    },
    'CH4_oxidation_Pd': {
        'family':   'Methane oxidation',
        'equation': 'CH4 + 2O2 -> CO2 + 2H2O',
        'stoich':   {'CH4': -1, 'O2': -2, 'CO2': +1, 'H2O': +2},
        'A':        5.00e14, 'b': 0.0, 'Ea': 95000,
        'catalyst': 'Pd', 'T_range': (600, 1000),
        'note':     'Pd more active than Pt for CH4 oxidation below 400C.',
        'source':   '[7] Mhadeshwar & Vlachos, J. Phys. Chem. B, 2004'
    },
    'CH4_partial_oxidation': {
        'family':   'Methane oxidation',
        'equation': 'CH4 + 1/2 O2 -> CO + 2H2',
        'stoich':   {'CH4': -1, 'O2': -0.5, 'CO': +1, 'H2': +2},
        'A':        1.00e12, 'b': 0.0, 'Ea': 115000,
        'catalyst': 'Pt', 'T_range': (700, 1100),
        'note':     'Partial oxidation — produces syngas. Competes with combustion.',
        'source':   '[1] Deutschmann et al., Catal. Today, 2000'
    },
    'SCR_standard': {
        'family':   'NOx SCR',
        'equation': '4NO + 4NH3 + O2 -> 4N2 + 6H2O',
        'stoich':   {'NO': -4, 'NH3': -4, 'O2': -1, 'N2': +4, 'H2O': +6},
        'A':        3.50e14, 'b': 0.0, 'Ea': 89300,
        'catalyst': 'V2O5/WO3/TiO2', 'T_range': (473, 773),
        'note':     'Standard SCR. Optimal window 300-500C.',
        'source':   '[3] Chatterjee et al., SAE 2006-01-0468, 2006'
    },
    'SCR_fast': {
        'family':   'NOx SCR',
        'equation': '2NH3 + NO + NO2 -> 2N2 + 3H2O',
        'stoich':   {'NH3': -2, 'NO': -1, 'NO2': -1, 'N2': +2, 'H2O': +3},
        'A':        1.33e14, 'b': 0.0, 'Ea': 71700,
        'catalyst': 'V2O5/WO3/TiO2', 'T_range': (373, 673),
        'note':     'Fast SCR — twice the rate at low T. Needs NO2:NO ~ 1:1.',
        'source':   '[3] Chatterjee et al., SAE 2006-01-0468, 2006'
    },
    'SCR_NO2': {
        'family':   'NOx SCR',
        'equation': '8NH3 + 6NO2 -> 7N2 + 12H2O',
        'stoich':   {'NH3': -8, 'NO2': -6, 'N2': +7, 'H2O': +12},
        'A':        5.00e13, 'b': 0.0, 'Ea': 83000,
        'catalyst': 'V2O5/WO3/TiO2', 'T_range': (423, 723),
        'note':     'NO2-SCR. Can produce N2O at low temperatures.',
        'source':   '[3] Chatterjee et al., SAE 2006-01-0468, 2006'
    },
    'NH3_oxidation': {
        'family':   'NOx SCR',
        'equation': '4NH3 + 3O2 -> 2N2 + 6H2O',
        'stoich':   {'NH3': -4, 'O2': -3, 'N2': +2, 'H2O': +6},
        'A':        2.10e13, 'b': 0.0, 'Ea': 113000,
        'catalyst': 'V2O5/WO3/TiO2', 'T_range': (523, 823),
        'note':     'NH3 slip oxidation. Include to model NOx-NH3 trade-off.',
        'source':   '[11] Sjoblom et al., Top. Catal., 2013'
    },
    'NO_oxidation': {
        'family':   'NOx SCR',
        'equation': '2NO + O2 -> 2NO2',
        'stoich':   {'NO': -2, 'O2': -1, 'NO2': +2},
        'A':        1.00e12, 'b': 0.0, 'Ea': 72000,
        'catalyst': 'Cu-zeolite', 'T_range': (373, 673),
        'note':     'NO oxidation — enables fast SCR. Equilibrium-limited above 450C.',
        'source':   '[4] Olsson et al., Chem. Eng. Sci., 2008'
    },
    'SCR_Cu_standard': {
        'family':   'NOx SCR',
        'equation': '4NO + 4NH3 + O2 -> 4N2 + 6H2O',
        'stoich':   {'NO': -4, 'NH3': -4, 'O2': -1, 'N2': +4, 'H2O': +6},
        'A':        5.72e15, 'b': 0.0, 'Ea': 97500,
        'catalyst': 'Cu-zeolite', 'T_range': (423, 823),
        'note':     'Standard SCR over Cu-zeolite. Broader T window than V2O5.',
        'source':   '[4] Olsson et al., Chem. Eng. Sci., 2008'
    },
    'DRM': {
        'family':   'Dry reforming',
        'equation': 'CH4 + CO2 -> 2CO + 2H2',
        'stoich':   {'CH4': -1, 'CO2': -1, 'CO': +2, 'H2': +2},
        'A':        4.65e13, 'b': 0.0, 'Ea': 189000,
        'catalyst': 'Ni', 'T_range': (973, 1273),
        'note':     'Dry reforming. Strongly endothermic. High coking below 800C.',
        'source':   '[5] Pakhare & Spivey, Chem. Soc. Rev., 2014'
    },
    'DRM_Rh': {
        'family':   'Dry reforming',
        'equation': 'CH4 + CO2 -> 2CO + 2H2',
        'stoich':   {'CH4': -1, 'CO2': -1, 'CO': +2, 'H2': +2},
        'A':        3.00e13, 'b': 0.0, 'Ea': 165000,
        'catalyst': 'Rh', 'T_range': (873, 1173),
        'note':     'Rh more coke-resistant than Ni for DRM.',
        'source':   '[5] Pakhare & Spivey, Chem. Soc. Rev., 2014'
    },
    'DRM_carbon_deposition': {
        'family':   'Dry reforming',
        'equation': 'CH4 -> C + 2H2',
        'stoich':   {'CH4': -1, 'H2': +2},
        'A':        1.00e11, 'b': 0.0, 'Ea': 154000,
        'catalyst': 'Ni', 'T_range': (700, 1000),
        'note':     'Methane cracking — primary coking pathway in DRM.',
        'source':   '[5] Pakhare & Spivey, Chem. Soc. Rev., 2014'
    },
    'ethylene_oxidation_complete': {
        'family':   'Partial oxidation',
        'equation': 'C2H4 + 3O2 -> 2CO2 + 2H2O',
        'stoich':   {'C2H4': -1, 'O2': -3, 'CO2': +2, 'H2O': +2},
        'A':        1.80e12, 'b': 0.0, 'Ea': 135000,
        'catalyst': 'Ag', 'T_range': (473, 573),
        'note':     'Complete combustion of ethylene — undesired side reaction.',
        'source':   '[6] NIST Chemical Kinetics Database, 2024'
    },
    'ethylene_oxide_synthesis': {
        'family':   'Partial oxidation',
        'equation': 'C2H4 + 1/2 O2 -> C2H4O',
        'stoich':   {'C2H4': -1, 'O2': -0.5, 'C2H4O': +1},
        'A':        5.60e11, 'b': 0.0, 'Ea': 71000,
        'catalyst': 'Ag', 'T_range': (473, 573),
        'note':     'Ethylene oxide synthesis — desired partial oxidation product.',
        'source':   '[6] NIST Chemical Kinetics Database, 2024'
    },
    'CO2_methanation': {
        'family':   'CO2 hydrogenation',
        'equation': 'CO2 + 4H2 -> CH4 + 2H2O',
        'stoich':   {'CO2': -1, 'H2': -4, 'CH4': +1, 'H2O': +2},
        'A':        3.46e10, 'b': 0.0, 'Ea': 77500,
        'catalyst': 'Ni', 'T_range': (473, 773),
        'note':     'Sabatier reaction. Thermodynamically favored below ~550C.',
        'source':   '[12] Grabow & Mavrikakis, ACS Catal., 2011'
    },
    'CO_methanation': {
        'family':   'CO2 hydrogenation',
        'equation': 'CO + 3H2 -> CH4 + H2O',
        'stoich':   {'CO': -1, 'H2': -3, 'CH4': +1, 'H2O': +1},
        'A':        7.81e10, 'b': 0.0, 'Ea': 80300,
        'catalyst': 'Ni', 'T_range': (473, 773),
        'note':     'CO methanation. Include alongside CO2_methanation.',
        'source':   '[12] Grabow & Mavrikakis, ACS Catal., 2011'
    },
    'methanol_synthesis': {
        'family':   'CO2 hydrogenation',
        'equation': 'CO2 + 3H2 -> CH3OH + H2O',
        'stoich':   {'CO2': -1, 'H2': -3, 'CH3OH': +1, 'H2O': +1},
        'A':        1.07e10, 'b': 0.0, 'Ea': 70200,
        'catalyst': 'Cu/ZnO/Al2O3', 'T_range': (473, 573),
        'note':     'Methanol synthesis. At atmospheric pressure conversion is low.',
        'source':   '[12] Grabow & Mavrikakis, ACS Catal., 2011'
    },
}

# =============================================================================
# SECTION 2: DATABASE HELPERS
# =============================================================================

def get_reaction(key):
    if key not in REACTION_DB:
        available = '\n  '.join(REACTION_DB.keys())
        raise KeyError(f"Reaction '{key}' not found. Available:\n  {available}")
    return REACTION_DB[key]

def get_reactions(keys):
    if len(keys) > 10:
        raise ValueError(f"Maximum 10 reactions supported. Got {len(keys)}.")
    return [get_reaction(k) for k in keys]

def get_all_species(keys):
    species = set()
    for key in keys:
        species.update(get_reaction(key)['stoich'].keys())
    return sorted(species)

def list_reactions(family=None):
    families = sorted(set(r['family'] for r in REACTION_DB.values()))
    if family and family not in families:
        print(f"Family '{family}' not found. Available families:")
        for f in families:
            print(f"  {f}")
        return
    print(f"\n  {'KEY':<35} {'FAMILY':<28} {'CATALYST':<22} T RANGE (K)")
    print("  " + "-" * 97)
    for key, rxn in REACTION_DB.items():
        if family and rxn['family'] != family:
            continue
        t = f"{rxn['T_range'][0]}-{rxn['T_range'][1]}"
        print(f"  {key:<35} {rxn['family']:<28} {rxn['catalyst']:<22} {t}")
    print()

def summarize_reaction(key):
    rxn = get_reaction(key)
    print(f"\n  Key:      {key}")
    print(f"  Equation: {rxn['equation']}")
    print(f"  Family:   {rxn['family']}")
    print(f"  Catalyst: {rxn['catalyst']}")
    print(f"  Ea:       {rxn['Ea']/1000:.1f} kJ/mol")
    print(f"  T range:  {rxn['T_range'][0]}-{rxn['T_range'][1]} K")
    print(f"  Note:     {rxn['note']}")
    print(f"  Source:   {rxn['source']}\n")

# =============================================================================
# SECTION 3: THERMODYNAMICS VIA CANTERA
# =============================================================================

CANTERA_NAME_MAP = {
    'CO':    'CO',    'CO2':   'CO2',  'O2':    'O2',
    'H2':    'H2',    'H2O':   'H2O',  'CH4':   'CH4',
    'NO':    'NO',    'NO2':   'NO2',  'NH3':   'NH3',
    'N2':    'N2',    'C2H4':  'C2H4', 'C2H4O': None,
    'CH3OH': None,
}

FALLBACK_PROPS = {'Cp_mass': 1050, 'rho': 1.2}

FALLBACK_HF = {
    'CO': -110500, 'CO2': -393500, 'O2': 0,       'H2': 0,
    'H2O': -241800, 'CH4': -74800, 'NO': 91300,   'NO2': 33200,
    'NH3': -45900,  'N2': 0,       'C2H4': 52500, 'C2H4O': -52600,
    'CH3OH': -200700,
}

def get_thermo(species_names, concentrations, T, P=101325):
    gas = _get_gas()
    if gas is None:
        return FALLBACK_PROPS['Cp_mass'], FALLBACK_PROPS['rho']
    total_conc = sum(max(c, 0) for c in concentrations) + 1e-30
    X_dict = {}
    for sp, c in zip(species_names, concentrations):
        ct_name = CANTERA_NAME_MAP.get(sp, sp)
        if ct_name:
            X_dict[ct_name] = max(c, 0) / total_conc
    if not X_dict:
        return FALLBACK_PROPS['Cp_mass'], FALLBACK_PROPS['rho']
    try:
        gas.TPX = T, P, X_dict
        return gas.cp_mass, gas.density
    except Exception:
        return FALLBACK_PROPS['Cp_mass'], FALLBACK_PROPS['rho']

def compute_dH(stoich, T, P=101325):
    gas = _get_gas()
    if gas is None:
        return sum(nu * FALLBACK_HF.get(sp, 0.0) for sp, nu in stoich.items())
    try:
        X_dict = {CANTERA_NAME_MAP.get(sp, sp): 1.0
                  for sp in stoich if CANTERA_NAME_MAP.get(sp, sp)}
        if not X_dict:
            raise ValueError("No Cantera-mappable species")
        gas.TPX = T, P, X_dict
        h_map = dict(zip(gas.species_names, gas.partial_molar_ / 1000)) # Convert J/mol to kJ/mol
        dH = 0.0
        for sp, nu in stoich.items():
            ct_name = CANTERA_NAME_MAP.get(sp, sp)
            dH += nu * (h_map[ct_name] if ct_name and ct_name in h_map
                        else FALLBACK_HF.get(sp, 0.0))
        return dH
    except Exception:
        return sum(nu * FALLBACK_HF.get(sp, 0.0) for sp, nu in stoich.items())

# =============================================================================
# SECTION 4: REACTION RATE
# =============================================================================

def reaction_rate(rxn, conc_dict, T):
    k = rxn['A'] * (T ** rxn.get('b', 0.0)) * np.exp(-rxn['Ea'] / (R_GAS * T))
    for sp, nu in rxn['stoich'].items():
        if nu < 0:
            k *= max(conc_dict.get(sp, 0.0), 0.0)
    return max(k, 0.0)

# =============================================================================
# SECTION 5: DERIVATIVES
# =============================================================================

def pfr_derivatives(species_names, active_reactions, C_arr, T, u):
    conc = {sp: C_arr[i] for i, sp in enumerate(species_names)}
    cp, rho = get_thermo(species_names, C_arr, T)
    dC = np.zeros(len(species_names))
    dT = 0.0
    for rxn in active_reactions:
        r  = reaction_rate(rxn, conc, T)
        dH = compute_dH(rxn['stoich'], T)
        for sp, nu in rxn['stoich'].items():
            if sp in species_names:
                dC[species_names.index(sp)] += nu * r / u
        dT += (-dH * r) / (rho * cp * u)
    return dC, dT

def batch_derivatives(species_names, active_reactions, C_arr, T):
    conc = {sp: C_arr[i] for i, sp in enumerate(species_names)}
    cp, rho = get_thermo(species_names, C_arr, T)
    dC = np.zeros(len(species_names))
    dT = 0.0
    for rxn in active_reactions:
        r  = reaction_rate(rxn, conc, T)
        dH = compute_dH(rxn['stoich'], T)
        for sp, nu in rxn['stoich'].items():
            if sp in species_names:
                dC[species_names.index(sp)] += nu * r
        dT += (-dH * r) / (rho * cp)
    return dC, dT

# =============================================================================
# SECTION 6: RK4 INTEGRATORS
# =============================================================================

def run_rk4(species_names, C0_arr, T0, active_reactions, u, L, n):
    dz     = L / n
    z_vals = np.linspace(0, L, n + 1)
    C      = C0_arr.copy().astype(float)
    T      = float(T0)
    hist_C = [C.copy()]
    hist_T = [T]
    for i in range(n):
        dC1, dT1 = pfr_derivatives(species_names, active_reactions, C,              T,              u)
        dC2, dT2 = pfr_derivatives(species_names, active_reactions, C + dz/2*dC1,   T + dz/2*dT1,   u)
        dC3, dT3 = pfr_derivatives(species_names, active_reactions, C + dz/2*dC2,   T + dz/2*dT2,   u)
        dC4, dT4 = pfr_derivatives(species_names, active_reactions, C + dz*dC3,     T + dz*dT3,     u)
        C = C + (dz/6) * (dC1 + 2*dC2 + 2*dC3 + dC4)
        T = T + (dz/6) * (dT1 + 2*dT2 + 2*dT3 + dT4)
        C = np.maximum(C, 0.0)
        hist_C.append(C.copy())
        hist_T.append(T)
    return z_vals, np.array(hist_C), np.array(hist_T)

def run_batch(species_names, C0_arr, T0, active_reactions, t_end, n):
    dt     = t_end / n
    t_vals = np.linspace(0, t_end, n + 1)
    C      = C0_arr.copy().astype(float)
    T      = float(T0)
    hist_C = [C.copy()]
    hist_T = [T]
    for i in range(n):
        dC1, dT1 = batch_derivatives(species_names, active_reactions, C,              T)
        dC2, dT2 = batch_derivatives(species_names, active_reactions, C + dt/2*dC1,   T + dt/2*dT1)
        dC3, dT3 = batch_derivatives(species_names, active_reactions, C + dt/2*dC2,   T + dt/2*dT2)
        dC4, dT4 = batch_derivatives(species_names, active_reactions, C + dt*dC3,     T + dt*dT3)
        C = C + (dt/6) * (dC1 + 2*dC2 + 2*dC3 + dC4)
        T = T + (dt/6) * (dT1 + 2*dT2 + 2*dT3 + dT4)
        C = np.maximum(C, 0.0)
        hist_C.append(C.copy())
        hist_T.append(T)
    return t_vals, np.array(hist_C), np.array(hist_T)

def run_ivp_stiff(species_names, C0_arr, T0, active_reactions, u, L, n):
    def derivatives(z, y):
        n_sp = len(species_names)
        C_arr = y[:n_sp]
        T = y[n_sp]
        dC, dT = pfr_derivatives(species_names, active_reactions, C_arr, T, u)
        return np.concatenate([dC, [dT]])
    y0 = np.concatenate([C0_arr, [T0]])
    z_eval = np.linspace(0, L, n + 1)
    sol = solve_ivp(derivatives, (0, L), y0, method='BDF',
                    t_eval=z_eval, rtol=1e-6, atol=1e-9)
    if not sol.success:
        print(f"IVP solver failed: {sol.message} — falling back to RK4")
        return run_rk4(species_names, C0_arr, T0, active_reactions, u, L, n)
    hist_C = np.maximum(sol.y[:len(species_names)].T, 0.0)
    hist_T = sol.y[len(species_names)]
    return sol.t, hist_C, hist_T

# =============================================================================
# SECTION 6b: CSTR SOLVER (Newton-Raphson)
# =============================================================================

def _cstr_residual(x, species_names, active_reactions, C_in, T_in, V, Q):
    n_sp = len(species_names)
    C = np.maximum(x[:n_sp], 0.0)
    T = x[n_sp]
    conc = {sp: C[i] for i, sp in enumerate(species_names)}
    cp, rho = get_thermo(species_names, C, T)
    F = np.zeros(n_sp + 1)
    for i, sp in enumerate(species_names):
        F[i] = Q * (C_in[i] - C[i])
    heat = 0.0
    for rxn in active_reactions:
        r  = reaction_rate(rxn, conc, T)
        dH = compute_dH(rxn['stoich'], T)
        for sp, nu in rxn['stoich'].items():
            if sp in species_names:
                F[species_names.index(sp)] += V * nu * r
        heat += -dH * r
    F[n_sp] = Q * rho * cp * (T_in - T) + V * heat
    return F

def _numerical_jacobian(func, x, eps=1e-7):
    f0 = func(x)
    n = len(x)
    J = np.empty((len(f0), n))
    for j in range(n):
        x_pert = x.copy()
        x_pert[j] += eps
        J[:, j] = (func(x_pert) - f0) / eps
    return J

def solve_cstr(species_names, C0_arr, T_in, active_reactions, V, Q,
               T0_guess=None, max_iter=50, tol=1e-8):
    n_sp = len(species_names)
    C_in = C0_arr.copy().astype(float)
    x = np.concatenate([C_in.copy(), [T0_guess if T0_guess else T_in]])
    residual_func = lambda x_: _cstr_residual(
        x_, species_names, active_reactions, C_in, T_in, V, Q)
    residual_history = []
    converged = False
    for k in range(max_iter):
        F = residual_func(x)
        res_norm = np.linalg.norm(F)
        residual_history.append(res_norm)
        if res_norm < tol:
            converged = True
            break
        J = _numerical_jacobian(residual_func, x)
        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break
        x = x + delta
        x[:n_sp] = np.maximum(x[:n_sp], 0.0)
    return {
        'C_out': x[:n_sp], 'T_out': x[n_sp],
        'C_in': C_in, 'T_in': T_in,
        'species': species_names, 'converged': converged,
        'residual_history': np.array(residual_history),
    }

# =============================================================================
# SECTION 7: PLOTTING (CLI only)
# =============================================================================

def _print_summary(hist_C, hist_T, species_names):
    print("\n--- Exit summary ---")
    for i, sp in enumerate(species_names):
        c_in  = hist_C[0, i]
        c_out = hist_C[-1, i]
        if c_in > 1e-8:
            conv = (c_in - c_out) / c_in * 100
            print(f"  {sp:>8s}: {c_in:.4f} -> {c_out:.4f} mol/m³   (conversion: {conv:.1f}%)")
        else:
            print(f"  {sp:>8s}: {c_in:.4f} -> {c_out:.4f} mol/m³   (produced)")
    dT = hist_T[-1] - hist_T[0]
    print(f"  {'T':>8s}: {hist_T[0]:.1f} -> {hist_T[-1]:.1f} K   (ΔT = {dT:+.2f} K)\n")

def plot_pfr_results(z_vals, hist_C, hist_T, species_names, reaction_keys):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle('PFR Simulation — ' + ', '.join(reaction_keys), fontsize=12)
    for i, sp in enumerate(species_names):
        ax1.plot(z_vals, hist_C[:, i], label=sp, linewidth=2)
    ax1.set_ylabel('Concentration (mol/m³)')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(z_vals, hist_T, color='#E24B4A', linewidth=2)
    ax2.set_ylabel('Temperature (K)'); ax2.set_xlabel('Reactor Length (m)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
    _print_summary(hist_C, hist_T, species_names)

def plot_batch_results(t_vals, hist_C, hist_T, species_names, reaction_keys):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle('Batch Reactor Simulation — ' + ', '.join(reaction_keys), fontsize=12)
    for i, sp in enumerate(species_names):
        ax1.plot(t_vals, hist_C[:, i], label=sp, linewidth=2)
    ax1.set_ylabel('Concentration (mol/m³)')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(t_vals, hist_T, color='#E24B4A', linewidth=2)
    ax2.set_ylabel('Temperature (K)'); ax2.set_xlabel('Time (s)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
    _print_summary(hist_C, hist_T, species_names)

def plot_cstr_results(cstr_result, reaction_keys):
    species  = cstr_result['species']
    C_in     = cstr_result['C_in']
    C_out    = cstr_result['C_out']
    res_hist = cstr_result['residual_history']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('CSTR Simulation — ' + ', '.join(reaction_keys), fontsize=12)
    x_pos = np.arange(len(species)); width = 0.35
    ax1.bar(x_pos - width/2, C_in,  width, label='Inlet',  color='#3B82F6')
    ax1.bar(x_pos + width/2, C_out, width, label='Outlet', color='#E24B4A')
    ax1.set_xticks(x_pos); ax1.set_xticklabels(species)
    ax1.set_ylabel('Concentration (mol/m³)'); ax1.legend(); ax1.grid(True, alpha=0.3, axis='y')
    ax2.semilogy(range(len(res_hist)), res_hist, 'o-', color='#6366F1', markersize=5)
    ax2.set_xlabel('Newton-Raphson Iteration'); ax2.set_ylabel('||F(x)||')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

# =============================================================================
# SECTION 8: RUN FUNCTIONS
# =============================================================================

def run_pfr_sim(reaction_keys, initial_concentrations, T0=500, u=0.1, L=1.0, n=1000):
    active_reactions = get_reactions(reaction_keys)
    species_names    = get_all_species(reaction_keys)
    C0_arr           = np.array([initial_concentrations.get(sp, 0.0) for sp in species_names])
    is_stiff = T0 > 900 or any('DRM' in key for key in reaction_keys)
    if is_stiff:
        z_vals, hist_C, hist_T = run_ivp_stiff(species_names, C0_arr, T0, active_reactions, u, L, n)
    else:
        z_vals, hist_C, hist_T = run_rk4(species_names, C0_arr, T0, active_reactions, u, L, n)
    plot_pfr_results(z_vals, hist_C, hist_T, species_names, reaction_keys)
    return z_vals, hist_C, hist_T, species_names

def run_batch_sim(reaction_keys, initial_concentrations, T0=500, t_end=10.0, n=1000):
    active_reactions = get_reactions(reaction_keys)
    species_names    = get_all_species(reaction_keys)
    C0_arr           = np.array([initial_concentrations.get(sp, 0.0) for sp in species_names])
    t_vals, hist_C, hist_T = run_batch(species_names, C0_arr, T0, active_reactions, t_end, n)
    plot_batch_results(t_vals, hist_C, hist_T, species_names, reaction_keys)
    return t_vals, hist_C, hist_T, species_names

def run_cstr_sim(reaction_keys, initial_concentrations, T_in=500, V=0.01, Q=0.001):
    active_reactions = get_reactions(reaction_keys)
    species_names    = get_all_species(reaction_keys)
    C0_arr           = np.array([initial_concentrations.get(sp, 0.0) for sp in species_names])
    result = solve_cstr(species_names, C0_arr, T_in, active_reactions, V, Q)
    plot_cstr_results(result, reaction_keys)
    return result

# =============================================================================
# SECTION 9: COST DATABASE & FINANCIAL ESTIMATION
# =============================================================================

CHEMICAL_PRICES = {
    'CO': 0.50, 'O2': 0.05, 'CO2': 0.03, 'H2': 2.50, 'H2O': 0.002,
    'CH4': 0.30, 'NH3': 0.40, 'NO': 1.50, 'NO2': 1.50, 'N2': 0.02,
    'C2H4': 0.80, 'C2H4O': 1.50, 'CH3OH': 0.35,
}

MOLAR_MASS_KG = {
    'CO': 0.02801, 'O2': 0.03200, 'CO2': 0.04401, 'H2': 0.00202,
    'H2O': 0.01802, 'CH4': 0.01604, 'NH3': 0.01703, 'NO': 0.03001,
    'NO2': 0.04601, 'N2': 0.02801, 'C2H4': 0.02805, 'C2H4O': 0.04405,
    'CH3OH': 0.03204,
}

CATALYST_PRICES = {
    'Pt': 30_000, 'Pd': 50_000, 'Ni': 15, 'Rh': 150_000,
    'Ag': 700, 'Cu-zeolite': 50, 'V2O5/WO3/TiO2': 30,
    'Cu/ZnO/Al2O3': 25, 'Pt/Rh': 60_000,
}

DEFAULT_CATALYST_DENSITY = 1200.0
VESSEL_COST_REF = {
    'PFR':  {'C_ref': 50_000, 'V_ref': 1.0},
    'CSTR': {'C_ref': 65_000, 'V_ref': 1.0},
}
DEFAULT_ELECTRICITY_PRICE = 0.07

def _mass_flow_cost(species, conc_dict, volumetric_flow):
    return {sp: conc_dict.get(sp, 0.0) * volumetric_flow
            * MOLAR_MASS_KG.get(sp, 0.030) for sp in species}

def compute_feedstock_cost(species, conc_dict, volumetric_flow):
    mf = _mass_flow_cost(species, conc_dict, volumetric_flow)
    bk = {sp: kg_s * CHEMICAL_PRICES.get(sp, 0.0) * 3600 for sp, kg_s in mf.items()}
    return sum(bk.values()), bk

def compute_product_value(species, conc_dict, volumetric_flow):
    return compute_feedstock_cost(species, conc_dict, volumetric_flow)

def compute_catalyst_cost(catalyst_name, bed_volume, catalyst_density=DEFAULT_CATALYST_DENSITY):
    price = CATALYST_PRICES.get(catalyst_name, 0.0)
    mass_kg = bed_volume * catalyst_density
    return price * mass_kg, mass_kg

def compute_vessel_cost(reactor_type, volume):
    ref = VESSEL_COST_REF.get(reactor_type, VESSEL_COST_REF['PFR'])
    if volume <= 0:
        return 0.0
    return ref['C_ref'] * (volume / ref['V_ref']) ** 0.6

def compute_energy_cost(thermal_duty_W, electricity_price=DEFAULT_ELECTRICITY_PRICE):
    return abs(thermal_duty_W) / 1000.0 * electricity_price

def estimate_thermal_duty(T_in, T_out, volumetric_flow, rho=1.0, cp=1010.0):
    return abs(rho * volumetric_flow * cp * (T_out - T_in))

def compute_total_cost(species, inlet_conc, outlet_conc, volumetric_flow,
                       reactor_type, reactor_volume, catalyst_name,
                       T_in, T_out,
                       electricity_price=DEFAULT_ELECTRICITY_PRICE,
                       catalyst_density=DEFAULT_CATALYST_DENSITY):
    feed_total, feed_bk = compute_feedstock_cost(species, inlet_conc, volumetric_flow)
    prod_total, prod_bk = compute_product_value(species, outlet_conc, volumetric_flow)
    duty   = estimate_thermal_duty(T_in, T_out, volumetric_flow)
    energy = compute_energy_cost(duty, electricity_price)
    vessel = compute_vessel_cost(reactor_type, reactor_volume)
    cat_cost, cat_mass = compute_catalyst_cost(catalyst_name, reactor_volume, catalyst_density)
    return {
        'feedstock_total': feed_total, 'feedstock_breakdown': feed_bk,
        'product_total': prod_total,   'product_breakdown': prod_bk,
        'net_operating': feed_total - prod_total + energy,
        'energy_cost': energy,
        'vessel_capex': vessel, 'catalyst_capex': cat_cost,
        'catalyst_mass': cat_mass, 'total_capex': vessel + cat_cost,
        'thermal_duty_W': duty,
    }

def print_cost_summary(costs, catalyst_name='', reactor_type='PFR', reactor_volume=0.0):
    print("\n--- Cost Estimate ---")
    print(f"  Feedstock cost:      ${costs['feedstock_total']:.4f}/hr")
    print(f"  Product value:       ${costs['product_total']:.4f}/hr")
    print(f"  Energy cost:         ${costs['energy_cost']:.4f}/hr")
    print(f"  Net operating cost:  ${costs['net_operating']:.4f}/hr")
    print(f"  Vessel CapEx:        ${costs['vessel_capex']:,.0f}")
    print(f"  Catalyst CapEx:      ${costs['catalyst_capex']:,.0f}")
    print(f"  Total CapEx:         ${costs['total_capex']:,.0f}")
    print()

def cost_sensitivity_sweep(species, inlet_conc, outlet_conc,
                           reactor_type, catalyst_name, T_in, T_out,
                           sweep_var='flow_rate', base_Q=0.001, base_V=0.01,
                           n_pts=20, **kw):
    if sweep_var == 'flow_rate':
        vals = np.linspace(max(base_Q*0.2, 1e-6), base_Q*3.0, n_pts)
        label = 'Volumetric flow rate (m³/s)'
    else:
        vals = np.linspace(max(base_V*0.2, 1e-6), base_V*3.0, n_pts)
        label = 'Reactor volume (m³)'
    net_ops, capexes = [], []
    for v in vals:
        q, vol = (v, base_V) if sweep_var == 'flow_rate' else (base_Q, v)
        c = compute_total_cost(species, inlet_conc, outlet_conc, q,
                               reactor_type, vol, catalyst_name, T_in, T_out, **kw)
        net_ops.append(c['net_operating'])
        capexes.append(c['total_capex'])
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(vals, net_ops, 'b-o', markersize=4, label='Net operating ($/hr)')
    ax1.set_xlabel(label); ax1.set_ylabel('Net operating cost ($/hr)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(vals, capexes, 'r--s', markersize=4, label='Total CapEx ($)')
    ax2.set_ylabel('Total CapEx ($)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    fig.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(); plt.show()

# =============================================================================
# SECTION 10: INTERACTIVE ENTRY POINT
# =============================================================================

def _select_reactions():
    list_reactions()
    print("Enter reaction keys one at a time (max 10).")
    print("Press Enter with no input when done.\n")
    reaction_keys = []
    while len(reaction_keys) < 10:
        key = input(f"  Reaction {len(reaction_keys) + 1}: ").strip()
        if key == '':
            if len(reaction_keys) == 0:
                print("  Please enter at least one reaction.")
                continue
            break
        if key not in REACTION_DB:
            print(f"  '{key}' not found. Check the table above and try again.")
            continue
        summarize_reaction(key)
        reaction_keys.append(key)
        print(f"  Added. ({len(reaction_keys)} reaction(s) selected)\n")
    return reaction_keys

def _collect_concentrations(reaction_keys):
    species = get_all_species(reaction_keys)
    print(f"\nSpecies involved: {', '.join(species)}")
    print("Enter inlet concentrations in mol/m³.")
    print("Press Enter to default to 0.\n")
    initial_concentrations = {}
    for sp in species:
        val = input(f"  {sp} concentration: ").strip()
        initial_concentrations[sp] = float(val) if val else 0.0
    return initial_concentrations

def start():
    print("=" * 60)
    print("  CATALYTIC REACTOR SIMULATOR — interactive setup")
    print("=" * 60)
    print("\nSelect reactor type:")
    print("  [1] PFR\n  [2] Batch\n  [3] CSTR\n")
    while True:
        choice = input("  Your choice (1/2/3): ").strip()
        if choice in ('1', '2', '3'):
            break
        print("  Invalid choice.")
    reactor_type = {'1': 'PFR', '2': 'Batch', '3': 'CSTR'}[choice]
    reaction_keys = _select_reactions()
    initial_concentrations = _collect_concentrations(reaction_keys)
    cats = set(get_reaction(k)['catalyst'] for k in reaction_keys)
    primary_cat = sorted(cats)[0]
    cat_label = ', '.join(sorted(cats))
    print("\n--- Reactor conditions (press Enter for defaults) ---")
    T0 = float(input("  Inlet temperature in K        [500]: ").strip() or 500)
    if reactor_type == 'PFR':
        u    = float(input("  Velocity (m/s)                [0.1]: ").strip() or 0.1)
        L    = float(input("  Length (m)                    [1.0]: ").strip() or 1.0)
        n    = int(input(  "  RK4 steps                    [1000]: ").strip() or 1000)
        A_cs = float(input("  Cross-sectional area (m²)    [0.01]: ").strip() or 0.01)
        z_vals, hist_C, hist_T, sp = run_pfr_sim(reaction_keys, initial_concentrations, T0=T0, u=u, L=L, n=n)
        flow_Q = u * A_cs; vol = L * A_cs
        inlet_conc  = {s: hist_C[0, i]  for i, s in enumerate(sp)}
        outlet_conc = {s: hist_C[-1, i] for i, s in enumerate(sp)}
        T_out = hist_T[-1]
    elif reactor_type == 'Batch':
        t_end   = float(input("  Simulation time (s)          [10.0]: ").strip() or 10.0)
        n       = int(input(  "  RK4 steps                    [1000]: ").strip() or 1000)
        V_batch = float(input("  Vessel volume (m³)           [0.01]: ").strip() or 0.01)
        t_vals, hist_C, hist_T, sp = run_batch_sim(reaction_keys, initial_concentrations, T0=T0, t_end=t_end, n=n)
        flow_Q = V_batch / t_end; vol = V_batch
        inlet_conc  = {s: hist_C[0, i]  for i, s in enumerate(sp)}
        outlet_conc = {s: hist_C[-1, i] for i, s in enumerate(sp)}
        T_out = hist_T[-1]
    elif reactor_type == 'CSTR':
        V = float(input("  Reactor volume (m³)          [0.01]: ").strip() or 0.01)
        Q = float(input("  Flow rate (m³/s)            [0.001]: ").strip() or 0.001)
        result = run_cstr_sim(reaction_keys, initial_concentrations, T_in=T0, V=V, Q=Q)
        flow_Q = Q; vol = V; sp = result['species']
        inlet_conc  = {s: result['C_in'][i]  for i, s in enumerate(sp)}
        outlet_conc = {s: result['C_out'][i] for i, s in enumerate(sp)}
        T_out = result['T_out']
    costs = compute_total_cost(sp, inlet_conc, outlet_conc, flow_Q,
        reactor_type if reactor_type != 'Batch' else 'PFR',
        vol, primary_cat, T0, T_out)
    print_cost_summary(costs, catalyst_name=cat_label,
                       reactor_type=reactor_type, reactor_volume=vol)
    cost_sensitivity_sweep(sp, inlet_conc, outlet_conc,
        reactor_type if reactor_type != 'Batch' else 'PFR',
        primary_cat, T0, T_out, sweep_var='flow_rate', base_Q=flow_Q, base_V=vol)

if __name__ == '__main__':
    start()