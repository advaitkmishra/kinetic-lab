from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

from simulator import (
    REACTION_DB,
    get_all_species,
    get_reactions,
    get_reaction,
    run_rk4,
    run_batch,
    solve_cstr,
    compute_dH,
    get_thermo,
    compute_total_cost,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# REQUEST MODEL
# =============================================================================

class SimulationRequest(BaseModel):
    reactor_type: str
    reaction_keys: List[str]
    initial_concentrations: Dict[str, float]
    T0: float = 500
    u: float = 0.1
    L: float = 1.0
    n: int = 1000
    A_cs: float = 0.01
    V: float = 0.01
    Q: float = 0.001
    t_end: float = 10.0

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/reactions")
def get_reaction_list():
    result = []
    for key, rxn in REACTION_DB.items():
        result.append({
            "key":      key,
            "family":   rxn["family"],
            "equation": rxn["equation"],
            "catalyst": rxn["catalyst"],
            "T_range":  rxn["T_range"],
            "note":     rxn["note"],
            "source":   rxn["source"],
        })
    return result


@app.post("/simulate")
def simulate(req: SimulationRequest):
    active_reactions = get_reactions(req.reaction_keys)
    species_names    = get_all_species(req.reaction_keys)
    C0_arr           = np.array([req.initial_concentrations.get(sp, 0.0)
                                  for sp in species_names])

    # --- Run simulation ---
    if req.reactor_type == "PFR":
        z_vals, hist_C, hist_T = run_rk4(
            species_names, C0_arr, req.T0,
            active_reactions, req.u, req.L, req.n
        )
        x_vals  = z_vals.tolist()
        x_label = "Reactor Length (m)"
        T_out   = float(hist_T[-1])
        inlet_conc  = {sp: float(hist_C[0, i])  for i, sp in enumerate(species_names)}
        outlet_conc = {sp: float(hist_C[-1, i]) for i, sp in enumerate(species_names)}
        flow_Q = req.u * req.A_cs
        vol    = req.L * req.A_cs

    elif req.reactor_type == "Batch":
        t_vals, hist_C, hist_T = run_batch(
            species_names, C0_arr, req.T0,
            active_reactions, req.t_end, req.n
        )
        x_vals  = t_vals.tolist()
        x_label = "Time (s)"
        T_out   = float(hist_T[-1])
        inlet_conc  = {sp: float(hist_C[0, i])  for i, sp in enumerate(species_names)}
        outlet_conc = {sp: float(hist_C[-1, i]) for i, sp in enumerate(species_names)}
        flow_Q = req.V / req.t_end
        vol    = req.V

    elif req.reactor_type == "CSTR":
        result = solve_cstr(
            species_names, C0_arr, req.T0,
            active_reactions, req.V, req.Q
        )
        T_out       = float(result["T_out"])
        inlet_conc  = {sp: float(C0_arr[i])            for i, sp in enumerate(species_names)}
        outlet_conc = {sp: float(result["C_out"][i])   for i, sp in enumerate(species_names)}
        flow_Q = req.Q
        vol    = req.V

        conversions = {}
        for sp in species_names:
            c_in  = inlet_conc[sp]
            c_out = outlet_conc[sp]
            conversions[sp] = round((c_in - c_out) / c_in * 100, 2) if c_in > 1e-8 else None

        costs = _compute_costs(
            req, species_names, active_reactions, inlet_conc, outlet_conc,
            flow_Q, vol, req.T0, T_out
        )
        return {
            "reactor_type": "CSTR",
            "species":      species_names,
            "inlet":        C0_arr.tolist(),
            "outlet":       result["C_out"].tolist(),
            "T_in":         req.T0,
            "T_out":        T_out,
            "converged":    result["converged"],
            "conversions":  conversions,
            "costs":        costs,
        }

    # --- Concentration profiles ---
    concentration_profiles = {}
    for i, sp in enumerate(species_names):
        concentration_profiles[sp] = hist_C[:, i].tolist()

    # --- Conversions ---
    conversions = {}
    for i, sp in enumerate(species_names):
        c_in  = hist_C[0, i]
        c_out = hist_C[-1, i]
        conversions[sp] = round((c_in - c_out) / c_in * 100, 2) if c_in > 1e-8 else None

    # --- Cost analysis ---
    costs = _compute_costs(
        req, species_names, active_reactions, inlet_conc, outlet_conc,
        flow_Q, vol, req.T0, T_out
    )

    return {
        "reactor_type":            req.reactor_type,
        "species":                 species_names,
        "x_vals":                  x_vals,
        "x_label":                 x_label,
        "concentration_profiles":  concentration_profiles,
        "temperature":             hist_T.tolist(),
        "T_in":                    req.T0,
        "T_out":                   T_out,
        "conversions":             conversions,
        "costs":                   costs,
    }


def _compute_costs(req, species_names, active_reactions, inlet_conc, outlet_conc,
                   flow_Q, vol, T_in, T_out):
    cats = set(rxn["catalyst"] for rxn in active_reactions)
    primary_cat = sorted(cats)[0]
    reactor_type_for_cost = req.reactor_type if req.reactor_type != "Batch" else "PFR"
    try:
        c = compute_total_cost(
            species_names, inlet_conc, outlet_conc, flow_Q,
            reactor_type_for_cost, vol, primary_cat, T_in, T_out
        )
        return {
            "feedstock_cost":   round(c["feedstock_total"], 4),
            "product_value":    round(c["product_total"], 4),
            "energy_cost":      round(c["energy_cost"], 4),
            "net_operating":    round(c["net_operating"], 4),
            "vessel_capex":     round(c["vessel_capex"], 0),
            "catalyst_capex":   round(c["catalyst_capex"], 0),
            "catalyst_mass_kg": round(c["catalyst_mass"], 2),
            "total_capex":      round(c["total_capex"], 0),
            "thermal_duty_W":   round(c["thermal_duty_W"], 2),
            "catalyst":         primary_cat,
            "flow_Q":           round(flow_Q, 6),
            "volume_m3":        round(vol, 6),
        }
    except Exception as e:
        return {"error": str(e)}