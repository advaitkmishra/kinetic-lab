import { useState, useEffect } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from "recharts";

const API = "http://127.0.0.1:8000";

const COLORS = [
  "#6dbf67", "#60a5fa", "#f97316", "#e879f9",
  "#facc15", "#34d399", "#f87171", "#a78bfa"
];

const TEMP_GLOBAL_MIN = 373;
const TEMP_GLOBAL_MAX = 1273;

const DISPLAY_NAMES = {
  'CO_oxidation':               'CO Oxidation (Pt)',
  'CO_oxidation_Pd':            'CO Oxidation (Pd)',
  'CO_oxidation_inhibited':     'CO Oxidation — High Concentration',
  'WGS_forward':                'Water-Gas Shift',
  'WGS_reverse':                'Reverse Water-Gas Shift',
  'SMR_R1':                     'Steam Reforming — Primary',
  'SMR_R2':                     'Steam Reforming — Full',
  'SMR_WGS':                    'Steam Reforming + WGS',
  'CH4_oxidation_Pt':           'Methane Combustion (Pt)',
  'CH4_oxidation_Pd':           'Methane Combustion (Pd)',
  'CH4_partial_oxidation':      'Methane Partial Oxidation',
  'SCR_standard':               'NOx Reduction — Standard',
  'SCR_fast':                   'NOx Reduction — Fast',
  'SCR_NO2':                    'NOx Reduction — NO2 Rich',
  'NH3_oxidation':              'Ammonia Slip Oxidation',
  'NO_oxidation':               'NO to NO2 Oxidation',
  'SCR_Cu_standard':            'NOx Reduction (Cu-Zeolite)',
  'DRM':                        'Dry Reforming (Ni)',
  'DRM_Rh':                     'Dry Reforming (Rh)',
  'DRM_carbon_deposition':      'Methane Cracking',
  'ethylene_oxidation_complete':'Ethylene Combustion',
  'ethylene_oxide_synthesis':   'Ethylene Oxide Synthesis',
  'CO2_methanation':            'CO2 Methanation (Sabatier)',
  'CO_methanation':             'CO Methanation',
  'methanol_synthesis':         'Methanol Synthesis',
};

const FAMILY_CONFIG = {
  'CO oxidation':            { cardBg: '#1a1208', border: '#BA7517', label: '#EF9F27' },
  'Water-gas shift':         { cardBg: '#081522', border: '#185FA5', label: '#60a5fa' },
  'Steam methane reforming': { cardBg: '#071a10', border: '#0F6E56', label: '#1D9E75' },
  'Methane oxidation':       { cardBg: '#1a0d08', border: '#993C1D', label: '#D85A30' },
  'NOx SCR':                 { cardBg: '#100d1a', border: '#534AB7', label: '#a78bfa' },
  'Dry reforming':           { cardBg: '#0d1a07', border: '#3B6D11', label: '#97C459' },
  'Partial oxidation':       { cardBg: '#1a1408', border: '#854F0B', label: '#facc15' },
  'CO2 hydrogenation':       { cardBg: '#180b15', border: '#993556', label: '#e879f9' },
};

const CATALYST_CONFIG = {
  'Pt':             { bg: '#2a2a2a', color: '#d4d4d4', border: '#888' },
  'Pd':             { bg: '#0c2a45', color: '#60a5fa', border: '#185FA5' },
  'Ni':             { bg: '#0d1f08', color: '#86efac', border: '#3B6D11' },
  'Rh':             { bg: '#1a1030', color: '#c4b5fd', border: '#534AB7' },
  'Ag':             { bg: '#1e1e1e', color: '#e2e8f0', border: '#64748b' },
  'Cu-zeolite':     { bg: '#1f1200', color: '#fbbf24', border: '#854F0B' },
  'V2O5/WO3/TiO2': { bg: '#1a0a0a', color: '#fca5a5', border: '#993C1D' },
  'Cu/ZnO/Al2O3':  { bg: '#051a14', color: '#5eead4', border: '#0F6E56' },
  'Pt/Rh':          { bg: '#1a0d18', color: '#f0abfc', border: '#993556' },
};

function tempToColor(T) {
  const t = Math.max(0, Math.min(1, (T - TEMP_GLOBAL_MIN) / (TEMP_GLOBAL_MAX - TEMP_GLOBAL_MIN)));
  const r = Math.round(30 + t * 210);
  const g = Math.round(80 - t * 60);
  const b = Math.round(220 - t * 190);
  return `rgb(${r},${g},${b})`;
}

function TempBar({ tMin, tMax }) {
  const startPct = ((tMin - TEMP_GLOBAL_MIN) / (TEMP_GLOBAL_MAX - TEMP_GLOBAL_MIN)) * 100;
  const endPct   = ((tMax - TEMP_GLOBAL_MIN) / (TEMP_GLOBAL_MAX - TEMP_GLOBAL_MIN)) * 100;
  const widthPct = endPct - startPct;
  const startColor = tempToColor(tMin);
  const endColor   = tempToColor(tMax);
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#4a6a52', marginBottom: 3 }}>
        <span>{TEMP_GLOBAL_MIN}K</span>
        <span>{TEMP_GLOBAL_MAX}K</span>
      </div>
      <div style={{ position: 'relative', height: 6, background: '#1e3a28', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', left: `${startPct}%`, width: `${widthPct}%`,
          height: '100%', borderRadius: 4,
          background: `linear-gradient(to right, ${startColor}, ${endColor})`
        }} />
      </div>
      <div style={{ position: 'relative', height: 16, marginTop: 3 }}>
        <span style={{
          position: 'absolute', left: `clamp(0%, ${startPct}%, 85%)`,
          fontSize: 10, color: startColor,
          transform: 'translateX(-10%)', whiteSpace: 'nowrap'
        }}>{tMin}K</span>
        <span style={{
          position: 'absolute', left: `clamp(15%, ${endPct}%, 100%)`,
          fontSize: 10, color: endColor,
          transform: 'translateX(-90%)', whiteSpace: 'nowrap'
        }}>{tMax}K</span>
      </div>
    </div>
  );
}

function CatalystBadge({ catalyst }) {
  const cfg = CATALYST_CONFIG[catalyst] || { bg: '#1e3a28', color: '#8aaa90', border: '#1e3a28' };
  return (
    <span style={{
      background: cfg.bg, color: cfg.color, border: `1px solid ${cfg.border}`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 500
    }}>{catalyst}</span>
  );
}

function CostPanel({ costs }) {
  if (!costs || costs.error) return null;

  const netPositive = costs.net_operating <= 0;

  const metricCard = (label, value, sub, color) => (
    <div style={{ background: "#0d1f14", border: "1px solid #1e3a28", borderRadius: 8, padding: "12px 16px" }}>
      <p style={{ fontSize: 11, color: "#8aaa90", marginBottom: 4 }}>{label}</p>
      <p style={{ fontSize: 20, fontFamily: "monospace", color: color || "white", fontWeight: 600 }}>{value}</p>
      {sub && <p style={{ fontSize: 11, color: "#8aaa90", marginTop: 2 }}>{sub}</p>}
    </div>
  );

  return (
    <div style={{ background: "#112218", border: "1px solid #1e3a28", borderRadius: 8, padding: 16, marginTop: 16 }}>
      <p style={{ fontSize: 12, color: "#8aaa90", marginBottom: 14, letterSpacing: 1 }}>ECONOMIC ANALYSIS</p>

      {/* Operating costs row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 12 }}>
        {metricCard("Feedstock cost", `$${costs.feedstock_cost.toFixed(4)}/hr`, "inlet chemicals", "#f87171")}
        {metricCard("Product value", `$${costs.product_value.toFixed(4)}/hr`, "outlet chemicals", "#6dbf67")}
        {metricCard("Energy cost", `$${costs.energy_cost.toFixed(4)}/hr`, `${costs.thermal_duty_W.toFixed(1)} W thermal`, "#facc15")}
        {metricCard(
          "Net operating",
          `$${Math.abs(costs.net_operating).toFixed(4)}/hr`,
          netPositive ? "net profit" : "net cost",
          netPositive ? "#6dbf67" : "#f87171"
        )}
      </div>

      {/* CapEx row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 12 }}>
        {metricCard("Vessel CapEx", `$${Number(costs.vessel_capex).toLocaleString()}`, `${costs.volume_m3} m³ reactor`, "#60a5fa")}
        {metricCard("Catalyst CapEx", `$${Number(costs.catalyst_capex).toLocaleString()}`, `${costs.catalyst_mass_kg} kg ${costs.catalyst}`, "#a78bfa")}
        {metricCard("Total CapEx", `$${Number(costs.total_capex).toLocaleString()}`, "one-time capital cost", "#e879f9")}
      </div>

      {/* Per-24h projection */}
      <div style={{
        background: "#0a1a0f", border: "1px solid #1e3a28", borderRadius: 6,
        padding: "10px 14px", display: "flex", gap: 32, alignItems: "center"
      }}>
        <span style={{ fontSize: 11, color: "#8aaa90" }}>Per 24 hr projection:</span>
        <span style={{ fontSize: 13, color: "#f87171", fontFamily: "monospace" }}>
          Feedstock: ${(costs.feedstock_cost * 24).toFixed(2)}
        </span>
        <span style={{ fontSize: 13, color: "#6dbf67", fontFamily: "monospace" }}>
          Product: ${(costs.product_value * 24).toFixed(2)}
        </span>
        <span style={{ fontSize: 13, color: "#facc15", fontFamily: "monospace" }}>
          Energy: ${(costs.energy_cost * 24).toFixed(2)}
        </span>
        <span style={{
          fontSize: 14, fontFamily: "monospace", fontWeight: 700,
          color: netPositive ? "#6dbf67" : "#f87171"
        }}>
          Net: {netPositive ? "+" : "-"}${Math.abs(costs.net_operating * 24).toFixed(2)}
        </span>
      </div>
    </div>
  );
}

export default function App() {
  const [reactions, setReactions]           = useState([]);
  const [selectedRxns, setSelectedRxns]     = useState([]);
  const [reactorType, setReactorType]       = useState("PFR");
  const [concentrations, setConcentrations] = useState({});
  const [conditions, setConditions]         = useState({
    T0: 500, u: 0.1, L: 1.0, n: 1000,
    V: 0.01, Q: 0.001, t_end: 10.0, A_cs: 0.01
  });
  const [results, setResults]     = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [activeTab, setActiveTab] = useState("simulator");
  const [familyFilter, setFamilyFilter] = useState("All");

  useEffect(() => {
    axios.get(`${API}/reactions`).then(r => setReactions(r.data));
  }, []);

  const families = ["All", ...new Set(reactions.map(r => r.family))];

  const toggleReaction = (key) => {
    setSelectedRxns(prev =>
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
    );
  };

  const runSimulation = async () => {
    if (selectedRxns.length === 0) { setError("Select at least one reaction."); return; }
    setLoading(true); setError(null);
    try {
      const res = await axios.post(`${API}/simulate`, {
        reactor_type: reactorType,
        reaction_keys: selectedRxns,
        initial_concentrations: concentrations,
        ...conditions
      });
      setResults(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || "Simulation failed.");
    }
    setLoading(false);
  };

  const chartData = results && results.x_vals
    ? results.x_vals.map((x, i) => {
        const point = { x: +x.toFixed(4) };
        results.species.forEach(sp => { point[sp] = +(results.concentration_profiles[sp][i]).toFixed(5); });
        return point;
      })
    : [];

  const tempData = results && results.temperature
    ? results.x_vals.map((x, i) => ({ x: +x.toFixed(4), T: +results.temperature[i].toFixed(2) }))
    : [];

  const inputStyle = {
    background: "#0d1f14", border: "1px solid #1e3a28", color: "white",
    padding: "6px 10px", borderRadius: 6, fontSize: 13, width: "100%", outline: "none"
  };

  return (
    <div style={{ minHeight: "100vh" }}>

      {/* Navbar */}
      <nav style={{
        background: "#0d1f14", borderBottom: "1px solid #1e3a28",
        padding: "0 24px", display: "flex", alignItems: "center",
        justifyContent: "space-between", height: 52
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
          <span style={{ color: "#6dbf67", fontWeight: 700, fontSize: 16, letterSpacing: 1 }}>KINETIC LAB</span>
          {["simulator", "registry"].map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)} style={{
              background: "none", color: activeTab === tab ? "#6dbf67" : "#8aaa90",
              fontWeight: activeTab === tab ? 600 : 400,
              borderBottom: activeTab === tab ? "2px solid #6dbf67" : "none",
              borderRadius: 0, padding: "4px 0", fontSize: 13,
              textTransform: "uppercase", letterSpacing: 1
            }}>{tab}</button>
          ))}
        </div>
        <button onClick={runSimulation} style={{
          background: "#6dbf67", color: "#0a1a0f", fontWeight: 700, padding: "8px 20px", fontSize: 13
        }}>
          {loading ? "Running..." : "Run Simulation"}
        </button>
      </nav>

      {/* ── SIMULATOR TAB ── */}
      {activeTab === "simulator" && (
        <div style={{ display: "flex", height: "calc(100vh - 52px)" }}>

          <div style={{
            width: 240, background: "#0d1f14", borderRight: "1px solid #1e3a28",
            padding: 16, overflowY: "auto", flexShrink: 0
          }}>
            <div style={{ marginBottom: 16 }}>
              <label style={{ fontSize: 11, color: "#8aaa90", textTransform: "uppercase", letterSpacing: 1 }}>Reactor Type</label>
              <select value={reactorType} onChange={e => setReactorType(e.target.value)}
                style={{ ...inputStyle, marginTop: 6 }}>
                <option value="PFR">Plug Flow Reactor</option>
                <option value="Batch">Batch Reactor</option>
                <option value="CSTR">CSTR</option>
              </select>
            </div>

            <div style={{ marginBottom: 16 }}>
              <label style={{ fontSize: 11, color: "#8aaa90", textTransform: "uppercase", letterSpacing: 1 }}>Selected Reactions</label>
              {selectedRxns.length === 0
                ? <p style={{ fontSize: 12, color: "#8aaa90", marginTop: 6 }}>None — pick from Registry</p>
                : selectedRxns.map(k => {
                    const rxn = reactions.find(r => r.key === k);
                    const cfg = rxn ? (FAMILY_CONFIG[rxn.family] || {}) : {};
                    return (
                      <div key={k} style={{
                        display: "flex", justifyContent: "space-between", alignItems: "center",
                        background: cfg.cardBg || "#112218",
                        border: `1px solid ${cfg.border || '#1e3a28'}`,
                        borderRadius: 6, padding: "6px 10px", marginTop: 6, fontSize: 12
                      }}>
                        <span style={{ color: cfg.label || "#6dbf67" }}>{DISPLAY_NAMES[k] || k}</span>
                        <button onClick={() => toggleReaction(k)} style={{ background: "none", color: "#8aaa90", padding: "0 4px", fontSize: 14 }}>×</button>
                      </div>
                    );
                  })
              }
            </div>

            <div style={{ marginBottom: 16 }}>
              <label style={{ fontSize: 11, color: "#8aaa90", textTransform: "uppercase", letterSpacing: 1 }}>
                Inlet Concentrations (mol/m³)
              </label>
              {selectedRxns.length === 0
                ? <p style={{ fontSize: 12, color: "#8aaa90", marginTop: 6 }}>Select reactions first</p>
                : reactions
                    .filter(r => selectedRxns.includes(r.key))
                    .flatMap(r => Object.entries(
                      Object.fromEntries([...r.equation.matchAll(/([A-Z][A-Za-z0-9]*)/g)].map(m => [m[1], 1]))
                    ))
                    .filter((v, i, a) => a.findIndex(x => x[0] === v[0]) === i)
                    .map(([sp]) => (
                      <div key={sp} style={{ marginTop: 8 }}>
                        <label style={{ fontSize: 11, color: "#8aaa90" }}>{sp}</label>
                        <input type="number" step="0.01" value={concentrations[sp] ?? ""} placeholder="0.0"
                          onChange={e => setConcentrations(prev => ({ ...prev, [sp]: parseFloat(e.target.value) || 0 }))}
                          style={{ ...inputStyle, marginTop: 3 }}
                        />
                      </div>
                    ))
              }
            </div>

            <div style={{ marginBottom: 16 }}>
              <label style={{ fontSize: 11, color: "#8aaa90", textTransform: "uppercase", letterSpacing: 1 }}>Conditions</label>
              {[
                { label: "Inlet Temp (K)", key: "T0" },
                ...(reactorType === "PFR"
                  ? [{ label: "Velocity (m/s)", key: "u" },
                     { label: "Length (m)", key: "L" },
                     { label: "Cross-section (m²)", key: "A_cs" },
                     { label: "RK4 Steps", key: "n" }]
                  : reactorType === "Batch"
                  ? [{ label: "Time (s)", key: "t_end" },
                     { label: "Volume (m³)", key: "V" },
                     { label: "RK4 Steps", key: "n" }]
                  : [{ label: "Volume (m³)", key: "V" },
                     { label: "Flow Rate (m³/s)", key: "Q" }])
              ].map(({ label, key }) => (
                <div key={key} style={{ marginTop: 8 }}>
                  <label style={{ fontSize: 11, color: "#8aaa90" }}>{label}</label>
                  <input type="number" value={conditions[key]}
                    onChange={e => setConditions(prev => ({ ...prev, [key]: parseFloat(e.target.value) || 0 }))}
                    style={{ ...inputStyle, marginTop: 3 }}
                  />
                </div>
              ))}
            </div>

            {error && <p style={{ color: "#f87171", fontSize: 12 }}>{error}</p>}
          </div>

          {/* Results */}
          <div style={{ flex: 1, padding: 20, overflowY: "auto" }}>
            {!results && !loading && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", flexDirection: "column", gap: 12 }}>
                <span style={{ fontSize: 32 }}>⚗️</span>
                <p style={{ color: "#8aaa90", fontSize: 14 }}>Select reactions from the Registry tab, set conditions, and run.</p>
              </div>
            )}
            {loading && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
                <p style={{ color: "#6dbf67", fontSize: 14 }}>Running simulation...</p>
              </div>
            )}
            {results && !loading && (
              <>
                {/* Conversion metric cards */}
                <div style={{ display: "grid", gridTemplateColumns: `repeat(${Math.min(results.species.length + 1, 6)}, 1fr)`, gap: 10, marginBottom: 20 }}>
                  <div style={{ background: "#112218", border: "1px solid #1e3a28", borderRadius: 8, padding: "12px 16px" }}>
                    <p style={{ fontSize: 11, color: "#8aaa90" }}>EXIT TEMP</p>
                    <p style={{ fontSize: 22, fontFamily: "monospace", color: "#6dbf67", marginTop: 4 }}>{results.T_out.toFixed(1)} K</p>
                    <p style={{ fontSize: 11, color: "#8aaa90" }}>ΔT = {(results.T_out - results.T_in).toFixed(2)} K</p>
                  </div>
                  {results.species.map((sp, i) => (
                    <div key={sp} style={{ background: "#112218", border: "1px solid #1e3a28", borderRadius: 8, padding: "12px 16px" }}>
                      <p style={{ fontSize: 11, color: "#8aaa90" }}>{sp}</p>
                      <p style={{ fontSize: 22, fontFamily: "monospace", color: COLORS[i % COLORS.length], marginTop: 4 }}>
                        {results.conversions[sp] !== null ? `${results.conversions[sp]}%` : "—"}
                      </p>
                      <p style={{ fontSize: 11, color: "#8aaa90" }}>conversion</p>
                    </div>
                  ))}
                </div>

                {/* Concentration chart */}
                <div style={{ background: "#112218", border: "1px solid #1e3a28", borderRadius: 8, padding: 16, marginBottom: 16 }}>
                  <p style={{ fontSize: 12, color: "#8aaa90", marginBottom: 12 }}>SPECIES CONCENTRATION PROFILES</p>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e3a28" />
                      <XAxis dataKey="x" stroke="#8aaa90" fontSize={11}
                        label={{ value: results.x_label, position: "insideBottom", offset: -4, fill: "#8aaa90", fontSize: 11 }} />
                      <YAxis stroke="#8aaa90" fontSize={11} />
                      <Tooltip contentStyle={{ background: "#0d1f14", border: "1px solid #1e3a28", fontSize: 12 }} />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      {results.species.map((sp, i) => (
                        <Line key={sp} type="monotone" dataKey={sp} stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={2} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Temperature chart */}
                <div style={{ background: "#112218", border: "1px solid #1e3a28", borderRadius: 8, padding: 16 }}>
                  <p style={{ fontSize: 12, color: "#8aaa90", marginBottom: 12 }}>TEMPERATURE PROFILE</p>
                  <ResponsiveContainer width="100%" height={180}>
                    <LineChart data={tempData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e3a28" />
                      <XAxis dataKey="x" stroke="#8aaa90" fontSize={11}
                        label={{ value: results.x_label, position: "insideBottom", offset: -4, fill: "#8aaa90", fontSize: 11 }} />
                      <YAxis stroke="#8aaa90" fontSize={11} />
                      <Tooltip contentStyle={{ background: "#0d1f14", border: "1px solid #1e3a28", fontSize: 12 }} />
                      <Line type="monotone" dataKey="T" stroke="#f97316" dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Economic analysis */}
                <CostPanel costs={results.costs} />
              </>
            )}
          </div>
        </div>
      )}

      {/* ── REGISTRY TAB ── */}
      {activeTab === "registry" && (
        <div style={{ padding: 24 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <div>
              <h2 style={{ fontSize: 20, fontWeight: 700 }}>Reaction Registry</h2>
              <p style={{ fontSize: 13, color: "#8aaa90", marginTop: 4 }}>{reactions.length} reactions available</p>
            </div>
            <select value={familyFilter} onChange={e => setFamilyFilter(e.target.value)}
              style={{ background: "#0d1f14", border: "1px solid #1e3a28", color: "white", padding: "6px 12px", borderRadius: 6, fontSize: 13 }}>
              {families.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>

          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
            {Object.entries(FAMILY_CONFIG).map(([family, cfg]) => (
              <span key={family} onClick={() => setFamilyFilter(family === familyFilter ? "All" : family)}
                style={{
                  background: cfg.cardBg, border: `1px solid ${cfg.border}`,
                  color: cfg.label, borderRadius: 20, padding: "4px 12px",
                  fontSize: 12, cursor: "pointer", fontWeight: 500,
                  opacity: familyFilter !== "All" && familyFilter !== family ? 0.35 : 1,
                  transition: "opacity 0.15s"
                }}>
                {family}
              </span>
            ))}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
            <span style={{ fontSize: 11, color: "#8aaa90" }}>Temperature scale:</span>
            <div style={{
              width: 180, height: 8, borderRadius: 4,
              background: `linear-gradient(to right, ${tempToColor(373)}, ${tempToColor(800)}, ${tempToColor(1273)})`
            }} />
            <span style={{ fontSize: 11, color: tempToColor(373) }}>373 K — cold</span>
            <span style={{ fontSize: 11, color: tempToColor(1273) }}>1273 K — hot</span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 12 }}>
            {reactions
              .filter(r => familyFilter === "All" || r.family === familyFilter)
              .map(r => {
                const selected = selectedRxns.includes(r.key);
                const cfg = FAMILY_CONFIG[r.family] || { cardBg: "#112218", border: "#1e3a28", label: "#6dbf67" };
                return (
                  <div key={r.key} onClick={() => toggleReaction(r.key)} style={{
                    background: selected ? cfg.cardBg : "#0f1a12",
                    border: `1px solid ${selected ? cfg.border : '#1e3a28'}`,
                    boxShadow: selected ? `0 0 0 1px ${cfg.border}` : "none",
                    borderRadius: 8, padding: 14, cursor: "pointer", transition: "all 0.15s"
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                      <span style={{ fontSize: 10, color: cfg.label, textTransform: "uppercase", letterSpacing: 1, fontWeight: 600 }}>
                        {r.family}
                      </span>
                      {selected && <span style={{ color: cfg.label, fontSize: 11, fontWeight: 600 }}>✓</span>}
                    </div>
                    <p style={{ fontWeight: 600, fontSize: 13, marginBottom: 4, color: "white" }}>
                      {DISPLAY_NAMES[r.key] || r.key}
                    </p>
                    <p style={{ fontSize: 11, color: cfg.label, fontFamily: "monospace", marginBottom: 10, lineHeight: 1.6 }}>
                      {r.equation}
                    </p>
                    <CatalystBadge catalyst={r.catalyst} />
                    <TempBar tMin={r.T_range[0]} tMax={r.T_range[1]} />
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}