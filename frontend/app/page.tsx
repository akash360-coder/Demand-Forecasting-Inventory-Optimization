"use client";

import { useEffect, useState } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type TrendPoint = { date: string; forecast: number };
type DashboardData = {
  demand_today: number;
  inventory_risk: number;
  average_forecast_error: number;
  recommended_reorder: number;
  risk_breakdown: Record<string, number>;
  trend: TrendPoint[];
};
type ExplanationData = {
  prediction: number;
  top_features: { feature: string; shap_value: number; direction: string }[];
  summary: string;
};
type SimulationOptions = { products: { product_id: string; name: string }[]; stores: { store_id: number; name: string; region: string }[] };
type SimulationData = {
  baseline: { forecast_demand: number; forecast: { date: string; predicted_demand: number }[]; inventory: Record<string, number | string | null> };
  result: { forecast_demand: number; forecast: { date: string; predicted_demand: number }[]; inventory: Record<string, number | string | null> };
  impact: { demand_change: number; demand_change_percent: number | null };
  explanation: { features: { feature: string; shap_value: number; direction: string }[]; summary: string };
};
type MonitoringData = { status: string; data_quality: { row_count: number; duplicate_rate: number; invalid_count: number; status: string }; feature_drift: { feature: string; score: number; status: string }[]; target_drift: { score: number; status: string }; prediction_drift: { score: number; status: string }; model_performance: { metric: string; reference_value: number; current_value: number; change: number; status: string }[]; alerts: { severity: string; category: string; message: string; feature: string | null; score: number }[] };
type ModelRegistryData = { models: { model_version: string; model_name: string; model_type: string; status: string; is_production: boolean; metrics: Record<string, number> }[] };
type AccuracyData = { summary: { wmape: number; mae: number; rmse: number; mape: number; bias: number; over_forecast_rate: number; under_forecast_rate: number; observation_count: number }; breakdowns: Record<string, { product_id?: string; product_name?: string; store_id?: number; category?: string; region?: string; wmape: number; mae: number; bias: number; actual_demand: number; forecast_demand: number; status: string }[]>; trends: Record<string, { date: string; actual_demand: number; forecast_demand: number; wmape: number }[]>; bias: { label: string; over_forecast_count: number; under_forecast_count: number }; business_impact: { under_forecast_units: number; over_forecast_units: number; stockout_risk_count: number; excess_inventory_risk_count: number }; metadata: { model_name: string; model_version: string } };
type InventoryIntelligenceData = {
  summary: {
    total_products: number;
    total_stores: number;
    total_inventory_units: number;
    stockout_risk_count: number;
    excess_inventory_count: number;
    critical_inventory_count: number;
    average_health_score: number;
    abc_distribution: Record<string, number>;
    xyz_distribution: Record<string, number>;
    abc_xyz_distribution: Record<string, number>;
  };
  inventory_health: {
    average_score: number;
    health_band_counts: Record<string, number>;
    top_critical_products: string[];
  };
  risk: {
    stockout_risk_distribution: Record<string, number>;
    excess_inventory_distribution: Record<string, number>;
    risk_matrix_data: Record<string, number>;
  };
  abc_xyz: { class_: string; product_count: number; business_value: number; percentage_contribution: number; demand_variability: number }[];
  opportunities: { product_id: string; store_id: number; priority: string; opportunity_type: string; relevant_metric: string; current_value: number; threshold: number; explanation: string }[];
  service_level: { service_level: number; z_score: number; safety_stock: number; reorder_point: number; target_inventory: number; recommended_order: number }[];
};

const fallbackData: DashboardData = {
  demand_today: 316,
  inventory_risk: 0.41,
  average_forecast_error: 17.8,
  recommended_reorder: 845,
  risk_breakdown: { stockout: 0.22, excess: 0.19, on_time: 0.78 },
  trend: [
    { date: "2025-08-01", forecast: 180 },
    { date: "2025-08-02", forecast: 210 },
    { date: "2025-08-03", forecast: 240 },
    { date: "2025-08-04", forecast: 230 },
    { date: "2025-08-05", forecast: 260 },
  ],
};

function formatNumber(value: number, compact = false) {
  if (compact && Math.abs(value) >= 1000000) return `${(value / 1000000).toFixed(2).replace(/\.?0+$/, "")}M`;
  if (compact && Math.abs(value) >= 1000) return `${(value / 1000).toFixed(1).replace(/\.?0+$/, "")}K`;
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value);
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1).replace(/\.0$/, "")}%`;
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" }).format(new Date(value));
}

function StatCard({ label, value, description, accent }: { label: string; value: string; description: string; accent: string }) {
  return (
    <div className="min-w-0 rounded-xl border border-slate-800 bg-slate-900/80 p-5 shadow-lg shadow-slate-950/20">
      <div className="flex items-center gap-2 text-sm font-medium text-slate-400">
        <span className={`h-2 w-2 shrink-0 rounded-full ${accent}`} />
        <span className="truncate">{label}</span>
      </div>
      <div className="mt-3 truncate text-3xl font-semibold tracking-tight text-white">{value}</div>
      <div className="mt-2 truncate text-xs text-slate-500">{description}</div>
    </div>
  );
}

export default function Page() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [isDemo, setIsDemo] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [explanation, setExplanation] = useState<ExplanationData | null>(null);
  const [explanationUnavailable, setExplanationUnavailable] = useState(false);
  const [options, setOptions] = useState<SimulationOptions | null>(null);
  const [simulation, setSimulation] = useState<SimulationData | null>(null);
  const [simulationError, setSimulationError] = useState("");
  const [simulationLoading, setSimulationLoading] = useState(false);
  const [simProduct, setSimProduct] = useState("P100");
  const [simStore, setSimStore] = useState(1);
  const [simHorizon, setSimHorizon] = useState(7);
  const [simPrice, setSimPrice] = useState(100);
  const [simPromotion, setSimPromotion] = useState(false);
  const [simHoliday, setSimHoliday] = useState(false);
  const [simLeadTime, setSimLeadTime] = useState(5);
  const [simInventory, setSimInventory] = useState(120);
  const [monitoring, setMonitoring] = useState<MonitoringData | null>(null);
  const [registry, setRegistry] = useState<ModelRegistryData | null>(null);
  const [retraining, setRetraining] = useState("");
  const [accuracy, setAccuracy] = useState<AccuracyData | null>(null);
  const [inventory, setInventory] = useState<InventoryIntelligenceData | null>(null);
  const [inventoryError, setInventoryError] = useState("");
  const [inventoryLoading, setInventoryLoading] = useState(true);
  const [inventoryFilters, setInventoryFilters] = useState({ product_id: "", store_id: "", abc_class: "", xyz_class: "", risk_level: "", health_band: "" });

  async function loadInventoryData(filters = inventoryFilters) {
    setInventoryLoading(true);
    setInventoryError("");
    try {
      const params = new URLSearchParams();
      if (filters.product_id) params.set("product_id", filters.product_id);
      if (filters.store_id) params.set("store_id", String(filters.store_id));
      if (filters.abc_class) params.set("abc_class", filters.abc_class);
      if (filters.xyz_class) params.set("xyz_class", filters.xyz_class);
      if (filters.risk_level) params.set("risk_level", filters.risk_level);
      if (filters.health_band) params.set("health_band", filters.health_band);
      const response = await fetch(`http://localhost:8000/api/v1/analytics/inventory-intelligence${params.toString() ? `?${params.toString()}` : ""}`);
      if (!response.ok) throw new Error((await response.json()).detail || "Inventory intelligence request failed");
      setInventory(await response.json());
    } catch (error) {
      setInventoryError(error instanceof Error ? error.message : "Inventory intelligence unavailable");
      setInventory(null);
    } finally {
      setInventoryLoading(false);
    }
  }

  useEffect(() => {
    fetch("http://localhost:8000/api/v1/dashboard")
      .then((response) => {
        if (!response.ok) throw new Error("Dashboard request failed");
        return response.json();
      })
      .then((json: DashboardData) => setData(json))
      .catch(() => {
        setData(fallbackData);
        setIsDemo(true);
      })
      .finally(() => setIsLoading(false));
    fetch("http://localhost:8000/api/v1/explain?product_id=P100&store_id=1&forecast_horizon=7")
      .then((response) => {
        if (!response.ok) throw new Error("Explanation unavailable");
        return response.json();
      })
      .then((json: ExplanationData) => setExplanation(json))
      .catch(() => setExplanationUnavailable(true));
    fetch("http://localhost:8000/api/v1/simulation/options")
      .then((response) => { if (!response.ok) throw new Error("Options unavailable"); return response.json(); })
      .then((json: SimulationOptions) => { setOptions(json); if (json.products[0]) setSimProduct(json.products[0].product_id); if (json.stores[0]) setSimStore(json.stores[0].store_id); })
      .catch(() => setSimulationError("Simulator options are unavailable."));
    fetch("http://localhost:8000/api/v1/monitoring")
      .then((response) => { if (!response.ok) throw new Error("Monitoring unavailable"); return response.json(); })
      .then((json: MonitoringData) => setMonitoring(json))
      .catch(() => undefined);
    fetch("http://localhost:8000/api/v1/models")
      .then((response) => { if (!response.ok) throw new Error("Registry unavailable"); return response.json(); })
      .then((json: ModelRegistryData) => setRegistry(json))
      .catch(() => undefined);
    fetch("http://localhost:8000/api/v1/analytics/forecast-accuracy")
      .then((response) => { if (!response.ok) throw new Error("Accuracy unavailable"); return response.json(); })
      .then((json: AccuracyData) => setAccuracy(json))
      .catch(() => undefined);
    loadInventoryData();
  }, []);

  async function runSimulation() {
    setSimulationLoading(true);
    setSimulationError("");
    try {
      const response = await fetch("http://localhost:8000/api/v1/simulate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ product_id: simProduct, store_id: simStore, forecast_horizon: simHorizon, price: simPrice, promotion: simPromotion, holiday: simHoliday, lead_time_days: simLeadTime, current_inventory: simInventory }),
      });
      if (!response.ok) throw new Error((await response.json()).detail || "Simulation failed");
      setSimulation(await response.json());
    } catch (error) {
      setSimulationError(error instanceof Error ? error.message : "Simulation failed.");
    } finally { setSimulationLoading(false); }
  }

  if (isLoading) {
    return <main className="flex min-h-screen items-center justify-center bg-slate-950 p-6 text-slate-300"><div className="rounded-xl border border-slate-800 bg-slate-900 px-6 py-5 text-sm shadow-xl">Loading dashboard...</div></main>;
  }

  if (!data) {
    return <main className="flex min-h-screen items-center justify-center bg-slate-950 p-6 text-slate-300"><div className="max-w-md rounded-xl border border-rose-900/50 bg-slate-900 px-6 py-5 text-center text-sm"><p className="font-medium text-white">Unable to load live dashboard data.</p><p className="mt-1 text-slate-400">Please check that the backend is running.</p></div></main>;
  }

  return (
    <main className="min-h-screen overflow-x-hidden bg-slate-950 px-4 py-6 text-white sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <header className="mb-8 flex flex-col gap-5 border-b border-slate-800 pb-7 sm:flex-row sm:items-end sm:justify-between">
          <div className="min-w-0">
            <div className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-400">Demand Intelligence</div>
            <h1 className="mt-2 truncate text-2xl font-semibold tracking-tight sm:text-3xl">Demand Forecasting &amp; Inventory Optimization</h1>
            <p className="mt-2 text-sm text-slate-400">A concise view of demand signals, forecast accuracy, and replenishment exposure.</p>
          </div>
          <div className={`flex w-fit shrink-0 items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium ${isDemo ? "border-amber-500/30 bg-amber-500/10 text-amber-200" : "border-emerald-500/30 bg-emerald-500/10 text-emerald-200"}`}>
            <span className={`h-2 w-2 rounded-full ${isDemo ? "bg-amber-400" : "bg-emerald-400"}`} />
            {isDemo ? "Demo / fallback data" : "System online"}
          </div>
        </header>
        {isDemo && (
          <div className="mb-6 rounded-lg border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-sm text-amber-100">
            <span className="font-medium">Live dashboard data is unavailable.</span>{" "}
            Showing the configured demo dataset. Please check that the backend is running.
          </div>
        )}

        <section className="grid min-w-0 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <StatCard label="Demand today" value={`${formatNumber(data.demand_today, true)} units`} description="Current estimated demand" accent="bg-cyan-400" />
          <StatCard label="Inventory risk" value={formatPercent(data.inventory_risk)} description="Current inventory exposure" accent="bg-amber-400" />
          <StatCard label="Forecast error" value={`${formatNumber(data.average_forecast_error)} units`} description="Average monitored error" accent="bg-violet-400" />
          <StatCard label="Recommended reorder" value={`${formatNumber(data.recommended_reorder, true)} units`} description="Suggested replenishment" accent="bg-emerald-400" />
        </section>
        <section className="mt-6 rounded-xl border border-slate-800 bg-slate-900/80 p-5">
          <h2 className="text-base font-semibold">Why This Forecast?</h2>
          {explanation ? (
            <>
              <p className="mt-2 text-sm text-slate-300">Predicted demand: <span className="font-semibold text-white">{formatNumber(explanation.prediction)} units</span></p>
              <div className="mt-4 grid gap-2 sm:grid-cols-2">
                {explanation.top_features.map((feature) => (
                  <div key={feature.feature} className="flex min-w-0 justify-between gap-3 rounded-lg bg-slate-950/60 px-3 py-2 text-sm">
                    <span className="truncate text-slate-300">{feature.feature}</span>
                    <span className={feature.direction === "positive" ? "shrink-0 text-emerald-400" : "shrink-0 text-rose-400"}>{feature.shap_value >= 0 ? "+" : ""}{feature.shap_value.toFixed(2)}</span>
                  </div>
                ))}
              </div>
              <p className="mt-4 text-sm text-slate-400">{explanation.summary}</p>
            </>
          ) : (
            <p className="mt-2 text-sm text-slate-500">{explanationUnavailable ? "Explanation temporarily unavailable." : "Loading explanation..."}</p>
          )}
        </section>
        <section className="mt-6 rounded-xl border border-violet-900/60 bg-slate-900/80 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-base font-semibold">Forecast Accuracy Intelligence</h2><p className="mt-1 text-xs text-slate-500">Production-model error analysis and operational indicators.</p></div>{accuracy && <span className="text-xs text-slate-500">{accuracy.metadata.model_name} · {accuracy.metadata.model_version}</span>}</div>
          {accuracy ? <><div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5"><StatCard label="WMAPE" value={`${accuracy.summary.wmape.toFixed(1)}%`} description="Weighted error" accent="bg-violet-400" /><StatCard label="MAE" value={formatNumber(accuracy.summary.mae)} description="Mean absolute error" accent="bg-cyan-400" /><StatCard label="RMSE" value={formatNumber(accuracy.summary.rmse)} description="Root mean square error" accent="bg-amber-400" /><StatCard label="MAPE" value={`${accuracy.summary.mape.toFixed(1)}%`} description="Zero-safe percentage error" accent="bg-emerald-400" /><StatCard label="Bias" value={`${(accuracy.summary.bias * 100).toFixed(1)}%`} description={accuracy.bias.label} accent="bg-rose-400" /></div>
            <div className="mt-5 grid gap-5 lg:grid-cols-2"><div><h3 className="font-medium">Actual vs forecast</h3><div className="mt-2 h-56"><ResponsiveContainer width="100%" height="100%"><AreaChart data={accuracy.trends.month}><CartesianGrid stroke="#1e293b" strokeDasharray="3 3" /><XAxis dataKey="date" stroke="#64748b" /><YAxis stroke="#64748b" /><Tooltip /><Area type="monotone" dataKey="actual_demand" stroke="#a78bfa" fill="#4c1d95" /><Area type="monotone" dataKey="forecast_demand" stroke="#22d3ee" fill="transparent" /></AreaChart></ResponsiveContainer></div></div><div><h3 className="font-medium">Error trend</h3><div className="mt-2 h-56"><ResponsiveContainer width="100%" height="100%"><AreaChart data={accuracy.trends.month}><CartesianGrid stroke="#1e293b" strokeDasharray="3 3" /><XAxis dataKey="date" stroke="#64748b" /><YAxis stroke="#64748b" /><Tooltip /><Area type="monotone" dataKey="wmape" stroke="#f59e0b" fill="#78350f" /></AreaChart></ResponsiveContainer></div></div></div>
            <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><StatCard label="Under-forecast units" value={formatNumber(accuracy.business_impact.under_forecast_units, true)} description="Potential stockout exposure" accent="bg-rose-400" /><StatCard label="Over-forecast units" value={formatNumber(accuracy.business_impact.over_forecast_units, true)} description="Potential excess exposure" accent="bg-amber-400" /><StatCard label="Stockout indicators" value={formatNumber(accuracy.business_impact.stockout_risk_count)} description="Operational observations" accent="bg-rose-400" /><StatCard label="Excess indicators" value={formatNumber(accuracy.business_impact.excess_inventory_risk_count)} description="Operational observations" accent="bg-amber-400" /></div>
            <div className="mt-5 overflow-x-auto"><table className="w-full min-w-[650px] text-left text-sm"><thead className="text-slate-500"><tr><th className="p-2">Product</th><th className="p-2">Actual</th><th className="p-2">Forecast</th><th className="p-2">WMAPE</th><th className="p-2">MAE</th><th className="p-2">Bias</th><th className="p-2">Status</th></tr></thead><tbody>{accuracy.breakdowns.product.slice(0, 10).map((item) => <tr key={item.product_id} className="border-t border-slate-800"><td className="p-2">{item.product_id} · {item.product_name}</td><td className="p-2">{formatNumber(item.actual_demand, true)}</td><td className="p-2">{formatNumber(item.forecast_demand, true)}</td><td className="p-2">{item.wmape.toFixed(1)}%</td><td className="p-2">{formatNumber(item.mae)}</td><td className="p-2">{(item.bias * 100).toFixed(1)}%</td><td className="p-2">{item.status}</td></tr>)}</tbody></table></div>
          </> : <p className="mt-4 text-sm text-slate-500">Forecast accuracy data unavailable.</p>}
        </section>
        <section className="mt-6 rounded-xl border border-emerald-900/60 bg-slate-900/80 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-base font-semibold">Inventory Business Intelligence</h2>
              <p className="mt-1 text-xs text-slate-500">ABC / XYZ segmentation, health score, risk posture, opportunities, and service-level analytics.</p>
            </div>
            <button onClick={() => loadInventoryData(inventoryFilters)} className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-sm font-medium text-emerald-200">Refresh</button>
          </div>

          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-6">
            <label className="text-xs text-slate-400">Product<input value={inventoryFilters.product_id} onChange={(e) => setInventoryFilters((prev) => ({ ...prev, product_id: e.target.value }))} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-2 text-sm text-white" placeholder="P100" /></label>
            <label className="text-xs text-slate-400">Store<input value={inventoryFilters.store_id} onChange={(e) => setInventoryFilters((prev) => ({ ...prev, store_id: e.target.value }))} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-2 text-sm text-white" placeholder="1" /></label>
            <label className="text-xs text-slate-400">ABC<select value={inventoryFilters.abc_class} onChange={(e) => setInventoryFilters((prev) => ({ ...prev, abc_class: e.target.value }))} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-2 text-sm text-white"><option value="">All</option><option value="A">A</option><option value="B">B</option><option value="C">C</option></select></label>
            <label className="text-xs text-slate-400">XYZ<select value={inventoryFilters.xyz_class} onChange={(e) => setInventoryFilters((prev) => ({ ...prev, xyz_class: e.target.value }))} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-2 text-sm text-white"><option value="">All</option><option value="X">X</option><option value="Y">Y</option><option value="Z">Z</option></select></label>
            <label className="text-xs text-slate-400">Risk<select value={inventoryFilters.risk_level} onChange={(e) => setInventoryFilters((prev) => ({ ...prev, risk_level: e.target.value }))} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-2 text-sm text-white"><option value="">All</option><option value="Low">Low</option><option value="Medium">Medium</option><option value="High">High</option><option value="Critical">Critical</option></select></label>
            <label className="text-xs text-slate-400">Health<select value={inventoryFilters.health_band} onChange={(e) => setInventoryFilters((prev) => ({ ...prev, health_band: e.target.value }))} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-2 text-sm text-white"><option value="">All</option><option value="Excellent">Excellent</option><option value="Healthy">Healthy</option><option value="Watch">Watch</option><option value="Risk">Risk</option><option value="Critical">Critical</option></select></label>
          </div>

          {inventoryLoading ? <p className="mt-4 text-sm text-slate-500">Loading inventory intelligence...</p> : inventory ? (
            <>
              <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
                <StatCard label="Products" value={String(inventory.summary.total_products)} description="In scope" accent="bg-cyan-400" />
                <StatCard label="Stores" value={String(inventory.summary.total_stores)} description="Active stores" accent="bg-violet-400" />
                <StatCard label="Inventory units" value={formatNumber(inventory.summary.total_inventory_units, true)} description="On-hand units" accent="bg-emerald-400" />
                <StatCard label="Avg health" value={`${Number(inventory.summary.average_health_score || 0).toFixed(1)}/100`} description="Portfolio health score" accent="bg-amber-400" />
                <StatCard label="Critical" value={String(inventory.summary.critical_inventory_count)} description="High-risk items" accent="bg-rose-400" />
              </div>

              <div className="mt-5 grid gap-5 lg:grid-cols-2">
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4">
                  <h3 className="font-medium text-white">Health distribution</h3>
                  <div className="mt-3 space-y-2 text-sm">
                    {Object.entries(inventory.inventory_health.health_band_counts).map(([band, count]) => (
                      <div key={band} className="flex items-center justify-between gap-3 rounded-md bg-slate-900 px-3 py-2">
                        <span className="text-slate-300">{band}</span>
                        <span className="font-medium text-white">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4">
                  <h3 className="font-medium text-white">Risk posture</h3>
                  <div className="mt-3 grid gap-2 sm:grid-cols-2">
                    {Object.entries(inventory.risk.stockout_risk_distribution).map(([level, count]) => (
                      <div key={level} className="rounded-md bg-slate-900 px-3 py-2 text-sm"><div className="text-slate-400">Stockout {level}</div><div className="mt-1 font-semibold text-white">{count}</div></div>
                    ))}
                    {Object.entries(inventory.risk.excess_inventory_distribution).map(([level, count]) => (
                      <div key={level} className="rounded-md bg-slate-900 px-3 py-2 text-sm"><div className="text-slate-400">Excess {level}</div><div className="mt-1 font-semibold text-white">{count}</div></div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="mt-5 grid gap-5 lg:grid-cols-2">
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4">
                  <h3 className="font-medium text-white">ABC / XYZ mix</h3>
                  <div className="mt-3 space-y-2 text-sm">
                    {inventory.abc_xyz.map((item) => (
                      <div key={item.class_} className="flex items-center justify-between gap-3 rounded-md bg-slate-900 px-3 py-2">
                        <span className="text-slate-300">{item.class_}</span>
                        <span className="font-medium text-white">{item.product_count} products · {item.percentage_contribution.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4">
                  <h3 className="font-medium text-white">Service levels</h3>
                  <div className="mt-3 space-y-2 text-sm">
                    {inventory.service_level.map((service) => (
                      <div key={service.service_level} className="flex items-center justify-between gap-3 rounded-md bg-slate-900 px-3 py-2">
                        <span className="text-slate-300">{(service.service_level * 100).toFixed(0)}%</span>
                        <span className="font-medium text-white">Safety stock {formatNumber(service.safety_stock, true)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="mt-5 rounded-lg border border-slate-800 bg-slate-950/60 p-4">
                <h3 className="font-medium text-white">Opportunity ranking</h3>
                <div className="mt-3 overflow-x-auto">
                  <table className="w-full min-w-[700px] text-left text-sm">
                    <thead className="text-slate-500"><tr><th className="p-2">Product</th><th className="p-2">Type</th><th className="p-2">Priority</th><th className="p-2">Current</th><th className="p-2">Threshold</th><th className="p-2">Why</th></tr></thead>
                    <tbody>{inventory.opportunities.slice(0, 8).map((opportunity) => (
                      <tr key={`${opportunity.product_id}-${opportunity.opportunity_type}`} className="border-t border-slate-800"><td className="p-2 text-white">{opportunity.product_id}</td><td className="p-2 text-slate-300">{opportunity.opportunity_type}</td><td className="p-2 text-slate-300">{opportunity.priority}</td><td className="p-2 text-white">{formatNumber(opportunity.current_value, true)}</td><td className="p-2 text-white">{formatNumber(opportunity.threshold, true)}</td><td className="p-2 text-slate-300">{opportunity.explanation}</td></tr>
                    ))}</tbody>
                  </table>
                </div>
              </div>
            </>
          ) : (
            <p className="mt-4 text-sm text-slate-500">{inventoryError || "Inventory intelligence data unavailable."}</p>
          )}
        </section>
        <section className="mt-6 rounded-xl border border-slate-800 bg-slate-900/80 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-base font-semibold">Model Registry</h2><p className="mt-1 text-xs text-slate-500">Registered champion and challenger artifacts.</p></div><button onClick={async () => { setRetraining("Running retraining..."); const response = await fetch("http://localhost:8000/api/v1/models/retrain", { method: "POST" }); const result = await response.json(); setRetraining(response.ok ? `${result.decision}: ${result.reason}` : result.detail); }} className="rounded-lg border border-cyan-500/40 px-3 py-2 text-sm text-cyan-300">Run Retraining</button></div>
          {retraining && <p className="mt-3 text-sm text-slate-300">{retraining}</p>}
          {registry ? <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[620px] text-left text-sm"><thead className="text-slate-500"><tr><th className="p-2">Version</th><th className="p-2">Model</th><th className="p-2">WMAPE</th><th className="p-2">Status</th><th className="p-2">Production</th></tr></thead><tbody>{registry.models.map((item) => <tr key={item.model_version} className="border-t border-slate-800"><td className="p-2">{item.model_version}</td><td className="p-2">{item.model_name} ({item.model_type})</td><td className="p-2">{item.metrics.wmape?.toFixed(2)}</td><td className="p-2">{item.status}</td><td className="p-2">{item.is_production ? "Yes" : "No"}</td></tr>)}</tbody></table></div> : <p className="mt-4 text-sm text-slate-500">Registry unavailable.</p>}
        </section>

        <section className="mt-6 rounded-xl border border-cyan-900/60 bg-slate-900/80 p-5">
          <div className="mb-5"><h2 className="text-base font-semibold">What-If Simulator</h2><p className="mt-1 text-xs text-slate-500">Run a real production-model scenario without changing historical data.</p></div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <label className="text-sm text-slate-300">Product<select value={simProduct} onChange={(e) => setSimProduct(e.target.value)} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white">{options?.products.map((item) => <option key={item.product_id} value={item.product_id}>{item.product_id} - {item.name}</option>)}</select></label>
            <label className="text-sm text-slate-300">Store<select value={simStore} onChange={(e) => setSimStore(Number(e.target.value))} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white">{options?.stores.map((item) => <option key={item.store_id} value={item.store_id}>{item.name} ({item.region})</option>)}</select></label>
            <label className="text-sm text-slate-300">Horizon<select value={simHorizon} onChange={(e) => setSimHorizon(Number(e.target.value))} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white">{[1, 7, 14, 30].map((value) => <option key={value} value={value}>{value} days</option>)}</select></label>
            <label className="text-sm text-slate-300">Price<input type="number" min="0.01" step="0.01" value={simPrice} onChange={(e) => setSimPrice(Number(e.target.value))} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white" /></label>
            <label className="text-sm text-slate-300">Lead time (days)<input type="number" min="1" value={simLeadTime} onChange={(e) => setSimLeadTime(Number(e.target.value))} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white" /></label>
            <label className="text-sm text-slate-300">Current inventory<input type="number" min="0" value={simInventory} onChange={(e) => setSimInventory(Number(e.target.value))} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white" /></label>
            <label className="flex items-center gap-2 pt-6 text-sm text-slate-300"><input type="checkbox" checked={simPromotion} onChange={(e) => setSimPromotion(e.target.checked)} /> Promotion</label>
            <label className="flex items-center gap-2 pt-6 text-sm text-slate-300"><input type="checkbox" checked={simHoliday} onChange={(e) => setSimHoliday(e.target.checked)} /> Holiday</label>
          </div>
          <button onClick={runSimulation} disabled={simulationLoading || !options} className="mt-5 rounded-lg bg-cyan-500 px-4 py-2 text-sm font-semibold text-slate-950 disabled:cursor-not-allowed disabled:opacity-50">{simulationLoading ? "Running..." : "Run Simulation"}</button>
          {simulationError && <p className="mt-3 text-sm text-rose-400">{simulationError}</p>}
          {simulation && <div className="mt-6 grid gap-5 lg:grid-cols-2">
            <div><h3 className="font-medium">Forecast impact</h3><div className="mt-2 grid grid-cols-3 gap-2 text-center text-sm"><div className="rounded-lg bg-slate-950 p-3"><div className="text-slate-500">Baseline</div><b>{formatNumber(simulation.baseline.forecast_demand)}</b></div><div className="rounded-lg bg-slate-950 p-3"><div className="text-slate-500">Scenario</div><b>{formatNumber(simulation.result.forecast_demand)}</b></div><div className="rounded-lg bg-slate-950 p-3"><div className="text-slate-500">Change</div><b className={simulation.impact.demand_change >= 0 ? "text-emerald-400" : "text-rose-400"}>{simulation.impact.demand_change >= 0 ? "+" : ""}{formatNumber(simulation.impact.demand_change)} ({simulation.impact.demand_change_percent == null ? "n/a" : `${simulation.impact.demand_change_percent.toFixed(1)}%`})</b></div></div><div className="mt-4 h-56"><ResponsiveContainer width="100%" height="100%"><AreaChart data={simulation.result.forecast.map((point, index) => ({ ...point, baseline_demand: simulation.baseline.forecast[index]?.predicted_demand }))}><CartesianGrid stroke="#1e293b" strokeDasharray="3 3" /><XAxis dataKey="date" tickFormatter={formatDate} stroke="#64748b" /><YAxis stroke="#64748b" /><Tooltip formatter={(value, name) => [formatNumber(Number(value)), name === "baseline_demand" ? "Baseline" : "Scenario"]} /><Area type="monotone" dataKey="baseline_demand" stroke="#64748b" fill="transparent" /><Area type="monotone" dataKey="predicted_demand" stroke="#22d3ee" fill="#164e63" /></AreaChart></ResponsiveContainer></div></div>
            <div><h3 className="font-medium">Why Did the Scenario Change?</h3><div className="mt-2 grid grid-cols-2 gap-2 text-sm">{[["Safety stock", "safety_stock"], ["Reorder point", "reorder_point"], ["Recommended order", "recommended_order"], ["Coverage", "coverage_days"], ["Stockout risk", "stockout_label"], ["Excess risk", "excess_inventory_label"]].map(([label, key]) => <div key={key} className="rounded-lg bg-slate-950 p-3"><div className="text-slate-500">{label}</div><b>{String(simulation.result.inventory[key] ?? "n/a")}</b></div>)}</div><div className="mt-4 space-y-2">{simulation.explanation.features.map((feature) => <div key={feature.feature} className="flex justify-between text-sm"><span className="truncate text-slate-300">{feature.feature}</span><span className={feature.direction === "positive" ? "text-emerald-400" : "text-rose-400"}>{feature.shap_value >= 0 ? "+" : ""}{feature.shap_value.toFixed(2)}</span></div>)}</div><p className="mt-3 text-sm text-slate-400">{simulation.explanation.summary}</p></div>
          </div>}
        </section>
        <section className="mt-6 rounded-xl border border-slate-800 bg-slate-900/80 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-base font-semibold">Model Monitoring</h2><p className="mt-1 text-xs text-slate-500">Current data quality, drift, and performance signals.</p></div>{monitoring && <span className={`rounded-full px-3 py-1 text-xs font-semibold ${monitoring.status === "HEALTHY" ? "bg-emerald-500/15 text-emerald-300" : monitoring.status === "WARNING" ? "bg-amber-500/15 text-amber-300" : "bg-rose-500/15 text-rose-300"}`}>{monitoring.status}</span>}</div>
          {monitoring ? <><div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><StatCard label="Rows monitored" value={formatNumber(monitoring.data_quality.row_count)} description={`Quality: ${monitoring.data_quality.status}`} accent="bg-cyan-400" /><StatCard label="Duplicate rate" value={formatPercent(monitoring.data_quality.duplicate_rate)} description={`${monitoring.data_quality.invalid_count} invalid values`} accent="bg-amber-400" /><StatCard label="Target drift" value={monitoring.target_drift.score.toFixed(3)} description={monitoring.target_drift.status} accent="bg-violet-400" /><StatCard label="Prediction drift" value={monitoring.prediction_drift.score.toFixed(3)} description={monitoring.prediction_drift.status} accent="bg-emerald-400" /></div><div className="mt-5 overflow-x-auto"><table className="w-full min-w-[560px] text-left text-sm"><thead className="text-slate-500"><tr><th className="p-2">Metric</th><th className="p-2">Reference</th><th className="p-2">Current</th><th className="p-2">Change</th><th className="p-2">Status</th></tr></thead><tbody>{monitoring.model_performance.map((item) => <tr key={item.metric} className="border-t border-slate-800"><td className="p-2">{item.metric}</td><td className="p-2">{item.reference_value.toFixed(2)}</td><td className="p-2">{item.current_value.toFixed(2)}</td><td className="p-2">{item.change >= 0 ? "+" : ""}{item.change.toFixed(2)}</td><td className="p-2">{item.status}</td></tr>)}</tbody></table></div><div className="mt-5 grid gap-5 lg:grid-cols-2"><div><h3 className="font-medium">Feature Drift</h3><div className="mt-2 space-y-2">{monitoring.feature_drift.filter((item) => item.status !== "HEALTHY").slice(0, 8).map((item) => <div key={item.feature} className="flex justify-between text-sm"><span>{item.feature}</span><span>{item.score.toFixed(3)} · {item.status}</span></div>)}</div></div><div><h3 className="font-medium">Monitoring Alerts</h3><div className="mt-2 space-y-2 text-sm">{monitoring.alerts.length ? monitoring.alerts.slice(0, 8).map((item, index) => <div key={`${item.category}-${index}`} className="rounded-lg bg-slate-950 p-2"><span className="font-semibold">{item.severity}</span> {item.message}</div>) : <p className="text-slate-500">No active alerts.</p>}</div></div></div></> : <p className="mt-4 text-sm text-slate-500">Monitoring data unavailable.</p>}
        </section>

        <section className="mt-6 grid min-w-0 gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(280px,1fr)]">
          <div className="min-w-0 rounded-xl border border-slate-800 bg-slate-900/80 p-5">
            <div className="mb-5">
              <h2 className="text-base font-semibold">7-Day Demand Forecast</h2>
              <p className="mt-1 text-xs text-slate-500">Predicted demand for the selected forecast horizon</p>
            </div>
            <div className="h-72 min-w-0 sm:h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.trend} margin={{ top: 5, right: 8, left: -18, bottom: 0 }}>
                  <defs>
                    <linearGradient id="forecastFill" x1="0" x2="0" y1="0" y2="1">
                      <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.1} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={formatDate} stroke="#64748b" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={42} />
                  <Tooltip labelFormatter={(label) => formatDate(String(label))} formatter={(value) => [formatNumber(Number(value)), "Forecast"]} contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, color: "#e2e8f0" }} />
                  <Area type="monotone" dataKey="forecast" stroke="#22d3ee" fill="url(#forecastFill)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="min-w-0 rounded-xl border border-slate-800 bg-slate-900/80 p-5">
            <div className="mb-5">
              <h2 className="text-base font-semibold">Inventory risk</h2>
              <p className="mt-1 text-xs text-slate-500">Exposure by operational signal</p>
            </div>
            <div className="space-y-5">
              {Object.entries(data.risk_breakdown).map(([key, value]) => (
                <div key={key}>
                  <div className="mb-2 flex min-w-0 justify-between gap-3 text-sm text-slate-300">
                    <span className="truncate capitalize">{key.replace(/_/g, " ")}</span>
                    <span className="shrink-0 font-medium text-white">{formatPercent(Number(value))}</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                    <div className="h-full max-w-full rounded-full bg-cyan-400 transition-all" style={{ width: `${Math.min(Math.max(Number(value) * 100, 0), 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
