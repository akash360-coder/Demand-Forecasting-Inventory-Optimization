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
  }, []);

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
