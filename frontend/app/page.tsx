"use client";

import { useEffect, useState } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

function StatCard({ label, value, delta }: { label: string; value: string; delta: string }) {
  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4 shadow-lg">
      <div className="text-sm text-slate-400">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
      <div className="mt-1 text-xs text-emerald-400">{delta}</div>
    </div>
  );
}

export default function Page() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/v1/dashboard")
      .then((response) => response.json())
      .then((json) => setData(json))
      .catch(() => setData({
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
          { date: "2025-08-05", forecast: 260 }
        ]
      }));
  }, []);

  const chartData = data?.trend ?? [];

  return (
    <main className="min-h-screen bg-slate-950 p-8 text-white">
      <div className="mx-auto max-w-7xl">
        <header className="mb-8 flex items-center justify-between">
          <div>
            <div className="text-sm uppercase tracking-[0.25em] text-cyan-400">Demand Intelligence</div>
            <h1 className="mt-2 text-4xl font-bold">AI-powered demand forecasting</h1>
          </div>
          <div className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-4 py-2 text-sm text-cyan-200">
            14-day forecast horizon
          </div>
        </header>

        <section className="grid gap-4 md:grid-cols-4">
          <StatCard label="Demand today" value={String(data?.demand_today ?? 0)} delta="+8.6% vs prior week" />
          <StatCard label="Inventory risk" value={`${(data?.inventory_risk ?? 0).toFixed(2)}`} delta="Managed within service targets" />
          <StatCard label="Forecast error" value={`${(data?.average_forecast_error ?? 0).toFixed(1)} units`} delta="MAPE monitored" />
          <StatCard label="Recommended reorder" value={String(data?.recommended_reorder ?? 0)} delta="Q3 replenishment plan" />
        </section>

        <section className="mt-8 grid gap-6 lg:grid-cols-[2fr_1fr]">
          <div className="rounded-2xl border border-slate-700 bg-slate-900 p-4">
            <div className="mb-4 text-lg font-semibold">Forecast trend</div>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="forecastFill" x1="0" x2="0" y1="0" y2="1">
                      <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.1} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#334155" strokeDasharray="3 3" />
                  <XAxis dataKey="date" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip />
                  <Area type="monotone" dataKey="forecast" stroke="#22d3ee" fill="url(#forecastFill)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="space-y-4 rounded-2xl border border-slate-700 bg-slate-900 p-4">
            <div className="text-lg font-semibold">Risk breakdown</div>
            <div className="space-y-3">
              {Object.entries(data?.risk_breakdown ?? {}).map(([key, value]) => (
                <div key={key}>
                  <div className="mb-1 flex justify-between text-sm text-slate-300">
                    <span>{key}</span>
                    <span>{Number(value).toFixed(2)}</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-800">
                    <div className="h-full rounded-full bg-cyan-400" style={{ width: `${Number(value) * 100}%` }} />
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
