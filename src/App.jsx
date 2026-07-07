/**
 * Om Gold Intelligence — App.jsx  v4
 * ─────────────────────────────────────────────────────────────────
 * Key upgrades over v3:
 *  • Header: true 3-col CSS grid → brand is centred focal point
 *  • Y-axis: intelligent domain scaling (min/max ± 7% padding)
 *  • Y-axis ticks: Indian lakh notation  ₹1.44L  ₹1.47L
 *  • Confidence band: semi-transparent area around forecast using MAE
 *  • Chart legend: History / Forecast / Confidence band
 *  • Tooltip: shows ₹INR + USD side by side
 *  • Forecast chart: passes currentPrice + MAE so band is drawn
 *  • All existing API calls and state unchanged
 * ─────────────────────────────────────────────────────────────────
 */

import { useState, useEffect, useCallback, useMemo } from "react";
import {
  ResponsiveContainer,
  AreaChart, Area,
  LineChart, Line,
  XAxis, YAxis,
  Tooltip, CartesianGrid,
  ReferenceLine,
} from "recharts";
import "./App.css";

/* ─── Constants ──────────────────────────────────────────────────── */
const API_BASE = "http://localhost:5000";

// MAE values mirrored from app.py — used for confidence band
const MAE = {
    "1D":1.57,
    "7D":4.15,
    "14D":5.9,
    "21D":7.1,
    "30D":8.6
};

/* ─── Formatting helpers ─────────────────────────────────────────── */
const fmtINR = (n, dec = 0) =>
  n == null
    ? "—"
    : Number(n).toLocaleString("en-IN", {
        minimumFractionDigits: dec,
        maximumFractionDigits: dec,
      });

const fmtUSD = (n) =>
  n == null
    ? "—"
    : Number(n).toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      });

// Indian lakh notation for chart Y-axis  →  ₹1.44L
const fmtLakh = (v) => {
  if (v == null) return "";
  const l = v / 100000;
  return `₹${l.toFixed(2)}L`;
};

/* ─── Compute smart Y-axis domain ───────────────────────────────── */
const smartDomain = (values, padPct = 0.07) => {
  const clean = values.filter((v) => v != null && isFinite(v));
  if (!clean.length) return ["auto", "auto"];
  const lo = Math.min(...clean);
  const hi = Math.max(...clean);
  const pad = (hi - lo) * padPct || hi * padPct;
  return [Math.floor(lo - pad), Math.ceil(hi + pad)];
};

/* ─── Direction helper ───────────────────────────────────────────── */
const dirLabel = (change) => {
  if (change == null)  return { text: "Stable",    cls: "neutral" };
  if (change >  0.3)  return { text: "Rising",    cls: "up"      };
  if (change < -0.3)  return { text: "Declining", cls: "down"    };
  return { text: "Stable", cls: "neutral" };
};

/* ─── Confidence colour ──────────────────────────────────────────── */
const confColor = (c) =>
  ({ High: "var(--gold-deep)", Moderate: "var(--ink-lo)",
     Low: "var(--ink-mute)", "Very Low": "var(--ink-mute)" }[c] ?? "var(--ink-mute)");

/* ═══════════════════════════════════════════════════════════════════
   CHART TOOLTIP
═══════════════════════════════════════════════════════════════════ */
const ChartTooltip = ({ active, payload, label, usdInr }) => {
  if (!active || !payload?.length) return null;

  // Pull the "price" entry (not the band entries)
  const entry = payload.find((p) => p.dataKey === "priceINR" || p.dataKey === "price");
  if (!entry) return null;

  const priceINR = entry.value;
  const priceUSD = usdInr ? priceINR / ((usdInr * 10) / 31.1035) : null;

  return (
    <div className="chart-tooltip-box">
      <p className="chart-tooltip-label">{label}</p>
      <p className="chart-tooltip-price">₹{fmtINR(priceINR)}</p>
      {priceUSD && (
        <p className="chart-tooltip-usd">≈ USD {fmtUSD(priceUSD)}</p>
      )}
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════════════
   ORNAMENT DIVIDER
═══════════════════════════════════════════════════════════════════ */
const Ornament = () => (
  <div className="ornament">
    <div className="ornament-line" />
    <div className="ornament-gem-sm" />
    <div className="ornament-gem" />
    <div className="ornament-gem-sm" />
    <div className="ornament-line" />
  </div>
);

/* ═══════════════════════════════════════════════════════════════════
   PREDICTION CARD
═══════════════════════════════════════════════════════════════════ */
function PredictionCard({ horizonId, horizonLabel, sublabel, data, isActive, onClick }) {
  if (!data) return null;
  const dir    = dirLabel(data.change);
  const absPct = Math.abs(data.change ?? 0).toFixed(2);

  return (
    <button
      className={`prediction-card${isActive ? " prediction-card--active" : ""}`}
      onClick={onClick}
      aria-pressed={isActive}
    >
      <div className="card-horizon">{horizonLabel}</div>
      <div className="card-sublabel">{sublabel}</div>
      <div className="card-divider" />

      <div className="card-price">₹{fmtINR(data.priceRetailINR)}</div>
      <div className="card-usd">USD {fmtUSD(data.priceUSD)}</div>

      <div className={`card-direction card-direction--${dir.cls}`}>
        <span className="direction-dot" />
        {dir.text}{Number(absPct) > 0.01 ? ` · ${absPct}%` : ""}
      </div>
      <span className="card-confidence" style={{ color: confColor(data.confidence) }}>
        {data.confidence} Confidence
      </span>
    </button>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   FORECAST CHART
   — solid gold line for the prediction
   — semi-transparent band = price ± MAE%
   — smart Y-axis domain (no zero baseline)
═══════════════════════════════════════════════════════════════════ */
function ForecastChart({ days, activeTabId, currentPriceINR, usdInr }) {
  const [rawData, setRawData]   = useState([]);
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/forecast/${days}`)
      .then((r) => r.json())
      .then((d) => {
        setRawData(d.forecast ?? []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [days]);

  const mae = MAE[activeTabId] ?? 4;

  // Build chart data including upper/lower band values
  const chartData = useMemo(() => {
    if (!rawData.length) return [];
    // Prepend "Today" point so chart starts at current price
    const today = currentPriceINR
      ? [{ day: "Today", priceINR: currentPriceINR, upper: currentPriceINR * (1 + mae / 100), lower: currentPriceINR * (1 - mae / 100) }]
      : [];
    const rest = rawData.map((f) => ({
      day   : `Day ${f.day}`,
      priceINR: f.priceINR,
      upper : f.priceINR * (1 + mae / 100),
      lower : f.priceINR * (1 - mae / 100),
    }));
    return [...today, ...rest];
  }, [rawData, currentPriceINR, mae]);

  const allValues = useMemo(
    () => chartData.flatMap((d) => [d.lower, d.upper]),
    [chartData]
  );
  const domain = useMemo(() => smartDomain(allValues, 0.06), [allValues]);

  if (loading) return <div className="chart-loading">Calculating forecast…</div>;
  if (!chartData.length) return <div className="chart-loading">No data available.</div>;

  return (
    <>
      {/* Legend */}
      <div className="chart-legend">
        <span className="legend-item">
          <span className="legend-line" style={{ background: "var(--gold-mid)" }} />
          Forecast
        </span>
        <span className="legend-item">
          <span className="legend-band" />
          Confidence band (±{mae}% MAE)
        </span>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={chartData} margin={{ top: 10, right: 24, left: 8, bottom: 4 }}>
          <defs>
            {/* Confidence band gradient */}
            <linearGradient id="bandGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="var(--gold-mid)" stopOpacity={0.12} />
              <stop offset="100%" stopColor="var(--gold-mid)" stopOpacity={0.04} />
            </linearGradient>
            {/* Main line gradient */}
            <linearGradient id="lineGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="var(--gold-mid)" stopOpacity={0.18} />
              <stop offset="100%" stopColor="var(--gold-mid)" stopOpacity={0.02} />
            </linearGradient>
          </defs>

          <CartesianGrid
            strokeDasharray="3 3"
            stroke="var(--bg-deep)"
            vertical={false}
            strokeOpacity={0.7}
          />

          <XAxis
            dataKey="day"
            tick={{ fontSize: 11, fill: "var(--ink-mute)", fontFamily: "Inter, sans-serif" }}
            axisLine={false}
            tickLine={false}
            interval="preserveStartEnd"
            padding={{ left: 8, right: 8 }}
          />

          <YAxis
            domain={domain}
            tick={{ fontSize: 11, fill: "var(--ink-mute)", fontFamily: "Inter, sans-serif" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={fmtLakh}
            width={60}
            tickCount={6}
          />

          <Tooltip
            content={<ChartTooltip usdInr={usdInr} />}
            cursor={{ stroke: "var(--gold-pale)", strokeWidth: 1, strokeDasharray: "4 2" }}
          />

          {/* Upper bound of confidence band (invisible stroke, fills to lower) */}
          <Area
            type="monotone"
            dataKey="upper"
            stroke="none"
            fill="url(#bandGrad)"
            fillOpacity={1}
            dot={false}
            legendType="none"
            isAnimationActive={true}
            animationDuration={800}
          />

          {/* Lower bound — fills upward to close the band */}
          <Area
            type="monotone"
            dataKey="lower"
            stroke="none"
            fill="var(--bg-surface)"
            fillOpacity={1}
            dot={false}
            legendType="none"
            isAnimationActive={true}
            animationDuration={800}
          />

          {/* Main forecast line */}
          <Area
            type="monotone"
            dataKey="priceINR"
            stroke="var(--gold-mid)"
            strokeWidth={2}
            strokeDasharray="6 3"
            fill="url(#lineGrad)"
            dot={false}
            activeDot={{ r: 5, fill: "var(--gold-mid)", strokeWidth: 0 }}
            isAnimationActive={true}
            animationDuration={1000}
            animationEasing="ease-out"
          />
        </AreaChart>
      </ResponsiveContainer>
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   HISTORY CHART
   — solid line for actual data
   — smart Y-axis domain
═══════════════════════════════════════════════════════════════════ */
function HistoryChart({ data, usdInr }) {
  // Sample every 2nd point on dense datasets for performance
  const display = useMemo(
    () => data.filter((_, i) => i % 2 === 0),
    [data]
  );

  const domain = useMemo(
    () => smartDomain(display.map((d) => d.priceINR), 0.05),
    [display]
  );

  const tickInterval = Math.max(1, Math.floor(display.length / 8));

  return (
    <>
      <div className="chart-legend">
        <span className="legend-item">
          <span className="legend-line" style={{ background: "var(--gold-mid)" }} />
          Gold price (INR per 10g)
        </span>
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={display} margin={{ top: 10, right: 24, left: 8, bottom: 4 }}>
          <defs>
            <linearGradient id="histGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="var(--gold-mid)" stopOpacity={0.16} />
              <stop offset="100%" stopColor="var(--gold-mid)" stopOpacity={0.02} />
            </linearGradient>
          </defs>

          <CartesianGrid
            strokeDasharray="3 3"
            stroke="var(--bg-deep)"
            vertical={false}
            strokeOpacity={0.7}
          />

          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: "var(--ink-mute)", fontFamily: "Inter, sans-serif" }}
            axisLine={false}
            tickLine={false}
            interval={tickInterval}
            tickFormatter={(d) =>
              new Date(d).toLocaleDateString("en-IN", { month: "short", year: "2-digit" })
            }
          />

          <YAxis
            domain={domain}
            tick={{ fontSize: 10, fill: "var(--ink-mute)", fontFamily: "Inter, sans-serif" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={fmtLakh}
            width={60}
            tickCount={5}
          />

          <Tooltip
            content={<ChartTooltip usdInr={usdInr} />}
            cursor={{ stroke: "var(--gold-pale)", strokeWidth: 1, strokeDasharray: "4 2" }}
          />

          <Area
            type="monotone"
            dataKey="priceINR"
            stroke="var(--gold-mid)"
            strokeWidth={1.5}
            fill="url(#histGrad)"
            dot={false}
            activeDot={{ r: 4, fill: "var(--gold-mid)", strokeWidth: 0 }}
            isAnimationActive={true}
            animationDuration={900}
            animationEasing="ease-out"
          />
        </AreaChart>
      </ResponsiveContainer>
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   MAIN APP
═══════════════════════════════════════════════════════════════════ */
export default function App() {
  const [predict, setPredict]       = useState(null);
  const [history, setHistory]       = useState([]);
  const [activeTab, setActiveTab]   = useState("1D");
  const [forecastDays, setFDays]    = useState(7);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState(null);

  const today = new Date().toLocaleDateString("en-IN", {
    weekday: "long", day: "numeric", month: "long", year: "numeric",
  });

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/predict`).then((r) => r.json()),
      fetch(`${API_BASE}/history`).then((r) => r.json()),
    ])
      .then(([p, h]) => { setPredict(p); setHistory(h); setLoading(false); })
      .catch(() => {
        setError("Unable to reach the Om Gold Intelligence server. Please ensure the backend is running.");
        setLoading(false);
      });
  }, []);
  const tabs = [

  {
    id: "1D",
    label: "Tomorrow",
    sublabel: "Tomorrow's Estimate",
    days: 1,
    data: predict?.prediction1D,
  },

  {
    id: "7D",
    label: "7 Days",
    sublabel: "7-Day Estimate",
    days: 7,
    data: predict?.prediction7D,
  },

  {
    id: "14D",
    label: "14 Days",
    sublabel: "14-Day Estimate",
    days: 14,
    data: predict?.prediction14D,
  },

  {
    id: "21D",
    label: "21 Days",
    sublabel: "21-Day Estimate",
    days: 21,
    data: predict?.prediction21D,
  },

  {
    id: "30D",
    label: "30 Days",
    sublabel: "30-Day Estimate",
    days: 30,
    data: predict?.prediction30D,
  },

];
  const activeTabObj = tabs.find((t) => t.id === activeTab);
  const histPrices   = history.map((h) => h.priceINR);
  const usdInr       = predict?.usdInr;

  /* ── Splash states ─────────────────────────────────────────── */
  if (loading) return (
    <div className="splash">
      <div className="splash-brand">
        <span className="splash-om">ॐ</span>
        <span className="splash-name">Om Gold Intelligence</span>
        <div className="splash-rule" />
        <span className="splash-tagline">Om Jewellers · Price Forecasting</span>
      </div>
      <div className="splash-line" />
      <p className="splash-sub">Analysing market signals…</p>
    </div>
  );

  if (error) return (
    <div className="splash">
      <div className="splash-brand">
        <span className="splash-om">ॐ</span>
        <span className="splash-name">Om Gold Intelligence</span>
        <div className="splash-rule" />
      </div>
      <p className="splash-error">{error}</p>
    </div>
  );

  /* ── Main layout ───────────────────────────────────────────── */
  return (
    <div className="app">

      {/* ════════ HEADER ════════════════════════════════════════ */}
      <header className="header">
        <div className="header-inner">

          {/* Left — date */}
          <div className="header-left">
            <span className="header-date">{today}</span>
          </div>

          {/* Centre — maison mark (primary focal point) */}
          <div className="brand">
            <span className="brand-om">ॐ</span>
            <span className="brand-name">Om Gold Intelligence</span>
            <div className="brand-rule" />
            <span className="brand-sub">Om Jewellers · Price Forecasting</span>
          </div>

          {/* Right — live badge */}
          <div className="header-right">
            <div className="live-badge">
              <span className="live-dot" />
              Live
            </div>
          </div>

        </div>
      </header>

      {/* ════════ HERO — current price ═════════════════════════ */}
      <section className="hero">
        <div className="hero-inner">

          <p className="hero-eyebrow">Current Market Price · 24K Gold · 10g</p>

          <div className="hero-price">
            <span className="hero-currency">₹</span>
            <span className="hero-amount">
              {fmtINR(predict?.currentPriceRetailINR)}
            </span>
          </div>

          <div className="hero-pills">
            <span className="hero-pill">USD {fmtUSD(predict?.currentPriceUSD)}</span>
            <span className="hero-pill">1 USD = ₹{fmtUSD(usdInr)}</span>
          </div>

          <p className="hero-note">
            Estimated retail price · incl. import duty + GST · live USD / INR
          </p>

        </div>
      </section>

      <Ornament />

      {/* ════════ PRICE OUTLOOK ════════════════════════════════ */}
      <section className="section">
        <div className="section-inner">

          <div className="section-header">
            <p className="section-eyebrow">Forecast</p>
            <h2 className="section-title">Price Outlook</h2>
            <p className="section-desc">
              Model forecasts derived from technical indicators,
              global market signals, and historical gold price patterns.
              Select a horizon to explore the projection chart below.
            </p>
          </div>

          <div className="cards-grid">
            {tabs.map((t) => (
              <PredictionCard
                key={t.id}
                horizonId={t.id}
                horizonLabel={t.label}
                sublabel={t.sublabel}
                data={t.data}
                isActive={activeTab === t.id}
                onClick={() => {
                  setActiveTab(t.id);
                  setFDays(t.days === 1 ? 7 : t.days);
                }}
              />
            ))}
          </div>

        </div>
      </section>

      <Ornament />

      {/* ════════ FORECAST CHART ═══════════════════════════════ */}
      <section className="section section--alt">
        <div className="section-inner">

          <div className="section-head-split">
            <div>
              <p className="section-eyebrow">Projection</p>
              <h2 className="section-title">Forecast Chart</h2>
              <p className="section-desc">
                Projected price path · {activeTabObj?.label} horizon.
                Shaded band reflects model confidence range.
              </p>
            </div>
            <div className="chart-tabs">
              {[
                { label: "7 Days",  val: 7  },
                {label:"14 Days", val:14},
                {label:"21 Days", val:21},
                { label: "30 Days", val: 30 },
              
              ].map((t) => (
                <button
                  key={t.val}
                  className={`chart-tab${forecastDays === t.val ? " chart-tab--active" : ""}`}
                  onClick={() => { setFDays(t.val); setActiveTab(`${t.val}D`); }}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          <div className="chart-wrap">
            <ForecastChart
              days={forecastDays}
              activeTabId={activeTab}
              currentPriceINR={predict?.currentPriceINR}
              usdInr={usdInr}
            />
          </div>

          {/* Stat strip below chart */}
          {activeTabObj?.data && (
            <div className="forecast-stats">
              <div className="stat-card">
                <div className="stat-label">Expected Change</div>
                <div className={`stat-value ${activeTabObj.data.change >= 0 ? "up" : "down"}`}>
                  {activeTabObj.data.change >= 0 ? "+" : ""}
                  {Number(activeTabObj.data.change).toFixed(2)}%
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Model Accuracy</div>
                <div className="stat-value">{activeTabObj.data.accuracy}%</div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Confidence Level</div>
                <div
                  className="stat-value"
                  style={{ color: confColor(activeTabObj.data.confidence) }}
                >
                  {activeTabObj.data.confidence}
                </div>
              </div>
            </div>
          )}

        </div>
      </section>

      {/* ════════ HISTORICAL PERFORMANCE ═══════════════════════ */}
      {histPrices.length > 0 && (
        <>
          <Ornament />
          <section className="section">
            <div className="section-inner">

              <div className="section-header">
                <p className="section-eyebrow">Market History</p>
                <h2 className="section-title">Historical Performance</h2>
                <p className="section-desc">
                  Gold price movement over the past 12 months · INR per 10g
                </p>
              </div>

              <div className="chart-wrap">
                <HistoryChart data={history} usdInr={usdInr} />
              </div>

              <div className="history-stats">
                <div className="stat-card">
                  <div className="stat-label">52-Week High</div>
                  <div className="stat-value">₹{fmtINR(Math.max(...histPrices))}</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">52-Week Low</div>
                  <div className="stat-value">₹{fmtINR(Math.min(...histPrices))}</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Period Average</div>
                  <div className="stat-value">
                    ₹{fmtINR(histPrices.reduce((a, b) => a + b, 0) / histPrices.length)}
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Data Points</div>
                  <div className="stat-value">{history.length} days</div>
                </div>
              </div>

            </div>
          </section>
        </>
      )}

      {/* ════════ FOOTER ════════════════════════════════════════ */}
      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-brand">
            <span className="footer-om">ॐ</span>
            <span className="footer-name">Om Jewellers</span>
            <div className="footer-rule" />
          </div>
          <p className="footer-disclaimer">
            These forecasts are generated by machine learning models for business
            planning purposes only. Gold prices are influenced by global factors
            beyond any model's scope. All prices are indicative and should not
            be treated as financial advice.
          </p>
          <p className="footer-copy">
            © {new Date().getFullYear()} Om Jewellers · Om Gold Intelligence
          </p>
        </div>
      </footer>

    </div>
  );
}