import { useState, useEffect } from "react";
import {
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  Area, AreaChart, CartesianGrid,
} from "recharts";
import "./App.css";

const API_BASE = "http://localhost:5000";

const fmt = (n, decimals = 2) =>
  n == null ? "—" : Number(n).toLocaleString("en-IN", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

const directionLabel = (change) => {
  if (change == null) return { text: "Stable", cls: "neutral" };
  if (change > 0.3)  return { text: "Rising",   cls: "up"      };
  if (change < -0.3) return { text: "Declining", cls: "down"    };
  return { text: "Stable", cls: "neutral" };
};

const confidenceColor = (conf) => ({
  High:       "var(--gold)",
  Moderate:   "var(--slate)",
  Low:        "var(--mist)",
  "Very Low": "var(--mist)",
}[conf] ?? "var(--mist)");

/* ── Tooltip ─────────────────────────────────────────────────────── */
const GoldTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="tooltip-label">{label}</p>
      {payload.map((p, i) => (
        <p key={i} className="tooltip-value">₹{fmt(p.value)}</p>
      ))}
    </div>
  );
};

/* ── Prediction Card ─────────────────────────────────────────────── */
function PredictionCard({ horizon, label, data, isActive, onClick }) {
  if (!data) return null;
  const dir = directionLabel(data.change);
  const absChange = Math.abs(data.change ?? 0);

  return (
    <button
      className={`prediction-card${isActive ? " prediction-card--active" : ""}`}
      onClick={onClick}
      aria-pressed={isActive}
    >
      <div className="card-horizon">{horizon}</div>
      <div className="card-label">{label}</div>
      <div className="card-divider" />
      <div className="card-price">₹{fmt(data.priceINR, 0)}</div>
      <div className="card-usd">USD {fmt(data.priceUSD)}</div>
      <div className={`card-direction card-direction--${dir.cls}`}>
        <span className="direction-dot" />
        {dir.text}{absChange > 0.01 ? ` · ${absChange.toFixed(2)}%` : ""}
      </div>
      <div className="card-confidence" style={{ color: confidenceColor(data.confidence) }}>
        {data.confidence} Confidence
      </div>
    </button>
  );
}

/* ── Forecast Chart ──────────────────────────────────────────────── */
function ForecastChart({ days }) {
  const [data, setData]   = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/forecast/${days}`)
      .then((r) => r.json())
      .then((d) => {
        setData(d.forecast.map((f) => ({ day: `Day ${f.day}`, priceINR: f.priceINR })));
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [days]);

  if (loading) return <div className="chart-loading">Calculating forecast…</div>;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart data={data} margin={{ top: 12, right: 24, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="goldGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#B8965A" stopOpacity={0.18} />
            <stop offset="95%" stopColor="#B8965A" stopOpacity={0}    />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="4 4" stroke="#E8E1D4" vertical={false} />
        <XAxis
          dataKey="day"
          tick={{ fontSize: 11, fill: "#A39D92", fontFamily: "Inter, sans-serif" }}
          axisLine={false} tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fontSize: 11, fill: "#A39D92", fontFamily: "Inter, sans-serif" }}
          axisLine={false} tickLine={false}
          tickFormatter={(v) => `₹${(v / 1000).toFixed(0)}k`}
          width={54}
        />
        <Tooltip content={<GoldTooltip />} />
        <Area
          type="monotone" dataKey="priceINR" name="Price (INR)"
          stroke="#B8965A" strokeWidth={1.5}
          fill="url(#goldGrad)" dot={false}
          activeDot={{ r: 4, fill: "#B8965A", strokeWidth: 0 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* ── History Chart ───────────────────────────────────────────────── */
function HistoryChart({ data }) {
  const display = data.filter((_, i) => i % 3 === 0);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={display} margin={{ top: 12, right: 24, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="histGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#B8965A" stopOpacity={0.14} />
            <stop offset="95%" stopColor="#B8965A" stopOpacity={0}    />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="4 4" stroke="#E8E1D4" vertical={false} />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 10, fill: "#A39D92", fontFamily: "Inter, sans-serif" }}
          axisLine={false} tickLine={false}
          interval={Math.floor(display.length / 6)}
          tickFormatter={(d) => new Date(d).toLocaleDateString("en-IN", { month: "short", year: "2-digit" })}
        />
        <YAxis
          tick={{ fontSize: 10, fill: "#A39D92", fontFamily: "Inter, sans-serif" }}
          axisLine={false} tickLine={false}
          tickFormatter={(v) => `₹${(v / 1000).toFixed(0)}k`}
          width={54}
        />
        <Tooltip content={<GoldTooltip />} />
        <Area
          type="monotone" dataKey="priceINR" name="Price (INR)"
          stroke="#B8965A" strokeWidth={1.5}
          fill="url(#histGrad)" dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* ── App ─────────────────────────────────────────────────────────── */
export default function App() {
  const [predict, setPredict]   = useState(null);
  const [history, setHistory]   = useState([]);
  const [activeTab, setActiveTab] = useState("1D");
  const [forecastDays, setForecastDays] = useState(7);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(null);

  const now = new Date().toLocaleDateString("en-IN", {
    weekday: "long", day: "numeric", month: "long", year: "numeric",
  });

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/predict`).then((r) => r.json()),
      fetch(`${API_BASE}/history`).then((r) => r.json()),
    ])
      .then(([p, h]) => { setPredict(p); setHistory(h); setLoading(false); })
      .catch(() => { setError("Unable to connect to Om Gold Intelligence server."); setLoading(false); });
  }, []);

  const tabConfig = [
    { id: "1D",  label: "Tomorrow", sublabel: "Tomorrow's Estimate", days: 1,  data: predict?.prediction1D  },
    { id: "7D",  label: "7 Days",   sublabel: "7D Estimate",         days: 7,  data: predict?.prediction7D  },
    { id: "30D", label: "30 Days",  sublabel: "30D Estimate",        days: 30, data: predict?.prediction30D },
  ];

  const activeData = tabConfig.find((t) => t.id === activeTab);

  /* ── Loading ── */
  if (loading) return (
    <div className="splash">
      <div className="splash-logo">
        <span className="splash-om">ॐ</span>
        <span className="splash-name">Om Gold Intelligence</span>
      </div>
      <div className="splash-shimmer" />
      <p className="splash-sub">Analysing market signals…</p>
    </div>
  );

  if (error) return (
    <div className="splash">
      <div className="splash-logo">
        <span className="splash-om">ॐ</span>
        <span className="splash-name">Om Gold Intelligence</span>
      </div>
      <p className="splash-error">{error}</p>
    </div>
  );

  const histPrices = history.map((h) => h.priceINR);

  return (
    <div className="app">

      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-inner">
          <div className="brand">
            <span className="brand-om">ॐ</span>
            <div className="brand-text">
              <span className="brand-name">Om Gold Intelligence</span>
              <span className="brand-sub">Om Jewellers · Price Forecasting</span>
            </div>
          </div>
          <div className="header-meta">
            <span className="header-date">{now}</span>
            <div className="live-badge">
              <span className="live-dot" />
              Live
            </div>
          </div>
        </div>
      </header>

      {/* ── Hero ────────────────────────────────────────────────── */}
      <section className="hero">
        <div className="hero-inner">
          <p className="hero-eyebrow">Current Market Price · 24K Gold (10g)</p>
          <div className="hero-price-row">
            <div className="hero-price-primary">
              <span className="hero-currency">₹</span>
              <span className="hero-amount">
                {Number(predict?.currentPriceINR ?? 0).toLocaleString("en-IN", { maximumFractionDigits: 0 })}
              </span>
            </div>
            <div className="hero-price-secondary">
              <span className="hero-secondary-pill">USD {fmt(predict?.currentPriceUSD)}</span>
              <span className="hero-secondary-pill">1 USD = ₹{fmt(predict?.usdInr)}</span>
            </div>
          </div>
          <p className="hero-note">
            Prices reflect international spot rates converted at live USD/INR
          </p>
        </div>
      </section>

      {/* ── Price Outlook ───────────────────────────────────────── */}
      <section className="section">
        <div className="section-inner">
          <h2 className="section-title">Price Outlook</h2>
          <p className="section-sub">
            Model forecasts based on technical indicators, global market signals,
            and historical gold patterns
          </p>
          <div className="cards-grid">
            {tabConfig.map((t) => (
              <PredictionCard
                key={t.id}
                horizon={t.label}
                label={t.sublabel}
                data={t.data}
                isActive={activeTab === t.id}
                onClick={() => {
                  setActiveTab(t.id);
                  setForecastDays(t.days === 1 ? 7 : t.days);
                }}
              />
            ))}
          </div>
        </div>
      </section>

      {/* ── Forecast Chart ──────────────────────────────────────── */}
      <section className="section section--alt">
        <div className="section-inner">
          <div className="section-head">
            <div>
              <h2 className="section-title">Forecast Chart</h2>
              <p className="section-sub" style={{ marginBottom: 0 }}>
                Projected price path · {activeData?.label} horizon
              </p>
            </div>
            <div className="chart-tabs">
              {[{ label: "7 Days", val: 7 }, { label: "30 Days", val: 30 }, { label: "90 Days", val: 90 }].map((t) => (
                <button
                  key={t.val}
                  className={`chart-tab${forecastDays === t.val ? " chart-tab--active" : ""}`}
                  onClick={() => setForecastDays(t.val)}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          <div className="chart-wrap">
            <ForecastChart days={forecastDays} />
          </div>

          {activeData?.data && (
            <div className="forecast-stats">
              <div className="fstat">
                <span className="fstat-label">Expected Change</span>
                <span className={`fstat-value ${activeData.data.change >= 0 ? "up" : "down"}`}>
                  {activeData.data.change >= 0 ? "+" : ""}{fmt(activeData.data.change)}%
                </span>
              </div>
              <div className="fstat">
                <span className="fstat-label">Model Accuracy</span>
                <span className="fstat-value">{activeData.data.accuracy}%</span>
              </div>
              <div className="fstat">
                <span className="fstat-label">Confidence Level</span>
                <span className="fstat-value" style={{ color: confidenceColor(activeData.data.confidence) }}>
                  {activeData.data.confidence}
                </span>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* ── Historical Performance ───────────────────────────────── */}
      {history.length > 0 && (
        <section className="section">
          <div className="section-inner">
            <h2 className="section-title">Historical Performance</h2>
            <p className="section-sub">
              Gold price movement over the past 12 months · INR per 10g
            </p>
            <div className="chart-wrap">
              <HistoryChart data={history} />
            </div>
            <div className="history-stats">
              <div className="hstat">
                <span className="hstat-label">52-Week High</span>
                <span className="hstat-value">₹{Math.max(...histPrices).toLocaleString("en-IN", { maximumFractionDigits: 0 })}</span>
              </div>
              <div className="hstat">
                <span className="hstat-label">52-Week Low</span>
                <span className="hstat-value">₹{Math.min(...histPrices).toLocaleString("en-IN", { maximumFractionDigits: 0 })}</span>
              </div>
              <div className="hstat">
                <span className="hstat-label">Period Average</span>
                <span className="hstat-value">
                  ₹{(histPrices.reduce((a, b) => a + b, 0) / histPrices.length).toLocaleString("en-IN", { maximumFractionDigits: 0 })}
                </span>
              </div>
              <div className="hstat">
                <span className="hstat-label">Data Points</span>
                <span className="hstat-value">{history.length} days</span>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ── Footer ──────────────────────────────────────────────── */}
      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-brand">
            <span className="footer-om">ॐ</span>
            <span className="footer-name">Om Jewellers</span>
          </div>
          <p className="footer-disclaimer">
            These forecasts are generated by machine learning models for business planning purposes only.
            Gold prices are influenced by global factors beyond any model's scope.
            All prices are indicative and should not be treated as financial advice.
          </p>
          <p className="footer-copy">© {new Date().getFullYear()} Om Jewellers</p>
        </div>
      </footer>

    </div>
  );
}