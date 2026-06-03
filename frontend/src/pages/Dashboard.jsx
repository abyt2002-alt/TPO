import { Link } from 'react-router-dom'
import {
  BarChart3,
  BrainCircuit,
  CalendarDays,
  LineChart,
  Percent,
  Sparkles,
  ArrowRight,
  Target,
  Layers,
  Workflow,
  Cpu,
  GitBranch,
} from 'lucide-react'
import Layout from '../components/Layout'

const pipeline = [
  {
    number: '01',
    title: 'Store Segmentation',
    description: 'Scope the business and segment outlets.',
    link: '/rfm',
    icon: BarChart3,
    ring: 'group-hover:border-blue-300',
    iconCls: 'bg-blue-50 text-blue-600',
  },
  {
    number: '02',
    title: 'Discount Analysis',
    description: 'Estimate structural scheme depth and slab calendar.',
    link: '/rfm?step=2',
    icon: Percent,
    ring: 'group-hover:border-amber-300',
    iconCls: 'bg-amber-50 text-amber-600',
  },
  {
    number: '03',
    title: 'Modeling & ROI',
    description: 'Fit slab demand models and evaluate ROI.',
    link: '/rfm?step=3',
    icon: LineChart,
    ring: 'group-hover:border-emerald-300',
    iconCls: 'bg-emerald-50 text-emerald-600',
  },
  {
    number: '04',
    title: 'Scenario Planner',
    description: 'Plan the horizon with live impact.',
    link: '/rfm?step=4',
    icon: CalendarDays,
    ring: 'group-hover:border-violet-300',
    iconCls: 'bg-violet-50 text-violet-600',
  },
  {
    number: '05',
    title: 'TrinityAI Generator',
    description: 'Intent to ranked, ready-to-act scenarios.',
    link: '/rfm?step=5',
    icon: BrainCircuit,
    ring: 'group-hover:border-fuchsia-300',
    iconCls: 'bg-fuchsia-50 text-fuchsia-600',
  },
]

const engineLayers = [
  {
    tag: 'Top layer',
    title: 'AI Scenario Intelligence',
    description: 'Agentic framework with Monte Carlo simulation — business intent turned into ranked, implementable QPS scenarios at scale.',
    icon: Sparkles,
    accent: 'from-violet-600 to-blue-500',
    border: 'border-violet-200',
    bg: 'bg-violet-50/50',
  },
  {
    tag: 'Backbone',
    title: 'Multi-Step Demand Modeling',
    description: 'Own discount, lag carryover, cross-slab effects and STL structural trend — fit per slab.',
    icon: Cpu,
    accent: 'from-emerald-500 to-teal-500',
    border: 'border-emerald-200',
    bg: 'bg-emerald-50/50',
  },
  {
    tag: 'Projection',
    title: 'Forecast & Trend Layer',
    description: 'Holt\'s trend extrapolation and STL decomposition anchor the 3-month forward baseline.',
    icon: Layers,
    accent: 'from-amber-500 to-orange-500',
    border: 'border-amber-200',
    bg: 'bg-amber-50/50',
  },
  {
    tag: 'Coupling',
    title: 'Cross-Pack & Slab Dynamics',
    description: 'Pack-to-pack and slab-to-slab interactions modeled so every discount move stays consistent.',
    icon: GitBranch,
    accent: 'from-blue-500 to-cyan-500',
    border: 'border-blue-200',
    bg: 'bg-blue-50/50',
  },
]

const heroStats = [
  { label: 'Granularity', value: 'Any pack, channel or region' },
  { label: 'Planning horizon', value: '3, 6 or custom months' },
  { label: 'Scenario scale', value: '100K+ implementable options' },
]

const Dashboard = () => {
  return (
    <Layout>
      <style>{`
        @keyframes lpGradient { 0%,100% { background-position:0% 50% } 50% { background-position:100% 50% } }
        @keyframes lpFloat { 0%,100% { transform:translateY(0) } 50% { transform:translateY(-8px) } }
        @keyframes lpFadeUp { from { opacity:0; transform:translateY(10px) } to { opacity:1; transform:translateY(0) } }
        .lp-gradient { background:linear-gradient(120deg,#4f46e5,#7c3aed,#2563eb,#8b5cf6); background-size:300% 300%; animation:lpGradient 7s ease infinite; }
        .lp-float { animation:lpFloat 5s ease-in-out infinite; }
        .lp-fade { animation:lpFadeUp .5s ease both; }
        .lp-grid-overlay { background-image:linear-gradient(rgba(255,255,255,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.08) 1px,transparent 1px); background-size:34px 34px; }
      `}</style>

      <div className="flex flex-col gap-3">
        {/* ── HERO ── */}
        <section className="lp-gradient relative overflow-hidden rounded-2xl shadow-lg">
          <div className="lp-grid-overlay absolute inset-0 opacity-60" />
          <div className="absolute -right-16 -top-16 h-52 w-52 rounded-full bg-white/10 blur-2xl lp-float" />
          <div className="absolute -bottom-20 left-1/3 h-48 w-48 rounded-full bg-fuchsia-400/20 blur-3xl" />

          <div className="relative grid grid-cols-1 gap-6 px-7 py-6 xl:grid-cols-[1.3fr_0.7fr] xl:px-10">
            <div className="lp-fade">
              <div className="inline-flex items-center gap-2 rounded-full border border-white/30 bg-white/10 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-white backdrop-blur">
                <Sparkles className="h-3 w-3 text-yellow-300" />
                Powered by TrinityAI
              </div>

              <h1 className="mt-3 text-3xl font-bold leading-[1.12] tracking-tight text-white sm:text-4xl">
                Turn business intent into
                <br />
                winning QPS scenarios.
              </h1>

              <p className="mt-2.5 max-w-xl text-sm leading-6 text-white/80">
                One intelligent workspace that unifies demand modeling, structural scheme logic and
                AI scenario generation — so every discount decision is grounded in evidence and aligned
                to the targets you set.
              </p>

              <div className="mt-4 flex flex-wrap items-center gap-3">
                <Link
                  to="/rfm"
                  className="inline-flex items-center gap-2 rounded-xl bg-white px-4 py-2.5 text-sm font-bold text-violet-700 shadow-lg transition-transform hover:-translate-y-0.5"
                >
                  Start Planning
                  <ArrowRight className="h-4 w-4" />
                </Link>
                <Link
                  to="/rfm?step=5"
                  className="inline-flex items-center gap-2 rounded-xl border border-white/40 bg-white/10 px-4 py-2.5 text-sm font-bold text-white backdrop-blur transition-colors hover:bg-white/20"
                >
                  <Sparkles className="h-4 w-4 text-yellow-300" />
                  Generate AI Scenarios
                </Link>
              </div>

              <div className="mt-5 grid max-w-xl grid-cols-1 gap-2.5 sm:grid-cols-3">
                {heroStats.map((stat) => (
                  <div key={stat.label} className="rounded-xl border border-white/20 bg-white/10 px-3 py-2 backdrop-blur">
                    <div className="text-[9px] font-semibold uppercase tracking-[0.16em] text-white/60">
                      {stat.label}
                    </div>
                    <div className="mt-0.5 text-[13px] font-bold text-white">{stat.value}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Intent → Scenarios visual */}
            <div className="lp-fade hidden items-center justify-center xl:flex" style={{ animationDelay: '120ms' }}>
              <div className="w-full max-w-sm rounded-2xl border border-white/20 bg-white/10 p-4 backdrop-blur-md">
                <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-white/70">
                  <Target className="h-3 w-3" /> Your intent
                </div>
                <p className="mt-1.5 rounded-lg bg-white/15 px-3 py-2 text-xs leading-relaxed text-white">
                  "Protect margin but keep 12-ML volume growing — avoid deep discounting on 18-ML."
                </p>

                <div className="my-2 flex items-center justify-center">
                  <div className="flex items-center gap-1.5 rounded-full bg-white/20 px-2.5 py-0.5 text-[10px] font-bold text-white">
                    <Sparkles className="h-3 w-3 text-yellow-300" />
                    TrinityAI
                    <ArrowRight className="h-3 w-3" />
                  </div>
                </div>

                <div className="space-y-1.5">
                  {[
                    { name: 'Balanced Growth', vol: '+18.4%' },
                    { name: 'Margin Guard', vol: '+9.1%' },
                    { name: 'Volume Push', vol: '+24.2%' },
                  ].map((row, i) => (
                    <div
                      key={row.name}
                      className="flex items-center justify-between rounded-lg border border-white/15 bg-white/10 px-3 py-1.5"
                      style={{ animation: 'lpFadeUp .5s ease both', animationDelay: `${250 + i * 100}ms` }}
                    >
                      <span className="text-xs font-medium text-white">{row.name}</span>
                      <span className="text-xs font-bold text-emerald-300">{row.vol}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── DECISION ENGINE ── */}
        <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-3 flex items-center gap-2.5">
            <div className="rounded-lg bg-slate-900 p-1.5 text-white">
              <Layers className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900">The Decision Engine</h2>
              <p className="text-xs text-slate-500">
                AI planning sits on top; scheme logic, modeling and cross effects power it underneath.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {engineLayers.map((layer, i) => {
              const Icon = layer.icon
              return (
                <div
                  key={layer.title}
                  className={`lp-fade rounded-xl border ${layer.border} ${layer.bg} p-3.5`}
                  style={{ animationDelay: `${i * 70}ms` }}
                >
                  <div className="flex items-center gap-2.5">
                    <div className={`inline-flex rounded-lg bg-gradient-to-br ${layer.accent} p-2 text-white shadow-sm`}>
                      <Icon className="h-4 w-4" />
                    </div>
                    <span className="text-[9px] font-bold uppercase tracking-[0.16em] text-slate-400">
                      {layer.tag}
                    </span>
                  </div>
                  <h3 className="mt-2 text-sm font-bold text-slate-900">{layer.title}</h3>
                  <p className="mt-1 text-xs leading-5 text-slate-600">{layer.description}</p>
                </div>
              )
            })}
          </div>
        </section>

        {/* ── END-TO-END PIPELINE ── */}
        <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-3 flex items-center gap-2.5">
            <div className="rounded-lg bg-slate-900 p-1.5 text-white">
              <Workflow className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900">End-to-End Pipeline</h2>
              <p className="text-xs text-slate-500">From scope selection to ranked scenarios — one connected flow.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
            {pipeline.map((step, idx) => {
              const Icon = step.icon
              return (
                <Link
                  key={step.number}
                  to={step.link}
                  className={`group relative flex flex-col rounded-xl border border-slate-200 bg-white p-3.5 transition-all hover:-translate-y-1 hover:shadow-md ${step.ring}`}
                >
                  {idx < pipeline.length - 1 && (
                    <div className="absolute -right-2.5 top-1/2 z-10 hidden -translate-y-1/2 xl:block">
                      <div className="flex h-5 w-5 items-center justify-center rounded-full border border-slate-200 bg-white text-slate-300 shadow-sm group-hover:text-slate-400">
                        <ArrowRight className="h-3 w-3" />
                      </div>
                    </div>
                  )}

                  <div className="flex items-center justify-between">
                    <div className={`rounded-lg p-2 ${step.iconCls}`}>
                      <Icon className="h-4 w-4" />
                    </div>
                    <span className="text-xl font-black text-slate-100 transition-colors group-hover:text-slate-200">
                      {step.number}
                    </span>
                  </div>

                  <h3 className="mt-2 text-sm font-bold text-slate-900">{step.title}</h3>
                  <p className="mt-0.5 text-xs leading-5 text-slate-500">{step.description}</p>
                </Link>
              )
            })}
          </div>
        </section>
      </div>
    </Layout>
  )
}

export default Dashboard
