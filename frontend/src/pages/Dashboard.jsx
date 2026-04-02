import { Link } from 'react-router-dom'
import {
  BarChart3,
  BrainCircuit,
  CalendarDays,
  LineChart,
  Percent,
  Sparkles,
  TrendingUp,
} from 'lucide-react'
import Layout from '../components/Layout'

const workflowSteps = [
  {
    number: '01',
    title: 'Store Segmentation',
    description: 'Scope the business and run outlet segmentation.',
    link: '/rfm',
    icon: BarChart3,
    tone: 'bg-blue-50 text-blue-700 border-blue-200',
  },
  {
    number: '02',
    title: 'Discount Analysis',
    description: 'Estimate structural discount and slab calendar.',
    link: '/rfm?step=2',
    icon: Percent,
    tone: 'bg-amber-50 text-amber-700 border-amber-200',
  },
  {
    number: '03',
    title: 'Modeling & ROI',
    description: 'Fit slab models and evaluate ROI logic.',
    link: '/rfm?step=3',
    icon: LineChart,
    tone: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  },
  {
    number: '04',
    title: 'Scenario Planner',
    description: 'Plan the next 3 months across 12-ML and 18-ML.',
    link: '/rfm?step=4',
    icon: CalendarDays,
    tone: 'bg-violet-50 text-violet-700 border-violet-200',
  },
  {
    number: '05',
    title: 'Scenario Generator',
    description: 'Compare fixed and AI-generated options.',
    link: '/rfm?step=5',
    icon: BrainCircuit,
    tone: 'bg-rose-50 text-rose-700 border-rose-200',
  },
  {
    number: '06',
    title: 'Built-in EDA',
    description: 'Validate price, discount, MRP, and volume signals.',
    link: '/rfm?step=2',
    icon: TrendingUp,
    tone: 'bg-cyan-50 text-cyan-700 border-cyan-200',
  },
]

const engineItems = [
  {
    label: 'AI-led planning',
    value: 'Prompt to ranked scenarios',
  },
  {
    label: 'Scheme backbone',
    value: 'Structural slab discount logic',
  },
  {
    label: 'Modeling layer',
    value: 'Own, lag, weighted slab effects',
  },
  {
    label: 'Cross effects',
    value: '12-ML / 18-ML interaction',
  },
]

const Dashboard = () => {
  return (
    <Layout>
      <div className="space-y-4">
        <section className="rounded-3xl border border-slate-200 bg-white shadow-sm overflow-hidden">
          <div className="grid grid-cols-1 xl:grid-cols-[1.15fr_0.85fr]">
            <div className="px-8 py-7">
              <div className="inline-flex items-center gap-2 rounded-full bg-blue-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-blue-700">
                <Sparkles className="h-3.5 w-3.5" />
                QPS
              </div>
              <h1 className="mt-4 text-4xl font-bold tracking-tight text-slate-950">
                QPS Optimization
              </h1>
              <p className="mt-3 max-w-2xl text-base leading-7 text-slate-600">
                Structural scheme estimation, slab-level modeling, cross-pack planning, and AI-led scenario evaluation in one connected workflow.
              </p>
            </div>

            <div className="border-t border-slate-200 bg-slate-50 px-8 py-7 xl:border-l xl:border-t-0">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                Focus
              </div>
              <div className="mt-3 text-2xl font-semibold leading-tight text-slate-900">
                AI-led planning is the top layer. Scheme logic, modeling, and cross effects are the decision engine below it.
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-sm font-medium text-slate-700">12-ML / 18-ML</span>
                <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-sm font-medium text-slate-700">3-Month Planning</span>
                <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-sm font-medium text-slate-700">AI + MC Scenarios</span>
              </div>
            </div>
          </div>
        </section>

        <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center justify-between gap-4">
            <div>
              <h2 className="text-xl font-bold text-slate-900">Decision Engine</h2>
              <p className="mt-1 text-sm text-slate-500">
                The four layers that drive planning outputs.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
            {engineItems.map((item) => (
              <div key={item.label} className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-4">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                  {item.label}
                </div>
                <div className="mt-2 text-base font-semibold text-slate-900">
                  {item.value}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-4">
            <h2 className="text-xl font-bold text-slate-900">Workflow</h2>
            <p className="mt-1 text-sm text-slate-500">
              Compact path from scope selection to scenario ranking.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 xl:grid-cols-3">
            {workflowSteps.map((step) => {
              const Icon = step.icon
              return (
                <Link
                  key={step.number}
                  to={step.link}
                  className="group rounded-2xl border border-slate-200 bg-white p-4 transition-all hover:-translate-y-0.5 hover:border-slate-300 hover:shadow-md"
                >
                  <div className="flex items-start gap-4">
                    <div className={`rounded-2xl border p-3 ${step.tone}`}>
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="min-w-0">
                      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                        Step {step.number}
                      </div>
                      <h3 className="mt-1 text-lg font-semibold text-slate-900 group-hover:text-blue-700">
                        {step.title}
                      </h3>
                      <p className="mt-1 text-sm leading-6 text-slate-600">
                        {step.description}
                      </p>
                    </div>
                  </div>
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
