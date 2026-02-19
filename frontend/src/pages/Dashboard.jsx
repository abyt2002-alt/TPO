import { Link } from 'react-router-dom'
import { Users, TrendingUp, DollarSign, Calendar } from 'lucide-react'
import Layout from '../components/Layout'

const Dashboard = () => {
  const features = [
    {
      icon: Users,
      title: 'Store Segmentation',
      description: 'Segment outlets by Recency, Frequency, and Monetary value',
      link: '/rfm',
      color: 'bg-primary',
      isLive: true,
    },
    {
      icon: TrendingUp,
      title: 'Discount Analysis (Base Depth)',
      description: 'Step 2 is available inside Store Segmentation',
      link: '/rfm?step=2',
      color: 'bg-secondary',
      disabled: false,
      isLive: true,
    },
    {
      icon: DollarSign,
      title: 'Modeling & ROI',
      description: 'Step 3 is available inside Store Segmentation',
      link: '/rfm?step=3',
      color: 'bg-accent',
      disabled: false,
      isLive: true,
    },
    {
      icon: Calendar,
      title: '12-Month Planner',
      description: 'Step 4 is available inside Store Segmentation',
      link: '/rfm?step=4',
      color: 'bg-primary',
      disabled: false,
      isLive: true,
    },
  ]

  return (
    <Layout>
      <div className="space-y-8">
        {/* Welcome Section */}
        <div className="bg-gradient-to-r from-primary to-secondary text-white rounded-lg shadow-lg p-8">
          <h1 className="text-4xl font-bold mb-4">Welcome to Trade Promo Optimization Tool</h1>
          <p className="text-xl opacity-90">
            Comprehensive analytics platform for retail outlet performance and promotional optimization
          </p>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-muted text-sm">Total Records</p>
                <p className="text-3xl font-bold text-primary">3.1M+</p>
              </div>
              <Users className="text-primary opacity-20" size={48} />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-muted text-sm">RFM Segments</p>
                <p className="text-3xl font-bold text-success">8</p>
              </div>
              <TrendingUp className="text-success opacity-20" size={48} />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-muted text-sm">Data Files</p>
                <p className="text-3xl font-bold text-warning">20</p>
              </div>
              <DollarSign className="text-warning opacity-20" size={48} />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-muted text-sm">Data Range</p>
                <p className="text-3xl font-bold text-accent">14 Mo</p>
              </div>
              <Calendar className="text-accent opacity-20" size={48} />
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-body">Available Features</h2>
            <div className="inline-flex items-center rounded-full bg-secondary px-3 py-1">
              <span className="text-xs font-semibold text-white">
                {features.filter((f) => f.isLive).length} Live Now
              </span>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {features.map((feature) => {
              const Icon = feature.icon
              const isDisabled = feature.disabled

              const card = (
                <div
                  className={`relative rounded-lg p-6 transition-all ${
                    isDisabled
                      ? 'bg-white/80 border border-gray-200 opacity-60 cursor-not-allowed'
                      : 'bg-white border border-secondary/50 shadow-md ring-1 ring-secondary/20 hover:shadow-xl hover:-translate-y-1 cursor-pointer'
                  }`}
                >
                  {!isDisabled && (
                    <span className="absolute top-3 right-3 text-[10px] bg-secondary text-white px-2 py-1 rounded-full font-semibold tracking-wide">
                      Available Now
                    </span>
                  )}
                  <div className="flex items-start space-x-4">
                    <div className={`${feature.color} p-3 rounded-lg`}>
                      <Icon className="text-white" size={32} />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-body mb-2">
                        {feature.title}
                        {isDisabled && (
                          <span className="ml-2 text-xs bg-accent-light text-muted px-2 py-1 rounded">
                            Coming Soon
                          </span>
                        )}
                      </h3>
                      <p className="text-muted">{feature.description}</p>
                      {!isDisabled && (
                        <div className="mt-3">
                          <span className="inline-flex items-center text-xs font-semibold text-secondary">
                            Open Feature
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )

              return isDisabled ? (
                <div key={feature.title}>{card}</div>
              ) : (
                <Link key={feature.title} to={feature.link} className="block">
                  {card}
                </Link>
              )
            })}
          </div>
        </div>

        {/* About Section */}
        <div className="bg-white rounded-lg shadow-md p-8">
          <h2 className="text-2xl font-bold text-body mb-4">About Store Segmentation</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <h3 className="font-semibold text-lg text-primary mb-2">Recency (R)</h3>
              <p className="text-muted">
                How recently did the outlet make a purchase? Recent customers are more likely to respond to offers.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-lg text-success mb-2">Frequency (F)</h3>
              <p className="text-muted">
                How often does the outlet purchase? Frequent buyers show higher engagement and loyalty.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-lg text-warning mb-2">Monetary (M)</h3>
              <p className="text-muted">
                How much does the outlet spend per order? High-value customers contribute more to revenue.
              </p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  )
}

export default Dashboard
