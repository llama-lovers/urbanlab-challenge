import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { HeatmapMap } from '../components/HeatmapLayer'
import { BarChart } from '../components/BarChart'
import { StatCard } from '../components/StatCard'

export function Dashboard() {
  const [category, setCategory] = useState<string | undefined>()

  const { data: heatmap = [], isLoading: heatmapLoading } = useQuery({
    queryKey: ['heatmap', category],
    queryFn: () => api.heatmap(category),
  })

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: api.stats,
  })

  const categoryData = stats
    ? Object.entries(stats.by_category).map(([label, value]) => ({ label, value }))
    : []

  return (
    <div className="min-h-screen bg-gray-950 text-white p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold">Urban Analytics — Lublin</h1>
        <select
          className="bg-gray-800 text-white text-sm rounded-lg px-3 py-1.5 border border-gray-700"
          value={category ?? ''}
          onChange={(e) => setCategory(e.target.value || undefined)}
        >
          <option value="">All categories</option>
          {Object.keys(stats?.by_category ?? {}).map((cat) => (
            <option key={cat} value={cat}>
              {cat}
            </option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard title="Total events" value={stats?.total ?? '—'} />
        <StatCard title="Shown on map" value={heatmap.length} />
        <StatCard title="Categories" value={Object.keys(stats?.by_category ?? {}).length} />
        <StatCard title="Active filter" value={category ?? 'none'} />
      </div>

      <div className="flex gap-4 flex-1 min-h-0">
        <div className="relative flex-1 rounded-xl overflow-hidden" style={{ height: 520 }}>
          <HeatmapMap data={heatmap} />
          {heatmapLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900/60 text-gray-400 text-sm">
              Loading…
            </div>
          )}
        </div>

        <div className="w-72 flex flex-col gap-4">
          <BarChart title="Events by category" data={categoryData.slice(0, 8)} color="#6366f1" />
        </div>
      </div>
    </div>
  )
}
