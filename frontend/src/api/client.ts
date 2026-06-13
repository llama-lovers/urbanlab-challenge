import type { HeatmapPoint, Event, StatsResponse, NearbyEvent } from '../types'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`)
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)))
  }
  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const api = {
  heatmap: (category?: string) =>
    get<HeatmapPoint[]>('/api/geo/heatmap', category ? { category } : undefined),

  nearby: (lat: number, lon: number, radiusM = 1000) =>
    get<NearbyEvent[]>('/api/geo/nearby', { lat, lon, radius_m: radiusM }),

  events: (category?: string, limit = 1000) =>
    get<Event[]>('/api/data/', category ? { category, limit } : { limit }),

  stats: () => get<StatsResponse>('/api/data/stats'),
}
