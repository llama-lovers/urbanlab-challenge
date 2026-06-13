export interface HeatmapPoint {
  lat: number
  lon: number
  weight: number
}

export interface Event {
  id: number
  lat: number
  lon: number
  timestamp: string
  category: string
  value: number
  attributes: Record<string, unknown> | null
}

export interface StatsResponse {
  total: number
  by_category: Record<string, number>
}

export interface NearbyEvent extends Event {
  distance_m: number
}
