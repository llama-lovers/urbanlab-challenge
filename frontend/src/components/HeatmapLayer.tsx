import DeckGL from '@deck.gl/react'
import { HeatmapLayer } from '@deck.gl/aggregation-layers'
import Map from 'react-map-gl/maplibre'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { HeatmapPoint } from '../types'

const LUBLIN = { longitude: 22.5684, latitude: 51.2465, zoom: 12, pitch: 0, bearing: 0 }

const BASEMAP = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'

interface Props {
  data: HeatmapPoint[]
  radiusPixels?: number
}

export function HeatmapMap({ data, radiusPixels = 60 }: Props) {
  const layers = [
    new HeatmapLayer({
      id: 'heatmap',
      data,
      getPosition: (d: HeatmapPoint) => [d.lon, d.lat],
      getWeight: (d: HeatmapPoint) => d.weight,
      radiusPixels,
      intensity: 1,
      threshold: 0.05,
    }),
  ]

  return (
    <DeckGL
      initialViewState={LUBLIN}
      controller
      layers={layers}
      style={{ width: '100%', height: '100%' }}
    >
      <Map mapStyle={BASEMAP} reuseMaps />
    </DeckGL>
  )
}
