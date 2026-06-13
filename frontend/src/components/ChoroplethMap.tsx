import DeckGL from '@deck.gl/react'
import { GeoJsonLayer } from '@deck.gl/layers'
import Map from 'react-map-gl/maplibre'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { FeatureCollection, Geometry } from 'geojson'

const LUBLIN = { longitude: 22.5684, latitude: 51.2465, zoom: 11, pitch: 0, bearing: 0 }

interface DistrictProps {
  name: string
  value: number
  event_count: number
  [key: string]: unknown
}

interface Props {
  geojson: FeatureCollection<Geometry, DistrictProps> | null
}

export function ChoroplethMap({ geojson }: Props) {
  if (!geojson) return null

  const values = geojson.features.map((f) => f.properties.value)
  const maxValue = Math.max(...values, 1)

  const layer = new GeoJsonLayer({
    id: 'choropleth',
    data: geojson,
    getFillColor: (f) => {
      const t = f.properties.value / maxValue
      return [255 * t, 100 * (1 - t), 0, 180] as [number, number, number, number]
    },
    getLineColor: [255, 255, 255, 80] as [number, number, number, number],
    lineWidthMinPixels: 1,
    pickable: true,
  })

  return (
    <DeckGL
      initialViewState={LUBLIN}
      controller
      layers={[layer]}
      style={{ width: '100%', height: '100%' }}
      getTooltip={({ object }) =>
        object && `${object.properties.name}: ${object.properties.event_count} events`
      }
    >
      <Map mapStyle="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json" reuseMaps />
    </DeckGL>
  )
}
