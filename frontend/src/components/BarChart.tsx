import {
  BarChart as ReBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

interface DataPoint {
  label: string
  value: number
}

interface Props {
  data: DataPoint[]
  title?: string
  color?: string
}

export function BarChart({ data, title, color = '#6366f1' }: Props) {
  return (
    <div className="bg-gray-900 rounded-xl p-4">
      {title && <h3 className="text-sm font-medium text-gray-400 mb-3">{title}</h3>}
      <ResponsiveContainer width="100%" height={200}>
        <ReBarChart data={data} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="label" tick={{ fill: '#9ca3af', fontSize: 12 }} />
          <YAxis tick={{ fill: '#9ca3af', fontSize: 12 }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: 8 }}
            labelStyle={{ color: '#f9fafb' }}
          />
          <Bar dataKey="value" fill={color} radius={[4, 4, 0, 0]} />
        </ReBarChart>
      </ResponsiveContainer>
    </div>
  )
}
