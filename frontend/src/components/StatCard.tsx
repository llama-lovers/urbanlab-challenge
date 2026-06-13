interface Props {
  title: string
  value: number | string
  unit?: string
  trend?: 'up' | 'down' | 'neutral'
}

export function StatCard({ title, value, unit }: Props) {
  return (
    <div className="bg-gray-900 rounded-xl p-4">
      <p className="text-sm text-gray-400">{title}</p>
      <p className="text-2xl font-bold text-white mt-1">
        {value}
        {unit && <span className="text-sm font-normal text-gray-400 ml-1">{unit}</span>}
      </p>
    </div>
  )
}
