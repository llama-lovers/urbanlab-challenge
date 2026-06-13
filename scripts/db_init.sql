-- Run automatically on first Postgres container start.
-- Add any additional schema setup here.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Example: generic events table for point data
CREATE TABLE IF NOT EXISTS events (
    id          BIGSERIAL PRIMARY KEY,
    geom        GEOMETRY(POINT, 4326) NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    category    VARCHAR(100) NOT NULL DEFAULT 'default',
    value       DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    attributes  JSONB
);

CREATE INDEX IF NOT EXISTS events_geom_idx   ON events USING GIST(geom);
CREATE INDEX IF NOT EXISTS events_cat_idx    ON events (category);
CREATE INDEX IF NOT EXISTS events_time_idx   ON events (timestamp DESC);

-- Seed a few sample points around Lublin (51.2465, 22.5684)
INSERT INTO events (geom, category, value) VALUES
    (ST_SetSRID(ST_MakePoint(22.5684, 51.2465), 4326), 'sample', 1.0),
    (ST_SetSRID(ST_MakePoint(22.5550, 51.2510), 4326), 'sample', 2.5),
    (ST_SetSRID(ST_MakePoint(22.5800, 51.2400), 4326), 'sample', 0.8),
    (ST_SetSRID(ST_MakePoint(22.5700, 51.2600), 4326), 'sample', 3.2),
    (ST_SetSRID(ST_MakePoint(22.5450, 51.2350), 4326), 'sample', 1.7)
ON CONFLICT DO NOTHING;
