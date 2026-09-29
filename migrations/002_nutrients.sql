-- Parsed "## Nutrients per 100 g" table of an ingredient spec: {column: {kcal, protein_g, ...}}.
-- NULL for documents without one. Idempotent like 001: every *.sql runs on each app start.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS nutrients_per_100g JSONB;
