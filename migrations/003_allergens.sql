-- Parsed allergens table at the top of an ingredient spec's "## Allergens" section:
-- {column: {"allergens": [EU names], "vegan": true | false | null}}, columns as in
-- nutrients_per_100g. NULL = no table, i.e. status unknown. Idempotent like 001 and 002.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS allergens JSONB;
