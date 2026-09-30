-- Full-text side of hybrid search: a tsvector per chunk, kept in sync by Postgres itself.
-- The 'english' config must be named: to_tsvector(text) with the session default is not
-- immutable, and a generated column requires an immutable expression. Adding a STORED column
-- computes it for the rows already there, so no re-ingest is needed. Idempotent like 001-003.
ALTER TABLE chunks
  ADD COLUMN IF NOT EXISTS tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
