-- Run only after creating a dedicated F0 database and applying the current Alembic migrations.
-- The baseline runner validates the database name separately and never creates this marker itself.
CREATE TABLE rag_f0_evaluation_marker (
    marker_key TEXT PRIMARY KEY,
    dataset_sha256 CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_rag_f0_evaluation_marker_key
        CHECK (marker_key = 'rag-f0-dedicated-database')
);

INSERT INTO rag_f0_evaluation_marker (marker_key, dataset_sha256)
VALUES (
    'rag-f0-dedicated-database',
    '0627e66b0b5b40cb1fbe326f2dfb980be2f43440d264056b5ebc774e157d26ec'
);
