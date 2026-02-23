CREATE TABLE IF NOT EXISTS datasets (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  file_path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  n_rows INT NOT NULL,
  n_cols INT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiment_runs (
  id SERIAL PRIMARY KEY,
  dataset_id INT REFERENCES datasets(id),
  model_name TEXT NOT NULL,
  pipeline_config JSONB,
  hyperparams JSONB,
  metrics JSONB,
  artifact_paths JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);