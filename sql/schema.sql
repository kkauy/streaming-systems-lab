  CREATE TABLE IF NOT EXISTS model_runs (
    id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(50),
    best_c FLOAT,
    cv_val_mean_roc FLOAT,
    cv_val_std_roc FLOAT,
    test_roc FLOAT,
    random_state INTEGER
);