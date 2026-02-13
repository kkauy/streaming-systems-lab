from db import get_conn


def insert_model_run(
    model_name: str,
    dataset_name: str,
    n_splits: int,
    best_c: float,
    cv_val_mean_roc: float,
    cv_val_std_roc: float,
    test_roc: float,
    random_state: int = 42,
):
    # transaction-safe context manager
    # atomic experiment logging
    # automatic rollback on failure
    # reproducible and integrity-safe research records
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_runs
                (model_name, dataset_name, n_splits, best_c,
                 cv_val_mean_roc, cv_val_std_roc, test_roc, random_state)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    model_name,
                    dataset_name,
                    int(n_splits),
                    float(best_c),
                    float(cv_val_mean_roc),
                    float(cv_val_std_roc),
                    float(test_roc),
                    int(random_state),
                ),
            )
        conn.commit()