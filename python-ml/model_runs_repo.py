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
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO model_runs
        (model_name, dataset_name, n_splits, best_c, cv_val_mean_roc, cv_val_std_roc, test_roc, random_state)
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
    cur.close()
    conn.close()
