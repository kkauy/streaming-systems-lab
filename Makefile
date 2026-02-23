DB_URL=postgresql://research:research@localhost:5432/experiments

db-up:
	docker compose up -d

db-down:
	docker compose down

db-init:
	psql "$(DB_URL)" -f sql/schema.sql

train:
	python python-ml/train.py

top-runs:
	psql "$(DB_URL)" -c "SELECT id, (metrics->>'auc_test')::float AS auc_test, created_at FROM experiment_runs ORDER BY auc_test DESC LIMIT 5;"