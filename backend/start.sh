#!/bin/sh
set -e

echo "==> Running Alembic migrations..."
if alembic upgrade head; then
    echo "==> Migrations complete."
else
    echo "==> WARNING: Migrations failed (DB may not be reachable). Starting server anyway..."
    echo "==> Fix DATABASE_URL in .env then run: docker compose restart"
fi

echo "==> Starting AskMyDocs API..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
