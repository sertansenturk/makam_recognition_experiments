#!/usr/bin/env bash

HOST=0.0.0.0
PORT=5000
BACKEND_STORE_URI=file:/server/mlflow-store
DEFAULT_ARTIFACT_ROOT=file:/server/mlflow-artifacts

printf "Backend Store: ${BACKEND_STORE_URI}\nArtifact root: ${DEFAULT_ARTIFACT_ROOT}\n"
ls -la /server/mlflow-store
ls -la /server/mlflow-artifacts

case "$1" in
    db_upgrade)
        mlflow db upgrade ${BACKEND_STORE_URI}
        ;;
    server)
        mlflow server \
            --host "${HOST}" \
            --port "${PORT}" \
            --backend-store-uri "${BACKEND_STORE_URI}" \
            --default-artifact-root "${DEFAULT_ARTIFACT_ROOT}"
        ;;
    *)
    echo "Unknown command \"$1\""
    ;;
esac