#!/bin/bash
set -e

./wait-for-it.sh $1:$2 -t $3 -- \
    echo "Running pytest as $USER with uid $UID" && \
    pytest -vv "$(pwd)/docker"

# test that package is installed in editable mode
# i.e. if the .egg-link file is in the site-packeges
ls /opt/conda/lib/python3.7/site-packages/mre.egg-linkas