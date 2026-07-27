#!/bin/bash

docker rm -f hypertf-ood-detection 2>/dev/null || true

docker run -it --rm \
    --gpus all \
    --name hypertf-ood-detection \
    -v $(pwd):/workspace \
    -e PYTHONPATH="/workspace/ood_detection:/workspace" \
    hypertf-ood-detection:latest \
    bash
