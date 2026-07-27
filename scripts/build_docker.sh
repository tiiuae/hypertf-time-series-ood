#!/bin/bash

# Build the Docker image for the project
DOCKER_BUILDKIT=1 docker build \
  --build-arg USERNAME=mluser \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -t  hypertf-ood-detection:latest .
