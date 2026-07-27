# ---- Base Image ----
FROM python:3.11.9-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="/workspace/src"

WORKDIR /workspace

# Install deps for builds + debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl wget git openssh-client sudo \
    dnsutils iputils-ping net-tools traceroute \
    && git config --global --add safe.directory /workspace \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# ---- Install Dependencies ----
FROM base AS install-deps

# Copy dependency files
COPY requirements.txt /workspace/
COPY tests/requirements.txt /workspace/tests/requirements.txt

# Install runtime, test, and dev dependencies
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r tests/requirements.txt


# ---- Final Stage ----
FROM install-deps AS final

ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=1000

# Create non-root user inside the container
RUN set -eux; \
    if ! getent group "${USER_GID}" >/dev/null; then \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
    fi; \
    if ! id -u "${USERNAME}" >/dev/null 2>&1; then \
        useradd \
            --uid "${USER_UID}" \
            --gid "${USER_GID}" \
            --create-home \
            --shell /bin/bash \
            "${USERNAME}"; \
    fi; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"; \
    chown -R "${USER_UID}:${USER_GID}" /workspace

USER ${USERNAME}

WORKDIR /workspace
