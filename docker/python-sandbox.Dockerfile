FROM ubuntu:22.04

ARG APT_MIRROR=
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV VIRTUAL_ENV=/opt/lychee-case-venv
ENV PATH=/opt/lychee-case-venv/bin:${PATH}
ENV HOME=/tmp/lychee-home
ENV XDG_CACHE_HOME=/tmp/lychee-cache

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN if [[ -n "${APT_MIRROR}" ]]; then \
        sed -i "s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g; s|http://security.ubuntu.com/ubuntu|${APT_MIRROR}|g" /etc/apt/sources.list; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade -i "${PIP_INDEX_URL}" pip setuptools wheel \
    && mkdir -p "${HOME}" "${XDG_CACHE_HOME}" /opt/lychee-sandbox \
    && chmod 1777 "${HOME}" "${XDG_CACHE_HOME}" \
    && chmod -R a+rwX "${VIRTUAL_ENV}" \
    && rm -rf /var/lib/apt/lists/*

ARG LYCHEE_SANDBOX_FINGERPRINT=unknown

LABEL org.lychee-mas.sandbox.profile="python_sandbox"
LABEL org.lychee-mas.sandbox.schema="1"
LABEL org.lychee-mas.sandbox.fingerprint="${LYCHEE_SANDBOX_FINGERPRINT}"

WORKDIR /workspace
