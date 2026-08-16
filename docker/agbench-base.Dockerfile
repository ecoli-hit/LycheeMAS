FROM mcr.microsoft.com/devcontainers/python:3.11

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV VIRTUAL_ENV=/opt/lychee-case-venv
ENV PATH=/opt/lychee-case-venv/bin:${PATH}
ENV HOME=/tmp/lychee-home
ENV XDG_CACHE_HOME=/tmp/lychee-cache
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Keep the base OS additions and timezone aligned with AgBench's Dockerfile.
RUN apt-get update \
    && apt-get install -y ffmpeg exiftool \
    && ln -snf /usr/share/zoneinfo/US/Pacific /etc/localtime \
    && echo "US/Pacific" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade -i "${PIP_INDEX_URL}" pip setuptools wheel \
    && mkdir -p "${HOME}" "${XDG_CACHE_HOME}" /opt/lychee-sandbox \
    && chmod 1777 "${HOME}" "${XDG_CACHE_HOME}"

# AgBench preloads AutoGen's transitive dependencies, then removes the three
# AutoGen distributions. Benchmark-specific requirements add them back later.
RUN python -m pip install --no-cache-dir -i "${PIP_INDEX_URL}" \
        autogen-core autogen-agentchat autogen-ext pyyaml \
    && python -m pip uninstall --yes autogen-core autogen-agentchat autogen-ext

COPY docker/agbench-base.requirements.txt /opt/lychee-sandbox/agbench-base.requirements.txt

RUN python -m pip install --no-cache-dir -i "${PIP_INDEX_URL}" \
        -r /opt/lychee-sandbox/agbench-base.requirements.txt \
    && python -m playwright install --with-deps chromium \
    && python -m pip freeze > /opt/lychee-sandbox/pip-freeze.txt \
    && chmod -R a+rX /ms-playwright \
    && chmod -R a+rwX "${VIRTUAL_ENV}"

COPY docker/agbench-base-capabilities.json /opt/lychee-sandbox/capabilities.json
COPY docker/verify_benchmark_sandbox.py /opt/lychee-sandbox/verify_benchmark_sandbox.py

RUN chmod -R a+rX /opt/lychee-sandbox \
    && python /opt/lychee-sandbox/verify_benchmark_sandbox.py --json

ARG LYCHEE_SANDBOX_FINGERPRINT=unknown

LABEL org.lychee-mas.sandbox.profile="agbench_base"
LABEL org.lychee-mas.sandbox.schema="1"
LABEL org.lychee-mas.sandbox.fingerprint="${LYCHEE_SANDBOX_FINGERPRINT}"

WORKDIR /workspace
