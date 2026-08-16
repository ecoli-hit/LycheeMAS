FROM lychee-agbench-base:local

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# AgBench mounts the same vendored tree at /autogen_python and installs these
# requirements for every GAIA case. This layer pre-installs that exact source.
COPY src/autogen/python/packages/autogen-core /autogen_python/packages/autogen-core
COPY src/autogen/python/packages/autogen-ext /autogen_python/packages/autogen-ext
COPY src/autogen/python/packages/autogen-agentchat /autogen_python/packages/autogen-agentchat
COPY docker/agbench-gaia.requirements.txt /opt/lychee-sandbox/agbench-gaia.requirements.txt

RUN python -m pip install --no-cache-dir -i "${PIP_INDEX_URL}" \
        -r /opt/lychee-sandbox/agbench-gaia.requirements.txt \
    && python -m pip freeze > /opt/lychee-sandbox/pip-freeze.txt \
    && chmod -R a+rwX "${VIRTUAL_ENV}"

COPY docker/agbench-gaia-capabilities.json /opt/lychee-sandbox/capabilities.json
COPY docker/verify_benchmark_sandbox.py /opt/lychee-sandbox/verify_benchmark_sandbox.py

RUN chmod -R a+rX /opt/lychee-sandbox \
    && python /opt/lychee-sandbox/verify_benchmark_sandbox.py --json

ARG LYCHEE_SANDBOX_FINGERPRINT=unknown

LABEL org.lychee-mas.sandbox.profile="agbench_gaia"
LABEL org.lychee-mas.sandbox.schema="1"
LABEL org.lychee-mas.sandbox.fingerprint="${LYCHEE_SANDBOX_FINGERPRINT}"

WORKDIR /workspace
