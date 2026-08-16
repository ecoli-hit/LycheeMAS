FROM lychee-python-sandbox:local

ARG LYCHEE_SANDBOX_FINGERPRINT=unknown

LABEL org.lychee-mas.sandbox.profile="human_eval"
LABEL org.lychee-mas.sandbox.schema="1"
LABEL org.lychee-mas.sandbox.fingerprint="${LYCHEE_SANDBOX_FINGERPRINT}"

WORKDIR /workspace
