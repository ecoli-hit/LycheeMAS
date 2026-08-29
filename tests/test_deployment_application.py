from __future__ import annotations

from unittest.mock import Mock

import pytest
from lychee_mas.eval.apis.registry import normalize_api_instance, normalize_api_spec
from lychee_mas.eval.application.deployments import DeploymentApplicationService


def _service() -> tuple[DeploymentApplicationService, Mock, Mock, Mock, Mock]:
    deployments = Mock()
    models = Mock()
    api_access = Mock()
    team_instances = Mock()
    service = DeploymentApplicationService(
        deployments=deployments,
        models=models,
        api_access=api_access,
        team_instances=team_instances,
    )
    return service, deployments, models, api_access, team_instances


def test_local_deployment_spec_is_bound_to_its_model_spec() -> None:
    service, _, models, _, _ = _service()
    models.get_spec.return_value = {"id": "model-a"}

    value = service.validate_spec_references(
        {"id": "deployment-a", "source_spec": {"type": "model", "id": "model-a"}}
    )

    assert value["model_spec_id"] == "model-a"
    models.get_spec.assert_called_once_with("model-a")


def test_api_deployment_rejects_a_model_outside_the_api_access_allowlist() -> None:
    service, _, models, api_access, _ = _service()
    models.get_spec.return_value = {"id": "model-b"}
    api_access.get_spec.return_value = {
        "id": "api-a",
        "allowed_model_spec_ids": ["model-a"],
    }

    with pytest.raises(ValueError, match="does not allow"):
        service.validate_spec_references(
            {
                "id": "deployment-a",
                "model_spec_id": "model-b",
                "source_spec": {"type": "api", "id": "api-a"},
            }
        )


@pytest.mark.parametrize(
    "field",
    ["model_id", "model_info", "capabilities", "thinking_protocol"],
)
def test_api_access_contract_rejects_copied_model_semantics(field: str) -> None:
    spec = {
        "id": "api-a",
        "provider_key": "provider-a",
        "allowed_model_spec_ids": ["model-a"],
        "default_base_url": "https://example.invalid/v1",
        field: {} if field in {"model_info", "capabilities"} else "obsolete",
    }
    instance = {
        "id": "api-instance-a",
        "api_spec_id": "api-a",
        "api_spec_fingerprint": "abc",
        "base_url": "https://example.invalid/v1",
        "auth_mode": "none",
        field: {} if field in {"model_info", "capabilities"} else "obsolete",
    }

    with pytest.raises(ValueError, match="cannot define model semantics"):
        normalize_api_spec(spec)
    with pytest.raises(ValueError, match="cannot copy model semantics"):
        normalize_api_instance(instance)


def test_instantiate_applies_transport_independent_defaults() -> None:
    service, deployments, _, _, _ = _service()
    spec = {"id": "deployment-a"}
    deployments.specs.return_value = [spec]
    deployments.deploy.return_value = {"id": "instance-deployment-a"}
    service.resolve_source = Mock(
        return_value=({"type": "model", "id": "model-instance"}, {"kind": "hf"})
    )

    result = service.instantiate(
        "deployment-a",
        {
            "source_instance_id": "model-instance",
            "actual_pricing_instance_id": "pricing-instance",
        },
    )

    assert result["id"] == "instance-deployment-a"
    deployments.deploy.assert_called_once_with(
        spec,
        instance_id="instance-deployment-a",
        source_instance={"type": "model", "id": "model-instance"},
        source_fields={"kind": "hf"},
        actual_pricing_instance_id="pricing-instance",
        api_equivalent_pricing_instance_id=None,
        api_key="",
    )


def test_referenced_deployment_instance_cannot_be_deleted() -> None:
    service, deployments, _, _, team_instances = _service()
    team_instances.all.return_value = [
        {
            "resource_bindings": [
                {
                    "resource_instance_type": "DeploymentInstance",
                    "resource_instance_id": "deployment-instance-a",
                }
            ]
        }
    ]

    with pytest.raises(ValueError, match="referenced by a TeamInstance"):
        service.delete_instance("deployment-instance-a")

    deployments.delete_instance.assert_not_called()
