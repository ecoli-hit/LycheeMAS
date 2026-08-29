"""Inspect one framework-neutral TeamSpec without starting a model backend."""

from __future__ import annotations

from pathlib import Path


def print_team_structure(*, team_spec_path: str | None) -> None:
    """Print explicit Nodes, Relations, and adapter-owned bindings."""

    if not team_spec_path:
        raise SystemExit("--list-team-structure requires --team-spec")

    from lychee_mas.eval.teams.compiler import load_team_spec, team_details_from_graph

    path = Path(team_spec_path).expanduser().resolve()
    graph, document = load_team_spec(path, rounds=1)
    details = team_details_from_graph(graph)
    print(f"TeamSpec: {document['id']} ({path})")
    print(
        f"Nodes: {len(document.get('nodes') or [])}; "
        f"Relations: {len(document.get('edges') or [])}; "
        f"Completion: {(document.get('execution_policy') or {}).get('completion')}"
    )
    print("\nNODES")
    for node in document.get("nodes") or []:
        implementation = node.get("implementation") or {}
        operations = ",".join((implementation.get("operations") or {}).keys()) or "-"
        print(
            f"  {node['id']}: {node['node_type']}/{implementation.get('kind')}; "
            f"operations={operations}"
        )
    print("\nRELATIONS")
    for edge in document.get("edges") or []:
        print(
            f"  {edge['id']}: {edge['edge_type']} "
            f"{edge['source']['node']}.{edge['source']['port']} -> "
            f"{edge['target']['node']}.{edge['target']['port']}"
        )
    print("\nFRAMEWORK BINDINGS")
    for framework, plan in details["adapter_plans"].items():
        print(
            f"  {framework}: {plan.get('implementation')}; "
            f"mapping={plan.get('mapping_level')}"
        )
        for binding in plan.get("node_bindings") or []:
            print(
                f"    {binding['node_id']}: {binding.get('runtime_implementation')} "
                f"[{binding.get('mapping_level')}]"
            )


__all__ = ["print_team_structure"]
