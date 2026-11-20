"""Business-domain loader.

WHAT
    A business domain is a folder under `domains/`:

        domains/<name>/
            domain_manifest.yaml   title, description, thresholds, documented rules, decision guide
            server.py              the domain's MCP server (launched as a subprocess)
            policies.py            REQUIRED_THRESHOLDS + create_policy_engine(thresholds)
            benchmark.json         evaluation questions
            data/                  whatever the server reads

    `load_domain(name)` turns that folder into a `DomainSpec` the runtime can
    use: server specs for `build_single_agent_runtime(server_modules=...)`
    and a ready `PolicyEngine`.

WHY THIS EXISTS (the structural bugs it replaces)
    - The manifests looked like configuration but nothing read them. Their
      `condition` strings, `active_tools` and `database` sections were
      documentation that had already drifted from the code: the product
      analytics manifest said rollouts above 20% need approval while the code
      enforced 50%, and the operations manifest described a rule that did not
      exist. Manifests now hold the thresholds the code actually uses, and the
      loader refuses a manifest whose thresholds or rule names do not match
      the policy module.
    - Domain servers lived inside the generic runtime package
      (agent_runtime/mcp_layer/servers/) while their data and policies lived
      in domains/, and evaluation scripts did `import domains...`, which only
      works when the repository root happens to be on sys.path (it is under
      pytest, not under `python scripts/...`). Each domain is now
      self-contained, and policies are loaded by file path.
"""

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

import yaml

from .agent.policy import PolicyEngine
from .config import REPO_ROOT

DOMAINS_DIR = REPO_ROOT / "domains"
MANIFEST_NAME = "domain_manifest.yaml"


class DomainConfigError(ValueError):
    """A domain folder or manifest is malformed or inconsistent with its code."""


@dataclass(frozen=True)
class PolicyRuleDoc:
    name: str
    on_violation: str
    description: str


@dataclass(frozen=True)
class DomainSpec:
    name: str
    title: str
    description: str
    root: Path
    thresholds: Mapping[str, float]
    policy_rules: tuple[PolicyRuleDoc, ...]
    # Prompt text telling the model this domain's action names, parameter
    # keys and the evidence its rules read (see PolicyEngine.guidance).
    decision_guide: str = ""
    entity_binding: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    permit_tools: tuple[str, ...] = field(default_factory=tuple)

    @property
    def server_script(self) -> Path:
        return self.root / "server.py"

    @property
    def benchmark_path(self) -> Path:
        return self.root / "benchmark.json"

    @property
    def server_modules(self) -> dict[str, str | Path]:
        """The `server_modules` argument for build_single_agent_runtime."""
        return {self.name: self.server_script}

    def create_policy_engine(self) -> PolicyEngine:
        """A fresh engine enforcing this domain's rules and thresholds."""
        engine = _policy_module(self.name, self.root / "policies.py").create_policy_engine(self.thresholds)
        engine.guidance = self.decision_guide
        engine.entity_binding = self.entity_binding
        engine.permit_tools = frozenset(self.permit_tools)
        documented = {rule.name for rule in self.policy_rules}
        implemented = set(engine.rule_names)
        if documented != implemented:
            raise DomainConfigError(
                f"Domain '{self.name}': manifest documents rules {sorted(documented)} but policies.py implements "
                f"{sorted(implemented)}."
            )
        return engine


@cache
def _policy_module(domain_name: str, path: Path) -> ModuleType:
    # A private module name, so this never collides with `domains.<name>.policies`
    # if a test imports the same file as a package module.
    spec = importlib.util.spec_from_file_location(f"_agent_runtime_domain_policies.{domain_name}", path)
    if spec is None or spec.loader is None:
        raise DomainConfigError(f"Cannot load policy module {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attribute in ("REQUIRED_THRESHOLDS", "create_policy_engine"):
        if not hasattr(module, attribute):
            raise DomainConfigError(f"{path} must define {attribute}.")
    return module


def list_domains() -> list[str]:
    """Names of every folder under domains/ that has a manifest."""
    if not DOMAINS_DIR.is_dir():
        return []
    return sorted(path.parent.name for path in DOMAINS_DIR.glob(f"*/{MANIFEST_NAME}"))


def _require_text(manifest: Mapping, key: str, manifest_path: Path) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DomainConfigError(f"{manifest_path}: '{key}' must be a non-empty string.")
    return value.strip()


def load_domain(name: str) -> DomainSpec:
    """Load and validate one domain. Raises DomainConfigError on any mismatch."""
    root = DOMAINS_DIR / name
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise DomainConfigError(f"Unknown domain '{name}'. Available: {list_domains()}")
    for required_file in ("server.py", "policies.py"):
        if not (root / required_file).is_file():
            raise DomainConfigError(f"Domain '{name}' is missing {required_file}.")

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise DomainConfigError(f"{manifest_path} must contain a YAML mapping.")
    if manifest.get("name") != name:
        raise DomainConfigError(f"{manifest_path}: name '{manifest.get('name')}' does not match folder '{name}'.")

    raw_thresholds = manifest.get("thresholds") or {}
    if not isinstance(raw_thresholds, Mapping):
        raise DomainConfigError(f"{manifest_path}: 'thresholds' must be a mapping.")
    thresholds: dict[str, float] = {}
    for key, value in raw_thresholds.items():
        # bool is an int subclass; `true` is not a threshold.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DomainConfigError(f"{manifest_path}: threshold '{key}' must be a number, got {value!r}.")
        thresholds[str(key)] = float(value)

    required = set(_policy_module(name, root / "policies.py").REQUIRED_THRESHOLDS)
    missing, unknown = required - thresholds.keys(), thresholds.keys() - required
    if missing or unknown:
        # Unknown keys are rejected too: a misspelt threshold would otherwise
        # be ignored while the reader of the manifest believes it applies.
        raise DomainConfigError(
            f"{manifest_path}: missing thresholds {sorted(missing)}, unknown thresholds {sorted(unknown)}."
        )

    rules = []
    for entry in manifest.get("policy_rules") or []:
        if not isinstance(entry, Mapping) or not entry.get("name"):
            raise DomainConfigError(f"{manifest_path}: every policy_rules entry needs a name.")
        rules.append(
            PolicyRuleDoc(
                name=str(entry["name"]),
                on_violation=str(entry.get("on_violation", "")),
                description=str(entry.get("description", "")),
            )
        )

    decision_guide = manifest.get("decision_guide") or ""
    if not isinstance(decision_guide, str):
        raise DomainConfigError(f"{manifest_path}: 'decision_guide' must be a string.")

    raw_entity_binding = manifest.get("entity_binding") or {}
    if not isinstance(raw_entity_binding, Mapping):
        raise DomainConfigError(f"{manifest_path}: 'entity_binding' must be a mapping.")

    raw_permit_tools = manifest.get("permit_tools") or []
    if not isinstance(raw_permit_tools, (list, tuple)):
        raise DomainConfigError(f"{manifest_path}: 'permit_tools' must be a list.")

    return DomainSpec(
        name=name,
        title=_require_text(manifest, "title", manifest_path),
        description=_require_text(manifest, "description", manifest_path),
        root=root,
        thresholds=MappingProxyType(thresholds),
        policy_rules=tuple(rules),
        decision_guide=decision_guide.strip(),
        entity_binding=MappingProxyType(dict(raw_entity_binding)),
        permit_tools=tuple(str(t) for t in raw_permit_tools),
    )
