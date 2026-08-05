"""
domain_registry.py — the single lookup point the rest of llm_extract uses
to get a domain's extraction schema, prompt, and target-builder.

Domains are defined entirely in YAML (domains/<name>.yaml) + an editable
prompt text file (referenced from the YAML). Adding a new catalysis topic —
methanol synthesis, MOF defect engineering, whatever's next — means adding
one YAML file and one prompt file. No Python changes anywhere in this
package are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel

from .domain_spec import (
    DomainSpec,
    build_binary_targets_generic,
    build_pydantic_model,
    field_docs,
    load_domain_spec,
    load_prompt_template,
)


@dataclass
class SchemaBundle:
    domain_name: str
    spec: DomainSpec
    ExtractionFields: type[BaseModel]
    PROMPT_TEMPLATE: str
    FIELD_DOCS: list[tuple[str, str, str]]
    build_binary_targets: Callable[[pd.DataFrame], pd.DataFrame]

    @property
    def multi_experiment(self) -> bool:
        return self.spec.multi_experiment

    @property
    def extract_claims(self) -> bool:
        return self.spec.extract_claims

    @property
    def ExperimentFields(self) -> type[BaseModel] | None:
        """Only set when multi_experiment is true; the per-experiment model
        nested inside ExtractionFields.experiments. None for flat domains."""
        return getattr(self.ExtractionFields, "__experiment_model__", None)


def get_schema(name: str, domains_dir: Path) -> SchemaBundle:
    """Load domains_dir/<name>.yaml and everything it points to."""
    yaml_path = Path(domains_dir) / f"{name}.yaml"
    if not yaml_path.exists():
        available = sorted(p.stem for p in Path(domains_dir).glob("*.yaml")) if Path(domains_dir).is_dir() else []
        raise ValueError(
            f"No domain spec found at {yaml_path}. Available domains in "
            f"{domains_dir}: {available or '(none found)'}. "
            "Add a new one by creating <domains_dir>/<name>.yaml — see "
            "domains/biodiesel.yaml for the format."
        )

    spec = load_domain_spec(yaml_path)
    extraction_fields = build_pydantic_model(spec)
    prompt_template = load_prompt_template(yaml_path, spec)

    def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
        return build_binary_targets_generic(df, spec)

    return SchemaBundle(
        domain_name=spec.domain_name,
        spec=spec,
        ExtractionFields=extraction_fields,
        PROMPT_TEMPLATE=prompt_template,
        FIELD_DOCS=field_docs(spec),
        build_binary_targets=_build_targets,
    )
