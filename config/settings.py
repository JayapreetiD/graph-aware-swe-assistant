"""
config/settings.py

Centralized path configuration for the Graph-Aware SWE Assistant.

Single responsibility: define, per target repository, where its
source lives on disk and where its derived artifacts (graph, vector
store) are written/read. Every other module imports from here instead
of hardcoding paths locally — this file is the single source of truth.

Switch active repo via the ACTIVE_REPO environment variable:
    export ACTIVE_REPO=click     # default
    export ACTIVE_REPO=django
"""

import os
from pathlib import Path

REPOS = {
    "click": {
        "repo_root": Path.home() / "repos" / "target-small" / "src" / "click",
        "graph_path": "data/graph/click/graph.gpickle",
        "vector_db_path": "data/vectors/click/qdrant_db",
    },
    "django": {
        "repo_root": Path.home() / "repos" / "target-medium",
        "graph_path": "data/graph/django/graph.gpickle",
        "vector_db_path": "data/vectors/django/qdrant_db",
    },
}

ACTIVE_REPO = os.environ.get("ACTIVE_REPO", "click")

if ACTIVE_REPO not in REPOS:
    raise ValueError(
        f"Unknown ACTIVE_REPO '{ACTIVE_REPO}'. Valid options: {list(REPOS.keys())}"
    )

_active = REPOS[ACTIVE_REPO]

REPO_ROOT = _active["repo_root"]
GRAPH_PATH = _active["graph_path"]
VECTOR_DB_PATH = _active["vector_db_path"]
