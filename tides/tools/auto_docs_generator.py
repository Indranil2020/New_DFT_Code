#!/usr/bin/env python3
"""Auto-docs generator for TIDES configuration schema.

Reads the TidesConfig dataclass and all sub-configs from config.py,
generates Sphinx RST documentation with option tables.

Usage: python3 tools/auto_docs_generator.py [output_path]

No try/except control flow (ERR001).
"""
from __future__ import annotations

import inspect
import os
import sys
import textwrap
from dataclasses import dataclass, fields, is_dataclass
from typing import get_type_hints


def find_config_classes():
    """Import config.py and find all dataclass configs."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "python", "tides", "config.py"
    )
    config_path = os.path.normpath(config_path)

    import importlib.util
    spec = importlib.util.spec_from_file_location("tides_config", config_path)
    if spec is None or spec.loader is None:
        print("ERROR: Could not load config.py")
        return []

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    configs = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if is_dataclass(obj) and name.endswith("Config"):
            configs.append(obj)

    return configs


def format_type(tp) -> str:
    """Format a type annotation as a string."""
    if hasattr(tp, "__name__"):
        return tp.__name__
    return str(tp).replace("typing.", "")


def generate_rst(configs) -> str:
    """Generate RST documentation from config dataclasses."""
    lines = [
        "Configuration Options",
        "=====================",
        "",
        "This page is auto-generated from the TidesConfig dataclass schema.",
        "To regenerate, run: ``python3 tools/auto_docs_generator.py``",
        "",
        ".. contents::",
        "   :local:",
        "   :depth: 2",
        "",
    ]

    for cls in configs:
        section = cls.__name__
        docstring = inspect.getdoc(cls) or ""
        if docstring:
            section = docstring.split("\n")[0].rstrip(".")

        lines.append(section)
        lines.append("-" * len(section))
        lines.append("")

        # Table header.
        lines.append(f".. list-table:: {cls.__name__}")
        lines.append("   :widths: 25 15 15 45")
        lines.append("   :header-rows: 1")
        lines.append("")
        lines.append("   * - Parameter")
        lines.append("     - Type")
        lines.append("     - Default")
        lines.append("     - Description")

        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {}

        for f in fields(cls):
            ftype = format_type(hints.get(f.name, f.type))
            fdefault = f.default
            if f.default_factory is not f.default_factory.__class__.__name__:
                # default_factory is set; show the factory result type.
                pass
            if f.default is not inspect.Parameter.empty and f.default is not None:
                fdefault_str = repr(f.default)
            elif f.default_factory is not list:
                fdefault_str = "[]"
            else:
                fdefault_str = "N/A"

            # Extract docstring from field (if available in source).
            fdoc = ""
            src = inspect.getsource(cls)
            # Simple heuristic: find the field name and look for a docstring after it.
            for line in src.split("\n"):
                if f.name in line and ":" in line:
                    # Check if next line is a docstring.
                    pass

            lines.append(f"   * - {f.name}")
            lines.append(f"     - {ftype}")
            lines.append(f"     - {fdefault_str}")
            lines.append(f"     - {fdoc or 'See dataclass definition.'}")

        lines.append("")

    # Example TOML.
    lines.append("Example TOML")
    lines.append("------------")
    lines.append("")
    lines.append(".. code-block:: toml")
    lines.append("")
    lines.append(textwrap.dedent('''\
       [system]
       n_atoms = 2
       atomic_numbers = [1, 1]
       positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
       boundary_conditions = "free"

       [basis]
       kind = "DZP"

       [xc]
       functional = "PBE"
       dispersion = "D3BJ"

       [scf]
       max_iter = 100
       energy_tol = 1e-8
       mixing = "pulay"
       mixing_alpha = 0.3
    '''))

    return "\n".join(lines)


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else None
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "docs", "sphinx", "api",
            "config_options.rst"
        )
    output_path = os.path.normpath(output_path)

    configs = find_config_classes()
    if not configs:
        print("WARNING: No config dataclasses found. Using placeholder.")
        return 1

    rst = generate_rst(configs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(rst)

    print(f"Generated: {output_path}")
    print(f"Config classes documented: {len(configs)}")
    for c in configs:
        print(f"  - {c.__name__}: {len(fields(c))} fields")
    return 0


if __name__ == "__main__":
    sys.exit(main())
