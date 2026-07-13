#!/usr/bin/env python3
"""TIDES auto-docs generator — introspects the config schema and produces RST.

T10.4: Every key is documented or the docs build fails.

Usage:
    python3 tides/tools/auto_docs_generator.py [--output PATH]

Reads tides/api/python/tides/config.py, extracts all dataclass config
sections, and generates a reStructuredText file with option tables.

No try/except (ERR001): uses if/else checks for robustness.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import textwrap
from dataclasses import fields, is_dataclass
from typing import get_type_hints, get_origin, get_args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_config_module():
    """Import the tides.config module from the API source tree."""
    api_root = os.path.join(
        os.path.dirname(__file__), "..", "api", "python"
    )
    api_root = os.path.abspath(api_root)
    if api_root not in sys.path:
        sys.path.insert(0, api_root)

    # Check that the module is importable before importing.
    spec = importlib.util.find_spec("tides.config")
    if spec is None:
        print(
            "ERROR: cannot find tides.config. "
            "Ensure tides/api/python is on the path.",
            file=sys.stderr,
        )
        sys.exit(1)
    return importlib.import_module("tides.config")


def _format_type(f) -> str:
    """Format a dataclass field type for display."""
    ftype = f.type
    # Handle string-form types (from __future__ annotations).
    if isinstance(ftype, str):
        return ftype
    # Handle typing generics (list[int], Optional[str], etc.).
    origin = get_origin(ftype)
    if origin is not None:
        args = get_args(ftype)
        arg_strs = [getattr(a, "__name__", str(a)) for a in args]
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{', '.join(arg_strs)}]" if arg_strs else origin_name
    return getattr(ftype, "__name__", str(ftype))


def _format_default(f) -> str:
    """Format a dataclass field default value for display."""
    from dataclasses import MISSING

    # Handle fields that use default_factory (e.g. field(default_factory=list)).
    if f.default_factory is not MISSING:
        factory = f.default_factory
        if callable(factory) and hasattr(factory, "__name__"):
            name = factory.__name__
            if name == "list":
                return "[]"
            elif name == "dict":
                return "{}"
            return f"{name}()"
        return str(factory)

    default = f.default
    if default is MISSING:
        return "(required)"
    if callable(default) and hasattr(default, "__name__"):
        return default.__name__
    if isinstance(default, str):
        return repr(default)
    if isinstance(default, (list, dict)):
        return repr(default)
    return str(default)


def _extract_field_doc(cls, field_name) -> str:
    """Extract a per-field docstring from the class source.

    Dataclass field docstrings appear as string literals immediately
    after the field assignment in the class body.
    """
    source = inspect.getsource(cls)
    lines = source.split("\n")
    in_field = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(field_name + ":") or stripped.startswith(
            field_name + " :"
        ):
            in_field = True
            continue
        if in_field:
            if stripped == "":
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                doc = stripped.strip('"\'')
                if doc:
                    return doc
                # Multi-line docstring — gather continuation.
                if i + 1 < len(lines):
                    return lines[i + 1].strip().strip('"\'')
            else:
                break
    return ""


def _section_rst(section_name: str, cls) -> list[str]:
    """Generate RST for one config section."""
    lines = []
    lines.append(f"[{section_name}]")
    lines.append("-" * (len(section_name) + 2))
    lines.append("")
    cls_doc = inspect.getdoc(cls) or ""
    if cls_doc:
        lines.append(cls_doc)
        lines.append("")

    lines.append(".. list-table::")
    lines.append("   :header-rows: 1")
    lines.append("   :widths: 20 20 15 45")
    lines.append("")
    lines.append("   * - Key")
    lines.append("     - Type")
    lines.append("     - Default")
    lines.append("     - Description")

    for f in fields(cls):
        ftype = _format_type(f)
        fdefault = _format_default(f)
        fdoc = _extract_field_doc(cls, f.name)
        lines.append(f"   * - {f.name}")
        lines.append(f"     - {ftype}")
        lines.append(f"     - {fdefault}")
        lines.append(f"     - {fdoc}")

    lines.append("")
    return lines


def _example_toml(section_name: str, cls) -> list[str]:
    """Generate an example TOML snippet for a section."""
    lines = []
    lines.append(f"Example TOML::")
    lines.append("")
    lines.append(f"   [{section_name}]")
    for f in fields(cls):
        fdefault = _format_default(f)
        if fdefault in ("[]", "list"):
            continue  # skip empty lists
        if fdefault.startswith("'") or fdefault.startswith('"'):
            lines.append(f"   {f.name} = {fdefault}")
        else:
            lines.append(f"   {f.name} = {fdefault}")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(output_path: str) -> str:
    """Generate the RST documentation file."""
    config_mod = _import_config_module()

    # Collect all dataclass config sections.
    sections = []
    for name in dir(config_mod):
        obj = getattr(config_mod, name)
        if is_dataclass(obj) and name.endswith("Config"):
            sections.append((name.replace("Config", "").lower(), obj))

    # Sort by name for deterministic output.
    sections.sort(key=lambda s: s[0])

    lines = []
    lines.append("TIDES Configuration Options (Auto-Generated)")
    lines.append("=" * 48)
    lines.append("")
    lines.append(
        "This page is auto-generated from the TIDES dataclass schema. "
        "To regenerate, run ``python3 tides/tools/auto_docs_generator.py``."
    )
    lines.append("")

    for section_name, cls in sections:
        lines.extend(_section_rst(section_name, cls))
        lines.extend(_example_toml(section_name, cls))
        lines.append("")

    content = "\n".join(lines)

    # Write to file.
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write(content)

    return content


def main():
    parser = argparse.ArgumentParser(
        description="Generate TIDES config documentation (RST)."
    )
    default_output = os.path.join(
        os.path.dirname(__file__),
        "..",
        "docs",
        "sphinx",
        "api",
        "config_options_generated.rst",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.abspath(default_output),
        help="Output RST file path (default: %(default)s)",
    )
    args = parser.parse_args()

    content = generate(args.output)
    print(f"Generated {len(content.splitlines())} lines → {args.output}")


if __name__ == "__main__":
    main()
