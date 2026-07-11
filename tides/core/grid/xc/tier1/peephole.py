#!/usr/bin/env python3
"""T-X4.2: Peephole optimization pass for libxc maple2c generated C sources.

Replaces pow(x, p/q) calls with cbrt/sqrt chains for rational exponents,
and precomputes pow(constant, rational) as compile-time constants.

This is a build-step tool, not a committed output. Run it on a copy of the
maple2c source before compiling with nvcc.

Transforms:
  pow(x, 1.0/3.0)  -> cbrt(x)           # POW_1_3
  pow(x, 2.0/3.0)  -> cbrt(x)*cbrt(x)   # POW_2_3
  pow(x, 4.0/3.0)  -> x*cbrt(x)         # POW_4_3
  pow(x, 5.0/3.0)  -> x*cbrt(x)*cbrt(x) # POW_5_3
  pow(x, 7.0/3.0)  -> x*x*cbrt(x)       # POW_7_3
  pow(x, 1.0/2.0)  -> sqrt(x)           # POW_1_2
  pow(x, 3.0/2.0)  -> x*sqrt(x)         # POW_3_2
  pow(x, 1.0/4.0)  -> sqrt(sqrt(x))     # POW_1_4
  pow(C, p/q)      -> <precomputed>     # constant base

Usage:
  python3 peephole.py input.c > output.c
  python3 peephole.py --inplace file1.c file2.c ...
"""

import re
import sys
import math
from pathlib import Path

# Patterns for pow(x, rational) -> cbrt/sqrt chains
# Match pow(var, numeric_fraction) where the fraction is a known rational
POW_PATTERNS = [
    # (regex_pattern, replacement_template, description)
    (r'pow\(([^,]+),\s*0\.1e1\s*/\s*0\.3e1\)', r'cbrt(\1)', 'pow(x,1/3)->cbrt'),
    (r'pow\(([^,]+),\s*0\.2e1\s*/\s*0\.3e1\)', r'(cbrt(\1)*cbrt(\1))', 'pow(x,2/3)->cbrt^2'),
    (r'pow\(([^,]+),\s*0\.4e1\s*/\s*0\.3e1\)', r'(\1*cbrt(\1))', 'pow(x,4/3)->x*cbrt'),
    (r'pow\(([^,]+),\s*0\.5e1\s*/\s*0\.3e1\)', r'(\1*cbrt(\1)*cbrt(\1))', 'pow(x,5/3)->x*cbrt^2'),
    (r'pow\(([^,]+),\s*0\.7e1\s*/\s*0\.3e1\)', r'(\1*\1*cbrt(\1))', 'pow(x,7/3)->x^2*cbrt'),
    (r'pow\(([^,]+),\s*0\.1e1\s*/\s*0\.2e1\)', r'sqrt(\1)', 'pow(x,1/2)->sqrt'),
    (r'pow\(([^,]+),\s*0\.3e1\s*/\s*0\.2e1\)', r'(\1*sqrt(\1))', 'pow(x,3/2)->x*sqrt'),
    (r'pow\(([^,]+),\s*0\.1e1\s*/\s*0\.4e1\)', r'sqrt(sqrt(\1))', 'pow(x,1/4)->sqrt(sqrt)'),
]

# Also match pow(constant, rational) -> precomputed value
CONST_POW_RE = re.compile(
    r'pow\(([0-9]+\.?[0-9]*e[+-]?[0-9]+|[0-9]+\.[0-9]+|[0-9]+),\s*'
    r'(0\.[0-9]+e[+-]?[0-9]+|[0-9]+\.[0-9]+)\s*/\s*'
    r'(0\.[0-9]+e[+-]?[0-9]+|[0-9]+\.[0-9]+|[0-9]+)\)'
)


def precompute_const_pow(base_str: str, num_str: str, den_str: str) -> str:
    """Precompute pow(base, num/den) as a decimal literal."""
    base = float(base_str)
    num = float(num_str)
    den = float(den_str)
    result = base ** (num / den)
    # Format with enough precision
    return f"{result:.16e}"


def transform(source: str) -> str:
    """Apply peephole transformations to source text."""
    out = source
    for pattern, replacement, desc in POW_PATTERNS:
        regex = re.compile(pattern)
        count = len(regex.findall(out))
        if count > 0:
            out = regex.sub(replacement, out)
    # Precompute constant-base pow calls
    def const_replacer(m):
        return precompute_const_pow(m.group(1), m.group(2), m.group(3))
    out = CONST_POW_RE.sub(const_replacer, out)
    return out


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    if args[0] == "--inplace":
        for path in args[1:]:
            src = Path(path).read_text()
            dst = transform(src)
            Path(path).write_text(dst)
            print(f"peephole: transformed {path}", file=sys.stderr)
    else:
        src = Path(args[0]).read_text()
        sys.stdout.write(transform(src))


if __name__ == "__main__":
    main()
