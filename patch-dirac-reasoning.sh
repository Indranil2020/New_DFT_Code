#!/usr/bin/env bash
# patch-dirac-reasoning.sh — Patches dirac CLI for GLM-5.2
#
# Run this after updating dirac-cli (npm update -g dirac-cli).
# Dirac CLI accepts: none|low|medium|high|xhigh
# GLM-5.2 API accepts: none|low|medium|high|max
# This patch does TWO things:
#   1. Maps xhigh→max (so dirac's xhigh becomes GLM-5.2's max)
#   2. Changes default from medium→xhigh (so all dirac calls get max reasoning)
#
# Usage: bash patch-dirac-reasoning.sh

set -euo pipefail

DIRAC_DIR="$(npm root -g 2>/dev/null)/dirac-cli"
if [[ ! -d "$DIRAC_DIR" ]]; then
  echo "Error: dirac-cli not found at $DIRAC_DIR"
  echo "Is dirac installed globally? Run: npm install -g dirac-cli"
  exit 1
fi

CLI_MJS="$DIRAC_DIR/dist/cli.mjs"
LIB_MJS="$DIRAC_DIR/dist/lib.mjs"

echo "Patching dirac CLI at: $DIRAC_DIR"
echo ""

# --- Patch 1: cli.mjs — function Pk ---
# Original:  function Pk(t){let e=(t||"medium").toLowerCase();return REe(e)?e:"medium"}
# Patched:   function Pk(t){let e=(t||"medium").toLowerCase();e=e==="xhigh"?"max":e;return REe(e)?e:"medium"}
# Also adds "max" to the ude validation array so REe("max") returns true.
if grep -q 'function Pk(t){let e=(t||"medium").toLowerCase();return REe(e)?e:"medium"}' "$CLI_MJS"; then
  sed -i 's/function Pk(t){let e=(t||"medium").toLowerCase();return REe(e)?e:"medium"}/function Pk(t){let e=(t||"medium").toLowerCase();e=e==="xhigh"?"max":e;return REe(e)?e:"medium"}/' "$CLI_MJS"
  echo "[OK] cli.mjs: Pk() patched (xhigh→max)"
elif grep -q 'function Pk(t){let e=(t||"medium").toLowerCase();e=e==="xhigh"?"max":e;return REe(e)?e:"medium"}' "$CLI_MJS"; then
  echo "[SKIP] cli.mjs: Pk() already patched"
else
  echo "[WARN] cli.mjs: Pk() pattern not found — dirac may have changed. Manual patch needed."
  echo "       Search for: function Pk(t){let e=(t||\"medium\").toLowerCase()"
  echo "       The validation function may be named differently (e.g. REe, TEe, etc.)"
fi

# Add "max" to ude array in cli.mjs (so REe validates "max" as valid)
if grep -q 'ude=\["none","low","medium","high","xhigh"\]' "$CLI_MJS"; then
  sed -i 's/ude=\["none","low","medium","high","xhigh"\]/ude=\["none","low","medium","high","xhigh","max"\]/' "$CLI_MJS"
  echo "[OK] cli.mjs: ude array patched (added \"max\")"
elif grep -q 'ude=\["none","low","medium","high","xhigh","max"\]' "$CLI_MJS"; then
  echo "[SKIP] cli.mjs: ude array already has \"max\""
else
  echo "[WARN] cli.mjs: ude array pattern not found — may have changed."
fi

# --- Patch 2: cli.mjs — direct reasoning_effort assignment ---
# Original:  reasoning_effort:this.options.reasoningEffort||"medium"
# Patched:   reasoning_effort:(this.options.reasoningEffort==="xhigh"?"max":this.options.reasoningEffort)||"max"
if grep -q 'reasoning_effort:this.options.reasoningEffort||"medium"' "$CLI_MJS"; then
  sed -i 's/reasoning_effort:this.options.reasoningEffort||"medium"/reasoning_effort:(this.options.reasoningEffort==="xhigh"?"max":this.options.reasoningEffort)||"max"/' "$CLI_MJS"
  echo "[OK] cli.mjs: direct reasoning_effort assignment patched (default→max)"
elif grep -q 'reasoning_effort:(this.options.reasoningEffort==="xhigh"?"max":this.options.reasoningEffort)||"max"' "$CLI_MJS"; then
  echo "[SKIP] cli.mjs: direct assignment already patched"
elif grep -q 'reasoning_effort:(this.options.reasoningEffort==="xhigh"?"max":this.options.reasoningEffort)||"medium"' "$CLI_MJS"; then
  # Old patch (fallback medium) — upgrade to max
  sed -i 's/reasoning_effort:(this.options.reasoningEffort==="xhigh"?"max":this.options.reasoningEffort)||"medium"/reasoning_effort:(this.options.reasoningEffort==="xhigh"?"max":this.options.reasoningEffort)||"max"/' "$CLI_MJS"
  echo "[OK] cli.mjs: direct assignment upgraded (fallback medium→max)"
else
  echo "[WARN] cli.mjs: direct assignment pattern not found — may have changed."
fi

# --- Patch 3: lib.mjs — function nP (was rP in older versions) ---
# Original:  function nP(t){let e=(t||"medium").toLowerCase();return qjt(e)?e:"medium"}
# Patched:   function nP(t){let e=(t||"medium").toLowerCase();e=e==="xhigh"?"max":e;return qjt(e)?e:"medium"}
# Also adds "max" to the Qmn validation array so qjt("max") returns true.
if grep -q 'function nP(t){let e=(t||"medium").toLowerCase();return qjt(e)?e:"medium"}' "$LIB_MJS"; then
  sed -i 's/function nP(t){let e=(t||"medium").toLowerCase();return qjt(e)?e:"medium"}/function nP(t){let e=(t||"medium").toLowerCase();e=e==="xhigh"?"max":e;return qjt(e)?e:"medium"}/' "$LIB_MJS"
  echo "[OK] lib.mjs: nP() patched (xhigh→max)"
elif grep -q 'function nP(t){let e=(t||"medium").toLowerCase();e=e==="xhigh"?"max":e;return qjt(e)?e:"medium"}' "$LIB_MJS"; then
  echo "[SKIP] lib.mjs: nP() already patched"
else
  echo "[WARN] lib.mjs: nP() pattern not found — dirac may have changed. Manual patch needed."
  echo "       Search for: function nP(t){let e=(t||\"medium\").toLowerCase()"
  echo "       The validation function may be named differently (e.g. qjt, Ljt, rP, etc.)"
fi

# Add "max" to Qmn array in lib.mjs (so qjt validates "max" as valid)
if grep -q 'Qmn=\["none","low","medium","high","xhigh"\]' "$LIB_MJS"; then
  sed -i 's/Qmn=\["none","low","medium","high","xhigh"\]/Qmn=\["none","low","medium","high","xhigh","max"\]/' "$LIB_MJS"
  echo "[OK] lib.mjs: Qmn array patched (added \"max\")"
elif grep -q 'Qmn=\["none","low","medium","high","xhigh","max"\]' "$LIB_MJS"; then
  echo "[SKIP] lib.mjs: Qmn array already has \"max\""
else
  echo "[WARN] lib.mjs: Qmn array pattern not found — may have changed."
fi

# --- Patch 4: lib.mjs — direct reasoning_effort assignment ---
if grep -q 'reasoning_effort:this.options.reasoningEffort||"medium"' "$LIB_MJS"; then
  sed -i 's/reasoning_effort:this.options.reasoningEffort||"medium"/reasoning_effort:(this.options.reasoningEffort==="xhigh"?"max":this.options.reasoningEffort)||"max"/' "$LIB_MJS"
  echo "[OK] lib.mjs: direct reasoning_effort assignment patched (default→max)"
elif grep -q 'reasoning_effort:(this.options.reasoningEffort==="xhigh"?"max":this.options.reasoningEffort)||"max"' "$LIB_MJS"; then
  echo "[SKIP] lib.mjs: direct assignment already patched"
elif grep -q 'reasoning_effort:(this.options.reasoningEffort==="xhigh"?"max":this.options.reasoningEffort)||"medium"' "$LIB_MJS"; then
  # Old patch (fallback medium) — upgrade to max
  sed -i 's/reasoning_effort:(this.options.reasoningEffort==="xhigh"?"max":this.options.reasoningEffort)||"medium"/reasoning_effort:(this.options.reasoningEffort==="xhigh"?"max":this.options.reasoningEffort)||"max"/' "$LIB_MJS"
  echo "[OK] lib.mjs: direct assignment upgraded (fallback medium→max)"
else
  echo "[WARN] lib.mjs: direct assignment pattern not found — may have changed."
fi

echo ""
echo "--- Subagent patches ---"

# --- Patch 5: Subagent timeout default 300→0 (no timeout) ---
# cli.mjs
if grep -q 'timeout:e.timeout?parseInt(String(e.timeout),10):300,maxTurns' "$CLI_MJS"; then
  sed -i 's/timeout:e.timeout?parseInt(String(e.timeout),10):300,maxTurns/timeout:e.timeout?parseInt(String(e.timeout),10):0,maxTurns/' "$CLI_MJS"
  echo "[OK] cli.mjs: subagent timeout 300→0 (no timeout)"
elif grep -q 'timeout:e.timeout?parseInt(String(e.timeout),10):0,maxTurns' "$CLI_MJS"; then
  echo "[SKIP] cli.mjs: subagent timeout already 0"
else
  echo "[WARN] cli.mjs: subagent timeout pattern not found"
fi
# lib.mjs
if grep -q 'timeout:e.timeout?parseInt(String(e.timeout),10):300,maxTurns' "$LIB_MJS"; then
  sed -i 's/timeout:e.timeout?parseInt(String(e.timeout),10):300,maxTurns/timeout:e.timeout?parseInt(String(e.timeout),10):0,maxTurns/' "$LIB_MJS"
  echo "[OK] lib.mjs: subagent timeout 300→0 (no timeout)"
elif grep -q 'timeout:e.timeout?parseInt(String(e.timeout),10):0,maxTurns' "$LIB_MJS"; then
  echo "[SKIP] lib.mjs: subagent timeout already 0"
else
  echo "[WARN] lib.mjs: subagent timeout pattern not found"
fi

# --- Patch 6: Subagent retry count 3→10 ---
# cli.mjs: BBm=3 → BBm=10
if grep -q 'BBm=3,Kna=3,PBm=2e3' "$CLI_MJS"; then
  sed -i 's/BBm=3,Kna=3,PBm=2e3/BBm=10,Kna=3,PBm=2e3/' "$CLI_MJS"
  echo "[OK] cli.mjs: subagent retry count 3→10"
elif grep -q 'BBm=10,Kna=3,PBm=2e3' "$CLI_MJS"; then
  echo "[SKIP] cli.mjs: subagent retry count already 10"
else
  echo "[WARN] cli.mjs: retry count pattern not found (variable may be renamed)"
fi
# lib.mjs: gEm=3 → gEm=10 (variable name differs in lib.mjs)
if grep -q 'gEm=3,rYi=3,_Em=2e3' "$LIB_MJS"; then
  sed -i 's/gEm=3,rYi=3,_Em=2e3/gEm=10,rYi=3,_Em=2e3/' "$LIB_MJS"
  echo "[OK] lib.mjs: subagent retry count 3→10"
elif grep -q 'gEm=10,rYi=3,_Em=2e3' "$LIB_MJS"; then
  echo "[SKIP] lib.mjs: subagent retry count already 10"
else
  echo "[WARN] lib.mjs: retry count pattern not found (variable may be renamed)"
  echo "       Search for: =3,=3,=2e3 near 'SubagentRunner' or 'setTimeout'"
fi

# --- Patch 7: Add prompt_6 through prompt_10 (5→10 subagents) ---
PROMPT_ADDITION='{name:"prompt_6",required:!1,instruction:"Optional sixth subagent prompt."},{name:"prompt_7",required:!1,instruction:"Optional seventh subagent prompt."},{name:"prompt_8",required:!1,instruction:"Optional eighth subagent prompt."},{name:"prompt_9",required:!1,instruction:"Optional ninth subagent prompt."},{name:"prompt_10",required:!1,instruction:"Optional tenth subagent prompt."}'
for f in "$CLI_MJS" "$LIB_MJS"; do
  fname=$(basename "$f")
  if grep -q '"prompt_10"' "$f"; then
    echo "[SKIP] $fname: prompt_6-10 already added"
  elif grep -q '{name:"prompt_5",required:!1,instruction:"Optional fifth subagent prompt."}' "$f"; then
    sed -i "s|{name:\"prompt_5\",required:!1,instruction:\"Optional fifth subagent prompt.\"}|{name:\"prompt_5\",required:!1,instruction:\"Optional fifth subagent prompt.\"},${PROMPT_ADDITION}|" "$f"
    echo "[OK] $fname: added prompt_6 through prompt_10"
  else
    echo "[WARN] $fname: prompt_5 pattern not found"
  fi
done

# --- Patch 8: Update prompt list parsing (5→10 prompts) ---
for f in "$CLI_MJS" "$LIB_MJS"; do
  fname=$(basename "$f")
  if grep -q '"prompt_1","prompt_2","prompt_3","prompt_4","prompt_5","prompt_6"' "$f"; then
    echo "[SKIP] $fname: prompt list already has 10 entries"
  elif grep -q '"prompt_1","prompt_2","prompt_3","prompt_4","prompt_5"' "$f"; then
    sed -i 's/"prompt_1","prompt_2","prompt_3","prompt_4","prompt_5"/"prompt_1","prompt_2","prompt_3","prompt_4","prompt_5","prompt_6","prompt_7","prompt_8","prompt_9","prompt_10"/g' "$f"
    echo "[OK] $fname: prompt list updated to 10"
  else
    echo "[WARN] $fname: prompt list pattern not found"
  fi
done

# --- Patch 9: Update timeout instruction text ---
for f in "$CLI_MJS" "$LIB_MJS"; do
  fname=$(basename "$f")
  if grep -q 'Defaults to 300 seconds' "$f"; then
    sed -i 's/Defaults to 300 seconds/Defaults to 0 (no timeout)/g' "$f"
    echo "[OK] $fname: timeout instruction text updated"
  elif grep -q 'Defaults to 0 (no timeout)' "$f"; then
    echo "[SKIP] $fname: timeout instruction already updated"
  else
    echo "[SKIP] $fname: timeout instruction text not found (may not exist)"
  fi
done

echo ""
echo "=== Patch complete ==="
echo "Test with: dirac -y \"Reply with exactly one word: hello, then stop.\""
echo ""
echo "If it works, you should see 'Task Completed' with no 400 errors."
echo "If it fails, check the [WARN] messages above for manual patch instructions."
