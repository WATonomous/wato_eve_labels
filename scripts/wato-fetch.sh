#!/bin/bash
# On the laptop: pull the status channel + newest eval tables from csc3.
set -u
D="${1:-$HOME/wato/agent_mirror}"
mkdir -p "$D"
scp -q csc3:/users/gnakamot/wato/agent/STATUS.md "$D/STATUS.md" 2>/dev/null
scp -q csc3:/users/gnakamot/wato/agent/HANDOFF.md "$D/HANDOFF.md" 2>/dev/null
scp -q csc3:'/users/gnakamot/wato/ablate*.json' "$D/" 2>/dev/null
scp -q csc3:'/users/gnakamot/wato/agent/outbox/*' "$D/" 2>/dev/null
scp -q csc3:'/users/gnakamot/wato/repo/wato-bevfusion/configs/dualhead/*.yaml' "$D/" 2>/dev/null
cat "$D/STATUS.md" 2>/dev/null || echo "no STATUS.md yet (start the agent on csc3)"
