#!/bin/bash
# Push the ACDC RGB archive to csc3 and extract it. Run when the browser
# download finishes: ~/Downloads/rgb_anon_trainvaltest.zip
set -eu
Z="${1:-$HOME/Downloads/rgb_anon_trainvaltest.zip}"
[ -f "$Z" ] || { echo "not found: $Z"; exit 1; }
echo "size: $(du -h "$Z" | cut -f1)"
scp -q "$Z" csc3:/users/gnakamot/wato/data/
ssh csc3 'cd /users/gnakamot/wato/data/acdc && python3 -c "
import zipfile; zipfile.ZipFile(\"/users/gnakamot/wato/data/rgb_anon_trainvaltest.zip\").extractall(\".\"); print(\"extracted\")
" && for c in fog night rain snow; do echo -n "$c/train: "; ls rgb_anon/$c/train/*/*_rgb_anon.png 2>/dev/null | wc -l; done'
echo "done — task 32-train-acdc is queued on csc3 and will pick up automatically"
