#!/usr/bin/env bash
# Download the relevant arXiv PDFs into refs/.  Existing files are skipped.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .

while read -r name id; do
    [ -z "${name}" ] && continue
    if [ -s "${name}.pdf" ]; then
        echo "[skip] ${name}.pdf"
        continue
    fi
    echo "[fetch] ${name} <- arXiv:${id}"
    curl -fL --retry 3 "https://arxiv.org/pdf/${id}" -o "${name}.pdf"
done <<'EOF'
engram 2601.07372
xgram 2604.21724
longcat 2601.21204
bov 2606.02780
mome 2609.15126
move 2601.22887
scone 2502.01637
over_tokenized 2501.16975
hash_layers 2106.04426
jtok 2602.00800
memory_attention 2609.28399
stem 2601.10639
adaptive_input 1809.10853
press_wolf 1608.05859
inan_tying 1611.01462
lopardo 2603.26663
leviathan 2601.22040
freq_imbalance 2508.15390
resformer 2410.17897
kaplan 2001.08361
chinchilla 2203.15556
adamw 1711.05101
hash_embeddings 1709.03933
feature_hashing 0902.2206
EOF

echo "Done. PDFs are in $(pwd)"
