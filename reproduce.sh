#!/bin/sh
# Vocabulary 6,000, depth 2 row of SHARE.md (each arm at its best embedding rate). About 10 s per run on an idle Apple-silicon laptop.
# Expected known-word loss: plain 5.7460, Engram-style 5.7041, per-word value rows 5.6819.
set -e
cd "$(dirname "$0")"
D="--dev mps --threads 1 --V 6000 --width 16 --layers 2 --steps 150 --batch 128"
nice -n 20 python3 pw.py $D --elr 0.1 --tag repro-plain
nice -n 20 python3 pw.py $D --elr 0.2 --ngram 6000 --nglayer 0 --tlr 0.03 --tag repro-engram
nice -n 20 python3 pw.py $D --elr 0.1 --ve 0,1 --tlr 0.1 --tag repro-word
