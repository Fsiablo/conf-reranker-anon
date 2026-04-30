# EDA-Centric Stress-Test Protocols

This repository documents the two EDA-centric stress tests used in the TCAD manuscript, but it intentionally does **not** ship a complete runnable reproduction pipeline. Raw third-party sources, generated intermediate artifacts, generator outputs, judge prompts, and one-shot build scripts are not included.

The goal of this directory is to make the paper's construction logic auditable without turning the public companion repository into a full benchmark release.

## 1. Tool-Version Shift / Stale-Document Admission

The version-shift split is constructed from public OpenROAD documentation snapshots and ORD-QA passages:

1. Fix two OpenROAD documentation snapshots: a current snapshot and tag `v2.0`.
2. Extract documented Tcl commands and argument signatures from both snapshots.
3. Mark commands as stale when they are removed, renamed, or have changed signatures.
4. Label ORD-QA passages as:
   - `current`: mentions only commands present in the current snapshot with matching signatures;
   - `stale`: mentions at least one stale command or originates from a removed document;
   - `distractor`: has no overlap with either command inventory.
5. For each query, assemble the first-stage candidate pool from the BM25 top-50 plus a controlled 30% stale-passage injection.
6. Evaluate nDCG@5, Stale@5, false-admission rate, and mean admitted context size.

Paper artifact names:

- `command_inventory_master.json`
- `command_inventory_v2.json`
- `stale_commands.json`
- `passages_labeled.jsonl`
- `version_shift_test.jsonl`

## 2. End-to-End EDA RAG Answer Quality

The end-to-end evaluation fixes the generator and prompt template, and varies only the context selector:

1. Extract a current OpenROAD command whitelist from the master documentation snapshot.
2. Build one query/context pack per context-selection method.
3. Generate answers with the fixed generator used in the paper.
4. Judge each answer for:
   - correctness;
   - unsupported claims relative to the retrieved context;
   - wrong commands not present in the whitelist;
   - Tcl/SDC script validity when a frontend is available.

Paper artifact names:

- `command_whitelist.json`
- `packs/<method>.jsonl`

## Released vs. withheld

Released here:

- protocol descriptions;
- expected artifact schemas and names;
- paper result tables in `data/paper_results.json` and `docs/results.md`.

Not released here:

- raw ORD-QA / HotpotQA / OpenROAD snapshots;
- one-shot reproduction scripts;
- generated intermediate artifacts;
- generator outputs and judge prompts;
- pretrained checkpoints.
