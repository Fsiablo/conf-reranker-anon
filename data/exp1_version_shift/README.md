# Tool-Version Shift Protocol

This directory intentionally contains the **protocol description only**, not the full runnable construction pipeline.

The TCAD manuscript constructs the version-shift split by:

1. extracting documented Tcl commands and argument signatures from a fixed current OpenROAD documentation snapshot and from tag `v2.0`;
2. diffing the two inventories to identify removed, renamed, or signature-changed commands;
3. labeling ORD-QA corpus passages as `current`, `stale`, or `distractor`;
4. injecting a controlled 30% stale-passage subset into each first-stage BM25 candidate pool;
5. evaluating nDCG@5, Stale@5, false-admission rate, and mean admitted context size.

The released repository does not include a one-command executable version of this pipeline, raw OpenROAD snapshots, raw ORD-QA data, or generated intermediate artifacts. This keeps the public companion package lightweight while documenting the construction logic needed to audit the paper.

Expected generated artifact names in the paper:

- `command_inventory_master.json`
- `command_inventory_v2.json`
- `stale_commands.json`
- `passages_labeled.jsonl`
- `version_shift_test.jsonl`
