# End-to-End EDA RAG Protocol

This directory intentionally contains the **evaluation protocol description only**, not the full runnable end-to-end generation and judging pipeline.

The TCAD manuscript evaluates context selectors by:

1. extracting a command whitelist from the current OpenROAD documentation snapshot;
2. packaging each ORD-QA test query with the context selected by each method;
3. using a fixed generator and prompt template so that only the retrieved context varies;
4. judging generated answers for correctness, unsupported claims, wrong commands, and Tcl/SDC script validity.

The public companion repository does not include generator outputs, judge prompts, raw ORD-QA files, or a one-command executable evaluation script. The result tables are mirrored in `../../docs/results.md` and `../paper_results.json` for review and consistency checking.

Expected generated artifact names in the paper:

- `command_whitelist.json`
- `packs/bm25_top5.jsonl`
- `packs/finetuned_top5.jsonl`
- `packs/tempscale_top5.jsonl`
- `packs/conf_reranker_topk.jsonl`
- `packs/conf_reranker_conformal.jsonl`
