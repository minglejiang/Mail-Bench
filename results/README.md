# Published records

Small records only, never data or weights. Three kinds of file live here:

- `*_receipt.json`: the verification receipt of a pinned reference checkpoint. Its `receipt` block carries the
  repository the weights come from, the revision the receipt was taken at (the checkpoint path within the repository
  for pi0.5, the git revision for GR00T N1.5), the critical files with their SHA-256 digests, and the full-inventory
  digest (`inventory_sha256`) a run is checked against; the roster pins the same digest.
  The per-file `size`, `inode` and `mtime_ns` entries describe one local copy and are not portable; the digests are.
  Raw run certificates record local paths and endpoints and are summarised, never published, in the score records.
- `*_healthy_report.json`: the healthy phase of a reference run: scenes, solved scenes and the median completion step
  per task, and the completion-step distribution over the suite.
- `*_score.json`: the score record the scorer produced for a reference run: per-task, per-condition counts and scores,
  the measurement identity, the certificate summary, the onset check and the pre-fault trajectory audit. The README's
  results table is generated from these files by `scripts/paper_results_table.py`.
