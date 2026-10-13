# Decisions

1. **Dataset and license.** Anthropic/hh-rlhf, `helpful-base/train`. The Hugging Face dataset card lists the MIT license (checked via the HF API on 2026-10-13).
2. **Disturbing content.** A deliberately over-inclusive keyword blocklist (violence, sex, drugs, self-harm, slurs) drops any pair where either side matches. This also drops some harmless items; that is accepted. Items over 1000 characters are dropped so they are quick to read.
3. **What gets rated.** The "chosen" side of each pair only: the conversation so far plus the final assistant reply. The annotator rates the helpfulness of that final reply from 1 to 5. The "rejected" side is not used.
4. **Classifier target.** Binary: rating >= 4 is "helpful". With a few hundred labels, five-way prediction is too sparse to learn or compare strategies on. Raw 1-5 ratings stay in MongoDB.
5. **Active learning protocol.** Replay on the human-labeled pool. A fixed test split is held out per repeat, the same split is used for both strategies, and results are averaged over 20 seeded repeats. Seed set is 10 items with both classes guaranteed, batch size 5. Uncertainty sampling picks items whose predicted probability is closest to 0.5.
6. **Second annotator.** `--second` mode only serves a deterministic 40-item subset of items the primary annotator already rated, and hides the first rating. Kappa is reported unweighted and quadratic weighted (ratings are ordinal).
7. **Schema.** Two collections: `items` (one per conversation) and `annotations` (one per item and annotator, unique index). Edits push the old rating and rationale into a `revisions` array. Model predictions can be added to items later as extra fields without a migration.
8. **Docker.** `docker compose up` starts MongoDB 7 and a one-shot loader. The labeling CLI is interactive, so it runs from the host virtual environment against the exposed port 27017.
9. **Tests.** Synthetic text appears only in tests and is labeled SYNTHETIC in the code. Reported results come only from real labels.
