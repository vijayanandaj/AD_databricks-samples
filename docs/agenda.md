# Three-Day Program Agenda

This is a hands-on, code-backed enablement for Databricks, Delta Lake, and the Medallion (Bronze/Silver/Gold) pattern. Sessions map directly to runnable notebooks in this repo.

**Audience:** Data/Analytics Engineers, Platform Engineers  
**Format:** 3 days, ~6–7 hours/day including breaks  
**Prereqs:** Basic SQL/Python; Databricks workspace access (or run locally for smoke tests)  
**Artifacts:** All notebooks + lab outputs stay in your fork/template copy

---

## Day 1 — Foundations & Why it Matters
**Themes:** Parquet fundamentals, the need for Medallion, Delta basics

- **Kickoff & goals**: outcomes, success criteria, repo tour
- **Parquet fundamentals**: columnar layout, row groups, statistics, predicate pushdown
- **Why Medallion Architecture?**: separating concerns, governance, reproducibility
- **Delta Lake basics**: transaction log, atomicity, schema-on-write vs. read, checkpoints

**Labs (intro & ingestion):**
- Basic table creation & checkpoints — `basic-deltatable-creation/Basic_delta_plus_Checkpoint.ipynb`
- Metadata & table properties tour — `basic-deltatable-creation/Metadata_mgmt.ipynb`
- Bronze ingestion starter — `medallion/01-covid-ingestion.ipynb`  
  _Optional:_ first Bronze step in layered demo — `layer-demo/Telemetry_0/01_Bronze.ipynb`

**Outcome:** Participants can articulate _why_ Medallion + Delta, and set up a clean starting point.

---

## Day 2 — Delta Deep Dive + Bronze→Silver
**Themes:** ACID, time travel, schema evolution/merge, CDF, transforming to Silver

- **ACID & safe mutation** (upserts, deletes, constraints)  
- **Time Travel & rollback**: auditing and point-in-time reads  
- **Schema change**: evolution vs. enforcement; `MERGE` with type changes  
- **Change Data Feed (CDF)**: incremental downstream processing  
- **Transforming to Silver**: cleansing/conformance from Bronze inputs

**Labs (Delta core + Silver):**
- Delta ACID operations — `labs/01-delta-acid.ipynb`
- Schema evolution — `labs/03-schema-evolution.ipynb`
- Time travel — `labs/04-time-travel.ipynb`
- Schema merge (types/columns) — `labs/05-schema-merge.ipynb`
- MERGE INTO patterns — `labs/09-merge-into.ipynb`
- Change Data Feed sample — `labs/08-cdf-sample.ipynb`
- Bronze→Silver pass — `medallion/02-e2e-events.ipynb`  
  _Optional quality pass:_ `data-quality/Data_Quality.ipynb`

**Outcome:** Participants can safely change data, recover, and promote curated Silver data.

---

## Day 3 — Performance, Optimizations & Layered Exercises
**Themes:** OPTIMIZE/Z-ORDER, indexing concepts (incl. bloom filters where supported), streaming & telemetry, end-to-end practice

- **Performance toolkit**: file sizing, OPTIMIZE, Z-ORDER; data skipping concepts  
  _Bloom filters_: conceptual/feature overview and demo where available
- **Streaming mini-lab**: unifying batch & stream, checkpoints, idempotence
- **Audit & lineage**: understanding commits and change history
- **Layered exercises**: apply Bronze→Silver→Gold, telemetry & DLT pipeline

**Labs (perf + end-to-end):**
- OPTIMIZE & Z-ORDER — `labs/06-zorder.ipynb`
- Audit & lineage — `labs/07-audit-lineage.ipynb`
- Unified batch/stream (mini-lab) — `labs/02-unified-batch-stream.ipynb`
- DLT/automation sample — `automation/DLT.ipynb`
- Telemetry & DLT example — `layer-demo/Telemetry_DLT_Example.ipynb`
- Layered exercises — `layer-demo/Exercise2/E2E_exercise.ipynb` and the `layer-demo/Telemetry_0` / `Telemetry_1` sets

**Outcome:** Participants can reason about performance, validate changes, and run an end-to-end layered pipeline.

---

## Success Metrics (you can track post-training)
- Time to first successful **Bronze→Silver→Gold** run
- % of pipelines with **schema evolution** and **rollback** tested
- Read latency improvement after **OPTIMIZE/Z-ORDER**
- Number of jobs moved under **governed workflows**

> **Note**: All datasets are synthetic. Do not upload client-confidential data to this repo.
