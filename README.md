# Databricks Medallion & Delta Labs — Template-ready
## Executive Overview

> Learn more in the **[Program Overview](docs/overview.md)**.


This repository is a **code-backed training portfolio** for Databricks & Delta Lake. Every concept is paired with runnable notebooks, synthetic datasets, and CI checks — so the material is **demonstrably real** (not slideware).

**Audience**: Data engineers, analytics engineers, platform teams  
**Formats**: 90-min demo • half-day workshop • 1–3 day enablement • custom coaching  
**Prereqs**: Basic SQL/Python; Databricks workspace access (or run locally for smoke tests)

### What your team will be able to do
- Apply **Medallion (Bronze/Silver/Gold)** patterns with Delta Lake
- Use **time travel, schema evolution/merge, CDF** safely in production
- Optimize with **OPTIMIZE & Z-ORDER**, and reason about cost/perf trade-offs
- Build a small **streaming pipeline** and reason about batch vs stream
- Ship **governed jobs** (optionally via Databricks Asset Bundles)

### Why this is credible
- ✅ **Runnable** notebooks with synthetic data (no PHI/PII)
- ✅ **Reproducible**: requirements, CI smoke run, optional Codespaces
- ✅ **Template-ready**: teams can clone their own copy without touching the original

### Program options (typical)
| Option | Outcomes | Agenda (high level) |
|---|---|---|
| **90-min demo** | Awareness; see value quickly | Medallion tour, Delta ACID/time travel, quick perf tips |
| **1-day workshop** | Hands-on skills | Labs 01–06; mini streaming lab; Q&A |
| **3-day enablement** | Build confidence + patterns | All labs + layered exercises; job orchestration; design reviews |

### Success metrics you can track
- Time to first successful **Bronze→Silver→Gold** run
- % of pipelines with **schema evolution** and rollback tested
- Read latency improvement after **OPTIMIZE/Z-ORDER**
- Number of jobs moved under **governed workflows**

> Security: Only synthetic data is included; participants are instructed **not** to upload client data. See `SECURITY.md`.


Run code-backed labs for Delta Lake and Medallion (Bronze/Silver/Gold).

**Databricks:** Repos → Add (URL) → attach cluster → open `labs/` or `medallion/`.  
**Local (smoke tests):**
    python3.11 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    jupyter lab

## Modules
| Area | Notebook | Est. time |
|---|---|---|
| Delta ACID | `labs/01-delta-acid.ipynb` | 30–45m |
| Unified batch + stream | `labs/02-unified-batch-stream.ipynb` | 20–30m |
| Schema evolution | `labs/03-schema-evolution.ipynb` | 20–30m |
| Time travel | `labs/04-time-travel.ipynb` | 20–30m |
| Schema merge | `labs/05-schema-merge.ipynb` | 20–30m |
| OPTIMIZE & Z-ORDER | `labs/06-zorder.ipynb` | 20–30m |
| Audit & lineage | `labs/07-audit-lineage.ipynb` | 15–25m |
| CDF sample | `labs/08-cdf-sample.ipynb` | 15–25m |
| MERGE INTO | `labs/09-merge-into.ipynb` | 20–30m |
| Medallion pipeline | `medallion/*` | 45–60m |
| Layered design demos | `layer-demo/*` | 60–90m |

> Synthetic data only; no PHI/PII.
# AD_databricks-samples

Multiple demo notebooks for Parquet, Delta, Medallion pipelines, etc.

# Delta Table Feature Samples

1. **01_acid_properties.ipynb** – Demonstrating transactions & isolation.  
2. **02_unified_batch_stream.ipynb** – Unified batch + stream reads/writes.  
3. **03_schema_evolution.ipynb** – How to add/remove columns.  
4. **04_time_travel.ipynb** – Version history & rollback.  
5. **05_optimize_zorder.ipynb** – Performance tuning (OPTIMIZE/Z-ORDER).

# Layer  Samples
1. **01_Bronze.ipynb** – Demonstrating creation of Bronze layer  
2. **02_Silver.ipynb** – Demonstrating creation of Silver layer.  
3. **03_Gold.ipynb** – Demonstrating creation of Gold layer.


*Sample folders organized for demo purposes.*
