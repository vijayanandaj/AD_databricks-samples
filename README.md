# Databricks Medallion & Delta Labs — Template-ready

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