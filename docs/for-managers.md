# For Managers

## What this portfolio demonstrates
- **Code-first training**: Every topic has a runnable notebook and a small dataset generator.
- **Production alignment**: Topics map to real operator concerns (governance, rollback, evolution, cost/perf).
- **Repeatability**: The repo is a GitHub **Template**, so each cohort or customer can spin up their own copy.

## Learning outcomes mapped to labs
| Outcome | Labs |
|---|---|
| Understand Delta ACID & safe mutation | `labs/01-delta-acid.ipynb`, `labs/09-merge-into.ipynb` |
| Confident schema change & evolution | `labs/03-schema-evolution.ipynb`, `labs/05-schema-merge.ipynb` |
| Point-in-time restore & audits | `labs/04-time-travel.ipynb`, `labs/07-audit-lineage.ipynb` |
| Optimize reads & cost | `labs/06-zorder.ipynb` |
| Change Data Feed for downstream syncs | `labs/08-cdf-sample.ipynb` |
| End-to-end layered design | `medallion/*`, `layer-demo/*` |

## Delivery plan (sample)
- **Day 1 (Foundations)**: Medallion patterns, Delta ACID, time travel, schema evolution; hands-on labs 01–04  
- **Day 2 (Performance & change)**: Z-ORDER, CDF, streaming mini-lab; design discussion  
- **Day 3 (Apply & govern)**: Layered exercises, packaging as jobs, review of team pipelines

## What “good” looks like after training
- Pipelines land Bronze→Silver→Gold with **rollback paths** and **tests** for schema changes  
- Developers can explain **why/when** to optimize files & Z-ORDER  
- At least one **governed job** scheduled, with on-call notes / runbooks started

## Customization options
- **Your catalog/permissions**: widgets for catalog/schema/paths; run against Volumes or DBFS
- **Your domain**: swap in your domain dataset (still synthetic) for the medallion labs
- **Your governance**: optional session on bundle-based deployments / approvals

## Logistics & safety
- **Environment**: Databricks Workspace Repos (or local Jupyter for smoke tests)
- **Data policy**: Synthetic only; no client-confidential data in the repo
- **Artifacts**: All code + workshop notes stay with your team in a fork/template copy
