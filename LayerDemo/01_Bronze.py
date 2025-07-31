# Databricks notebook source
#01_Bronze notebook
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import current_timestamp, lit, input_file_name
import os
import builtins

# Delete the old Bronze folder entirely
dbutils.fs.rm("dbfs:/tmp/bronze/cc_events/", recurse=True)

# 1) Derive a unique process_id
all_conf   = spark.conf.getAll   # note the ()
job_id     = all_conf.get("spark.databricks.job.id",   "interactive")
run_id     = all_conf.get("spark.databricks.job.runId", "interactive")
process_id = f"daily_ingest|{job_id}_{run_id}"


# 1) derive process_id with safe fallback


# 2) Define the Bronze schema (exactly what Silver expects)
bronze_schema = StructType([
    StructField("user_id",         StringType(), True),
    StructField("app_name",        StringType(), True),
    StructField("event_type",      StringType(), True),
    StructField("event_timestamp", StringType(), True),
    StructField("device", StructType([
        StructField("os",     StringType(), True),
        StructField("region", StringType(), True)
    ]), True)
])

# 3) Ensure the raw folder exists under DBFS /tmp
raw_dbfs = "dbfs:/tmp/raw/telemetry/"
raw_local = "/dbfs/tmp/raw/telemetry/"
if not os.path.exists(raw_local):
    dbutils.fs.mkdirs(raw_dbfs)
    # Optionally burst in sample JSON here for a first run:
    # dbutils.fs.put(f"{raw_dbfs}sample1.json", '{"user_id":"u1","app_name":"PS",...}', True)

# 4) Read the raw JSON into our Bronze schema
raw_df = spark.read \
    .schema(bronze_schema) \
    .option("multiline", True) \
    .json(raw_dbfs)

# 5) Enrich with audit columns
bronze_df = (raw_df
    .withColumn("ingest_ts",   current_timestamp())
    .withColumn("process_id",  lit(process_id))
)

# 6) Write to Bronze Delta (append-only)
bronze_path = "dbfs:/tmp/bronze/cc_events/"
bronze_df.write \
    .format("delta") \
    .mode("append") \
    .save(bronze_path)

# 7) Sanity-check your Bronze table
display(spark.read.format("delta").load(bronze_path))

# COMMAND ----------

#Gold layer code 
# ───────────────────────────────────────────────────────────
# Gold Layer: Curated, Consumption-Ready Fact Tables
# ───────────────────────────────────────────────────────────

from pyspark.sql.functions import count, countDistinct

# 1) Define paths
silver_path           = "dbfs:/tmp/silver/cc_events_enterprise/"
feature_usage_gold    = "dbfs:/tmp/gold/feature_usage_fact/"
user_activity_gold    = "dbfs:/tmp/gold/user_activity_fact/"

# 2) Read the Silver “enterprise view”
silver_df = spark.read.format("delta").load(silver_path)
display(silver_df.limit(5))

# 3) Feature Usage Fact
feature_usage = (
    silver_df
      .filter("feature_category IS NOT NULL")                  # only track known features
      .groupBy("event_date", "app_name", "feature_category")   # natural grain
      .agg(count("*").alias("usage_count"))                    # daily usage per feature
)

# 4) Write Feature Usage to Delta, partitioned by event_date
feature_usage.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("event_date") \
    .save(feature_usage_gold)

display(spark.read.format("delta").load(feature_usage_gold))

# 5) User Activity Fact
user_activity = (
    silver_df
      .groupBy("event_date", "app_name", "region")
      .agg(countDistinct("user_id").alias("active_users"))     # daily unique users
)

# 6) Write User Activity to Delta, partitioned by event_date
user_activity.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("event_date") \
    .save(user_activity_gold)

display(spark.read.format("delta").load(user_activity_gold))

# 7) Optimize Gold tables for speed (Z-Order on high-cardinality column)
spark.sql(f"OPTIMIZE delta.`{feature_usage_gold}` ZORDER BY (feature_category)")
spark.sql(f"OPTIMIZE delta.`{user_activity_gold}`   ZORDER BY (region)")
