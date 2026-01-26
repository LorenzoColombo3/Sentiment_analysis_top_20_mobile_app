from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time
import os

# --- CONFIGURAZIONE ---
APP_NAME = "1_Data_Preparation_Augmentation"
INPUT_CSV_PATH = "hdfs://10.0.1.5:9000/progetto/data/*.csv"
OUTPUT_PARQUET_PATH = "hdfs://10.0.1.5:9000/progetto/data_parquet"

TARGET_ROWS = 20000000

spark = SparkSession.builder.appName(APP_NAME).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

print(f"AVVIO PREPARAZIONE DATI")

# 1. INGESTIONE E TAGGING
print("\n[1] Ingestione Dati Grezzi")
try:
    df_base = spark.read.csv(INPUT_CSV_PATH, header=True, inferSchema=True)
    df_raw = df_base.withColumn("full_path", input_file_name()) \
        .withColumn("app_name", 
                    regexp_replace(element_at(split(col("full_path"), "/"), -1), "\.csv", "")
                   ) \
        .drop("full_path")
    current_rows = df_raw.count()
    print("Anteprima")
    df_raw.select("app_name").distinct().show(5, truncate=False)
except Exception as e:
    print(f"ERRORE: {e}")
    spark.stop()
    exit(1)

# 2. DATA AUGMENTATION
print("\n[2] Data Augmentation")
FACTOR = int(TARGET_ROWS / current_rows) + 1
print(f"   -> Moltiplicazione x{FACTOR} per raggiungere target {TARGET_ROWS}")
df_expanded = df_raw.withColumn("dummy", explode(array_repeat(lit(0), FACTOR))) \
                .drop("dummy")
df_final = df_expanded.withColumn("unique_id", monotonically_increasing_id())
total_rows = df_final.count()

# 4. SALVATAGGIO
print("\n[3] Scrittura su Disco")
start_pq = time.time()
df_final.write.mode("overwrite").parquet(OUTPUT_PARQUET_PATH)

time_pq = time.time() - start_pq

spark.stop()
