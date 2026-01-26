from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim, length, explode, desc
from pyspark.ml.feature import Tokenizer, StopWordsRemover, NGram
import pandas as pd
import os
import time
import sys

# CONFIGURAZIONE
APP_NAME = "Analysis_Pipeline_Parquet_Optimized"

# PERCORSi IN/OUT
INPUT_PATH = "hdfs://10.0.1.5:9000/progetto/data_parquet" 
OUTPUT_DIR = "/home/Colomboadmin/progetto/output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

spark = SparkSession.builder \
    .appName(APP_NAME) \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
def now(): return time.time()
metrics = []
t_global_start = now()
print(f"AVVIO ANALISI")

# FASE 1: CARICAMENTO (I/O Puro)
print("\n[1] Caricamento Dati")
t0 = now()

df = spark.read.parquet(INPUT_PATH).select("content", "score")

df_dataset = df.select(
        col("content").alias("review_text"),
        col("score").cast("int").alias("rating")
    ).na.drop(subset=["review_text"])

row_count = df_dataset.count()

dt = now() - t0
metrics.append({"stage": "1_Load_IO", "seconds": dt, "format": "parquet"})

# FASE 2: DEFINIZIONE NLP (Lazy)
print("\n[2] Definizione Pipeline NLP")
t0 = now()
df_clean = df_dataset.withColumn("clean_text", lower(col("review_text"))) \
    .withColumn("clean_text", regexp_replace("clean_text", "[^a-z]", " ")) \
    .withColumn("clean_text", regexp_replace("clean_text", "\\s+", " "))\
    .withColumn("clean_text", trim(col("clean_text"))).filter(length(col("clean_text")) > 1)
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
df_tok = tokenizer.transform(df_clean)
custom_stopwords = [
    "candy", "crush", "saga", "king",
    "dropbox", "box",
    "facebook", "lite", "fb", "meta",
    "messenger", "message", "messaging",
    "flipboard", "flip",
    "instagram", "insta", "ig", "reel", "reels",
    "line", "lineapp",
    "microsoft", "powerpoint", "ppt", "pptx",
    "word", "doc", "docx", "office", "whatsapp",
    "slide", "slides", "document", "file",
    "app", "apps", "application", "apk",
    "android", "ios", "google", "play", "store",
    "phone", "mobile", "cell", "device", "tablet", "screen",
    "install", "installed", "uninstall", "download", "downloading",
    "update", "updating", "updated", "version",
    "internet", "wifi", "data", "connection",
    "account", "login", "log", "sign",
    "use", "using", "used", "user",
    "open", "opening", "start", "load", "loading", "working", "works",
    "just", "really", "very", "much", "many", "lot", "bit", "little",
    "get", "got", "getting", "make", "made", "making",
    "try", "tried", "trying",
    "want", "wanted", "wants",
    "know", "think", "say", "said",
    "way", "thing", "things", "stuff",
    "also", "even", "still", "back", "well",
    "one", "two", "first",
    "time", "times", "day", "days", "today", "yesterday", "tomorrow",
    "week", "weeks", "month", "months", "year", "years",
    "now", "minute", "minutes", "hour", "hours",
    "since", "ago", "long",
    "please", "plz", "pls",
    "help", "fix", "fixed", "fixing",
    "issue", "problem",
    "thanks", "thank", "thx"
]
remover = StopWordsRemover(
    inputCol="words", 
    outputCol="filtered", 
    stopWords=StopWordsRemover().getStopWords() + custom_stopwords
)
df_filt = remover.transform(df_tok)
ngram = NGram(n=2, inputCol="filtered", outputCol="ngrams")
df_nlp = ngram.transform(df_filt)
metrics.append({"stage": "2_NLP_Definition", "seconds": now() - t0, "format": "parquet"})

# FASE 3: CALCOLO MAPREDUCE
print("\n[3] MapReduce Execution")
t0 = now()
df_repartitioned = df_nlp.repartition(200)
df_exploded = df_repartitioned.withColumn("bigram", explode(col("ngrams")))
df_analysis = df_exploded.withColumn("bigram", trim(col("bigram"))).filter(length("bigram") > 3).filter(~col("bigram").contains("  ")).groupBy("rating", "bigram").count()
df_analysis.cache()
unique_ngrams = df_analysis.count()
dt = now() - t0
metrics.append({"stage": "3_MapReduce_Calc", "seconds": dt, "format": "parquet"})
print(f"Calcolo completato in {dt:.2f}s | Bigrammi unici trovati: {unique_ngrams}")

# FASE 4: EXPORT
print("\n[4] Export Risultati")
t0 = now()

#salvataggio frequenze
top_results = df_analysis.orderBy(desc("count")).limit(1000)
pdf_results = top_results.toPandas()
out_res_file = os.path.join(OUTPUT_DIR, "word_freq_results_parquet.csv")
pdf_results.to_csv(out_res_file, index=False)

# Salvataggio Metriche
t_total = now() - t_global_start
metrics.append({"stage": "Total_Execution", "seconds": t_total, "format": "parquet"})
pdf_metrics = pd.DataFrame(metrics)
metrics_file = os.path.join(OUTPUT_DIR, f"metrics_parquet_{time.strftime('%Y%m%d-%H%M%S')}.csv")
pdf_metrics.to_csv(metrics_file, index=False)
print(f"Risultati: {out_res_file}")
print(f"Metriche: {metrics_file}")
spark.stop()
