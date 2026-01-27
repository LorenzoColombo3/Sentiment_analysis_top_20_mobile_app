from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim, length, explode, desc, row_number
from pyspark.sql.window import Window
from pyspark.ml.feature import Tokenizer, StopWordsRemover, NGram
import pandas as pd
import os
import time

# CONFIGURAZIONE
APP_NAME = "3_App_Specific_Insights"
INPUT_PATH = "hdfs://10.0.1.5:9000/progetto/data_parquet"
OUTPUT_DIR = "/home/Colomboadmin/progetto/output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

spark = SparkSession.builder.appName(APP_NAME).config("spark.sql.shuffle.partitions", "200").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

def now(): return time.time()
print(f"{'='*50}")
print(f"AVVIO ANALISI PER APPLICAZIONE")
print(f"{'='*50}")

# 1. CARICANMENTO
print("\n[1] Caricamento Dati")
t0 = now()
df = spark.read.parquet(INPUT_PATH).select("app_name", "content", "score")
df_dataset = df.select(
        col("app_name"),
        col("content").alias("review_text"),
        col("score").cast("int").alias("rating")
    ).na.drop(subset=["review_text", "app_name"])

# 2. PIPELINE NLP
print("\n[2] NLP Processing")
t0 = now()
df_clean = df_dataset.withColumn("clean_text", lower(col("review_text"))) \
    .withColumn("clean_text", regexp_replace("clean_text", "[^a-z]", " ")) \
    .withColumn("clean_text", regexp_replace("clean_text", "\\s+", " ")) \
    .withColumn("clean_text", trim(col("clean_text"))) \
    .filter(length(col("clean_text")) > 1)
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

# 3. ANALISI PER APP
print("\n[3] Aggregazione per App e Ranking")
t0 = now()
df_exploded = df_nlp.select("app_name", "rating", explode(col("ngrams")).alias("bigram"))
df_clean_bigrams = df_exploded.withColumn("bigram", trim(col("bigram"))) \
    .filter(length("bigram") > 3).filter(~col("bigram").contains("  "))
df_counts = df_clean_bigrams.groupBy("app_name", "rating", "bigram").count()
windowSpec = Window.partitionBy("app_name", "rating").orderBy(desc("count"))
df_ranked = df_counts.withColumn("rank", row_number().over(windowSpec)).filter(col("rank") <= 10)
df_ranked.cache()
count_results = df_ranked.count()
print(f"Calcolo completato in {now() - t0:.2f}s")

# 4. EXPORT
print("\n[4] Salvataggio Report")
pdf_results = df_ranked.select("app_name", "rating", "rank", "bigram", "count") \
                       .orderBy("app_name", "rating", "rank") \
                       .toPandas()
out_file = os.path.join(OUTPUT_DIR, "app_insights_drilldown.csv")
pdf_results.to_csv(out_file, index=False)
print(f"Report salvato")
spark.stop()
