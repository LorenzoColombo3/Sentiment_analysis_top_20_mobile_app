# Benchmarking e Sentiment Analysis su Cluster Spark

Questo progetto implementa una pipeline di Big Data Analytics end-to-end utilizzando **Apache Spark** e **HDFS** su un cluster distribuito. L'obiettivo principale è analizzare un dataset massivo di recensioni del Google Play Store artificialmente espanso a oltre 20 milioni di record per estrarre insight sul sentiment degli utenti e, parallelamente, misurare le prestazioni di scalabilità del cluster.

## Obiettivi del Progetto

* **Analisi del Sentiment (NLP):** Estrarre pattern semantici (bigrammi) dalle recensioni per identificare i driver di insoddisfazione (Rating 1) e i fattori di successo (Rating 5) sia a livello globale che per singola applicazione.
* **Benchmarking Infrastrutturale:** Misurare la capacità del cluster di scalare variando il numero di nodi (da 1 a 4) a parità di volume dati, calcolando Speedup ed Efficienza.
* **Gestione dello Storage:** Implementare un Data Lake distribuito utilizzando Hadoop HDFS per garantire fault tolerance e data locality.

## Architettura

Il sistema è deployato su un cluster di macchine virtuali su cloud Azure:

* **Cluster Manager:** Spark Standalone
* **Storage Layer:** Apache Hadoop HDFS (Replication Factor: 3)
* **Nodi:** 1 Master + 3 Worker (totale 16 vCPU, 40 GB RAM)
* **Linguaggio:** Python (PySpark)
* **Formato Dati:** Apache Parquet (ottimizzato per letture colonnari)

## Struttura del Codice

Il repository è organizzato nei seguenti script principali:

* `1_Data_Preparation_Augmentation.py`: Script che ingerisce i file CSV grezzi, applica Data Augmentation per raggiungere i 20 milioni di record e salva il risultato in formato Parquet su HDFS.
* `analysis_pipeline.py`: Pipeline "Macro" che analizza l'intero dataset come un unico corpus per identificare trend trasversali.
* `app_insights.py`: Pipeline che utilizza Window Functions per isolare le keyword specifiche di ogni applicazione.
* `visualize_results.py`: Script locale per la generazione dei grafici a partire dai CSV dei risultati.

## Risultati Chiave

* **Scalabilità:** Il cluster ha mostrato uno Speedup di **3.04x** passando da 1 a 4 nodi, con un'efficienza ottimale (90%) nella configurazione a 3 nodi.
* **Insight di Business:** L'analisi ha permesso di distinguere chiaramente tra problemi tecnici e problemi di design/business.

## Requisiti

* Apache Spark 3.x
* Apache Hadoop 3.x
* Python 3.x (Librerie: `pyspark`, `pandas`, `matplotlib`, `seaborn`)

## Autore

**Lorenzo Colombo**
Corso di Large Scale Data Management
Università degli Studi di Milano-Bicocca
