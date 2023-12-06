#!/usr/bin/env python
""" \
    Лабораторна робота №1
    Виконав студент 543 групи Лунгу Денис
"""
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import NGram
from time import strftime
from pyspark.sql.functions import (
    lower,
    explode,
    regexp_replace,
    collect_list,
    split,
    flatten,
    size,
    concat_ws,
    expr,
)

# Визначимо основні константи
EVENT_TYPE = "PushEvent"
INPUT_FILE_PATH = "10K.github.jsonl"
NGRAM_FACTOR = 3
NGRAMS_COLUMN_NAME = "ngrams"

# Створимо основні об'єкти для роботи програми
ctx = SparkContext.getOrCreate()
spark = (
    SparkSession(ctx)
    .builder.master("local[*]")
    .appName("Лабораторна робота #1")
    .getOrCreate()
)
ngram = NGram(n=NGRAM_FACTOR, inputCol="words", outputCol=NGRAMS_COLUMN_NAME)

# Отримаємо повідомлення комітів і конвертуємо їх у колекцію
# пара ключ-значення AuthorName:CommitMessageWords
githubEventsDf = (
    spark.read.json(INPUT_FILE_PATH)
    .filter(f"type = '{EVENT_TYPE}'")
    .select(explode("payload.commits").alias("commit"))
    .select(
        lower("commit.author.name").alias("author"),
        lower("commit.message").alias("message"),
    )
    .withColumn("message", regexp_replace("message", "[^a-zA-Z0-9\\s]", ""))
    .withColumn("message", (split("message", "\\s+")))
    .withColumn("message", expr("filter(message, element -> element != '')"))
    .groupBy("author")
    .agg(flatten(collect_list("message")).alias("words"))
)

# Перетворимо слова в n-grams
result = (
    ngram.transform(githubEventsDf)
    .select("author", NGRAMS_COLUMN_NAME)
    .filter(size(NGRAMS_COLUMN_NAME) > 0)
    .withColumn(NGRAMS_COLUMN_NAME, concat_ws(", ", NGRAMS_COLUMN_NAME))
)

# Збережемо результат у CSV
resultFileName = "ngram-{timestamp}".format(timestamp=strftime("%Y%m%d-%H%M%S"))
result.write.option("header", True).option("delimiter", ";").csv(resultFileName)
