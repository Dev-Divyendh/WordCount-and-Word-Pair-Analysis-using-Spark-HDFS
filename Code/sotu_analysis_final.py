import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode, split, lower, avg, stddev, count, coalesce, lit, row_number
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, StructType, StructField
from pyspark.sql import Window

# Initializing Spark Session
spark = SparkSession.builder.appName("SOTUAnalysis").getOrCreate()

# HDFS file path
hdfs_path = "hdfs:///user/tsaminen/Assignment1_StateOfUnion/sotu.txt"

# Reading data from HDFS directly
sotu_rdd = spark.sparkContext.textFile(hdfs_path)

# Function to extract years, presidents, and speeches
def extract_speeches(text):
    years, presidents, speeches = [], [], []
    current_speech, capture_speech = [], False

    for line in text:
        line = line.strip()
        if not capture_speech:
            if line == "***":
                capture_speech = True
            continue

        if line.startswith("State of the Union Address"):
            if current_speech:
                speeches.append(" ".join(current_speech).strip())
                current_speech = []
            try:
                president_line = next(text).strip()
                year_line = next(text).strip()
                year_match = re.search(r'\b\d{4}\b', year_line)
                year = year_match.group(0) if year_match else None
                if year:
                    presidents.append(president_line)
                    years.append(int(year))
            except StopIteration:
                break
        elif line != '***' and line:
            current_speech.append(line)

    if current_speech:
        speeches.append(" ".join(current_speech).strip())

    return years, presidents, speeches

# Converting into a DataFrame
data_rdd = sotu_rdd.mapPartitions(lambda lines: [extract_speeches(lines)])
schema = StructType([StructField("year", IntegerType(), True),
                     StructField("president", StringType(), True),
                     StructField("speech", StringType(), True)])
sotu_spark_df = spark.createDataFrame(data_rdd, schema)

# Partition data based on year to improve performance
sotu_spark_df = sotu_spark_df.repartition(10, col("year"))

# Cleaning Data - Text pre-processing
def clean_text(text):
    stopwords = set(["the", "and", "for", "to", "of", "a", "in", "is", "on", "with", "as", "by", "it", "at", "this", "that", "an", "be", "or", "from", "which", "but", "not", "are", "was", "were", "has", "had", "have", "will", "would", "could", "shall", "should", "do", "did", "can", "if", "we", "they", "their", "our", "you", "your", "i", "he", "she", "his", "her", "its", "them", "all", "any", "been", "more", "no", "than", "so", "us", "who", "what", "when", "where", "why"])
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s.!?]', '', text)  # Remove punctuation
    words = [word for word in text.lower().split() if word not in stopwords]
    return " ".join(words)

clean_text_udf = udf(clean_text, StringType())

# Apply cleaning to speeches
sotu_spark_df_cleaned = sotu_spark_df.withColumn("cleaned_speech", clean_text_udf(col("speech")))

# Tokenize the cleaned text into words for further processing
sotu_words_df = sotu_spark_df_cleaned.withColumn("words", explode(split(lower(col("cleaned_speech")), "\s+")))

# Group by year and word to count word frequencies
word_counts = sotu_words_df.groupBy("year", "words").agg(count("words").alias("word_count"))

# Create 4-year windows from 2009
def get_window(year):
    if year >= 2009:
        start_year = ((year - 2009) // 4) * 4 + 2009
        return f"{start_year}-{start_year + 3}"
    return None

get_window_udf = udf(get_window, StringType())
word_counts_windowed = word_counts.withColumn("window", get_window_udf(col("year"))).filter(col("window").isNotNull())

word_counts_windowed = word_counts_windowed.repartition(10, col("window"))

# Aggregate word counts over the window
window_agg = word_counts_windowed.groupBy("window", "words").agg(
    avg("word_count").alias("avg_count"),
    stddev("word_count").alias("std_count")
)

# Handle NULL stddev values (if no variation, stddev will be NULL)
window_agg = window_agg.withColumn("std_count", coalesce(col("std_count"), lit(0)))

# Calculate word spikes
spike_threshold = window_agg.withColumn("threshold", col("avg_count") + 2 * col("std_count"))

# Save word spikes to a CSV file
spike_threshold.write.csv("hdfs:///user/tsaminen/word_spikes_output", mode="overwrite")

# Read the CSV back and print the top 10 rows
spike_threshold_df = spark.read.csv("hdfs:///user/tsaminen/word_spikes_output", header=True, inferSchema=True)
spike_threshold_df.show(10)

# Function to count syllables
def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    if len(word) == 0:
        return 0
    if word[0] in vowels:
        count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count = 1
    return count

# Calculate Flesch-Kincaid readability score
def flesch_kincaid(text):
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    words = text.split()
    word_count = len(words)
    syllable_count = sum([count_syllables(word) for word in words])
    if word_count == 0 or sentence_count == 0:
        return None
    return (0.39 * (word_count / sentence_count)) + (11.8 * (syllable_count / word_count)) - 15.59

flesch_kincaid_udf = udf(flesch_kincaid, FloatType())
sotu_spark_df_cleaned = sotu_spark_df_cleaned.withColumn("flesch_kincaid", flesch_kincaid_udf(col("cleaned_speech")))

# Save Flesch-Kincaid results to a CSV file
sotu_spark_df_cleaned.select("year", "president", "flesch_kincaid").write.csv("hdfs:///user/tsaminen/flesch_kincaid_output", mode="overwrite")

# Read the CSV back and print the top 10 rows
flesch_kincaid_df = spark.read.csv("hdfs:///user/tsaminen/flesch_kincaid_output", header=True, inferSchema=True)
flesch_kincaid_df.show(10)

# Word co-occurrence within the same sentence
sotu_sentences_df = sotu_spark_df_cleaned.withColumn("sentence", explode(split(col("cleaned_speech"), r'[.!?]')))
sotu_words_df = sotu_sentences_df.withColumn("word", explode(split(col("sentence"), "\s+")))

# Create word pairs using self-join and compute word pair frequencies
window_spec = Window.partitionBy("sentence").orderBy(monotonically_increasing_id())
sotu_words_df = sotu_words_df.withColumn("word_index", row_number().over(window_spec))
sotu_word_pairs_df = sotu_words_df.alias("df1").join(
    sotu_words_df.alias("df2"),
    (col("df1.sentence") == col("df2.sentence")) & (col("df1.word_index") < col("df2.word_index")),
    "inner"
)
word_pairs_df = sotu_word_pairs_df.select(col("df1.word").alias("word1"), col("df2.word").alias("word2"))
pair_counts_df = word_pairs_df.groupBy("word1", "word2").agg(count("*").alias("pair_count"))

# Filter word pairs that occur more than 10 times
frequent_pairs_df = pair_counts_df.filter(col("pair_count") > 10)

# Compute lift for word pairs
word_totals = sotu_words_df.groupBy("word").agg(count("*").alias("total_count"))
frequent_pairs_with_totals = frequent_pairs_df.join(
    word_totals.alias("word1_totals"), frequent_pairs_df["word1"] == col("word1_totals.word"), "inner"
).join(
    word_totals.alias("word2_totals"), frequent_pairs_df["word2"] == col("word2_totals.word"), "inner"
)

# Calculate lift for word pairs
frequent_pairs_with_lift = frequent_pairs_with_totals.withColumn(
    "lift", col("pair_count") / (col("word1_totals.total_count") * col("word2_totals.total_count"))
)

# Save word pairs with lift > 3.0 to a CSV file
high_lift_pairs_df = frequent_pairs_with_lift.filter(col("lift") > 3.0)
high_lift_pairs_df.select("word1", "word2", "pair_count", "lift").write.csv("hdfs:///user/tsaminen/high_lift_pairs_output", mode="overwrite")

# Read the CSV back and print the top 10 rows
high_lift_pairs_df = spark.read.csv("hdfs:///user/tsaminen/high_lift_pairs_output", header=True, inferSchema=True)
high_lift_pairs_df.show(10)
