# 📊 Modified WordCount and Word Pair Analysis — CS657 Assignment 1

This PySpark-based project analyzes the **State of the Union (SOTU)** speech transcripts from 1790 to 2021 using distributed processing techniques. It extends the traditional WordCount program with advanced statistical, linguistic, and co-occurrence-based computations, fully optimized for scalability and large dataset performance.

---

## 👥 Contributors

- **Dev Divyendh Dhinakaran** (G01450299)  
- **Tejaswi Samineni** (G01460925)

---

## ⚙️ Technologies & Tools

- **PySpark** (RDD + DataFrame APIs)
- **HDFS** (for distributed file storage)
- **Regular Expressions** (text cleaning)
- **Matplotlib / Plotly** (for visualization)
- **Linux + Spark Cluster (Perseus)**

---

## 📋 Features

### ✅ Text Cleaning and Preprocessing
- HTML tag removal using regex
- URL elimination
- Punctuation stripping
- Stopword filtering
- Fully implemented using PySpark transformations (no non-scalable libraries)

### 📈 Word Frequency Analysis
- Analyze word counts in 4-year sliding windows (starting 2009)
- Compute average and standard deviation of word usage
- Detect "spiked" words in the year following each window (using `avg + 2*std` threshold)

### 📚 Flesch-Kincaid Readability Score
- Calculated per speech using:
  ```
  Score = 0.39 * (words/sentence) + 11.8 * (syllables/word) - 15.59
  ```
- Results plotted as a bar chart, labeled with the president's last name for each year

### 🔁 Word Pair Co-Occurrence + Lift
- Co-occurrence of word pairs within same sentence
- Output 20 frequent word pairs (frequency > 10)
- Compute **lift** for each pair:
  ```
  lift(A, B) = P(A ∩ B) / (P(A) * P(B))
  ```
- Output word pairs with `lift > 3.0`

---

## 🗃️ Data Source

- **Dataset:** State of the Union Addresses (1790–2021)  
- 📥 [Source](https://stateoftheunion.onetwothree.net/appendices.html)  
- 📄 File format: `yyyymmdd.txt`, processed as a **single file in HDFS**

---

## 🚀 Execution Workflow

```bash
# Run on GMU Perseus (Spark Cluster)
spark-submit assignment1_modified_wordcount.py
```

### 📂 Input Location (HDFS)
```
hdfs:///user/<netid>/Assignment1_StateOfUnion/sotu.txt
```

### 📊 Sample Output
- Cleaned word counts by year
- Spiked words list per year window
- Flesch-Kincaid scores in bar chart
- 20 frequent word pairs with lift > 3.0

---

## 🔍 Optimization Techniques

- Repartitioning by year and window for parallelism
- No use of non-scalable libraries (e.g., BeautifulSoup)
- Efficient `explode`, `split`, and `withColumn` chaining
- Lift computation parallelized using joins and broadcast variables

---


## 🧠 Key Learnings

- Leveraging distributed computing with PySpark for large-scale text analysis
- Implementing custom scalable transformations
- Using statistical reasoning for spike detection
- Applying NLP readability metrics at scale
- Performing association mining using lift

---
