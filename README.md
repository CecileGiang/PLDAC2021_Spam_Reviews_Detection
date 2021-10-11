# Spam Reviews Detection

An interactive platform for spam reviews detection on Amazon data.

## Instructions

**STEP 1: Parse JSON file**
```
data, fields = parse('data/data.json', 65000)
data = doublonsNonSpams(data, fields)
```

**STEP 2: PageRank algorithm and spamicity score**

```
rg = ReviewGraph(data, fields, window = 1)
rg.computeScores(niter=50)
worst = rg.get_k_worst(50)
best = rg.get_k_best(50)
```

**STEP 3: Natural Language Processing**

```
tp = TextProcessor(corpus)
tp.process(lower=True, remove_punc=True, remove_digits=True, normalize=True, remove_stopwords=True, stemming=True)
tp.vectorize(vtype='tf-idf', analyzer='word', ngram_range=(1,1))
sim_matrix = tp.similarity_matrix()
sim_threshold = tp.similarity_threshold(threshold=0)
sum_sim_threshold = tp.overall_threshold(threshold=6)
```

**Command line for creating a reviews worpus**

`corpus = create_corpus(data, fields, id_prods, month_min, month_max)`
