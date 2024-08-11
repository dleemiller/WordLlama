
## MTEB Results l3_supercat, BS512
Llama3-based models improve over l2 supercat in clustering, reranking and classification tasks,
while performing slightly worse in pair classification and similarity tasks. The larger vocabulary
is also 4x larger on disk.

Supercat models are trained from concatenated token embeddings from several models using the same tokenizer
(eg Llama 3.1 405B, Llama3 70B, Llama guard 2)

| Metric                 | WL64        | WL128        | WL256        | WL512        | WL1024        | GloVe 300d | Komninos | all-MiniLM-L6-v2 |
|------------------------|-------------|--------------|--------------|--------------|---------------|------------|----------|------------------|
| Clustering             | 33.00       | 35.09        | 35.91        | 36.16        | 36.48         | 27.73      | 26.57    | 42.35            |
| Reranking              | 51.25       | 52.33        | 52.68        | 52.82        | 52.86         | 43.29      | 44.75    | 58.04            |
| Classification         | 53.37       | 56.66        | 58.72        | 59.65        | 59.92         | 57.29      | 57.65    | 63.05            |
| Pair Classification    | 75.68       | 77.17        | 77.81        | 77.94        | 78.07         | 70.92      | 72.94    | 82.37            |
| STS                    | 65.65       | 66.51        | 66.93        | 67.20        | 67.15         | 61.85      | 62.46    | 78.90            |
| CQA DupStack           | 18.67       | 22.51        | 24.16        | 24.92        | 25.10         | 15.47      | 16.79    | 41.32            |
| SummEval               | 30.11       | 30.08        | 30.11        | 30.88        | 29.96         | 28.87      | 30.49    | 30.81            |

