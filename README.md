# Sentiment analysis with Deep Atrous CNN networks

Implement of a Deep Atrous CNN architecture suitable for short text sentiment classification.

The model uses QueueRunner and Readers to prefetch training data. Information can be fetched from multiple feeds easily if they are put in the proper TSV format, which allows the sentiment analyser to be extended easily with more data sources.