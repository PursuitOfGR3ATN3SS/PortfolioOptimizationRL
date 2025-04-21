# Makefile

.PHONY: all get-data filter-data get-sentiments sentiment

# Default behavior
all:
	@echo "Please specify a target:"
	@echo "  make get-data               # Download dataset from Kaggle"
	@echo "  make filter-data [n=...]    # Filter by busiest month (optionally sample N rows)"
	@echo "  make get-sentiments         # Run sentiment analysis on filter data"
	@echo "  make sentiment [n=...]      # Run full pipeline: download --> filter --> analyze"

# Downloads dataset into /SentimentAnalysis/data/
get-data:
	@echo "Downloading dataset via Kaggle API..."
	python SentimentAnalysis/get_data.py

# Filters to busiest month, optionally random sample N rows: 'n=...'
filter-data:
	@echo "Filtering dataset to busiest year/month..."
	python SentimentAnalysis/filter_data.py $(if $(n),-n $(n),)

# Runs FinBERT sentiment analysis
get-sentiments:
	@echo "Running FinBERT sentiment analysis..."
	python SentimentAnalysis/get_sentiments.py

# full pipeline (data --> filter --> sentiment), supports 'n=...'
sentiment: get-data filter-data get-sentiments
