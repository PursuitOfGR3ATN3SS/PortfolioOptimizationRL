# Makefile

.PHONY: all get-notes run-pipeline sentiment

# Default behavior
all:
	@echo "Please specify a target:"
	@echo "	 make get-notes				# Download latest SEC dataset"
	@echo "	 make run-pipeline    # Run sentiment analysis"
	@echo "  make sentiment       # Download + Run sentiment analysis"

# Downloads the latest SEC dataset into SECData/<year>_<month>_notes
get-notes:
	@echo "Downloading latest SEC notes dataset..."
	python SentimentAnalysis/download_latest_notes.py

# Runs the sentiment analysis on the most recent dataset
run-pipeline:
	@echo "Running sentiment analysis on latest notes dataset..."
	python SentimentAnalysis/sentiment_notes_pipeline.py

# Download latest, then run sentiment analysis
sentiment: get-notes run-pipeline