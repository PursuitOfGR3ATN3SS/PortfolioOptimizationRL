import pandas as pd

# Load both CSVs
filing_df = pd.read_csv("SentimentAnalysis/outputs/2025_03_per_filing.csv")
note_df = pd.read_csv("SentimentAnalysis/outputs/2025_03_per_note.csv")

# -------------------------------
# Print column names
# -------------------------------
print("Columns in per_filing.csv:")
print(list(filing_df.columns))
print()

print("Columns in per_note.csv:")
print(list(note_df.columns))
print()

# -------------------------------
# Clean + show unique fye (from filing)
# -------------------------------
fye_cleaned_filing = (
    filing_df['fye']
    .dropna()
    .astype(int)
    .astype(str)
    .str.zfill(4)
)

fye_sorted_filing = sorted(fye_cleaned_filing.unique())

print("Unique fiscal year ends (fye) in per_filing.csv:")
print(fye_sorted_filing)
print()

# -------------------------------
# Clean + show unique fye (from notes)
# -------------------------------
if 'fye' in note_df.columns:
    fye_cleaned_note = (
        note_df['fye']
        .dropna()
        .astype(int)
        .astype(str)
        .str.zfill(4)
    )

    fye_sorted_note = sorted(fye_cleaned_note.unique())

    print("Unique fiscal year ends (fye) in per_note.csv:")
    print(fye_sorted_note)
    print()

# -------------------------------
# Clean + show unique ddate (from notes)
# -------------------------------
if 'ddate' in note_df.columns:
    ddate_cleaned = (
        note_df['ddate']
        .dropna()
        .astype(int)
        .astype(str)
        .str.zfill(8)
    )

    ddate_sorted = sorted(ddate_cleaned.unique())

    print("Unique disclosure dates (ddate) in per_note.csv:")
    print(ddate_sorted)
