"""
config.py
Central place for settings used across the application.
Changing a number here changes it everywhere.
"""

APP_TITLE = "AI-Powered Data Analyst Assistant"
APP_ICON = "📊"

# File upload rules
ALLOWED_EXTENSIONS = (".csv", ".xlsx")
MAX_FILE_SIZE_MB = 50          # reject files bigger than this
MAX_ROWS = 500_000             # reject datasets with more rows than this

# Preview settings
PREVIEW_ROWS = 10

# Data quality settings
CONVERSION_THRESHOLD = 0.9     # a text column is "really numbers/dates" if 90%+ of values convert
OUTLIER_IQR_MULTIPLIER = 1.5   # standard rule for detecting outliers
MIN_ROWS_FOR_OUTLIERS = 10     # too few values = unreliable outlier detection
MAX_UNIQUE_FOR_TEXT_CHECK = 1000  # skip spelling checks on ID-like / free-text columns
SAMPLE_SIZE = 1000             # rows used for a fast "is this column numbers/dates?" test
UNIQUE_VALUES_SHOWN = 20

# Columns whose names contain these words should normally never be negative
NON_NEGATIVE_KEYWORDS = ("quantity", "qty", "price", "revenue", "amount", "sales", "cost", "total")

# Dashboard and chart settings
DATE_PARSE_MIN_RATIO = 0.5     # a text column counts as "dates" if half its values parse as dates
DATE_SAMPLE_SIZE = 200         # how many values to test when looking for date columns
DEFAULT_TOP_N = 10             # default number of groups shown in "top N" charts
PIE_MAX_SLICES = 8             # pie charts group smaller slices into "Other"
MAX_SCATTER_POINTS = 5000      # scatter plots are sampled above this many points
MAX_HISTOGRAM_BINS = 100
# Analysis engine settings
MAX_FILTERS = 5
MAX_RESULT_ROWS = 50
MAX_CATEGORY_VALUES_SHOWN = 15
SAMPLE_VALUES_SHOWN = 3
# Insight and export settings
CONCENTRATION_PCT = 50         # a group holding this % of the total is flagged as "concentrated"
CHANGE_ALERT_PCT = 20          # a month-over-month change this big is mentioned
MISSING_ALERT_PCT = 5          # columns with this % of empty cells are flagged
EXCEL_EXPORT_MAX_ROWS = 100_000  # bigger datasets are offered as CSV only (Excel export is slow)