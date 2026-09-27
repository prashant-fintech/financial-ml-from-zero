"""Column positions in Freddie Mac's Single-Family Loan-Level Dataset files.

The origination and performance files are pipe-delimited with no header
row. Names come from Freddie Mac's user guide. Only the columns this post
uses are listed; everything else is left unread.

Check these positions against the user guide for the release you
download. Freddie Mac has appended columns over the years, but the
positions below have been stable at the front of each file.
"""

# Origination file: one row per loan, fixed at origination.
ORIGINATION_COLUMNS: dict[int, str] = {
    0: "credit_score",
    1: "first_payment_month",
    9: "orig_dti",
    10: "orig_upb",
    11: "orig_ltv",
    12: "orig_rate",
    19: "loan_id",
}
ORIGINATION_WIDTH = 32

# Monthly performance file: one row per loan per reporting month.
PERFORMANCE_COLUMNS: dict[int, str] = {
    0: "loan_id",
    1: "period",
    2: "current_upb",
    3: "dq_status_raw",
    4: "loan_age",
    8: "zero_balance_code",
    9: "zero_balance_month",
}
PERFORMANCE_WIDTH = 32

# Sentinel values from the user guide.
CREDIT_SCORE_MISSING = "9999"
DTI_MISSING = "999"
LTV_MISSING = "999"

# Delinquency status is months past due ("0", "1", "2", ...), or "RA" for
# REO acquired. Everything at 6 or more is treated as 6: past that point
# the loan is deep in default and the exact count adds nothing.
DQ_STATUS_CAP = 6
DQ_STATUS_REO = "RA"

# Zero balance codes that end a loan through a credit event, as opposed to
# 01 (prepaid or matured) or 96 (repurchased).
DEFAULT_ZERO_BALANCE_CODES = frozenset({"02", "03", "09", "15"})
