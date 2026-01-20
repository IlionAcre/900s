"""
Column Selector for 990 and 990-PF CSVs.

This script takes input CSVs and outputs them with only the specified columns,
matching the structure of the reference files. NO row filtering is performed.

Usage:
    python column_selector.py --input results_990.csv --output cleaned_990.csv --type 990
    python column_selector.py --input results_990pf.csv --output cleaned_990pf.csv --type 990pf
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Optional

# ============================================================
# COLUMN DEFINITIONS
# These match the reference files:
# - cleaned/id_cash_hist_990_dedup_latest.csv (for 990)
# - cleaned/id_cash_hist_990pf_dedup_latest.csv (for 990-PF)
# ============================================================

# 990-PF Columns (95 columns total including _EXTRA_FIELDS)
COLUMNS_990PF: List[str] = [
    "org_id",
    "Cash Boy Amt",
    "Cash Eoy Amt",
    "input_query",
    "org_name",
    "org_url",
    "city",
    "state",
    "filing_type",
    "xml_url",
    "OrganizationName",
    "EmployerIdentificationNumber",
    "Address",
    "ZipCode",
    "TaxPeriod",
    "AccountingEndMonth",
    "TaxExemptSince",
    "AssetsLatest",
    "RevenuesLatest",
    "IncomeLatest",
    "Tot Net Ast Or Fund Balances Boy Amt",
    "Tot Net Ast Or Fund Balances Eoy Amt",
    "Fmv Assets Eoy Amt",
    "Additional Paid In Capital Boy Amt",
    "Additional Paid In Capital Eoy Amt",
    "Capital Stock Boy Amt",
    "Capital Stock Eoy Amt",
    "Corporate Stock Boy Amt",
    "Corporate Stock Eoy Amt",
    "Other Assets Boy Amt",
    "Other Assets Eoy Amt",
    "Retained Earning Boy Amt",
    "Retained Earning Eoy Amt",
    "Sav And Temp Cash Invst Boy Amt",
    "Sav And Temp Cash Invst Eoy Amt",
    "Tot Net Ast Or Fund Balances Boy Amt 2",
    "Tot Net Ast Or Fund Balances Eoy Amt 2",
    "Total Assets Boy Amt",
    "Total Assets Eoy Amt",
    "Total Liabilities Boy Amt",
    "Total Liabilities Eoy Amt",
    "Total Liabilities Net Ast Boy Amt",
    "Total Liabilities Net Ast Eoy Amt",
    "Us Government Obligations Boy Amt",
    "Us Government Obligations Eoy Amt",
    "Eoy Book Value Amt",
    "State Local Sec Book Vl Eoy Amt",
    "Us Govt Obligations Book Vl Eoy Amt",
    "Acct Rcvbl Boy Amt",
    "Acct Rcvbl Eoy Amt",
    "Other Investments Boy Amt",
    "Other Investments Eoy Amt",
    "Boy Book Value Amt",
    "Eoy Book Value Amt 2",
    "Land Bldg Investments Boy Amt",
    "Land Bldg Investments Eoy Amt",
    "Prepaid Expenses Boy Amt",
    "Prepaid Expenses Eoy Amt",
    "Accounts Payable Boy Amt",
    "Accounts Payable Eoy Amt",
    "Land Boy Amt",
    "Land Eoy Amt",
    "No Donor Rstr Net Assests Boy Amt",
    "No Donor Rstr Net Assests Eoy Amt",
    "Corporate Bonds Boy Amt",
    "Corporate Bonds Eoy Amt",
    "Other Liabilities Boy Amt",
    "Other Liabilities Eoy Amt",
    "Inventories Boy Amt",
    "Inventories Eoy Amt",
    "Grants Payable Boy Amt",
    "Grants Payable Eoy Amt",
    "Other Nts And Loans Rcvbl Boy Amt",
    "Other Nts And Loans Rcvbl Eoy Amt",
    "Eoy Book Value Amt 3",
    "Boy Book Value Amt 2",
    "Eoy Book Value Amt 4",
    "Loans From Officers Eoy Amt",
    "Mortgages And Notes Payable Eoy Amt",
    "Rcvbl From Officers Eoy Amt",
    "Deferred Revenue Boy Amt",
    "Deferred Revenue Eoy Amt",
    "Donor Rstr Net Assets Boy Amt",
    "Donor Rstr Net Assets Eoy Amt",
    "Loans From Officers Boy Amt",
    "Pledges Rcvbl Boy Amt",
    "Pledges Rcvbl Eoy Amt",
    "Rcvbl From Officers Boy Amt",
    "Mortgages And Notes Payable Boy Amt",
    "Grants Receivable Boy Amt",
    "Grants Receivable Eoy Amt",
    "Mortgage Loans Boy Amt",
    "Mortgage Loans Eoy Amt",
    "Unrestricted Boy Amt",
    "Unrestricted Eoy Amt",
    "Permanently Restricted Boy Amt",
    "Permanently Restricted Eoy Amt",
    "Temporarily Restricted Boy Amt",
    "Temporarily Restricted Eoy Amt",
    "_EXTRA_FIELDS",
]

# 990 Columns - Same core structure, but uses different cash column names
COLUMNS_990: List[str] = [
    "input_query",
    "org_name",
    "org_url",
    "org_id",
    "city",
    "state",
    "filing_type",
    "xml_url",
    "OrganizationName",
    "EmployerIdentificationNumber",
    "Address",
    "ZipCode",
    "TaxPeriod",
    "AccountingEndMonth",
    "TaxExemptSince",
    "AssetsLatest",
    "RevenuesLatest",
    "IncomeLatest",
    "Cash Non Interest Bearing Group - BOY Amount",
    "Cash Non Interest Bearing Group - EOY Amount",
    "Tot Net Ast Or Fund Balances Boy Amt",
    "Tot Net Ast Or Fund Balances Eoy Amt",
    "Fmv Assets Eoy Amt",
    "Additional Paid In Capital Boy Amt",
    "Additional Paid In Capital Eoy Amt",
    "Capital Stock Boy Amt",
    "Capital Stock Eoy Amt",
    "Corporate Stock Boy Amt",
    "Corporate Stock Eoy Amt",
    "Other Assets Boy Amt",
    "Other Assets Eoy Amt",
    "Retained Earning Boy Amt",
    "Retained Earning Eoy Amt",
    "Sav And Temp Cash Invst Boy Amt",
    "Sav And Temp Cash Invst Eoy Amt",
    "Tot Net Ast Or Fund Balances Boy Amt 2",
    "Tot Net Ast Or Fund Balances Eoy Amt 2",
    "Total Assets Boy Amt",
    "Total Assets Eoy Amt",
    "Total Liabilities Boy Amt",
    "Total Liabilities Eoy Amt",
    "Total Liabilities Net Ast Boy Amt",
    "Total Liabilities Net Ast Eoy Amt",
    "Us Government Obligations Boy Amt",
    "Us Government Obligations Eoy Amt",
    "Eoy Book Value Amt",
    "State Local Sec Book Vl Eoy Amt",
    "Us Govt Obligations Book Vl Eoy Amt",
    "Acct Rcvbl Boy Amt",
    "Acct Rcvbl Eoy Amt",
    "Other Investments Boy Amt",
    "Other Investments Eoy Amt",
    "Boy Book Value Amt",
    "Eoy Book Value Amt 2",
    "Land Bldg Investments Boy Amt",
    "Land Bldg Investments Eoy Amt",
    "Prepaid Expenses Boy Amt",
    "Prepaid Expenses Eoy Amt",
    "Accounts Payable Boy Amt",
    "Accounts Payable Eoy Amt",
    "Land Boy Amt",
    "Land Eoy Amt",
    "No Donor Rstr Net Assests Boy Amt",
    "No Donor Rstr Net Assests Eoy Amt",
    "Corporate Bonds Boy Amt",
    "Corporate Bonds Eoy Amt",
    "Other Liabilities Boy Amt",
    "Other Liabilities Eoy Amt",
    "Inventories Boy Amt",
    "Inventories Eoy Amt",
    "Grants Payable Boy Amt",
    "Grants Payable Eoy Amt",
    "Other Nts And Loans Rcvbl Boy Amt",
    "Other Nts And Loans Rcvbl Eoy Amt",
    "Eoy Book Value Amt 3",
    "Boy Book Value Amt 2",
    "Eoy Book Value Amt 4",
    "Loans From Officers Eoy Amt",
    "Mortgages And Notes Payable Eoy Amt",
    "Rcvbl From Officers Eoy Amt",
    "Deferred Revenue Boy Amt",
    "Deferred Revenue Eoy Amt",
    "Donor Rstr Net Assets Boy Amt",
    "Donor Rstr Net Assets Eoy Amt",
    "Loans From Officers Boy Amt",
    "Pledges Rcvbl Boy Amt",
    "Pledges Rcvbl Eoy Amt",
    "Rcvbl From Officers Boy Amt",
    "Mortgages And Notes Payable Boy Amt",
    "Grants Receivable Boy Amt",
    "Grants Receivable Eoy Amt",
    "Mortgage Loans Boy Amt",
    "Mortgage Loans Eoy Amt",
    "Unrestricted Boy Amt",
    "Unrestricted Eoy Amt",
    "Permanently Restricted Boy Amt",
    "Permanently Restricted Eoy Amt",
    "Temporarily Restricted Boy Amt",
    "Temporarily Restricted Eoy Amt",
    "_EXTRA_FIELDS",
]


def get_columns_for_type(file_type: str) -> List[str]:
    """Return the column list for the specified type."""
    if file_type.lower() in ("990pf", "990-pf"):
        return COLUMNS_990PF
    elif file_type.lower() == "990":
        return COLUMNS_990
    else:
        raise ValueError(f"Unknown file type: {file_type}. Use '990' or '990pf'.")


def select_columns(
    input_path: Path,
    output_path: Path,
    target_columns: List[str],
    *,
    fill_missing: str = "-",
    chunksize: int = 10_000,
) -> int:
    """
    Read input CSV and write output with only target columns.
    
    - Columns in target_columns but missing from input get filled with `fill_missing`.
    - ALL rows are preserved (no filtering).
    - Returns number of rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows_written = 0
    header_written = False
    
    # Read header to see what columns exist
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        input_header = next(reader)
    
    input_cols_set = set(input_header)
    logging.info(f"Input has {len(input_header)} columns, target has {len(target_columns)} columns")
    
    # Process in chunks using pandas for efficiency
    try:
        import pandas as pd
        
        # Only read columns that exist
        usecols = [c for c in target_columns if c in input_cols_set]
        
        for chunk in pd.read_csv(
            input_path,
            chunksize=chunksize,
            usecols=usecols if usecols else None,
            low_memory=True,
            on_bad_lines="warn",
        ):
            # Reindex to get target column order, filling missing with placeholder
            chunk = chunk.reindex(columns=target_columns, fill_value=fill_missing)
            
            # Replace NaN with fill_missing
            chunk = chunk.fillna(fill_missing)
            
            chunk.to_csv(
                output_path,
                mode="a",
                index=False,
                header=not header_written,
            )
            header_written = True
            rows_written += len(chunk)
            
            if rows_written % 50_000 == 0:
                logging.info(f"Written {rows_written:,} rows...")
    
    except ImportError:
        # Fallback to pure CSV if pandas not available
        logging.warning("pandas not available, using pure CSV (slower)")
        
        col_indices = {}
        for i, col in enumerate(input_header):
            if col in target_columns:
                col_indices[col] = i
        
        with input_path.open("r", encoding="utf-8") as fin:
            reader = csv.reader(fin)
            next(reader)  # skip header
            
            with output_path.open("w", newline="", encoding="utf-8") as fout:
                writer = csv.writer(fout)
                writer.writerow(target_columns)
                
                for row in reader:
                    out_row = []
                    for col in target_columns:
                        if col in col_indices:
                            idx = col_indices[col]
                            val = row[idx] if idx < len(row) else fill_missing
                            out_row.append(val if val else fill_missing)
                        else:
                            out_row.append(fill_missing)
                    writer.writerow(out_row)
                    rows_written += 1
                    
                    if rows_written % 50_000 == 0:
                        logging.info(f"Written {rows_written:,} rows...")
    
    return rows_written


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    parser = argparse.ArgumentParser(
        description="Select specific columns from CSV to match reference structure. No row filtering."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--type", "-t",
        required=True,
        choices=["990", "990pf", "990-pf"],
        help="File type: 990 or 990pf (determines which columns to select)",
    )
    parser.add_argument(
        "--fill", "-f",
        default="-",
        help="Fill value for missing columns (default: '-')",
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return
    
    target_columns = get_columns_for_type(args.type)
    logging.info(f"Input: {input_path}")
    logging.info(f"Output: {output_path}")
    logging.info(f"Type: {args.type} ({len(target_columns)} columns)")
    
    rows = select_columns(input_path, output_path, target_columns, fill_missing=args.fill)
    
    logging.info(f"Done! Written {rows:,} rows to {output_path}")


if __name__ == "__main__":
    main()
