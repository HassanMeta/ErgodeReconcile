from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st


def validate_date_format_mmddyyyy(
    date_series: pd.Series, field_name: str
) -> tuple[bool, list[int]]:
    """
    Validate that dates are in MM/DD/YYYY format.

    Args:
        date_series: Series of date values (can be strings, datetime, etc.)
        field_name: Name of the field for error messages

    Returns:
        Tuple of (is_valid, list of invalid row indices)
    """
    invalid_rows = []

    # Convert to string for pattern matching
    date_str_series = date_series.astype(str)

    # Pattern for MM/DD/YYYY: 1-2 digits / 1-2 digits / 4 digits
    mmddyyyy_pattern = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")

    for idx, date_val in enumerate(date_str_series):
        # Skip NaN/null values (they'll be handled separately)
        if pd.isna(date_val) or date_val in ["nan", "None", "<NA>", "NaT", ""]:
            continue

        # Check if it matches MM/DD/YYYY pattern
        if not mmddyyyy_pattern.match(str(date_val).strip()):
            invalid_rows.append(idx)
            continue

        # Try to parse with MM/DD/YYYY format to ensure it's valid
        try:
            datetime.strptime(str(date_val).strip(), "%m/%d/%Y")
        except ValueError:
            invalid_rows.append(idx)

    return len(invalid_rows) == 0, invalid_rows


# PO Deduction Management Functions
def load_po_deductions() -> pd.DataFrame:
    """Load all PO deductions from storage"""
    deductions_path = Path("records/po_deductions")
    try:
        df = read_parquet_with_fallback(deductions_path)
        # Ensure required columns exist
        required_cols = [
            "PO_Number",
            "Deduction_Amount",
            "CC_Batch_ID",
            "Deduction_Date",
            "Reason",
            "Timestamp",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.NA
        return df
    except FileNotFoundError:
        return pd.DataFrame(
            columns=[
                "PO_Number",
                "Deduction_Amount",
                "CC_Batch_ID",
                "Deduction_Date",
                "Reason",
                "Timestamp",
            ]
        )
    except Exception:  # pylint: disable=broad-except
        return pd.DataFrame(
            columns=[
                "PO_Number",
                "Deduction_Amount",
                "CC_Batch_ID",
                "Deduction_Date",
                "Reason",
                "Timestamp",
            ]
        )


def get_po_available_balance(
    po_number: str, po_original_amount: float
) -> tuple[float, float]:
    """
    Calculate available balance for a PO after all deductions.

    Returns:
        tuple: (total_deductions, available_balance)
    """
    deductions_df = load_po_deductions()

    if deductions_df.empty:
        return 0.0, po_original_amount

    po_deductions = deductions_df[deductions_df["PO_Number"] == po_number]
    total_deductions = (
        po_deductions["Deduction_Amount"].sum() if not po_deductions.empty else 0.0
    )
    available_balance = po_original_amount - total_deductions

    return total_deductions, available_balance


def save_po_deduction(
    po_number: str,
    amount: float,
    po_original_amount: float,
    cc_batch_id: str,
    reason: str = "",
) -> float:
    """
    Save a new PO deduction with validation.

    Args:
        po_number: PO number to deduct from
        amount: Deduction amount
        po_original_amount: Original PO amount (for validation)
        cc_batch_id: CC Batch ID this deduction is associated with
        reason: Optional reason for the deduction

    Returns:
        float: New available balance after deduction

    Raises:
        ValueError: If insufficient balance or invalid amount
    """
    deductions_path = Path("records/po_deductions")

    # Validate amount
    if amount <= 0:
        raise ValueError("Deduction amount must be greater than 0")

    # Get current balance
    total_deductions, available_balance = get_po_available_balance(
        po_number, po_original_amount
    )

    # Check if sufficient balance
    if amount > available_balance:
        raise ValueError(
            f"Insufficient balance. Available: ${available_balance:.2f}, "
            f"Requested: ${amount:.2f}"
        )

    # Check if PO is already depleted
    if available_balance <= 0:
        raise ValueError(f"PO {po_number} is fully depleted. No deductions allowed.")

    # Create new deduction record
    existing = load_po_deductions()

    new_deduction = pd.DataFrame(
        {
            "PO_Number": [str(po_number)],
            "Deduction_Amount": [float(amount)],
            "CC_Batch_ID": [str(cc_batch_id)],
            "Deduction_Date": [pd.Timestamp.now().date()],
            "Reason": [str(reason) if reason else ""],
            "Timestamp": [pd.Timestamp.now()],
        }
    )

    combined = pd.concat([existing, new_deduction], ignore_index=True)

    # Ensure directory exists
    deductions_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the deduction
    try:
        write_parquet(combined, deductions_path)
    except Exception as e:
        raise ValueError(f"Failed to save deduction to file: {str(e)}")

    # Verify the save worked
    try:
        verify_df = load_po_deductions()
        if verify_df.empty or po_number not in verify_df["PO_Number"].values:
            raise ValueError("Deduction was not saved correctly - verification failed")
    except Exception as e:
        raise ValueError(f"Failed to verify deduction save: {str(e)}")

    # Return new available balance
    return available_balance - amount


# Parquet helper functions for fast data storage
def read_parquet_with_fallback(file_path: Path) -> pd.DataFrame:
    """
    Read data from Parquet file, with automatic CSV fallback and conversion.
    If Parquet doesn't exist but CSV does, reads CSV and converts to Parquet.

    Args:
        file_path: Path object (should be .parquet, but will check .csv as fallback)

    Returns:
        DataFrame with the data
    """
    parquet_path = file_path.with_suffix(".parquet")
    csv_path = file_path.with_suffix(".csv")

    # Try to read Parquet first
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path, engine="pyarrow")
        except Exception as exc:
            st.warning(
                f"Error reading Parquet file {parquet_path}: {exc}. Trying CSV fallback..."
            )

    # Fallback to CSV if Parquet doesn't exist
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # Convert to Parquet for future use
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path, engine="pyarrow", index=False)
            st.info(
                f"Converted {csv_path.name} to Parquet format for better performance."
            )
            return df
        except Exception as exc:
            st.error(f"Error reading CSV file {csv_path}: {exc}")
            raise

    # File doesn't exist
    raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} found")


def write_parquet(df: pd.DataFrame, file_path: Path) -> None:
    """
    Write DataFrame to Parquet format.

    Args:
        df: DataFrame to write
        file_path: Path object (will be saved as .parquet)
    """
    parquet_path = file_path.with_suffix(".parquet")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert columns to proper types to avoid type conversion errors
    df_to_write = df.copy()

    # Convert Payment_Terms to numeric if it exists
    if "Payment_Terms" in df_to_write.columns:
        df_to_write["Payment_Terms"] = pd.to_numeric(
            df_to_write["Payment_Terms"], errors="coerce"
        )

    # Convert string columns to ensure they're not mixed types (bool, int, etc.)
    string_columns = [
        "Vendor_Prefix",
        "Vendor_Name",
        "Category",
        "Dept",
        "PO_Number",
        "CC_Number",
        "CC_Description",
        "CC_Reference_ID",
        "Import_Batch_ID",
        "Reco_ID",
        "PREFIX",
        "VENDOR_NAME",
        "CATEGORY",
        "DEPT",
        "DESCRIPTION",
    ]
    for col in string_columns:
        if col in df_to_write.columns:
            # Convert to object type first to handle mixed types (bool, int, float, str)
            # Then convert to string, which will handle all types properly
            df_to_write[col] = df_to_write[col].astype(object)
            # Convert all values to string, preserving NaN
            df_to_write[col] = df_to_write[col].astype(str)
            # Clean up string representations of nulls and convert back to proper nulls
            null_strings = ["nan", "None", "<NA>", "NaT", ""]
            for null_str in null_strings:
                df_to_write[col] = df_to_write[col].replace(null_str, pd.NA)

    df_to_write.to_parquet(parquet_path, engine="pyarrow", index=False)

    # Optionally remove old CSV file if it exists
    csv_path = file_path.with_suffix(".csv")
    if csv_path.exists() and csv_path != parquet_path:
        try:
            csv_path.unlink()  # Remove old CSV file
        except Exception:
            pass  # Ignore errors when removing old CSV


def load_dept_mappings() -> pd.DataFrame:
    """Load saved Dept mappings for CC_Reference_IDs."""
    mappings_path = Path("records/common_dept_mappings.parquet")
    try:
        return read_parquet_with_fallback(mappings_path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["CC_Reference_ID", "Dept"])
    except Exception:  # pylint: disable=broad-except
        return pd.DataFrame(columns=["CC_Reference_ID", "Dept"])


def save_dept_mappings(ref_ids: list[str], dept: str) -> None:
    """Save Dept mapping for multiple CC_Reference_IDs."""
    if dept not in {"FBA", "FBM"} or not ref_ids:
        return
    mappings_path = Path("records/common_dept_mappings.parquet")
    try:
        existing_df = load_dept_mappings()

        # Normalize reference IDs
        ref_ids_normalized = [str(ref_id).strip() for ref_id in ref_ids]

        # Remove existing mappings for these reference IDs
        if not existing_df.empty and "CC_Reference_ID" in existing_df.columns:
            existing_df["CC_Reference_ID"] = (
                existing_df["CC_Reference_ID"].astype(str).str.strip()
            )
            existing_df = existing_df[
                ~existing_df["CC_Reference_ID"].isin(ref_ids_normalized)
            ]

        # Add new mappings
        new_rows = [
            {"CC_Reference_ID": ref_id, "Dept": dept.upper()}
            for ref_id in ref_ids_normalized
        ]
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            write_parquet(updated_df, mappings_path)
    except Exception:  # pylint: disable=broad-except
        pass  # Silently fail if can't save


def get_dept_mappings_for_common(common_df: pd.DataFrame) -> dict[str, str]:
    """
    Get saved Dept mappings for CC_Reference_IDs in common_df.
    Only considers CC_Reference_IDs that exist in the current common_df (batch-specific).
    Returns dict mapping normalized description to Dept (if all ref_ids in group agree).
    """
    if common_df.empty or "CC_Reference_ID" not in common_df.columns:
        return {}

    # Get all CC_Reference_IDs in current common_df (these are batch-specific)
    current_ref_ids = set(
        common_df["CC_Reference_ID"].dropna().astype(str).str.strip().tolist()
    )

    if not current_ref_ids:
        return {}

    saved_mappings_df = load_dept_mappings()
    if saved_mappings_df.empty:
        return {}

    # Filter to only mappings for CC_Reference_IDs in current common_df
    if "CC_Reference_ID" in saved_mappings_df.columns:
        saved_mappings_df["CC_Reference_ID"] = (
            saved_mappings_df["CC_Reference_ID"].astype(str).str.strip()
        )
        saved_mappings_df = saved_mappings_df[
            saved_mappings_df["CC_Reference_ID"].isin(current_ref_ids)
        ]
    else:
        return {}

    if saved_mappings_df.empty:
        return {}

    # Create mapping: CC_Reference_ID -> Dept
    ref_id_to_dept = dict(
        zip(
            saved_mappings_df["CC_Reference_ID"],
            saved_mappings_df["Dept"].astype(str).str.strip().str.upper(),
        )
    )

    # Get description column
    desc_column = next(
        (col for col in ("CC_Description", "Description") if col in common_df.columns),
        None,
    )
    if not desc_column:
        return {}

    # Group by normalized description and check if all CC_Reference_IDs have same Dept
    common_df_copy = common_df.copy()
    common_df_copy["_norm_desc"] = (
        common_df_copy[desc_column].astype(str).str.strip().str.upper()
    )
    common_df_copy["_ref_id"] = (
        common_df_copy["CC_Reference_ID"].astype(str).str.strip()
    )
    common_df_copy["_saved_dept"] = common_df_copy["_ref_id"].map(ref_id_to_dept)

    # For each description group, check if all have the same saved Dept
    desc_to_dept = {}
    for desc_norm, group in common_df_copy.groupby("_norm_desc"):
        saved_depts = group["_saved_dept"].dropna().unique()
        # If all reference IDs in this group have the same saved Dept, use it
        if len(saved_depts) == 1 and saved_depts[0] in {"FBA", "FBM"}:
            desc_to_dept[desc_norm] = saved_depts[0]
        # If most have the same Dept, use that (for cases where some weren't saved yet)
        elif len(saved_depts) > 0:
            dept_counts = group["_saved_dept"].value_counts()
            most_common = dept_counts.index[0] if not dept_counts.empty else None
            if most_common and most_common in {"FBA", "FBM"}:
                # Only use if majority (>= 50%) have this dept
                if dept_counts.iloc[0] / len(group) >= 0.5:
                    desc_to_dept[desc_norm] = most_common

    return desc_to_dept


# Custom CSS for modern dark theme UI
def apply_custom_css():
    """Apply modern dark theme styling to the entire application"""
    st.markdown(
        """
        <style>
        /* Force dark mode - override system theme */
        :root {
            --background-color: #0e1117 !important;
            --text-color: #fafafa !important;
        }

        /* Main theme - Dark mode */
        .stApp {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }

        /* Force dark mode on all Streamlit elements */
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117 !important;
        }

        /* Main headers */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #ff4b4b;
            text-align: left;
            padding: 1.5rem 0;
            margin-bottom: 2rem;
            border-bottom: 2px solid #262730;
        }

        /* Section headers */
        h1, h2, h3 {
            color: #fafafa !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1e1e1e;
            border-right: 1px solid #262730;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #fafafa;
        }

        /* Sidebar title */
        [data-testid="stSidebar"] h1 {
            color: #ff4b4b !important;
            text-align: center;
            padding-bottom: 1rem;
            border-bottom: 2px solid #262730;
        }

        /* Sidebar navigation section header */
        [data-testid="stSidebar"] h3 {
            color: #9ca3af !important;
            font-size: 0.875rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            margin-top: 1rem !important;
            margin-bottom: 0.75rem !important;
        }

        /* Navigation buttons styling */
        [data-testid="stSidebar"] .stButton {
            margin-bottom: 0.5rem;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            text-align: left !important;
            justify-content: flex-start !important;
            display: flex !important;
            align-items: center !important;
            font-weight: 500;
            font-size: 0.95rem;
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            transition: all 0.3s ease;
        }

        /* Force left alignment for button content */
        [data-testid="stSidebar"] .stButton > button > div,
        [data-testid="stSidebar"] .stButton > button > span {
            text-align: left !important;
            justify-content: flex-start !important;
            width: 100%;
        }

        /* Primary button (active/selected navigation) - More specific selectors */
        [data-testid="stSidebar"] .stButton > button[data-baseweb="button"][kind="primary"],
        [data-testid="stSidebar"] .stButton > button[kind="primary"],
        [data-testid="stSidebar"] button[data-baseweb="button"][kind="primary"] {
            background: linear-gradient(135deg, #ff4b4b 0%, #dc2626 100%) !important;
            background-color: #ff4b4b !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px rgba(255, 75, 75, 0.4) !important;
        }

        [data-testid="stSidebar"] .stButton > button[data-baseweb="button"][kind="primary"]:hover,
        [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
        [data-testid="stSidebar"] button[data-baseweb="button"][kind="primary"]:hover {
            background: linear-gradient(135deg, #ff6b6b 0%, #ff4b4b 100%) !important;
            background-color: #ff6b6b !important;
            transform: translateX(4px);
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.5) !important;
        }

        /* Secondary button (inactive navigation) - More specific selectors */
        [data-testid="stSidebar"] .stButton > button[data-baseweb="button"][kind="secondary"],
        [data-testid="stSidebar"] .stButton > button[kind="secondary"],
        [data-testid="stSidebar"] button[data-baseweb="button"][kind="secondary"] {
            background-color: #262730 !important;
            background: #262730 !important;
            color: #d1d5db !important;
            border: 1px solid #3a3a3a !important;
            font-weight: normal !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] .stButton > button[data-baseweb="button"][kind="secondary"]:hover,
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
        [data-testid="stSidebar"] button[data-baseweb="button"][kind="secondary"]:hover {
            background-color: #2d2d38 !important;
            background: #2d2d38 !important;
            color: #fafafa !important;
            border-color: #ff4b4b !important;
            transform: translateX(4px);
        }

        /* Ensure buttons respect their type attribute */
        [data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
            background-color: #262730 !important;
            background: #262730 !important;
        }

        /* Force secondary buttons to be dark */
        [data-testid="stSidebar"] button[kind="secondary"] {
            background-color: #262730 !important;
            background: #262730 !important;
            color: #d1d5db !important;
        }

        /* Force primary buttons to be red */
        [data-testid="stSidebar"] button[kind="primary"] {
            background: linear-gradient(135deg, #ff4b4b 0%, #dc2626 100%) !important;
            background-color: #ff4b4b !important;
            color: white !important;
        }

        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: bold;
            color: #ff4b4b;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #1e1e1e 0%, #262730 100%);
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid #3a3a3a;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }

        /* Buttons - General (only apply to main content, not sidebar) */
        section[data-testid="stAppViewContainer"] > div:first-child .stButton>button,
        .main .stButton>button {
            background: linear-gradient(135deg, #ff4b4b 0%, #dc2626 100%);
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1.5rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(255, 75, 75, 0.3);
        }

        section[data-testid="stAppViewContainer"] > div:first-child .stButton>button:hover,
        .main .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(255, 75, 75, 0.4);
            background: linear-gradient(135deg, #ff6b6b 0%, #ff4b4b 100%);
        }

        /* Form styling */
        .stForm {
            background-color: #1e1e1e;
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid #3a3a3a;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }

        /* Input fields */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stDateInput>div>div>input {
            background-color: #262730;
            color: #fafafa;
            border: 1px solid #3a3a3a;
            border-radius: 0.5rem;
        }

        .stTextInput>div>div>input:focus,
        .stNumberInput>div>div>input:focus {
            border-color: #ff4b4b;
            box-shadow: 0 0 0 1px #ff4b4b;
        }

        /* Select boxes */
        .stSelectbox>div>div>div,
        .stMultiSelect>div>div>div {
            background-color: #262730;
            color: #fafafa;
            border: 1px solid #3a3a3a;
            border-radius: 0.5rem;
        }

        /* Tabs - Using default Streamlit styling */

        /* Expander */
        .streamlit-expanderHeader {
            background-color: #262730;
            color: #fafafa;
            border: 1px solid #3a3a3a;
            border-radius: 0.5rem;
            font-weight: 600;
        }

        .streamlit-expanderHeader:hover {
            background-color: #2d2d38;
            border-color: #ff4b4b;
        }

        /* Data editor / DataFrame */
        .stDataFrame {
            background-color: #1e1e1e;
            border-radius: 0.5rem;
            border: 1px solid #3a3a3a;
        }

        /* File uploader */
        .uploadedFile,
        [data-testid="stFileUploader"] {
            background-color: #262730;
            border: 2px dashed #3a3a3a;
            border-radius: 0.75rem;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #ff4b4b;
        }

        /* Success/Info/Warning/Error messages */
        .stSuccess {
            background-color: #064e3b;
            border-left: 4px solid #10b981;
            color: #d1fae5;
        }

        .stInfo {
            background-color: #1e3a5f;
            border-left: 4px solid #3b82f6;
            color: #dbeafe;
        }

        .stWarning {
            background-color: #78350f;
            border-left: 4px solid #f59e0b;
            color: #fef3c7;
        }

        .stError {
            background-color: #7f1d1d;
            border-left: 4px solid #ef4444;
            color: #fee2e2;
        }

        /* Column configuration */
        .row-widget {
            background-color: #1e1e1e;
        }

        /* Divider */
        hr {
            border-color: #262730;
        }

        /* Tooltips */
        .stTooltipIcon {
            color: #9ca3af;
        }

        /* Checkbox */
        .stCheckbox {
            color: #fafafa;
        }

        /* Custom metric boxes */
        .metric-box {
            background: linear-gradient(135deg, #1e1e1e 0%, #262730 100%);
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid #3a3a3a;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            text-align: center;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #ff4b4b;
        }

        .metric-label {
            color: #9ca3af;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }

        /* Status flags */
        .green-flag {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);
        }

        .red-flag {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 2px 4px rgba(239, 68, 68, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_masters() -> None:
    st.markdown(
        '<div class="main-header">Masters Management</div>', unsafe_allow_html=True
    )
    st.markdown("*Manage vendor master data and description mappings*")

    tab1, tab2, tab3 = st.tabs(
        ["Vendor Master", "Description Mapping", "CC Reference ID Mappings"]
    )

    master_path = Path("data/master")
    mapping_path = Path("data/mappings")
    master_columns = [
        "PREFIX",
        "VENDOR_NAME",
        "CATEGORY",
        "DEPT",
        "PAYMENT TERMS",
        "CC FEE",
    ]
    category_options = ["ONLY FBA", "ONLY FBM", "COMMON", "NOT AVBL"]
    dept_options = ["FBA", "FBM"]
    payment_term_options = ["pre_payment", "Net 10", "Net 15", "Net 30"]

    with tab1:
        try:
            master_df = read_parquet_with_fallback(master_path)
        except FileNotFoundError:
            master_df = pd.DataFrame(columns=master_columns)
            st.warning(
                "Vendor master file not found. A new file will be created when you add an entry."
            )
        except Exception as exc:  # pylint: disable=broad-except
            master_df = pd.DataFrame(columns=master_columns)
            st.error(f"Unable to read vendor master file: {exc}")

        for column in master_columns:
            if column not in master_df.columns:
                master_df[column] = pd.NA
        master_df = master_df[master_columns]

        success_message = st.session_state.pop("vendor_master_success", None)
        if success_message:
            st.success(success_message)
        error_message = st.session_state.pop("vendor_master_error", None)
        if error_message:
            st.error(error_message)

        st.markdown("---")
        st.markdown("### Legacy Forms (for reference)")

        with st.expander("Add Vendor Master Entry (Legacy)", expanded=False):
            modal_error = st.session_state.pop("vendor_modal_error", None)
            if modal_error:
                st.error(modal_error)

            with st.form("vendor_master_form", clear_on_submit=True):
                prefix_input = st.text_input(
                    "Vendor Prefix",
                    max_chars=20,
                    help="Enter the vendor prefix (will be stored in uppercase).",
                )
                vendor_name_input = st.text_input(
                    "Vendor Name",
                    max_chars=100,
                    help="Enter the full vendor name.",
                )
                category_choice = st.selectbox(
                    "Category",
                    options=category_options,
                    help="Select the category for this vendor.",
                )
                dept_choice = st.selectbox(
                    "Dept",
                    options=dept_options,
                    help="Select the department (FBA or FBM).",
                )
                payment_choice = st.selectbox(
                    "Payment Terms",
                    options=payment_term_options,
                    help="Select a supported payment term.",
                )
                submitted = st.form_submit_button("Save Vendor", type="primary")

            if submitted:
                prefix_value = (prefix_input or "").strip().upper()
                validation_errors: list[str] = []
                if not prefix_value:
                    validation_errors.append("Vendor Prefix is required.")
                if prefix_value and any(
                    not char.isalnum() for char in prefix_value if not char.isspace()
                ):
                    validation_errors.append(
                        "Vendor Prefix should contain only letters, numbers, or spaces."
                    )

                duplicate_mask = (
                    (master_df["PREFIX"].astype(str).str.upper() == prefix_value)
                    & (
                        master_df["VENDOR_NAME"].astype(str).str.strip()
                        == (vendor_name_input or "").strip()
                    )
                    & (master_df["CATEGORY"].astype(str) == category_choice)
                    & (master_df["DEPT"].astype(str) == dept_choice)
                    & (master_df["PAYMENT TERMS"].astype(str) == payment_choice)
                )
                if prefix_value and duplicate_mask.any():
                    validation_errors.append(
                        "An identical vendor master entry already exists."
                    )

                if validation_errors:
                    st.session_state["vendor_modal_error"] = "\n".join(
                        validation_errors
                    )
                else:
                    new_entry = pd.DataFrame(
                        {
                            "PREFIX": [prefix_value],
                            "VENDOR_NAME": [(vendor_name_input or "").strip()],
                            "CATEGORY": [category_choice],
                            "DEPT": [dept_choice],
                            "PAYMENT TERMS": [payment_choice],
                        }
                    )
                    updated_master = pd.concat(
                        [master_df, new_entry], ignore_index=True
                    ).fillna("")
                    try:
                        write_parquet(updated_master, master_path)
                    except Exception as exc:  # pylint: disable=broad-except
                        st.session_state["vendor_modal_error"] = (
                            f"Unable to write to vendor master file: {exc}"
                        )
                    else:
                        st.session_state["vendor_master_success"] = (
                            f"Vendor prefix {prefix_value} added to master."
                        )
                if hasattr(st, "rerun"):
                    st.rerun()
                else:  # pragma: no cover
                    st.experimental_rerun()

        with st.expander("Edit Vendor Master Entry", expanded=False):
            edit_error = st.session_state.pop("vendor_edit_error", None)
            if edit_error:
                st.error(edit_error)

            if master_df.empty:
                st.info(
                    "No vendor entries available to edit. Please add a vendor first."
                )
            else:
                # Create dropdown options with PREFIX and VENDOR_NAME
                master_df_display = master_df.copy()
                master_df_display["PREFIX"] = (
                    master_df_display["PREFIX"].astype(str).str.strip()
                )
                master_df_display["VENDOR_NAME"] = (
                    master_df_display["VENDOR_NAME"].astype(str).str.strip()
                )

                vendor_options = []
                vendor_lookup = {}
                for idx, row in master_df_display.iterrows():
                    prefix = str(row["PREFIX"]).strip()
                    vendor_name = str(row["VENDOR_NAME"]).strip()
                    if prefix:
                        display_text = (
                            f"{prefix} - {vendor_name}" if vendor_name else prefix
                        )
                        vendor_options.append(display_text)
                        vendor_lookup[display_text] = idx

                if not vendor_options:
                    st.info("No valid vendor entries available to edit.")
                else:
                    selected_vendor = st.selectbox(
                        "Select Vendor to Edit",
                        options=["Select a vendor"] + sorted(vendor_options),
                        index=0,
                        key="edit_vendor_select",
                    )

                    if selected_vendor != "Select a vendor":
                        selected_idx = vendor_lookup[selected_vendor]
                        selected_row = master_df_display.loc[selected_idx]

                        with st.form("vendor_edit_form"):
                            edit_prefix_input = st.text_input(
                                "Vendor Prefix",
                                value=str(selected_row["PREFIX"]).strip(),
                                max_chars=20,
                                help="Enter the vendor prefix (will be stored in uppercase).",
                                key="edit_prefix",
                            )
                            edit_vendor_name_input = st.text_input(
                                "Vendor Name",
                                value=str(selected_row["VENDOR_NAME"]).strip(),
                                max_chars=100,
                                help="Enter the full vendor name.",
                                key="edit_vendor_name",
                            )
                            current_category = str(selected_row["CATEGORY"]).strip()
                            category_index = (
                                category_options.index(current_category)
                                if current_category in category_options
                                else 0
                            )
                            edit_category_choice = st.selectbox(
                                "Category",
                                options=category_options,
                                index=category_index,
                                help="Select the category for this vendor.",
                                key="edit_category",
                            )
                            current_dept = str(selected_row["DEPT"]).strip()
                            dept_index = (
                                dept_options.index(current_dept)
                                if current_dept in dept_options
                                else 0
                            )
                            edit_dept_choice = st.selectbox(
                                "Dept",
                                options=dept_options,
                                index=dept_index,
                                help="Select the department (FBA or FBM).",
                                key="edit_dept",
                            )
                            current_payment = str(selected_row["PAYMENT TERMS"]).strip()
                            payment_index = (
                                payment_term_options.index(current_payment)
                                if current_payment in payment_term_options
                                else 0
                            )
                            edit_payment_choice = st.selectbox(
                                "Payment Terms",
                                options=payment_term_options,
                                index=payment_index,
                                help="Select a supported payment term.",
                                key="edit_payment",
                            )
                            edit_submitted = st.form_submit_button(
                                "Update Vendor", type="primary"
                            )

                        if edit_submitted:
                            edit_prefix_value = (
                                (edit_prefix_input or "").strip().upper()
                            )
                            edit_validation_errors: list[str] = []

                            if not edit_prefix_value:
                                edit_validation_errors.append(
                                    "Vendor Prefix is required."
                                )
                            if edit_prefix_value and any(
                                not char.isalnum()
                                for char in edit_prefix_value
                                if not char.isspace()
                            ):
                                edit_validation_errors.append(
                                    "Vendor Prefix should contain only letters, numbers, or spaces."
                                )

                            # Check for duplicates (excluding the current row being edited)
                            if edit_prefix_value:
                                other_rows = master_df_display[
                                    master_df_display.index != selected_idx
                                ]
                                duplicate_mask = (
                                    (
                                        other_rows["PREFIX"].astype(str).str.upper()
                                        == edit_prefix_value
                                    )
                                    & (
                                        other_rows["VENDOR_NAME"]
                                        .astype(str)
                                        .str.strip()
                                        == (edit_vendor_name_input or "").strip()
                                    )
                                    & (
                                        other_rows["CATEGORY"].astype(str)
                                        == edit_category_choice
                                    )
                                    & (
                                        other_rows["DEPT"].astype(str)
                                        == edit_dept_choice
                                    )
                                    & (
                                        other_rows["PAYMENT TERMS"].astype(str)
                                        == edit_payment_choice
                                    )
                                )
                                if duplicate_mask.any():
                                    edit_validation_errors.append(
                                        "An identical vendor master entry already exists."
                                    )

                            if edit_validation_errors:
                                st.session_state["vendor_edit_error"] = "\n".join(
                                    edit_validation_errors
                                )
                            else:
                                # Update the row
                                master_df.loc[selected_idx, "PREFIX"] = (
                                    edit_prefix_value
                                )
                                master_df.loc[selected_idx, "VENDOR_NAME"] = (
                                    edit_vendor_name_input or ""
                                ).strip()
                                master_df.loc[selected_idx, "CATEGORY"] = (
                                    edit_category_choice
                                )
                                master_df.loc[selected_idx, "DEPT"] = edit_dept_choice
                                master_df.loc[selected_idx, "PAYMENT TERMS"] = (
                                    edit_payment_choice
                                )

                                try:
                                    write_parquet(master_df, master_path)
                                except Exception as exc:  # pylint: disable=broad-except
                                    st.session_state["vendor_edit_error"] = (
                                        f"Unable to write to vendor master file: {exc}"
                                    )
                                else:
                                    st.session_state["vendor_master_success"] = (
                                        f"Vendor {edit_prefix_value} updated successfully."
                                    )
                            if hasattr(st, "rerun"):
                                st.rerun()
                            else:  # pragma: no cover
                                st.experimental_rerun()

        st.markdown("### Existing Vendor Master Entries")
        st.caption(
            "View and edit Payment Terms and CC Fee for existing vendors. "
            "Other fields are read-only."
        )

        # Prepare dataframe for editing
        edit_df = master_df.copy()

        # Convert CC FEE to numeric if it exists
        if "CC FEE" in edit_df.columns:
            edit_df["CC FEE"] = pd.to_numeric(
                edit_df["CC FEE"], errors="coerce"
            ).fillna(0.0)
        else:
            edit_df["CC FEE"] = 0.0

        with st.form("vendor_master_edit_form"):
            edited_df = st.data_editor(
                edit_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "PREFIX": st.column_config.TextColumn(
                        "Vendor Prefix",
                        help="Vendor prefix code (read-only)",
                        disabled=True,
                    ),
                    "VENDOR_NAME": st.column_config.TextColumn(
                        "Vendor Name",
                        help="Full vendor name (read-only)",
                        disabled=True,
                    ),
                    "CATEGORY": st.column_config.TextColumn(
                        "Category",
                        help="Vendor category (read-only)",
                        disabled=True,
                    ),
                    "DEPT": st.column_config.TextColumn(
                        "Department",
                        help="Department (FBA or FBM) (read-only)",
                        disabled=True,
                    ),
                    "PAYMENT TERMS": st.column_config.SelectboxColumn(
                        "Payment Terms",
                        options=payment_term_options,
                        help="Select payment terms (editable)",
                        required=True,
                    ),
                    "CC FEE": st.column_config.NumberColumn(
                        "CC Fee",
                        help="Credit card fee rate (e.g., 0.03 for 3%) (editable)",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.001,
                        format="%.4f",
                    ),
                },
                key="vendor_master_editor",
            )

            save_button = st.form_submit_button(
                "💾 Save Changes", type="primary", use_container_width=True
            )

        if save_button:
            try:
                # Ensure CC FEE is numeric
                if "CC FEE" in edited_df.columns:
                    edited_df["CC FEE"] = pd.to_numeric(
                        edited_df["CC FEE"], errors="coerce"
                    ).fillna(0.0)

                write_parquet(edited_df, master_path)
                st.session_state["vendor_master_success"] = (
                    f"✅ Successfully updated {len(edited_df)} vendor master entries!"
                )
            except Exception as exc:
                st.session_state["vendor_master_error"] = (
                    f"Unable to save vendor master file: {exc}"
                )

            if hasattr(st, "rerun"):
                st.rerun()
            else:  # pragma: no cover
                st.experimental_rerun()

    with tab2:
        try:
            mapping_df = read_parquet_with_fallback(mapping_path)
        except FileNotFoundError:
            mapping_df = pd.DataFrame()
            st.warning(
                "Vendor mapping file not found. Upload the file to populate this view."
            )
        except Exception as exc:  # pylint: disable=broad-except
            mapping_df = pd.DataFrame()
            st.error(f"Unable to read vendor mapping file: {exc}")
        st.dataframe(mapping_df, hide_index=True, use_container_width=True)

    with tab3:
        st.markdown("### CC Reference ID to Dept Mappings")
        st.caption(
            "View and manage Dept (FBA/FBM) mappings for CC Reference IDs. "
            "These mappings are created when you select FBA or FBM in the Common tab."
        )
        try:
            dept_mappings_df = load_dept_mappings()
            if dept_mappings_df.empty:
                st.info(
                    "No CC Reference ID mappings found. Mappings will appear here after you select FBA/FBM in the Common tab."
                )
            else:
                # Display summary statistics
                total_mappings = len(dept_mappings_df)
                fba_count = len(dept_mappings_df[dept_mappings_df["Dept"] == "FBA"])
                fbm_count = len(dept_mappings_df[dept_mappings_df["Dept"] == "FBM"])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Mappings", total_mappings)
                with col2:
                    st.metric("FBA Mappings", fba_count)
                with col3:
                    st.metric("FBM Mappings", fbm_count)

                st.markdown("---")

                # Add filters
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    dept_filter = st.selectbox(
                        "Filter by Dept",
                        options=["<All>", "FBA", "FBM"],
                        key="dept_mapping_filter",
                    )
                with filter_col2:
                    search_ref_id = st.text_input(
                        "Search CC Reference ID",
                        key="search_ref_id",
                        placeholder="Enter CC Reference ID to search...",
                    )

                # Apply filters
                filtered_df = dept_mappings_df.copy()
                if dept_filter != "<All>":
                    filtered_df = filtered_df[filtered_df["Dept"] == dept_filter]
                if search_ref_id:
                    search_term = str(search_ref_id).strip().upper()
                    filtered_df = filtered_df[
                        filtered_df["CC_Reference_ID"]
                        .astype(str)
                        .str.upper()
                        .str.contains(search_term, na=False)
                    ]

                if filtered_df.empty:
                    st.info("No mappings match the current filters.")
                else:
                    st.markdown(
                        f"**Showing {len(filtered_df)} of {total_mappings} mappings**"
                    )

                    # Add a checkbox column for deletion selection
                    display_df = filtered_df.copy()
                    display_df["Delete"] = False

                    # Use data editor for editing and deletion
                    with st.form("edit_dept_mappings_form", clear_on_submit=False):
                        edited_df = st.data_editor(
                            display_df,
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "CC_Reference_ID": st.column_config.TextColumn(
                                    "CC Reference ID",
                                    help="Unique identifier for the credit card transaction",
                                    disabled=True,
                                ),
                                "Dept": st.column_config.SelectboxColumn(
                                    "Department",
                                    options=["FBA", "FBM"],
                                    help="Department assignment (FBA or FBM)",
                                    required=True,
                                ),
                                "Delete": st.column_config.CheckboxColumn(
                                    "Delete",
                                    help="Check to mark this mapping for deletion",
                                    default=False,
                                ),
                            },
                            key="dept_mappings_editor",
                        )

                        col1, col2, col3 = st.columns([1, 1, 2])
                        with col1:
                            save_changes = st.form_submit_button(
                                "Save Changes", type="primary", use_container_width=True
                            )
                        with col2:
                            delete_selected = st.form_submit_button(
                                "Delete Selected", use_container_width=True
                            )
                        with col3:
                            export_csv = st.form_submit_button(
                                "Export to CSV", use_container_width=True
                            )

                    # Handle save changes
                    if save_changes:
                        try:
                            # Get rows that were modified (Dept changed)
                            edited_df["Dept"] = (
                                edited_df["Dept"]
                                .fillna("")
                                .astype(str)
                                .str.strip()
                                .str.upper()
                            )

                            # Update mappings for changed rows
                            changes_made = False
                            for _, row in edited_df.iterrows():
                                ref_id = str(row["CC_Reference_ID"]).strip()
                                new_dept = row["Dept"]

                                if new_dept in {"FBA", "FBM"}:
                                    # Check if this is a change
                                    original_row = filtered_df[
                                        filtered_df["CC_Reference_ID"]
                                        .astype(str)
                                        .str.strip()
                                        == ref_id
                                    ]
                                    if not original_row.empty:
                                        original_dept = (
                                            str(original_row.iloc[0]["Dept"])
                                            .strip()
                                            .upper()
                                        )
                                        if new_dept != original_dept:
                                            # Update the mapping
                                            save_dept_mappings([ref_id], new_dept)
                                            changes_made = True

                            if changes_made:
                                st.success("Changes saved successfully!")
                                if hasattr(st, "rerun"):
                                    st.rerun()
                                else:  # pragma: no cover
                                    st.experimental_rerun()
                            else:
                                st.info("No changes detected.")
                        except Exception as exc:  # pylint: disable=broad-except
                            st.error(f"Error saving changes: {exc}")

                    # Handle delete selected
                    if delete_selected:
                        try:
                            rows_to_delete = edited_df[edited_df["Delete"]]
                            if rows_to_delete.empty:
                                st.warning("No rows selected for deletion.")
                            else:
                                # Get all current mappings
                                all_mappings_df = load_dept_mappings()

                                # Get reference IDs to delete
                                ref_ids_to_delete = set(
                                    rows_to_delete["CC_Reference_ID"]
                                    .astype(str)
                                    .str.strip()
                                    .tolist()
                                )

                                # Remove deleted mappings
                                if not all_mappings_df.empty:
                                    all_mappings_df["CC_Reference_ID"] = (
                                        all_mappings_df["CC_Reference_ID"]
                                        .astype(str)
                                        .str.strip()
                                    )
                                    updated_mappings_df = all_mappings_df[
                                        ~all_mappings_df["CC_Reference_ID"].isin(
                                            ref_ids_to_delete
                                        )
                                    ]

                                    # Save updated mappings
                                    mappings_path = Path(
                                        "records/common_dept_mappings.parquet"
                                    )
                                    write_parquet(updated_mappings_df, mappings_path)

                                    st.success(
                                        f"Successfully deleted {len(ref_ids_to_delete)} mapping(s)."
                                    )
                                    if hasattr(st, "rerun"):
                                        st.rerun()
                                    else:  # pragma: no cover
                                        st.experimental_rerun()
                        except Exception as exc:  # pylint: disable=broad-except
                            st.error(f"Error deleting mappings: {exc}")

                    # Handle export
                    if export_csv:
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"cc_reference_id_mappings_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_dept_mappings",
                        )
        except FileNotFoundError:
            st.info(
                "No CC Reference ID mappings found. Mappings will appear here after you select FBA/FBM in the Common tab."
            )
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to load CC Reference ID mappings: {exc}")


def run_reconciliation(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run reconciliation on a batch of CC transactions.
    Returns the reconciled dataframe without saving to file.

    Args:
        batch_df: DataFrame with CC transactions from a batch

    Returns:
        DataFrame with reconciliation results
    """
    try:
        mappings_df = read_parquet_with_fallback(Path("data/mappings"))
        master_df = read_parquet_with_fallback(Path("data/master"))
    except Exception as exc:  # pylint: disable=broad-except
        raise Exception(f"Unable to load mapping data: {exc}") from exc

    mappings_df["DESCRIPTION"] = (
        mappings_df["DESCRIPTION"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)  # Normalize multiple spaces
        .str.strip()
    )
    mappings_df["_norm_description"] = mappings_df["DESCRIPTION"].str.upper()

    batch_df["_norm_description"] = (
        batch_df["CC_Description"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)  # Normalize multiple spaces
        .str.strip()
        .str.upper()
    )

    reco_df = batch_df.merge(
        mappings_df[["PREFIX", "_norm_description"]],
        on="_norm_description",
        how="left",
    )

    master_trim = (
        master_df.dropna(subset=["PREFIX"])
        .drop_duplicates(subset=["PREFIX"])
        .set_index("PREFIX")
    )
    prefix_category_map = (
        master_trim["CATEGORY"].to_dict() if "CATEGORY" in master_trim.columns else {}
    )
    prefix_dept_map = (
        master_trim["DEPT"].to_dict() if "DEPT" in master_trim.columns else {}
    )
    prefix_terms_map = {}
    if "PAYMENT TERMS" in master_trim.columns:
        # Convert to numeric before creating dict to ensure proper type
        terms_series = pd.to_numeric(
            master_trim["PAYMENT TERMS"], errors="coerce"
        ).fillna(0)
        prefix_terms_map = terms_series.to_dict()
    prefix_vendor_name_map = (
        master_trim["VENDOR_NAME"].to_dict()
        if "VENDOR_NAME" in master_trim.columns
        else {}
    )

    reco_df["Vendor_Prefix"] = reco_df["PREFIX"].fillna("")
    reco_df["Vendor_Name"] = reco_df["Vendor_Prefix"].map(prefix_vendor_name_map)
    reco_df["Category"] = reco_df["Vendor_Prefix"].map(prefix_category_map)
    reco_df.loc[reco_df["Vendor_Prefix"] == "", "Category"] = "Unmapped"
    reco_df["Category"] = reco_df["Category"].fillna("Unmapped")

    reco_df["Dept"] = reco_df["Vendor_Prefix"].map(prefix_dept_map)
    reco_df["Payment_Terms"] = (
        reco_df["Vendor_Prefix"]
        .map(prefix_terms_map)
        .apply(lambda x: pd.to_numeric(x, errors="coerce") if pd.notna(x) else 0)
    )
    reco_df.loc[
        reco_df["Vendor_Prefix"] == "",
        ["Dept", "Payment_Terms", "Vendor_Name"],
    ] = pd.NA

    # Check for saved Dept mappings and update Category from COMMON to ONLY FBA/FBM
    if "CC_Reference_ID" in reco_df.columns:
        try:
            saved_mappings_df = load_dept_mappings()
            if (
                not saved_mappings_df.empty
                and "CC_Reference_ID" in saved_mappings_df.columns
            ):
                # Create mapping: CC_Reference_ID -> Dept
                saved_mappings_df["CC_Reference_ID"] = (
                    saved_mappings_df["CC_Reference_ID"].astype(str).str.strip()
                )
                ref_id_to_dept = dict(
                    zip(
                        saved_mappings_df["CC_Reference_ID"],
                        saved_mappings_df["Dept"].astype(str).str.strip().str.upper(),
                    )
                )

                # Update Category for transactions with saved Dept mappings
                reco_df["CC_Reference_ID"] = (
                    reco_df["CC_Reference_ID"].astype(str).str.strip()
                )
                saved_dept_mask = reco_df["CC_Reference_ID"].isin(ref_id_to_dept.keys())

                # Only update if current category is COMMON
                common_mask = reco_df["Category"].astype(str).str.upper() == "COMMON"
                update_mask = saved_dept_mask & common_mask

                if update_mask.any():
                    # Map saved Dept to Category
                    saved_depts = reco_df.loc[update_mask, "CC_Reference_ID"].map(
                        ref_id_to_dept
                    )
                    reco_df.loc[update_mask, "Category"] = saved_depts.apply(
                        lambda dept: "ONLY FBA" if dept == "FBA" else "ONLY FBM"
                    )
                    # Also update Dept column with saved mapping
                    reco_df.loc[update_mask, "Dept"] = saved_depts

                    # Update Payment_Terms based on saved Dept and Vendor_Prefix
                    for idx in reco_df.loc[update_mask].index:
                        vendor_prefix = reco_df.loc[idx, "Vendor_Prefix"]
                        saved_dept = saved_depts.loc[idx]
                        if vendor_prefix and pd.notna(vendor_prefix):
                            # Look up Payment_Terms from master for this vendor and dept
                            matched_rows = master_df[
                                (
                                    master_df.get("PREFIX", "").astype(str).str.strip()
                                    == str(vendor_prefix).strip()
                                )
                                & (
                                    master_df.get("DEPT", "")
                                    .astype(str)
                                    .str.strip()
                                    .str.upper()
                                    == saved_dept
                                )
                            ]
                            if not matched_rows.empty:
                                payment_terms = matched_rows.iloc[0].get(
                                    "PAYMENT TERMS", pd.NA
                                )
                                if pd.notna(payment_terms):
                                    payment_terms_numeric = pd.to_numeric(
                                        payment_terms, errors="coerce"
                                    )
                                    if pd.notna(payment_terms_numeric):
                                        reco_df.loc[idx, "Payment_Terms"] = (
                                            payment_terms_numeric
                                        )
        except Exception:  # pylint: disable=broad-except
            # Silently fail if can't load mappings - reconciliation should still work
            pass

    return reco_df


def render_po_data() -> None:
    st.markdown(
        '<div class="main-header">Purchase Order Management</div>',
        unsafe_allow_html=True,
    )
    st.markdown("*Import and manage purchase order records for reconciliation*")
    # st.write(
    #     "Inspect existing purchase order (PO) data and append additional records via CSV uploads."
    # )

    success_message = st.session_state.pop("po_upload_success", None)
    just_uploaded = st.session_state.get("po_just_uploaded", False)
    if success_message:
        st.success(success_message)
        # Increment upload counter to force file uploader to reset
        st.session_state["po_upload_counter"] = (
            st.session_state.get("po_upload_counter", 0) + 1
        )

    po_path = Path("records/po")
    required_fields = [
        "PO_Date",
        "PO_Number",
        "Vendor_Prefix",
        "PO_Amount",
        "Dept",
        "Import_Batch_ID",
        "CC_Fee",
        "Total_PO_Amount",  # PO_Amount including CC fee
    ]
    mapping_fields = ["PO_Date", "PO_Number", "Vendor_Prefix", "PO_Amount"]

    try:
        existing_df = read_parquet_with_fallback(po_path)
    except FileNotFoundError:
        existing_df = pd.DataFrame(columns=required_fields)
        st.info(
            "No existing PO data found. Upload a CSV or Excel file to create the dataset."
        )
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Unable to read existing PO data: {exc}")
        existing_df = pd.DataFrame(columns=required_fields)

    for field in required_fields:
        if field not in existing_df.columns:
            existing_df[field] = pd.NA
    existing_df = existing_df[required_fields]

    if not existing_df.empty:
        filtered_df = existing_df.copy()

        with st.expander("Filter PO Records", expanded=False):
            col_vendor, col_dept, col_date = st.columns(3)

            vendor_options = sorted(
                existing_df["Vendor_Prefix"].dropna().astype(str).unique().tolist()
            )
            with col_vendor:
                selected_vendors = st.multiselect(
                    "Vendor Prefix",
                    options=vendor_options,
                    placeholder="Select vendors",
                )
            if selected_vendors:
                filtered_df = filtered_df[
                    filtered_df["Vendor_Prefix"].astype(str).isin(selected_vendors)
                ]

            dept_options = sorted(
                existing_df["Dept"].dropna().astype(str).unique().tolist()
            )
            with col_dept:
                selected_depts = st.multiselect(
                    "Dept",
                    options=dept_options,
                    placeholder="Select departments",
                )
            if selected_depts:
                filtered_df = filtered_df[
                    filtered_df["Dept"].astype(str).isin(selected_depts)
                ]

            date_series_all = pd.to_datetime(existing_df["PO_Date"], errors="coerce")
            valid_dates_all = date_series_all.dropna()

            # Calculate min/max dates once
            min_date = (
                valid_dates_all.min().date() if not valid_dates_all.empty else None
            )
            max_date = (
                valid_dates_all.max().date() if not valid_dates_all.empty else None
            )

            with col_date:
                if min_date and max_date:
                    # Check if reset button was clicked - remove the key to reset
                    if st.session_state.get("po_reset_clicked", False):
                        st.session_state.po_reset_clicked = False
                        # Remove the key so widget uses default value
                        if "po_date_range" in st.session_state:
                            del st.session_state.po_date_range

                    # Initialize default value
                    default_date_range = (min_date, max_date)

                    # Date input without constraints - allows selecting any date range
                    date_input_result = st.date_input(
                        "PO Date Range",
                        value=default_date_range,
                        key="po_date_range",
                        help="Select a date range to filter records. Set to full range to show all records.",
                    )

                    # Handle both single date and tuple returns
                    if (
                        isinstance(date_input_result, tuple)
                        and len(date_input_result) == 2
                    ):
                        start_date, end_date = date_input_result
                    elif (
                        isinstance(date_input_result, tuple)
                        and len(date_input_result) == 1
                    ):
                        # Single date in tuple - use it as both start and end
                        start_date = end_date = date_input_result[0]
                    else:
                        # Single date value
                        start_date = end_date = date_input_result

                    # Button to reset to show all records
                    if st.button(
                        "Show All Records",
                        key="po_reset_date",
                        use_container_width=True,
                    ):
                        st.session_state.po_reset_clicked = True
                        st.rerun()
                else:
                    start_date = end_date = None
                    st.caption("No valid PO dates available.")

            # Only apply date filter if dates are selected AND the range is different from full range
            if start_date and end_date and min_date and max_date:
                # Check if selected range equals or exceeds full range - if so, don't filter (show all)
                if start_date <= min_date and end_date >= max_date:
                    # Full range selected, don't apply filter - show all records
                    pass
                else:
                    # Specific range selected, apply filter
                    date_series_filtered = pd.to_datetime(
                        filtered_df["PO_Date"], errors="coerce"
                    )
                    date_mask = (date_series_filtered >= pd.Timestamp(start_date)) & (
                        date_series_filtered <= pd.Timestamp(end_date)
                    )
                    filtered_df = filtered_df[date_mask.fillna(False)]

        with st.expander("Manage Imports", expanded=False):
            batch_series = existing_df["Import_Batch_ID"].fillna("Unknown")
            batch_counts = batch_series.value_counts().sort_index()
            batch_options = [
                f"{batch_id} ({count} rows)" for batch_id, count in batch_counts.items()
            ]

            if not batch_options:
                st.caption("No batch information available to manage.")
                selected_batches = []
            else:
                selected_display = st.multiselect(
                    "Select Import Batches to Roll Back",
                    options=batch_options,
                    default=st.session_state.get("po_selected_batches", []),
                    key="po_rollback_batches",
                )

                # Update session state
                st.session_state["po_selected_batches"] = selected_display

                # Extract batch IDs from selected display strings
                selected_batches = [
                    display.split(" (", maxsplit=1)[0] for display in selected_display
                ]

                rollback_clicked = st.button(
                    "Rollback Selected Batches",
                    disabled=len(selected_batches) == 0,
                    type="primary",
                )

                if rollback_clicked and selected_batches:
                    # Filter out all selected batches
                    updated_df = existing_df[
                        ~existing_df["Import_Batch_ID"]
                        .astype(str)
                        .isin(selected_batches)
                    ]
                    write_parquet(updated_df, po_path)
                    batch_list = ", ".join(selected_batches)
                    st.session_state["po_upload_success"] = (
                        f"Rolled back {len(selected_batches)} batch(es): {batch_list}"
                    )
                    st.session_state["po_selected_batches"] = []
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:  # pragma: no cover
                        st.experimental_rerun()

        st.markdown("### Existing PO Records")
        po_amount_series = pd.to_numeric(
            filtered_df.get("PO_Amount", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0.0)
        total_po_amount = po_amount_series.sum()
        st.caption(
            f"{len(filtered_df)} of {len(existing_df)} records shown · "
            f"Filtered PO Amount: ${total_po_amount:,.2f}"
        )
        st.dataframe(filtered_df, hide_index=True, use_container_width=True)

    # Use dynamic key based on upload counter to force reset after upload
    st.warning(
        "Please use CSV format and the **PO Date** should be in the format MM/DD/YYYY and **PO Amount** should be a number.",
        icon="⚠️",
    )
    upload_counter = st.session_state.get("po_upload_counter", 0)
    uploaded_file = st.file_uploader(
        "Upload CSV with PO records",
        type=["csv"],
        accept_multiple_files=False,
        help="Only the required fields will be imported; extra columns are ignored.",
        key=f"po_uploaded_file_{upload_counter}",
    )

    # Skip preview and mapping if we just successfully uploaded
    if just_uploaded:
        # Clear the flag
        st.session_state["po_just_uploaded"] = False
        return

    if uploaded_file is None:
        return

    try:
        # Handle both CSV and Excel files
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            incoming_df = pd.read_excel(uploaded_file)
        else:
            incoming_df = pd.read_csv(uploaded_file)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Unable to read the uploaded file: {exc}")
        return

    if incoming_df.empty:
        st.warning(
            "The uploaded file is empty. Please upload a CSV or Excel file with rows to append."
        )
        return

    st.subheader("Uploaded File Preview")
    st.caption(f"{len(incoming_df)} records found")

    st.markdown("### Map uploaded columns to required fields")
    st.caption(
        "Select the column in the uploaded file that corresponds to each required field. "
        "Only mapped columns will be imported."
    )

    with st.form("po_upload_form"):
        mapping: dict[str, str] = {}
        upload_columns = list(incoming_df.columns)

        for field in mapping_fields:
            default_index = (
                upload_columns.index(field) + 1 if field in upload_columns else 0
            )
            selection = st.selectbox(
                field,
                options=["Select a column"] + upload_columns,
                index=default_index,
                key=f"map_{field}",
            )
            if selection != "Select a column":
                mapping[field] = selection

        dept_value = st.selectbox(
            "Dept (applied to all imported rows)",
            options=["FBA", "FBM"],
            index=0,
        )

        submitted = st.form_submit_button("Append Records")

    if not submitted:
        return

    missing_fields = [field for field in mapping_fields if field not in mapping]
    if missing_fields:
        st.error(
            "Please map all required fields before appending records: "
            + ", ".join(missing_fields)
        )
        return

    if len(set(mapping.values())) != len(mapping.values()):
        st.error(
            "Each uploaded column can only be mapped once. Please adjust the selections."
        )
        return

    aligned_df = pd.DataFrame(
        {field: incoming_df[mapping[field]] for field in mapping_fields}
    )
    # Ensure all string columns are properly typed to avoid type conversion errors
    # Convert all string columns to handle mixed types (bool, int, float, str)
    # Note: PO_Date will be converted to datetime later, but we ensure it's string-safe here
    po_string_columns = ["Vendor_Prefix", "PO_Number"]
    for col in po_string_columns:
        if col in aligned_df.columns:
            # Convert to object first to handle mixed types, then to string
            aligned_df[col] = aligned_df[col].astype(object).astype(str).str.strip()
            # Clean up null representations
            aligned_df[col] = aligned_df[col].replace("nan", "").replace("None", "")

    # PO_Date - convert to string first (will be converted to datetime in write_parquet if needed)
    if "PO_Date" in aligned_df.columns:
        aligned_df["PO_Date"] = (
            aligned_df["PO_Date"].astype(object).astype(str).str.strip()
        )
        aligned_df["PO_Date"] = (
            aligned_df["PO_Date"].replace("nan", "").replace("None", "")
        )

    # Validate PO_Date format (MM/DD/YYYY)
    if "PO_Date" in aligned_df.columns:
        is_valid, invalid_rows = validate_date_format_mmddyyyy(
            aligned_df["PO_Date"], "PO_Date"
        )
        if not is_valid:
            invalid_dates = aligned_df.loc[invalid_rows, "PO_Date"].tolist()
            invalid_dates_str = ", ".join([str(d) for d in invalid_dates[:10]])
            suffix = "..." if len(invalid_dates) > 10 else ""
            st.error(
                f"PO_Date must be in MM/DD/YYYY format. Invalid dates found: {invalid_dates_str}{suffix}"
            )
            return

        # Convert PO_Date to datetime with explicit format, then back to string in YYYY-MM-DD format
        aligned_df["PO_Date"] = pd.to_datetime(
            aligned_df["PO_Date"], format="%m/%d/%Y", errors="coerce"
        )
        if aligned_df["PO_Date"].isna().any():
            st.error(
                "Unable to parse PO_Date for all rows. Please ensure dates are in MM/DD/YYYY format."
            )
            return
        # Store as string in YYYY-MM-DD format for consistency
        aligned_df["PO_Date"] = aligned_df["PO_Date"].dt.strftime("%Y-%m-%d")

    # Validate PO_Number uniqueness - check against existing data
    if "PO_Number" in aligned_df.columns:
        existing_po_numbers = (
            existing_df["PO_Number"].dropna().astype(str).str.strip().tolist()
            if not existing_df.empty and "PO_Number" in existing_df.columns
            else []
        )
        incoming_po_numbers = (
            aligned_df["PO_Number"].dropna().astype(str).str.strip().tolist()
        )
        duplicate_po_numbers = sorted(
            set(incoming_po_numbers).intersection(existing_po_numbers)
        )
        if duplicate_po_numbers:
            preview = ", ".join(duplicate_po_numbers[:10])
            suffix = "..." if len(duplicate_po_numbers) > 10 else ""
            st.error(
                "Import aborted. The following PO_Number values already exist: "
                + preview
                + suffix
            )
            return

    aligned_df["Dept"] = dept_value
    batch_id = datetime.utcnow().strftime("BATCH-%Y%m%d-%H%M%S")
    aligned_df["Import_Batch_ID"] = batch_id

    # Load master data to get CC fee
    try:
        master_df = read_parquet_with_fallback(Path("data/master"))
        # Normalize Vendor_Prefix for matching
        master_df["PREFIX"] = master_df["PREFIX"].astype(str).str.strip().str.upper()
        master_df["DEPT"] = master_df["DEPT"].astype(str).str.strip().str.upper()

        # Check if CC fee column exists (could be "CC fee", "CC_Fee", "cc fee", etc.)
        cc_fee_column = None
        for col in master_df.columns:
            if "cc" in col.lower() and "fee" in col.lower():
                cc_fee_column = col
                break

        if cc_fee_column:
            # Create lookup dictionary: (Vendor_Prefix, Dept) -> CC_Fee
            master_df["_lookup_key"] = (
                master_df["PREFIX"].astype(str) + "|" + master_df["DEPT"].astype(str)
            )
            cc_fee_lookup = dict(
                zip(
                    master_df["_lookup_key"],
                    pd.to_numeric(master_df[cc_fee_column], errors="coerce").fillna(
                        0.0
                    ),
                )
            )

            # Normalize aligned_df Vendor_Prefix and Dept for matching
            aligned_df["Vendor_Prefix"] = (
                aligned_df["Vendor_Prefix"].astype(str).str.strip().str.upper()
            )
            aligned_df["Dept"] = aligned_df["Dept"].astype(str).str.strip().str.upper()

            # Create lookup key for aligned_df
            aligned_df["_lookup_key"] = (
                aligned_df["Vendor_Prefix"].astype(str)
                + "|"
                + aligned_df["Dept"].astype(str)
            )

            # Get CC_Fee from master
            aligned_df["CC_Fee"] = (
                aligned_df["_lookup_key"].map(cc_fee_lookup).fillna(0.0)
            )

            # Calculate Total_PO_Amount = PO_Amount * (1 + CC_Fee)
            aligned_df["PO_Amount"] = pd.to_numeric(
                aligned_df["PO_Amount"], errors="coerce"
            ).fillna(0.0)
            aligned_df["Total_PO_Amount"] = aligned_df["PO_Amount"] * (
                1 + aligned_df["CC_Fee"]
            )

            # Remove temporary lookup key
            aligned_df = aligned_df.drop(columns=["_lookup_key"], errors="ignore")
        else:
            # CC fee column not found in master, set to 0
            aligned_df["CC_Fee"] = 0.0
            aligned_df["Total_PO_Amount"] = aligned_df["PO_Amount"]
            st.warning(
                "CC fee column not found in master data. CC_Fee and CC_Charge_Amount set to 0."
            )
    except Exception as exc:  # pylint: disable=broad-except
        # If master file can't be loaded, set CC fee to 0
        aligned_df["CC_Fee"] = 0.0
        aligned_df["Total_PO_Amount"] = pd.to_numeric(
            aligned_df["PO_Amount"], errors="coerce"
        ).fillna(0.0)
        st.warning(
            f"Unable to load master data for CC fee lookup: {exc}. CC_Fee set to 0, Total_PO_Amount = PO_Amount."
        )

    combined_df = pd.concat(
        [existing_df, aligned_df],
        ignore_index=True,
    )
    write_parquet(combined_df, po_path)

    st.session_state["po_upload_success"] = (
        f"New records appended to PO data (batch {batch_id})."
    )
    st.session_state["po_just_uploaded"] = True
    # Clear form keys to reset the form
    for field in mapping_fields:
        form_key = f"map_{field}"
        if form_key in st.session_state:
            del st.session_state[form_key]

    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - fallback for older Streamlit
        st.experimental_rerun()


def render_cc_data() -> None:
    st.markdown(
        '<div class="main-header">Credit Card Transactions</div>',
        unsafe_allow_html=True,
    )
    st.markdown("*Import and manage credit card transaction records*")
    # st.write(
    #     "Import, review, and manage credit card statement data for reconciliation workflows."
    # )

    success_message = st.session_state.pop("cc_upload_success", None)
    just_uploaded = st.session_state.get("cc_just_uploaded", False)
    if success_message:
        st.success(success_message)
        # Increment upload counter to force file uploader to reset
        st.session_state["cc_upload_counter"] = (
            st.session_state.get("cc_upload_counter", 0) + 1
        )

    cc_path = Path("records/cc")
    required_fields = [
        "CC_Txn_Date",
        "CC_Number",
        "CC_Description",
        "CC_Amt",
        "CC_Reference_ID",
        "Import_Batch_ID",
        "Reco_ID",
    ]
    mapping_fields = [
        "CC_Txn_Date",
        "CC_Number",
        "CC_Description",
        "CC_Amt",
        "CC_Reference_ID",
    ]

    try:
        existing_df = read_parquet_with_fallback(cc_path)
    except FileNotFoundError:
        existing_df = pd.DataFrame(columns=required_fields)
        st.info(
            "No credit card data found. Upload a CSV or Excel file to create the dataset."
        )
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Unable to read existing credit card data: {exc}")
        existing_df = pd.DataFrame(columns=required_fields)

    for field in required_fields:
        if field not in existing_df.columns:
            existing_df[field] = pd.NA
    existing_df = existing_df[required_fields]

    if not existing_df.empty:
        filtered_df = existing_df.copy()

        with st.expander("Filter Credit Card Records", expanded=False):
            col_batch, col_cc_number, col_date = st.columns((1, 1, 1))

            with col_batch:
                batch_series = (
                    existing_df["Import_Batch_ID"]
                    .fillna("Unknown")
                    .astype(str)
                    .str.strip()
                )
                batch_options = sorted(batch_series.unique().tolist())
                if batch_options:
                    batch_selection = st.selectbox(
                        "Import batch",
                        options=["All batches"] + batch_options,
                        index=0,
                        key="cc_batch_filter",
                    )
                else:
                    batch_selection = "All batches"
                    st.caption("No batch information available.")

            if batch_selection != "All batches":
                filtered_df = filtered_df[
                    filtered_df["Import_Batch_ID"]
                    .fillna("Unknown")
                    .astype(str)
                    .str.strip()
                    == batch_selection
                ]

            with col_cc_number:
                if "CC_Number" in existing_df.columns:
                    cc_number_options = sorted(
                        existing_df["CC_Number"].dropna().astype(str).unique().tolist()
                    )
                    if cc_number_options:
                        cc_number_selection = st.selectbox(
                            "CC Number",
                            options=["All CC Numbers"] + cc_number_options,
                            index=0,
                            key="cc_number_filter",
                        )
                    else:
                        cc_number_selection = "All CC Numbers"
                        st.caption("No CC numbers available.")
                else:
                    cc_number_selection = "All CC Numbers"

            if (
                cc_number_selection != "All CC Numbers"
                and "CC_Number" in filtered_df.columns
            ):
                filtered_df = filtered_df[
                    filtered_df["CC_Number"].astype(str) == cc_number_selection
                ]

            with col_date:
                date_series_all = pd.to_datetime(
                    existing_df["CC_Txn_Date"], errors="coerce"
                )
                valid_dates_all = date_series_all.dropna()
                if not valid_dates_all.empty:
                    min_date = valid_dates_all.min().date()
                    max_date = valid_dates_all.max().date()
                    start_date, end_date = st.date_input(
                        "Transaction date range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="cc_date_range",
                    )
                else:
                    start_date = end_date = None
                    st.caption("No valid transaction dates available.")

            if start_date and end_date:
                date_series_filtered = pd.to_datetime(
                    filtered_df["CC_Txn_Date"], errors="coerce"
                )
                date_mask = (date_series_filtered >= pd.Timestamp(start_date)) & (
                    date_series_filtered <= pd.Timestamp(end_date)
                )
                filtered_df = filtered_df[date_mask.fillna(False)]

        with st.expander("Manage Imports", expanded=False):
            batch_series = existing_df["Import_Batch_ID"].fillna("Unknown")
            batch_counts = batch_series.value_counts().sort_index()
            batch_options = [
                f"{batch_id} ({count} rows)"
                for batch_id, count in batch_counts.items()
                if batch_id != "Unknown"
            ]

            if not batch_options:
                st.caption("No batch information available to manage.")
                selected_batches = []
            else:
                selected_display = st.multiselect(
                    "Select Import Batches to Roll Back",
                    options=batch_options,
                    default=st.session_state.get("cc_selected_batches", []),
                    key="cc_rollback_batches",
                )

                # Update session state
                st.session_state["cc_selected_batches"] = selected_display

                # Extract batch IDs from selected display strings
                selected_batches = [
                    display.split(" (", maxsplit=1)[0] for display in selected_display
                ]

            rollback_clicked = st.button(
                "Rollback Selected Batches",
                disabled=len(selected_batches) == 0,
                type="secondary",
                key="cc_rollback_button",
                use_container_width=True,
            )

            if rollback_clicked and selected_batches:
                # Filter out all selected batches
                updated_df = existing_df[
                    ~existing_df["Import_Batch_ID"].astype(str).isin(selected_batches)
                ]
                write_parquet(updated_df, cc_path)
                batch_list = ", ".join(selected_batches)
                st.session_state["cc_upload_success"] = (
                    f"Rolled back {len(selected_batches)} batch(es): {batch_list}"
                )
                st.session_state["cc_selected_batches"] = []
                if hasattr(st, "rerun"):
                    st.rerun()
                else:  # pragma: no cover
                    st.experimental_rerun()

        st.subheader("Existing Credit Card Records")
        st.caption(f"{len(filtered_df)} of {len(existing_df)} records shown")
        # Display dataframe without Reco_ID column
        display_df = filtered_df.drop(columns=["Reco_ID"], errors="ignore")
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    st.warning(
        "Please use CSV format and the **CC Date** should be in the format MM/DD/YYYY and **CC Amount** should be a number.",
        icon="⚠️",
    )
    # Use dynamic key based on upload counter to force reset after upload
    upload_counter = st.session_state.get("cc_upload_counter", 0)
    uploaded_file = st.file_uploader(
        "Upload CSV with credit card records",
        type=["csv"],
        accept_multiple_files=False,
        help="Only the required fields will be imported; extra columns are ignored.",
        key=f"cc_uploaded_file_{upload_counter}",
    )

    # Skip preview and mapping if we just successfully uploaded
    if just_uploaded:
        # Clear the flag
        st.session_state["cc_just_uploaded"] = False
        return

    if uploaded_file is None:
        return

    try:
        # Handle both CSV and Excel files
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            incoming_df = pd.read_excel(uploaded_file)
        else:
            incoming_df = pd.read_csv(uploaded_file)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Unable to read the uploaded file: {exc}")
        return

    if incoming_df.empty:
        st.warning(
            "The uploaded file is empty. Please upload a CSV or Excel file with rows to append."
        )
        return

    st.subheader("Uploaded File Preview")
    st.caption(f"{len(incoming_df)} records found")

    st.markdown("### Map uploaded columns to required fields")
    st.caption(
        "Select the column in the uploaded file that corresponds to each required field. "
        "Only mapped columns will be imported."
    )

    with st.form("cc_upload_form"):
        mapping: dict[str, str] = {}
        upload_columns = list(incoming_df.columns)

        for field in mapping_fields:
            default_index = (
                upload_columns.index(field) + 1 if field in upload_columns else 0
            )
            selection = st.selectbox(
                field,
                options=["Select a column"] + upload_columns,
                index=default_index,
                key=f"cc_map_{field}",
            )
            if selection != "Select a column":
                mapping[field] = selection

        submitted = st.form_submit_button("Append Records")

    if not submitted:
        return

    missing_fields = [field for field in mapping_fields if field not in mapping]
    if missing_fields:
        st.error(
            "Please map all required fields before appending records: "
            + ", ".join(missing_fields)
        )
        return

    if len(set(mapping.values())) != len(mapping.values()):
        st.error(
            "Each uploaded column can only be mapped once. Please adjust the selections."
        )
        return

    aligned_df = pd.DataFrame(
        {field: incoming_df[mapping[field]] for field in mapping_fields}
    )

    # Ensure all string columns are properly typed to avoid type conversion errors
    # Convert all string columns to handle mixed types (bool, int, float, str)
    cc_string_columns = ["CC_Number", "CC_Description", "CC_Reference_ID"]
    for col in cc_string_columns:
        if col in aligned_df.columns:
            # Convert to object first to handle mixed types, then to string
            aligned_df[col] = aligned_df[col].astype(object).astype(str).str.strip()
            # Clean up null representations
            aligned_df[col] = aligned_df[col].replace("nan", "").replace("None", "")

    # Clean CC_Description: normalize multiple spaces to single space
    if "CC_Description" in aligned_df.columns:
        # Replace multiple spaces (2 or more) with single space
        aligned_df["CC_Description"] = (
            aligned_df["CC_Description"]
            .str.replace(
                r"\s+", " ", regex=True
            )  # Replace multiple spaces with single space
            .str.strip()  # Remove leading/trailing spaces
        )

    # Validate CC_Number
    if "CC_Number" in aligned_df.columns:
        if (
            aligned_df["CC_Number"].isna().any()
            or (aligned_df["CC_Number"] == "").any()
        ):
            st.error("All rows must have a CC_Number value.")
            return
    else:
        st.error("CC_Number field is required but not found in the mapped data.")
        return
    if (
        aligned_df["CC_Reference_ID"].isna().any()
        or (aligned_df["CC_Reference_ID"] == "").any()
    ):
        st.error("All rows must have a CC_Reference_ID value.")
        return

    duplicate_refs = aligned_df["CC_Reference_ID"].duplicated(keep=False)
    if duplicate_refs.any():
        duplicates = aligned_df.loc[duplicate_refs, "CC_Reference_ID"].unique().tolist()
        st.error(
            "Duplicate CC_Reference_ID values found within the uploaded file: "
            + ", ".join(duplicates[:10])
            + ("..." if len(duplicates) > 10 else "")
        )
        return

    # Convert CC_Txn_Date to string for validation
    if "CC_Txn_Date" in aligned_df.columns:
        aligned_df["CC_Txn_Date"] = (
            aligned_df["CC_Txn_Date"].astype(object).astype(str).str.strip()
        )
        aligned_df["CC_Txn_Date"] = (
            aligned_df["CC_Txn_Date"].replace("nan", "").replace("None", "")
        )

    # Validate CC_Txn_Date format (MM/DD/YYYY)
    if "CC_Txn_Date" in aligned_df.columns:
        is_valid, invalid_rows = validate_date_format_mmddyyyy(
            aligned_df["CC_Txn_Date"], "CC_Txn_Date"
        )
        if not is_valid:
            invalid_dates = aligned_df.loc[invalid_rows, "CC_Txn_Date"].tolist()
            invalid_dates_str = ", ".join([str(d) for d in invalid_dates[:10]])
            suffix = "..." if len(invalid_dates) > 10 else ""
            st.error(
                f"CC_Txn_Date must be in MM/DD/YYYY format. Invalid dates found: {invalid_dates_str}{suffix}"
            )
            return

    aligned_df["CC_Txn_Date"] = pd.to_datetime(
        aligned_df["CC_Txn_Date"], format="%m/%d/%Y", errors="coerce"
    )
    if aligned_df["CC_Txn_Date"].isna().any():
        st.error(
            "Unable to parse CC_Txn_Date for all rows. Please ensure the dates are in MM/DD/YYYY format."
        )
        return

    # Group by date to create separate batches for each date
    aligned_df["_date_group"] = aligned_df["CC_Txn_Date"].dt.date
    unique_dates = aligned_df["_date_group"].dropna().unique()

    # Convert date back to string format for storage
    aligned_df["CC_Txn_Date"] = aligned_df["CC_Txn_Date"].dt.strftime("%Y-%m-%d")

    aligned_df["CC_Amt"] = pd.to_numeric(aligned_df["CC_Amt"], errors="coerce")
    if aligned_df["CC_Amt"].isna().any():
        st.error(
            "Unable to parse CC_Amt for all rows. Please ensure the amounts are valid numbers."
        )
        return

    existing_references = (
        existing_df["CC_Reference_ID"].dropna().astype(str).str.strip().tolist()
        if not existing_df.empty
        else []
    )

    # Extract last 4 digits from CC_Number for batch ID
    def get_cc_last4(cc_number):
        """Extract last 4 digits from CC number, handling various formats."""
        if pd.isna(cc_number):
            return "0000"
        cc_str = str(cc_number).strip()
        # Remove any non-digit characters and get last 4 digits
        digits_only = "".join(filter(str.isdigit, cc_str))
        if len(digits_only) >= 4:
            return digits_only[-4:]
        # If less than 4 digits, pad with zeros
        return digits_only.zfill(4)[-4:]

    # Process each date group separately to create batches per date
    all_batches = []
    combined_df = existing_df.copy()
    datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    batch_counter = 0

    for date_group in sorted(unique_dates):
        date_df = aligned_df[aligned_df["_date_group"] == date_group].copy()

        # Check for duplicate references within this date group
        incoming_references = date_df["CC_Reference_ID"].tolist()
        duplicate_existing_refs = sorted(
            set(incoming_references).intersection(existing_references)
        )
        if duplicate_existing_refs:
            preview = ", ".join(duplicate_existing_refs[:10])
            suffix = "..." if len(duplicate_existing_refs) > 10 else ""
            st.warning(
                f"Skipping date {date_group.isoformat()}. The following CC_Reference_ID values already exist: "
                + preview
                + suffix
            )
            continue

        # Get the most common CC number's last 4 digits for this date group
        if "CC_Number" in date_df.columns:
            cc_numbers = date_df["CC_Number"].dropna()
            if not cc_numbers.empty:
                cc_number_counts = cc_numbers.value_counts()
                most_common_cc = (
                    cc_number_counts.index[0] if len(cc_number_counts) > 0 else None
                )
                cc_last4 = (
                    get_cc_last4(most_common_cc)
                    if most_common_cc is not None
                    else "0000"
                )
            else:
                cc_last4 = "0000"
        else:
            cc_last4 = "0000"

        # Create batch ID with timestamp, counter, and date suffix for uniqueness
        batch_counter += 1
        date_suffix = date_group.strftime("%m%d%Y")  # Add date to batch ID for clarity
        batch_id = f"CCBATCH-{cc_last4}-{date_suffix}"

        date_df["Import_Batch_ID"] = batch_id
        date_df["Reco_ID"] = pd.NA
        date_df = date_df.drop(columns=["_date_group"])  # Remove temporary column

        # Append to combined dataframe
        combined_df = pd.concat(
            [combined_df, date_df],
            ignore_index=True,
        )

        all_batches.append(batch_id)
        # Update existing_references to include newly added ones
        existing_references.extend(incoming_references)

    if not all_batches:
        st.error("No batches were created. All dates contained duplicate references.")
        return

    write_parquet(combined_df, cc_path)

    # Create success message with all batches
    if len(all_batches) == 1:
        batch_message = f"New credit card records appended (batch {all_batches[0]})."
    else:
        batch_list = ", ".join(all_batches)
        batch_message = f"Created {len(all_batches)} batches for {len(unique_dates)} date(s): {batch_list}"

    st.session_state["cc_upload_success"] = batch_message
    st.session_state["cc_just_uploaded"] = True
    # Clear form keys to reset the form
    for field in mapping_fields:
        form_key = f"cc_map_{field}"
        if form_key in st.session_state:
            del st.session_state[form_key]

    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - fallback for older Streamlit
        st.experimental_rerun()


def render_reco() -> None:
    st.markdown(
        '<div class="main-header">Reconciliation Center</div>',
        unsafe_allow_html=True,
    )
    st.markdown("*Reconcile credit card transactions with purchase orders*")

    success_message = st.session_state.pop("reco_success", None)
    if success_message:
        st.success(success_message)

    # Load CC data to get available batches
    cc_path = Path("records/cc")
    try:
        cc_df = read_parquet_with_fallback(cc_path)
    except FileNotFoundError:
        st.info("No credit card data found. Please upload CC transactions first.")
        return
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Unable to read CC data: {exc}")
        return

    if cc_df.empty:
        st.info("No credit card data available. Please upload CC transactions first.")
        return

    # Get available batches
    batch_series = cc_df["Import_Batch_ID"].fillna("Unknown")
    batch_counts = batch_series.value_counts().sort_index()
    batch_options = [
        f"{batch_id} ({count} rows)"
        for batch_id, count in batch_counts.items()
        if batch_id != "Unknown"
    ]

    if not batch_options:
        st.info("No batches available for reconciliation.")
        return

    # Batch selection
    selected_batch_display = st.selectbox(
        "Select CC Batch to Reconcile",
        options=["Select a batch"] + batch_options,
        index=0,
        key="reco_batch_selector",
    )

    if selected_batch_display == "Select a batch":
        st.info("Please select a batch to run reconciliation.")
        return

    selected_batch = selected_batch_display.split(" (", maxsplit=1)[0]

    # Get batch data
    batch_df = cc_df[cc_df["Import_Batch_ID"].astype(str) == selected_batch].copy()

    if batch_df.empty:
        st.error("No transactions found for the selected batch.")
        return

    # Check if we have in-memory reco data or need to run reconciliation
    current_reco_batch = st.session_state.get("current_reco_batch")
    reco_df = st.session_state.get("reco_df")

    # Clear reco data if batch changed
    if current_reco_batch is not None and current_reco_batch != selected_batch:
        st.session_state.pop("reco_df", None)
        st.session_state.pop("current_reco_batch", None)
        reco_df = None

    # Automatically run reconciliation if no reco data exists or batch changed
    if reco_df is None:
        try:
            with st.spinner(f"Running reconciliation for batch {selected_batch}..."):
                reco_df = run_reconciliation(batch_df)
                st.session_state["reco_df"] = reco_df
                st.session_state["current_reco_batch"] = selected_batch
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to run reconciliation: {exc}")
            return

    # Use the in-memory reco_df
    if reco_df is None or reco_df.empty:
        st.error("No reconciliation data available.")
        return
    display_df = reco_df.drop(columns=["Import_Batch_ID"], errors="ignore")
    category_series = display_df["Category"].astype(str).str.upper()
    common_df = display_df[category_series == "COMMON"]
    unmapped_df = display_df[category_series == "UNMAPPED"]
    others_df = display_df[category_series == "NOT AVBL"]
    mapped_df = display_df[~category_series.isin(["UNMAPPED", "COMMON", "NOT AVBL"])]

    results_group_cols = ["CC_Txn_Date", "Vendor_Prefix", "Dept", "Payment_Terms"]
    # Store data for analysis tab
    po_merge_data = None
    results_base_data = None
    # Get grace_days from session state if it exists, otherwise use default
    # Don't initialize session state here - let the widget handle it
    grace_days = int(st.session_state.get("results_grace_days", 5))
    if mapped_df.empty:
        results_df = pd.DataFrame(
            columns=results_group_cols
            + [
                "Total_CC_Amount",
                "CC_Transaction_Count",
                "Total_PO_Amount",
                "PO_Transaction_Count",
                "Total_Deductions",
                "CC_Fee",
                "Total_CC_Charge",
                "Flag",
                "Vendor_Name",
            ]
        )
        results_base_data = pd.DataFrame()
    else:
        results_base = mapped_df.assign(
            CC_Amt=pd.to_numeric(mapped_df["CC_Amt"], errors="coerce")
        ).fillna(
            {
                "CC_Txn_Date": "<missing date>",
                "Vendor_Prefix": "<missing prefix>",
                "Dept": "<missing dept>",
                "Payment_Terms": "<missing payment terms>",
            }
        )
        results_base_data = results_base.copy()
        results_base["CC_Txn_Date_dt"] = pd.to_datetime(
            results_base["CC_Txn_Date"], errors="coerce"
        )
        # Convert Payment_Terms to numeric, defaulting to 0 for missing/null values
        results_base["Payment_Terms_Numeric"] = pd.to_numeric(
            results_base["Payment_Terms"], errors="coerce"
        ).fillna(0)
        # Also ensure Payment_Terms column itself is numeric for grouping/merging
        results_base["Payment_Terms"] = results_base["Payment_Terms_Numeric"]
        # Calculate date window: Start = T - Payment_Terms - grace_days
        # End = T + grace_days (only if Payment_Terms = 0), else End = T
        results_base["Window_Start"] = (
            results_base["CC_Txn_Date_dt"]
            - pd.to_timedelta(results_base["Payment_Terms_Numeric"], unit="D")
            - pd.Timedelta(days=grace_days)
        )
        # Add grace_days to end date only when Payment_Terms is 0
        results_base["Window_End"] = results_base["CC_Txn_Date_dt"].where(
            results_base["Payment_Terms_Numeric"] > 0,
            results_base["CC_Txn_Date_dt"] + pd.Timedelta(days=grace_days),
        )
        # Deduplicate results_base by CC_Reference_ID to avoid double counting
        # This ensures each transaction is counted only once
        results_base_unique = results_base.drop_duplicates(
            subset=["CC_Reference_ID"], keep="first"
        )
        results_df = (
            results_base_unique.groupby(results_group_cols)
            .agg(
                Total_CC_Amount=("CC_Amt", "sum"),
                Transaction_Count=("CC_Reference_ID", "nunique"),
            )
            .reset_index()
        )
        results_df["Total_CC_Amount"] = results_df["Total_CC_Amount"].fillna(0.0)
        results_df["Transaction_Count"] = (
            results_df["Transaction_Count"].fillna(0).astype(int)
        )
        results_df["Vendor_Prefix"] = (
            results_df["Vendor_Prefix"].astype(str).str.strip().str.upper()
        )

        po_path = Path("records/po")
        try:
            po_df = read_parquet_with_fallback(po_path)
            po_df["Vendor_Prefix"] = (
                po_df["Vendor_Prefix"].astype(str).str.strip().str.upper()
            )
            po_df["PO_Amount"] = pd.to_numeric(
                po_df["PO_Amount"], errors="coerce"
            ).fillna(0.0)
            # Use Total_PO_Amount for reconciliation (PO_Amount including CC fee)
            # If Total_PO_Amount doesn't exist, fall back to PO_Amount
            if "Total_PO_Amount" in po_df.columns:
                po_df["Total_PO_Amount"] = pd.to_numeric(
                    po_df["Total_PO_Amount"], errors="coerce"
                ).fillna(0.0)
                # Use Total_PO_Amount if it exists and is valid
                po_df["Comparison_Amount"] = po_df["Total_PO_Amount"].where(
                    po_df["Total_PO_Amount"].notna(), po_df["PO_Amount"]
                )
            else:
                # Total_PO_Amount column doesn't exist, use PO_Amount
                po_df["Comparison_Amount"] = po_df["PO_Amount"]
            po_df["PO_Date"] = pd.to_datetime(po_df["PO_Date"], errors="coerce")

            # Apply PO deductions
            deductions_df = load_po_deductions()
            if not deductions_df.empty and "PO_Number" in po_df.columns:
                # Calculate total deductions per PO
                po_deductions = (
                    deductions_df.groupby("PO_Number")
                    .agg({"Deduction_Amount": "sum"})
                    .reset_index()
                )
                po_deductions.rename(
                    columns={"Deduction_Amount": "Total_Deductions"}, inplace=True
                )

                # Merge with PO data
                po_df = po_df.merge(po_deductions, on="PO_Number", how="left")
                po_df["Total_Deductions"] = po_df["Total_Deductions"].fillna(0.0)

                # Store original amounts
                po_df["Original_PO_Amount"] = po_df["PO_Amount"].copy()

                # Calculate adjusted amount (after deductions)
                po_df["Adjusted_PO_Amount"] = (
                    po_df["PO_Amount"] - po_df["Total_Deductions"]
                )

                # Update Comparison_Amount to use adjusted amount
                if "Total_PO_Amount" in po_df.columns:
                    # Recalculate Total_PO_Amount with adjusted amount
                    if "CC_Fee" in po_df.columns:
                        po_df["Adjusted_Total_PO_Amount"] = po_df[
                            "Adjusted_PO_Amount"
                        ] * (1 + po_df["CC_Fee"])
                    else:
                        po_df["Adjusted_Total_PO_Amount"] = po_df["Adjusted_PO_Amount"]
                    po_df["Comparison_Amount"] = po_df["Adjusted_Total_PO_Amount"]
                else:
                    po_df["Comparison_Amount"] = po_df["Adjusted_PO_Amount"]

                # Filter out fully depleted POs
                depleted_pos = (
                    po_df[po_df["Adjusted_PO_Amount"] <= 0]["PO_Number"]
                    .unique()
                    .tolist()
                )
                if depleted_pos:
                    st.info(
                        f"ℹ️ Excluded {len(depleted_pos)} fully depleted PO(s) from reconciliation: "
                        f"{', '.join(str(p) for p in depleted_pos[:5])}"
                        f"{'...' if len(depleted_pos) > 5 else ''}"
                    )
                po_df = po_df[po_df["Adjusted_PO_Amount"] > 0].copy()
            else:
                # No deductions - use original amounts
                po_df["Total_Deductions"] = 0.0
                po_df["Original_PO_Amount"] = po_df["PO_Amount"]
                po_df["Adjusted_PO_Amount"] = po_df["PO_Amount"]

            # Keep all PO rows including duplicates (multiple line items)
            # Don't aggregate - count every line item separately
            po_summary = pd.DataFrame(
                columns=results_group_cols + ["Comparison_Amount"]
            )
            # Use ALL CC transactions (not just unique groups) to match against POs
            # This ensures we capture all PO matching windows
            window_cols = results_base[
                results_group_cols
                + ["Window_Start", "Window_End", "CC_Txn_Date_dt", "CC_Reference_ID"]
            ].copy()
            window_cols = window_cols.dropna(
                subset=["CC_Txn_Date_dt", "Window_Start", "Window_End"]
            )
            # Keep all rows to allow POs to match against any CC transaction in the window
            # Don't deduplicate here - let POs match to all applicable transactions
            if not window_cols.empty and not po_df.empty:
                # Filter PO data by vendor prefix first to reduce merge size
                unique_vendors = window_cols["Vendor_Prefix"].dropna().unique()
                po_df_filtered = po_df[
                    po_df["Vendor_Prefix"].isin(unique_vendors)
                ].copy()

                if not po_df_filtered.empty:
                    # Ensure Dept column exists in PO data and normalize it
                    if "Dept" in po_df_filtered.columns:
                        po_df_filtered["Dept"] = (
                            po_df_filtered["Dept"].astype(str).str.strip().str.upper()
                        )
                    else:
                        # If Dept not in PO, we can't match by dept - this is a data issue
                        st.warning(
                            "PO data missing Dept column. PO matching may be inaccurate."
                        )

                    # Normalize Dept in window_cols for consistent matching
                    if "Dept" in window_cols.columns:
                        window_cols["Dept"] = (
                            window_cols["Dept"].astype(str).str.strip().str.upper()
                        )

                    # Merge on both Vendor_Prefix and Dept
                    merge_keys = ["Vendor_Prefix"]
                    if (
                        "Dept" in po_df_filtered.columns
                        and "Dept" in window_cols.columns
                    ):
                        merge_keys.append("Dept")

                    # Include deduction columns in merge
                    merge_columns = merge_keys + [
                        "PO_Date",
                        "Comparison_Amount",
                        "PO_Number",
                        "PO_Amount",
                        "Original_PO_Amount",
                        "Total_Deductions",
                        "Adjusted_PO_Amount",
                    ]
                    # Only include columns that exist
                    merge_columns = [
                        col for col in merge_columns if col in po_df_filtered.columns
                    ]

                    po_merge = window_cols.merge(
                        po_df_filtered[merge_columns],
                        on=merge_keys,
                        how="left",
                    )
                    valid_mask = (
                        po_merge["PO_Date"].notna()
                        & po_merge["Window_Start"].notna()
                        & po_merge["Window_End"].notna()
                        & (po_merge["PO_Date"] >= po_merge["Window_Start"])
                        & (po_merge["PO_Date"] <= po_merge["Window_End"])
                    )
                    if valid_mask.any():
                        po_merge_data = po_merge.loc[valid_mask].copy()
                        # Deduplicate by PO_Number to avoid counting the same PO multiple times
                        # when it matches multiple CC transaction windows
                        po_merge_unique = po_merge_data.drop_duplicates(
                            subset=["PO_Number"], keep="first"
                        )
                        po_summary = (
                            po_merge_unique.groupby(results_group_cols, dropna=False)
                            .agg(
                                Comparison_Amount=("Comparison_Amount", "sum"),
                                Base_PO_Amount=(
                                    "PO_Amount",
                                    "sum",
                                ),  # Base PO amount without CC fee
                                Original_PO_Amount_Sum=(
                                    "Original_PO_Amount",
                                    "sum",
                                )
                                if "Original_PO_Amount" in po_merge_unique.columns
                                else (
                                    "PO_Amount",
                                    "sum",
                                ),  # Original PO amount for CC charge calculation (before deductions)
                                PO_Transaction_Count=(
                                    "PO_Number",
                                    "nunique",
                                ),  # Count unique POs only
                            )
                            .reset_index()
                        )
                    else:
                        po_summary = pd.DataFrame(
                            columns=results_group_cols
                            + [
                                "Comparison_Amount",
                                "Base_PO_Amount",
                                "PO_Transaction_Count",
                            ]
                        )
                        po_merge_data = pd.DataFrame()
                else:
                    po_summary = pd.DataFrame(
                        columns=results_group_cols
                        + [
                            "Comparison_Amount",
                            "Base_PO_Amount",
                            "PO_Transaction_Count",
                        ]
                    )
                    po_merge_data = pd.DataFrame()
            else:
                po_summary = pd.DataFrame(
                    columns=results_group_cols
                    + ["Comparison_Amount", "Base_PO_Amount", "PO_Transaction_Count"]
                )
                po_merge_data = pd.DataFrame()
        except Exception as exc:  # pylint: disable=broad-except
            st.warning(f"Unable to load PO summary: {exc}")
            po_summary = pd.DataFrame(
                columns=results_group_cols
                + ["Comparison_Amount", "Base_PO_Amount", "PO_Transaction_Count"]
            )
            po_merge_data = pd.DataFrame()

        # Ensure Payment_Terms has consistent type in both dataframes before merging
        if "Payment_Terms" in results_df.columns:
            results_df["Payment_Terms"] = pd.to_numeric(
                results_df["Payment_Terms"], errors="coerce"
            ).fillna(0)
        if "Payment_Terms" in po_summary.columns:
            po_summary["Payment_Terms"] = pd.to_numeric(
                po_summary["Payment_Terms"], errors="coerce"
            ).fillna(0)

        # Ensure other merge columns have consistent types
        for col in results_group_cols:
            if col in results_df.columns and col in po_summary.columns:
                # Convert to string for text columns, numeric for Payment_Terms
                if col == "Payment_Terms":
                    continue  # Already handled above
                else:
                    results_df[col] = results_df[col].astype(str).str.strip()
                    po_summary[col] = po_summary[col].astype(str).str.strip()

        results_df = results_df.merge(
            po_summary,
            on=results_group_cols,
            how="left",
        ).rename(
            columns={
                "Comparison_Amount": "Total_PO_Amount",
                "Base_PO_Amount": "Base_PO_Amount",
                "Transaction_Count": "CC_Transaction_Count",
            }
        )
        results_df["Total_PO_Amount"] = results_df["Total_PO_Amount"].fillna(0.0)
        results_df["Base_PO_Amount"] = results_df["Base_PO_Amount"].fillna(0.0)
        # Store original PO amount sum for CC charge calculation
        if "Original_PO_Amount_Sum" in results_df.columns:
            results_df["Original_PO_Amount_Sum"] = results_df[
                "Original_PO_Amount_Sum"
            ].fillna(0.0)
        else:
            results_df["Original_PO_Amount_Sum"] = results_df["Base_PO_Amount"]
        results_df["PO_Transaction_Count"] = (
            results_df["PO_Transaction_Count"].fillna(0).astype(int)
        )

        # Calculate Total_Deductions per vendor/dept
        if not deductions_df.empty:
            # Group deductions by Vendor_Prefix and Dept from PO numbers
            # Get available columns from po_df for merging
            merge_cols = ["PO_Number", "Vendor_Prefix", "Dept"]
            if "Payment_Terms" in po_df.columns:
                merge_cols.append("Payment_Terms")

            # Group by columns that exist in the merged data (exclude CC_Txn_Date)
            deduction_group_cols = ["Vendor_Prefix", "Dept"]
            if "Payment_Terms" in po_df.columns:
                deduction_group_cols.append("Payment_Terms")

            # Merge deductions with PO data to get vendor/dept info
            # Use inner join to only include deductions for POs that exist
            deduction_merged = deductions_df.merge(
                po_df[merge_cols],
                on="PO_Number",
                how="inner",  # Only include deductions for POs that exist in po_df
            )

            # Ensure Deduction_Amount is numeric
            deduction_merged["Deduction_Amount"] = pd.to_numeric(
                deduction_merged["Deduction_Amount"], errors="coerce"
            ).fillna(0.0)

            # Group by vendor/dept/payment_terms and SUM all deductions
            deduction_summary = (
                deduction_merged.groupby(deduction_group_cols)
                .agg({"Deduction_Amount": "sum"})
                .reset_index()
                .rename(columns={"Deduction_Amount": "Total_Deductions"})
            )

            # Merge using only the common columns
            results_df = results_df.merge(
                deduction_summary, on=deduction_group_cols, how="left"
            )
            results_df["Total_Deductions"] = results_df["Total_Deductions"].fillna(0.0)
        else:
            results_df["Total_Deductions"] = 0.0

        # Calculate Total_CC_Charge from ORIGINAL amounts (before deductions)
        # This ensures deductions don't affect CC charge calculation
        # We'll calculate it after we have CC_Fee from master data
        # For now, set a placeholder that will be updated after CC_Fee is loaded
        results_df["Total_CC_Charge"] = 0.0
        results_df["Flag"] = np.where(
            results_df["Total_CC_Amount"] > results_df["Total_PO_Amount"],
            "Red Flag",
            "Green Flag",
        )

        # Add Vendor_Name and CC_Fee from master using Vendor_Prefix and Dept
        master_path = Path("data/master")
        try:
            master_df = read_parquet_with_fallback(master_path)
            if "PREFIX" in master_df.columns:
                master_df["PREFIX"] = (
                    master_df["PREFIX"].astype(str).str.strip().str.upper()
                )
                master_df["DEPT"] = (
                    master_df["DEPT"].astype(str).str.strip().str.upper()
                )

                # Get Vendor_Name (using first match per PREFIX)
                if "VENDOR_NAME" in master_df.columns:
                    master_df_unique = master_df.drop_duplicates(
                        subset=["PREFIX"], keep="first"
                    )
                    prefix_to_vendor_name = dict(
                        zip(master_df_unique["PREFIX"], master_df_unique["VENDOR_NAME"])
                    )
                    results_df["Vendor_Name"] = results_df["Vendor_Prefix"].map(
                        prefix_to_vendor_name
                    )
                    results_df["Vendor_Name"] = results_df["Vendor_Name"].fillna("")
                else:
                    results_df["Vendor_Name"] = ""

                # Get CC_Fee (using PREFIX + DEPT combination)
                cc_fee_column = None
                for col in master_df.columns:
                    if "cc" in col.lower() and "fee" in col.lower():
                        cc_fee_column = col
                        break

                if cc_fee_column:
                    # Create lookup: (PREFIX, DEPT) -> CC_Fee
                    master_df["_lookup_key"] = (
                        master_df["PREFIX"].astype(str)
                        + "|"
                        + master_df["DEPT"].astype(str)
                    )
                    cc_fee_lookup = dict(
                        zip(
                            master_df["_lookup_key"],
                            pd.to_numeric(
                                master_df[cc_fee_column], errors="coerce"
                            ).fillna(0.0),
                        )
                    )
                    # Create lookup key for results_df
                    results_df["_lookup_key"] = (
                        results_df["Vendor_Prefix"].astype(str).str.strip().str.upper()
                        + "|"
                        + results_df["Dept"].astype(str).str.strip().str.upper()
                    )
                    results_df["CC_Fee"] = (
                        results_df["_lookup_key"].map(cc_fee_lookup).fillna(0.0)
                    )
                    results_df = results_df.drop(
                        columns=["_lookup_key"], errors="ignore"
                    )

                    # Now calculate Total_CC_Charge from original amounts
                    # Total_CC_Charge = Original_PO_Amount * CC_Fee
                    results_df["Total_CC_Charge"] = (
                        results_df["Original_PO_Amount_Sum"] * results_df["CC_Fee"]
                    )
                else:
                    results_df["CC_Fee"] = 0.0
                    # If no CC_Fee, CC charge is 0
                    results_df["Total_CC_Charge"] = 0.0
            else:
                results_df["Vendor_Name"] = ""
                results_df["CC_Fee"] = 0.0
                results_df["Total_CC_Charge"] = 0.0
        except Exception:  # pylint: disable=broad-except
            results_df["Vendor_Name"] = ""
            results_df["CC_Fee"] = 0.0
            results_df["Total_CC_Charge"] = 0.0

    tab_labels = [
        f"Results ({len(results_df)})",
        f"Mapped ({mapped_df['CC_Reference_ID'].nunique()})",
        f"Common ({common_df['CC_Reference_ID'].nunique()})",
        f"Unmapped ({unmapped_df['CC_Reference_ID'].nunique()})",
        f"Others ({others_df['CC_Reference_ID'].nunique()})",
        "Analysis",
    ]
    tab_results, tab_mapped, tab_common, tab_unmapped, tab_others, tab_analysis = (
        st.tabs(tab_labels)
    )

    with tab_results:
        if results_df.empty:
            st.info("No reconciliation results to summarize yet.")
        else:
            (
                filters_vendor_col,
                filters_dept_col,
                filters_flag_col,
                filters_buffer_col,
            ) = st.columns(4)
            with filters_vendor_col:
                st.selectbox(
                    "Filter Vendor Prefix",
                    options=["<All>"]
                    + sorted(results_df["Vendor_Prefix"].dropna().unique().tolist()),
                    key="results_vendor_filter",
                )
            with filters_dept_col:
                st.selectbox(
                    "Filter Dept",
                    options=["<All>"]
                    + sorted(results_df["Dept"].dropna().unique().tolist()),
                    key="results_dept_filter",
                )
            with filters_flag_col:
                st.selectbox(
                    "Filter Flag",
                    options=["<All>"]
                    + sorted(results_df["Flag"].dropna().unique().tolist()),
                    key="results_flag_filter",
                )
            with filters_buffer_col:
                # Only set value parameter if key doesn't exist in session state
                # This avoids the conflict warning
                widget_kwargs = {
                    "label": "Grace Days Buffer",
                    "min_value": 0,
                    "max_value": 90,
                    "step": 1,
                    "key": "results_grace_days",
                    "help": "Adjust additional buffer days when comparing CC and PO dates.",
                }
                if "results_grace_days" not in st.session_state:
                    widget_kwargs["value"] = 5

                st.number_input(**widget_kwargs)
                # Update grace_days variable from session state after widget creation
                grace_days = int(st.session_state.get("results_grace_days", 5))
            vendor_filter = st.session_state.get("results_vendor_filter")
            if vendor_filter and vendor_filter != "<All>":
                results_df = results_df[
                    results_df["Vendor_Prefix"].astype(str) == vendor_filter
                ]
            dept_filter = st.session_state.get("results_dept_filter")
            if dept_filter and dept_filter != "<All>":
                results_df = results_df[results_df["Dept"].astype(str) == dept_filter]
            flag_filter = st.session_state.get("results_flag_filter")
            if flag_filter and flag_filter != "<All>":
                results_df = results_df[results_df["Flag"].astype(str) == flag_filter]
            results_display = results_df.copy()
            # Reorder columns to show Vendor_Name after Vendor_Prefix, and organize transaction counts
            if "Vendor_Name" in results_display.columns:
                cols = list(results_display.columns)
                if "Vendor_Prefix" in cols and "Vendor_Name" in cols:
                    prefix_idx = cols.index("Vendor_Prefix")
                    name_idx = cols.index("Vendor_Name")
                    # Remove Vendor_Name from current position and insert after Vendor_Prefix
                    cols.pop(name_idx)
                    cols.insert(prefix_idx + 1, "Vendor_Name")
                    results_display = results_display[cols]

            # Format amounts as currency
            results_display["Total_CC_Amount"] = results_display["Total_CC_Amount"].map(
                lambda x: f"${x:,.2f}"
            )
            if "Total_PO_Amount" in results_display.columns:
                results_display["Total_PO_Amount"] = results_display[
                    "Total_PO_Amount"
                ].map(lambda x: f"${x:,.2f}")

            # Format Total_Deductions as currency
            if "Total_Deductions" in results_display.columns:
                results_display["Total_Deductions"] = results_display[
                    "Total_Deductions"
                ].map(lambda x: f"${x:,.2f}")

            # Format amounts as currency (Total_CC_Charge already calculated above)
            if "Total_CC_Charge" in results_display.columns:
                results_display["Total_CC_Charge"] = results_display[
                    "Total_CC_Charge"
                ].map(lambda x: f"${x:,.2f}")

            # Format CC_Fee as actual value (not percentage)
            if "CC_Fee" in results_display.columns:
                results_display["CC_Fee"] = results_display["CC_Fee"].map(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "0.0000"
                )

            st.dataframe(results_display, hide_index=True, use_container_width=True)

    with tab_mapped:
        if mapped_df.empty:
            st.info("No mapped rows available.")
        else:
            mapped_vendor_col, mapped_dept_col = st.columns(2)
            with mapped_vendor_col:
                mapped_vendor_choice = st.selectbox(
                    "Filter Vendor Prefix",
                    options=["<All>"]
                    + sorted(mapped_df["Vendor_Prefix"].dropna().unique().tolist()),
                    key="mapped_vendor_filter",
                )
            with mapped_dept_col:
                mapped_dept_choice = st.selectbox(
                    "Filter Dept",
                    options=["<All>"]
                    + sorted(mapped_df["Dept"].dropna().unique().tolist()),
                    key="mapped_dept_filter",
                )
            mapped_display = mapped_df.copy()
            if mapped_vendor_choice != "<All>":
                mapped_display = mapped_display[
                    mapped_display["Vendor_Prefix"].astype(str) == mapped_vendor_choice
                ]
            if mapped_dept_choice != "<All>":
                mapped_display = mapped_display[
                    mapped_display["Dept"].astype(str) == mapped_dept_choice
                ]
            st.dataframe(mapped_display, hide_index=True, use_container_width=True)

    with tab_common:
        if common_df.empty:
            st.info("No common rows available.")
        else:
            desc_columns = [
                col
                for col in ("CC_Description", "Description")
                if col in common_df.columns
            ]
            if desc_columns:
                normalized_desc = (
                    common_df[desc_columns]
                    .astype(str)
                    .apply(lambda col: col.str.strip().str.upper())
                    .agg(
                        lambda row: next(
                            (value for value in row if value and value != "NAN"), ""
                        ),
                        axis=1,
                    )
                )
                ref_series_for_counts = common_df.get("CC_Reference_ID")
                if ref_series_for_counts is not None:
                    ref_series_for_counts = ref_series_for_counts.astype(
                        str
                    ).str.strip()
                    ref_series_for_counts = ref_series_for_counts.where(
                        ~ref_series_for_counts.str.lower().isin({"nan", "none", ""}),
                        "",
                    )
                    desc_ref_df = pd.DataFrame(
                        {
                            "normalized_desc": normalized_desc,
                            "CC_Reference_ID": ref_series_for_counts,
                        }
                    )
                    description_counts = (
                        desc_ref_df[
                            (desc_ref_df["normalized_desc"] != "")
                            & desc_ref_df["CC_Reference_ID"].ne("")
                        ]
                        .groupby("normalized_desc")["CC_Reference_ID"]
                        .nunique()
                        .sort_index()
                    )
                else:
                    description_counts = (
                        normalized_desc[normalized_desc != ""]
                        .value_counts()
                        .sort_index()
                    )
                description_option_map = {
                    f"{desc} ({count})": desc
                    for desc, count in description_counts.items()
                }
                description_options = ["<All>"] + list(description_option_map.keys())
            else:
                description_option_map = {}
                description_options = ["<All>"]

            description_filter = st.selectbox(
                "Filter Description",
                options=description_options,
                key="common_desc_filter",
                help="Select description to filter common rows.",
            )
            filtered_common = common_df.copy()
            if description_filter and description_filter != "<All>":
                selected_desc_key = description_option_map.get(description_filter)
                if selected_desc_key:
                    desc_columns = [
                        col
                        for col in ("CC_Description", "Description")
                        if col in filtered_common.columns
                    ]
                    if desc_columns:
                        combined_mask = pd.Series(False, index=filtered_common.index)
                        for col in desc_columns:
                            combined_mask = combined_mask | (
                                filtered_common[col].astype(str).str.strip().str.upper()
                                == selected_desc_key
                            )
                        filtered_common = filtered_common[combined_mask]
            if filtered_common.empty:
                st.info("No common rows match the current filter.")
            else:
                common_df = filtered_common
                ref_series = common_df.get("CC_Reference_ID")
                if ref_series is None:
                    st.warning(
                        "Unable to summarize common rows: column `CC_Reference_ID` is missing."
                    )
                else:
                    ref_series = ref_series.astype(str).str.strip()

                desc_column = next(
                    (
                        col
                        for col in ("CC_Description", "Description")
                        if col in common_df
                    ),
                    None,
                )
                amt_column = next(
                    (col for col in ("CC_Amt", "Amount") if col in common_df), None
                )

                if amt_column is None:
                    st.warning(
                        "Unable to summarize common rows: amount column `CC_Amt` (or `Amount`) is missing."
                    )
                    display_df = pd.DataFrame(
                        columns=["Description", "Amount", "Dept", "Transaction_Count"]
                    )
                else:
                    # Group by description (normalized) instead of CC_Reference_ID
                    working_df = common_df.copy()
                    # Deduplicate by CC_Reference_ID to avoid double counting amounts
                    if "CC_Reference_ID" in working_df.columns:
                        working_df = working_df.drop_duplicates(
                            subset=["CC_Reference_ID"], keep="first"
                        )
                    working_df["Amount"] = pd.to_numeric(
                        working_df[amt_column], errors="coerce"
                    )

                    if desc_column:
                        # Normalize description for grouping
                        working_df["_norm_desc"] = (
                            working_df[desc_column].astype(str).str.strip().str.upper()
                        )
                    else:
                        working_df["_norm_desc"] = ""

                    # Define aggregation function for reuse
                    def agg_desc_group(group: pd.DataFrame) -> pd.Series:
                        descriptions = (
                            [
                                value
                                for value in group[desc_column]
                                .dropna()
                                .astype(str)
                                .str.strip()
                                .unique()
                                .tolist()
                                if value and value.lower() not in {"nan", "none", ""}
                            ]
                            if desc_column
                            else []
                        )

                        description_value = descriptions[0] if descriptions else ""

                        valid_amounts = group["Amount"].dropna()
                        if valid_amounts.empty:
                            total_amount = pd.NA
                        else:
                            total_amount = valid_amounts.sum()

                        transaction_count = (
                            group["CC_Reference_ID"].nunique()
                            if "CC_Reference_ID" in group.columns
                            else len(group)
                        )

                        # Get Vendor_Prefix (should be same for all rows in group)
                        vendor_prefix_value = ""
                        if "Vendor_Prefix" in group.columns:
                            vendor_prefixes = (
                                group["Vendor_Prefix"]
                                .dropna()
                                .astype(str)
                                .str.strip()
                                .unique()
                                .tolist()
                            )
                            vendor_prefix_value = (
                                vendor_prefixes[0] if vendor_prefixes else ""
                            )

                        return pd.Series(
                            {
                                "Description": description_value,
                                "Amount": total_amount,
                                "Transaction_Count": transaction_count,
                                "Dept": "",
                                "Vendor_Prefix": vendor_prefix_value,
                            }
                        )

                    display_df = (
                        working_df.groupby("_norm_desc", as_index=False)
                        .apply(agg_desc_group)
                        .reset_index(drop=True)
                    )

                    # Remove the temporary column
                    if "_norm_desc" in display_df.columns:
                        display_df = display_df.drop(columns=["_norm_desc"])

                # Ensure correct column order
                if "Transaction_Count" not in display_df.columns:
                    display_df["Transaction_Count"] = 1

                # Ensure Vendor_Prefix column exists
                if "Vendor_Prefix" not in display_df.columns:
                    display_df["Vendor_Prefix"] = ""

                display_df = display_df[
                    [
                        "Description",
                        "Amount",
                        "Transaction_Count",
                        "Dept",
                        "Vendor_Prefix",
                    ]
                ]

                display_df["Amount"] = display_df["Amount"].apply(
                    lambda value: round(float(value), 2)
                    if pd.notna(value) and value != ""
                    else value
                )

                # Load saved Dept mappings from file (by CC_Reference_ID, aggregated by description)
                # Only considers CC_Reference_IDs in current common_df (batch-specific)
                saved_dept_map = get_dept_mappings_for_common(common_df)

                # Merge with session state (session state takes precedence for current session)
                existing_dept_choices: dict[str, str] = st.session_state.setdefault(
                    "common_dept_choices", {}
                )
                # Update session state with saved mappings if not already set
                for desc_norm, dept in saved_dept_map.items():
                    if desc_norm not in existing_dept_choices:
                        existing_dept_choices[desc_norm] = dept

                # Auto-assign FBM if total amount < 100
                def assign_dept(row):
                    desc_key = str(row["Description"]).strip().upper()
                    # Check session state first (user's current selection)
                    if desc_key in existing_dept_choices:
                        return existing_dept_choices[desc_key]
                    # Check saved mappings
                    if desc_key in saved_dept_map:
                        return saved_dept_map[desc_key]
                    # Auto-assign FBM if amount < 100
                    if pd.notna(row["Amount"]) and float(row["Amount"]) < 100:
                        return "FBM"
                    # Otherwise keep existing value or empty
                    return (
                        row["Dept"]
                        if pd.notna(row["Dept"]) and str(row["Dept"]).strip()
                        else ""
                    )

                display_df["Dept"] = display_df.apply(assign_dept, axis=1)

                existing_apply_flags: dict[str, bool] = st.session_state.setdefault(
                    "common_apply_flags", {}
                )
                # Auto-apply for amounts < 100
                display_df["Apply"] = display_df.apply(
                    lambda row: (
                        bool(
                            existing_apply_flags.get(
                                str(row["Description"]).strip().upper(), False
                            )
                        )
                        or (pd.notna(row["Amount"]) and float(row["Amount"]) < 100)
                    ),
                    axis=1,
                )

                # Auto-process descriptions with sum < 100 immediately (before showing form)
                auto_process_mask = display_df["Amount"].notna() & (
                    pd.to_numeric(display_df["Amount"], errors="coerce") < 100
                )
                auto_process_rows = display_df[auto_process_mask].copy()

                # Process auto-FBM items immediately if any exist
                if not auto_process_rows.empty:
                    # Force FBM and Apply for auto-process items
                    auto_process_rows["Dept"] = "FBM"
                    auto_process_rows["Apply"] = True

                    # Process auto-FBM items directly
                    try:
                        mappings_df = read_parquet_with_fallback(Path("data/mappings"))
                    except Exception as exc:  # pylint: disable=broad-except
                        st.error(f"Unable to load mappings file: {exc}")
                        mappings_df = None

                    try:
                        master_df = read_parquet_with_fallback(Path("data/master"))
                    except Exception as exc:  # pylint: disable=broad-except
                        st.error(f"Unable to load master file: {exc}")
                        master_df = None

                    if mappings_df is not None and master_df is not None:
                        # Process auto-FBM rows directly (same logic as below)
                        mappings_df["DESCRIPTION"] = (
                            mappings_df["DESCRIPTION"]
                            .astype(str)
                            .str.replace(
                                r"\s+", " ", regex=True
                            )  # Normalize multiple spaces
                            .str.strip()
                        )
                        mappings_df["_norm_description"] = mappings_df[
                            "DESCRIPTION"
                        ].str.upper()
                        prefix_lookup = (
                            mappings_df.groupby("_norm_description")["PREFIX"]
                            .apply(
                                lambda series: [
                                    str(prefix).strip()
                                    for prefix in series
                                    if str(prefix).strip()
                                ]
                            )
                            .to_dict()
                        )

                        master_df = master_df.copy()
                        if "PREFIX" in master_df.columns:
                            master_df["PREFIX"] = (
                                master_df["PREFIX"].astype(str).str.strip()
                            )
                        if "DEPT" in master_df.columns:
                            master_df["DEPT"] = (
                                master_df["DEPT"].astype(str).str.strip().str.upper()
                            )
                        if "PAYMENT TERMS" in master_df.columns:
                            master_df["PAYMENT TERMS"] = (
                                master_df["PAYMENT TERMS"].astype(str).str.strip()
                            )

                        auto_processed_descriptions: list[str] = []
                        for row in auto_process_rows.to_dict("records"):
                            description_value = str(row["Description"] or "").strip()
                            description_norm = description_value.upper()
                            dept_value = "FBM"  # Always FBM for auto-processed
                            amount_value = row["Amount"]
                            amount_numeric = pd.to_numeric(
                                pd.Series([amount_value], dtype="object"),
                                errors="coerce",
                            ).iloc[0]

                            # Find all rows in common_df with this description
                            desc_mask = pd.Series(False, index=common_df.index)
                            if desc_column:
                                desc_mask = (
                                    common_df[desc_column]
                                    .astype(str)
                                    .str.strip()
                                    .str.upper()
                                    == description_norm
                                )

                            matching_common_rows = common_df.loc[desc_mask].copy()

                            if matching_common_rows.empty:
                                continue

                            # Get prefix and payment terms (prioritize FBM)
                            prefix_candidates = prefix_lookup.get(description_norm, [])
                            chosen_prefix = ""
                            payment_terms_value: object = pd.NA
                            vendor_name_value: object = pd.NA

                            # Always search for FBM dept for auto-processed items
                            search_dept = "FBM"
                            for candidate in prefix_candidates:
                                matched_rows = master_df[
                                    (master_df.get("PREFIX", "") == candidate)
                                    & (master_df.get("DEPT", "") == search_dept)
                                ]
                                if not matched_rows.empty:
                                    chosen_prefix = candidate
                                    payment_terms_value = matched_rows.iloc[0].get(
                                        "PAYMENT TERMS", pd.NA
                                    )
                                    vendor_name_value = matched_rows.iloc[0].get(
                                        "VENDOR_NAME", pd.NA
                                    )
                                    break

                            if not chosen_prefix and prefix_candidates:
                                chosen_prefix = prefix_candidates[0]
                                matched_rows = master_df[
                                    master_df.get("PREFIX", "") == chosen_prefix
                                ]
                                if not matched_rows.empty:
                                    # Try to get FBM payment terms
                                    fbm_rows = matched_rows[
                                        matched_rows.get("DEPT", "") == "FBM"
                                    ]
                                    if not fbm_rows.empty:
                                        payment_terms_value = fbm_rows.iloc[0].get(
                                            "PAYMENT TERMS", pd.NA
                                        )
                                        vendor_name_value = fbm_rows.iloc[0].get(
                                            "VENDOR_NAME", pd.NA
                                        )
                                    else:
                                        payment_terms_value = matched_rows.iloc[0].get(
                                            "PAYMENT TERMS", pd.NA
                                        )
                                        vendor_name_value = matched_rows.iloc[0].get(
                                            "VENDOR_NAME", pd.NA
                                        )

                            if not chosen_prefix:
                                continue

                            # Set category to ONLY FBM for auto-processed items
                            category_value = "ONLY FBM"

                            # Get all reference IDs for this description
                            matching_ref_ids = (
                                matching_common_rows["CC_Reference_ID"]
                                .dropna()
                                .astype(str)
                                .str.strip()
                                .unique()
                                .tolist()
                            )

                            # Save Dept mapping for these reference IDs (batch-specific)
                            save_dept_mappings(matching_ref_ids, dept_value)

                            # Remove all existing rows with these reference IDs
                            ref_mask = (
                                reco_df["CC_Reference_ID"]
                                .astype(str)
                                .str.strip()
                                .isin(matching_ref_ids)
                            )

                            existing_rows = reco_df.loc[ref_mask].copy()
                            existing_reco_ids = (
                                existing_rows["Reco_ID"]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            )
                            reco_id_value = (
                                existing_reco_ids[0] if existing_reco_ids else pd.NA
                            )
                            existing_batch_ids = (
                                existing_rows.get(
                                    "Import_Batch_ID", pd.Series(dtype=str)
                                )
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            )
                            import_batch_value = (
                                existing_batch_ids[0] if existing_batch_ids else pd.NA
                            )

                            # Get the first transaction date from matching rows
                            first_txn_date = None
                            if "CC_Txn_Date" in matching_common_rows.columns:
                                txn_dates = matching_common_rows["CC_Txn_Date"].dropna()
                                if not txn_dates.empty:
                                    first_txn_date = txn_dates.iloc[0]

                            # Create consolidated entry
                            primary_ref_id = (
                                matching_ref_ids[0] if matching_ref_ids else ""
                            )

                            updated_df = reco_df.loc[~ref_mask].copy()

                            new_row = {column: pd.NA for column in reco_df.columns}
                            if "CC_Reference_ID" in new_row:
                                new_row["CC_Reference_ID"] = primary_ref_id
                            if "CC_Description" in new_row:
                                new_row["CC_Description"] = description_value
                            elif "Description" in new_row:
                                new_row["Description"] = description_value

                            if "CC_Amt" in new_row:
                                new_row["CC_Amt"] = (
                                    float(amount_numeric)
                                    if pd.notna(amount_numeric)
                                    else pd.NA
                                )
                            elif "Amount" in new_row:
                                new_row["Amount"] = (
                                    float(amount_numeric)
                                    if pd.notna(amount_numeric)
                                    else pd.NA
                                )

                            if "CC_Txn_Date" in new_row and first_txn_date:
                                new_row["CC_Txn_Date"] = first_txn_date
                            if "Import_Batch_ID" in new_row:
                                new_row["Import_Batch_ID"] = (
                                    import_batch_value
                                    if import_batch_value is not pd.NA
                                    else pd.NA
                                )

                            if "Vendor_Prefix" in new_row:
                                new_row["Vendor_Prefix"] = chosen_prefix
                            if "Vendor_Name" in new_row:
                                new_row["Vendor_Name"] = vendor_name_value
                            if "Category" in new_row:
                                new_row["Category"] = category_value
                            if "Dept" in new_row:
                                new_row["Dept"] = dept_value
                            if "Payment_Terms" in new_row:
                                new_row["Payment_Terms"] = (
                                    pd.to_numeric(payment_terms_value, errors="coerce")
                                    if pd.notna(payment_terms_value)
                                    else 0
                                )
                            if "PAYMENT TERMS" in new_row:
                                new_row["PAYMENT TERMS"] = (
                                    pd.to_numeric(payment_terms_value, errors="coerce")
                                    if pd.notna(payment_terms_value)
                                    else 0
                                )
                            if "Reco_ID" in new_row:
                                new_row["Reco_ID"] = reco_id_value

                            updated_df = pd.concat(
                                [updated_df, pd.DataFrame([new_row])],
                                ignore_index=True,
                            )
                            reco_df = updated_df
                            auto_processed_descriptions.append(description_value)
                            st.session_state["common_apply_flags"][description_norm] = (
                                False
                            )

                        if auto_processed_descriptions:
                            # Update session state with modified reco_df
                            st.session_state["reco_df"] = reco_df
                            summary_descs = ", ".join(auto_processed_descriptions[:5])
                            if len(auto_processed_descriptions) > 5:
                                summary_descs += ", ..."
                            st.success(
                                f"Auto-processed {len(auto_processed_descriptions)} description(s) with sum < $100 as FBM: "
                                + summary_descs
                            )
                            # Rerun to refresh the page
                            if hasattr(st, "rerun"):
                                st.rerun()
                            else:  # pragma: no cover
                                st.experimental_rerun()

                with st.form("common_apply_form", clear_on_submit=False):
                    edited_df = st.data_editor(
                        display_df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Description": st.column_config.TextColumn(
                                "Description",
                                disabled=True,
                            ),
                            "Amount": st.column_config.NumberColumn(
                                "Amount",
                                format="$%0.2f",
                                disabled=True,
                            ),
                            "Transaction_Count": st.column_config.NumberColumn(
                                "Transaction Count",
                                format="%d",
                                disabled=True,
                            ),
                            "Dept": st.column_config.SelectboxColumn(
                                "Dept",
                                options=["FBA", "FBM"],
                                required=False,
                                help="Choose the department. Amounts < $100 auto-assign FBM.",
                            ),
                            "Vendor_Prefix": st.column_config.TextColumn(
                                "Vendor Prefix",
                                disabled=True,
                                help="Current vendor prefix for this description group.",
                            ),
                            "Apply": st.column_config.CheckboxColumn(
                                "Apply",
                                help=(
                                    "Auto-checked for amounts < $100. Check to process this description group. "
                                    "All transactions with the same description will be consolidated."
                                ),
                            ),
                        },
                        key="common_summary_editor",
                    )

                    apply_submitted = st.form_submit_button(
                        "Apply Selected", type="primary", use_container_width=True
                    )

                # Update session state for display even if not submitted
                edited_df["Dept"] = (
                    edited_df["Dept"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .replace("NAN", "")
                )
                edited_df["Apply"] = edited_df["Apply"].fillna(False).astype(bool)

                # Save Dept selections to file by CC_Reference_ID (even if not applied)
                # Only saves for CC_Reference_IDs in current common_df (batch-specific)
                desc_column = next(
                    (
                        col
                        for col in ("CC_Description", "Description")
                        if col in common_df.columns
                    ),
                    None,
                )
                if desc_column:
                    for _, row in edited_df.iterrows():
                        if row["Dept"] in {"FBA", "FBM"}:
                            desc_value = str(row["Description"]).strip()
                            desc_norm = desc_value.upper()

                            # Find all CC_Reference_IDs with this description in current common_df
                            # (common_df is already filtered by selected batch)
                            desc_mask = (
                                common_df[desc_column]
                                .astype(str)
                                .str.strip()
                                .str.upper()
                                == desc_norm
                            )
                            matching_rows = common_df.loc[desc_mask]

                            if (
                                not matching_rows.empty
                                and "CC_Reference_ID" in matching_rows.columns
                            ):
                                ref_ids = (
                                    matching_rows["CC_Reference_ID"]
                                    .dropna()
                                    .astype(str)
                                    .str.strip()
                                    .unique()
                                    .tolist()
                                )
                                if ref_ids:
                                    # Save mapping for these specific CC_Reference_IDs (batch-specific)
                                    save_dept_mappings(ref_ids, row["Dept"])

                # Update session state
                st.session_state["common_dept_choices"] = {
                    str(row["Description"]).strip().upper(): row["Dept"]
                    for _, row in edited_df.iterrows()
                    if row["Dept"] in {"FBA", "FBM"}
                }
                st.session_state["common_apply_flags"] = {
                    str(row["Description"]).strip().upper(): bool(row["Apply"])
                    for _, row in edited_df.iterrows()
                }

                # Only process manually selected items if form was submitted
                apply_rows = (
                    edited_df[edited_df["Apply"]].copy()
                    if apply_submitted
                    else pd.DataFrame()
                )

                if not apply_rows.empty:
                    # Auto-assign FBM for amounts < 100 if not already assigned
                    for idx, row in apply_rows.iterrows():
                        if pd.notna(row["Amount"]) and float(row["Amount"]) < 100:
                            apply_rows.loc[idx, "Dept"] = "FBM"
                            st.info(
                                f"Auto-assigned FBM for description '{row['Description']}' "
                                f"(sum amount ${row['Amount']:.2f} < $100)"
                            )

                    missing_dept_refs = (
                        apply_rows[~apply_rows["Dept"].isin({"FBA", "FBM"})][
                            "Description"
                        ]
                        .astype(str)
                        .tolist()
                    )
                    if missing_dept_refs:
                        st.warning(
                            "Select either FBA or FBM before applying for: "
                            + ", ".join(missing_dept_refs[:10])
                            + ("..." if len(missing_dept_refs) > 10 else "")
                        )
                    else:
                        try:
                            mappings_df = read_parquet_with_fallback(
                                Path("data/mappings")
                            )
                        except Exception as exc:  # pylint: disable=broad-except
                            st.error(f"Unable to load mappings file: {exc}")
                            mappings_df = None

                        try:
                            master_df = read_parquet_with_fallback(Path("data/master"))
                        except Exception as exc:  # pylint: disable=broad-except
                            st.error(f"Unable to load master file: {exc}")
                            master_df = None

                        if mappings_df is not None and master_df is not None:
                            mappings_df["DESCRIPTION"] = (
                                mappings_df["DESCRIPTION"].astype(str).str.strip()
                            )
                            mappings_df["_norm_description"] = mappings_df[
                                "DESCRIPTION"
                            ].str.upper()
                            prefix_lookup = (
                                mappings_df.groupby("_norm_description")["PREFIX"]
                                .apply(
                                    lambda series: [
                                        str(prefix).strip()
                                        for prefix in series
                                        if str(prefix).strip()
                                    ]
                                )
                                .to_dict()
                            )

                            master_df = master_df.copy()
                            if "PREFIX" in master_df.columns:
                                master_df["PREFIX"] = (
                                    master_df["PREFIX"].astype(str).str.strip()
                                )
                            if "DEPT" in master_df.columns:
                                master_df["DEPT"] = (
                                    master_df["DEPT"]
                                    .astype(str)
                                    .str.strip()
                                    .str.upper()
                                )
                            if "PAYMENT TERMS" in master_df.columns:
                                master_df["PAYMENT TERMS"] = (
                                    master_df["PAYMENT TERMS"].astype(str).str.strip()
                                )

                            processed_descriptions: list[str] = []
                            missing_prefix_descriptions: list[str] = []
                            for row in apply_rows.to_dict("records"):
                                description_value = str(
                                    row["Description"] or ""
                                ).strip()
                                description_norm = description_value.upper()
                                dept_value = row["Dept"]
                                amount_value = row["Amount"]
                                amount_numeric = pd.to_numeric(
                                    pd.Series([amount_value], dtype="object"),
                                    errors="coerce",
                                ).iloc[0]

                                # If amount < 100, force FBM
                                if (
                                    pd.notna(amount_numeric)
                                    and float(amount_numeric) < 100
                                ):
                                    dept_value = "FBM"

                                if dept_value not in {"FBA", "FBM"}:
                                    continue

                                # Find all rows in common_df with this description
                                desc_mask = pd.Series(False, index=common_df.index)
                                if desc_column:
                                    desc_mask = (
                                        common_df[desc_column]
                                        .astype(str)
                                        .str.strip()
                                        .str.upper()
                                        == description_norm
                                    )

                                matching_common_rows = common_df.loc[desc_mask].copy()

                                if matching_common_rows.empty:
                                    continue

                                # Get prefix and payment terms
                                prefix_candidates = prefix_lookup.get(
                                    description_norm, []
                                )
                                chosen_prefix = ""
                                payment_terms_value: object = pd.NA
                                vendor_name_value: object = pd.NA
                                # When amount < 100, prioritize FBM matches
                                search_dept = dept_value
                                for candidate in prefix_candidates:
                                    matched_rows = master_df[
                                        (master_df.get("PREFIX", "") == candidate)
                                        & (master_df.get("DEPT", "") == search_dept)
                                    ]
                                    if not matched_rows.empty:
                                        chosen_prefix = candidate
                                        payment_terms_value = matched_rows.iloc[0].get(
                                            "PAYMENT TERMS", pd.NA
                                        )
                                        vendor_name_value = matched_rows.iloc[0].get(
                                            "VENDOR_NAME", pd.NA
                                        )
                                        break
                                if not chosen_prefix and prefix_candidates:
                                    chosen_prefix = prefix_candidates[0]
                                    matched_rows = master_df[
                                        master_df.get("PREFIX", "") == chosen_prefix
                                    ]
                                    if not matched_rows.empty:
                                        # If amount < 100, prioritize FBM dept match
                                        if (
                                            "DEPT" in matched_rows.columns
                                            and dept_value
                                            in matched_rows["DEPT"].values
                                        ):
                                            matched_dept_rows = matched_rows[
                                                matched_rows["DEPT"] == dept_value
                                            ]
                                            payment_terms_value = (
                                                matched_dept_rows.iloc[0].get(
                                                    "PAYMENT TERMS", pd.NA
                                                )
                                            )
                                            vendor_name_value = matched_dept_rows.iloc[
                                                0
                                            ].get("VENDOR_NAME", pd.NA)
                                        else:
                                            # If amount < 100, try to get FBM payment terms
                                            if (
                                                pd.notna(amount_numeric)
                                                and float(amount_numeric) < 100
                                            ):
                                                fbm_rows = matched_rows[
                                                    matched_rows.get("DEPT", "")
                                                    == "FBM"
                                                ]
                                                if not fbm_rows.empty:
                                                    payment_terms_value = fbm_rows.iloc[
                                                        0
                                                    ].get("PAYMENT TERMS", pd.NA)
                                                    vendor_name_value = fbm_rows.iloc[
                                                        0
                                                    ].get("VENDOR_NAME", pd.NA)
                                                else:
                                                    payment_terms_value = (
                                                        matched_rows.iloc[0].get(
                                                            "PAYMENT TERMS", pd.NA
                                                        )
                                                    )
                                                    vendor_name_value = (
                                                        matched_rows.iloc[0].get(
                                                            "VENDOR_NAME", pd.NA
                                                        )
                                                    )
                                            else:
                                                payment_terms_value = matched_rows.iloc[
                                                    0
                                                ].get("PAYMENT TERMS", pd.NA)
                                                vendor_name_value = matched_rows.iloc[
                                                    0
                                                ].get("VENDOR_NAME", pd.NA)

                                if not chosen_prefix:
                                    missing_prefix_descriptions.append(
                                        description_value
                                    )
                                    continue

                                # If amount < 100, ensure it's FBM and set category accordingly
                                if (
                                    pd.notna(amount_numeric)
                                    and float(amount_numeric) < 100
                                ):
                                    dept_value = "FBM"
                                    category_value = "ONLY FBM"
                                else:
                                    category_value = (
                                        "ONLY FBA"
                                        if dept_value == "FBA"
                                        else "ONLY FBM"
                                    )

                                # Get all reference IDs for this description
                                matching_ref_ids = (
                                    matching_common_rows["CC_Reference_ID"]
                                    .dropna()
                                    .astype(str)
                                    .str.strip()
                                    .unique()
                                    .tolist()
                                )

                                # Save Dept mapping for these reference IDs (batch-specific)
                                save_dept_mappings(matching_ref_ids, dept_value)

                                # Remove all existing rows with these reference IDs
                                ref_mask = (
                                    reco_df["CC_Reference_ID"]
                                    .astype(str)
                                    .str.strip()
                                    .isin(matching_ref_ids)
                                )

                                existing_rows = reco_df.loc[ref_mask].copy()
                                existing_reco_ids = (
                                    existing_rows["Reco_ID"]
                                    .dropna()
                                    .astype(str)
                                    .unique()
                                    .tolist()
                                )
                                reco_id_value = (
                                    existing_reco_ids[0] if existing_reco_ids else pd.NA
                                )
                                existing_batch_ids = (
                                    existing_rows.get(
                                        "Import_Batch_ID", pd.Series(dtype=str)
                                    )
                                    .dropna()
                                    .astype(str)
                                    .unique()
                                    .tolist()
                                )
                                import_batch_value = (
                                    existing_batch_ids[0]
                                    if existing_batch_ids
                                    else pd.NA
                                )

                                # Get the first transaction date from matching rows
                                first_txn_date = None
                                if "CC_Txn_Date" in matching_common_rows.columns:
                                    txn_dates = matching_common_rows[
                                        "CC_Txn_Date"
                                    ].dropna()
                                    if not txn_dates.empty:
                                        first_txn_date = txn_dates.iloc[0]

                                # Create consolidated entry for this description
                                # Use the first reference ID as the primary one
                                primary_ref_id = (
                                    matching_ref_ids[0] if matching_ref_ids else ""
                                )

                                updated_df = reco_df.loc[~ref_mask].copy()

                                new_row = {column: pd.NA for column in reco_df.columns}
                                if "CC_Reference_ID" in new_row:
                                    new_row["CC_Reference_ID"] = primary_ref_id
                                if "CC_Description" in new_row:
                                    new_row["CC_Description"] = description_value
                                elif "Description" in new_row:
                                    new_row["Description"] = description_value

                                if "CC_Amt" in new_row:
                                    new_row["CC_Amt"] = (
                                        float(amount_numeric)
                                        if pd.notna(amount_numeric)
                                        else pd.NA
                                    )
                                elif "Amount" in new_row:
                                    new_row["Amount"] = (
                                        float(amount_numeric)
                                        if pd.notna(amount_numeric)
                                        else pd.NA
                                    )

                                if "CC_Txn_Date" in new_row and first_txn_date:
                                    new_row["CC_Txn_Date"] = first_txn_date
                                if "Import_Batch_ID" in new_row:
                                    new_row["Import_Batch_ID"] = (
                                        import_batch_value
                                        if import_batch_value is not pd.NA
                                        else pd.NA
                                    )

                                if "Vendor_Prefix" in new_row:
                                    new_row["Vendor_Prefix"] = chosen_prefix
                                if "Vendor_Name" in new_row:
                                    new_row["Vendor_Name"] = vendor_name_value
                                if "Category" in new_row:
                                    new_row["Category"] = category_value
                                if "Dept" in new_row:
                                    new_row["Dept"] = dept_value
                                if "Payment_Terms" in new_row:
                                    new_row["Payment_Terms"] = payment_terms_value
                                if "PAYMENT TERMS" in new_row:
                                    new_row["PAYMENT TERMS"] = payment_terms_value
                                if "Reco_ID" in new_row:
                                    new_row["Reco_ID"] = reco_id_value

                                updated_df = pd.concat(
                                    [updated_df, pd.DataFrame([new_row])],
                                    ignore_index=True,
                                )
                                reco_df = updated_df
                                processed_descriptions.append(description_value)
                                st.session_state["common_apply_flags"][
                                    description_norm
                                ] = False

                            if missing_prefix_descriptions:
                                st.warning(
                                    "Unable to determine vendor prefix for: "
                                    + ", ".join(missing_prefix_descriptions[:10])
                                    + (
                                        "..."
                                        if len(missing_prefix_descriptions) > 10
                                        else ""
                                    )
                                )

                            if processed_descriptions:
                                # Update session state with modified reco_df
                                st.session_state["reco_df"] = reco_df
                                summary_descs = ", ".join(processed_descriptions[:5])
                                if len(processed_descriptions) > 5:
                                    summary_descs += ", ..."
                                st.session_state["reco_success"] = (
                                    f"Updated {len(processed_descriptions)} description group(s): "
                                    + summary_descs
                                )
                                # Don't set pending_active_page since we're already on Reco page
                                if hasattr(st, "rerun"):
                                    st.rerun()
                                else:  # pragma: no cover
                                    st.experimental_rerun()

    with tab_unmapped:
        if unmapped_df.empty:
            st.success("Great! No unmapped rows.")
        else:
            description_column = next(
                (
                    col
                    for col in ("CC_Description", "Description")
                    if col in unmapped_df.columns
                ),
                None,
            )
            amount_column = next(
                (col for col in ("CC_Amt", "Amount") if col in unmapped_df.columns),
                None,
            )

            if description_column is None:
                st.warning(
                    "Unable to group unmapped rows: description column (`CC_Description` or `Description`) is missing."
                )
                st.dataframe(unmapped_df, hide_index=True, use_container_width=True)
            else:
                master_path = Path("data/master")
                try:
                    master_df = read_parquet_with_fallback(master_path)
                except FileNotFoundError:
                    st.error("Vendor master file not found.")
                    master_df = pd.DataFrame()
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Unable to read vendor master file: {exc}")
                    master_df = pd.DataFrame()

                def normalize_description(series: pd.Series) -> pd.Series:
                    return series.fillna("").astype(str).str.strip().str.lower()

                normalized_unmapped_desc = normalize_description(
                    unmapped_df[description_column]
                )
                if description_column in reco_df.columns:
                    reco_description_series = normalize_description(
                        reco_df[description_column]
                    )
                else:
                    reco_description_series = pd.Series(
                        [""] * len(reco_df), index=reco_df.index
                    )

                unmapped_working = unmapped_df.copy()
                unmapped_working["_description_key"] = normalized_unmapped_desc

                grouped_records: list[dict[str, object]] = []

                for description_key, group in unmapped_working.groupby(
                    "_description_key", dropna=False
                ):
                    descriptions = (
                        group[description_column].dropna().astype(str).str.strip()
                    )
                    display_description = (
                        descriptions.iloc[0] if not descriptions.empty else ""
                    )
                    count_value = int(group.shape[0])
                    total_amount = None
                    if amount_column:
                        total_amount = float(
                            pd.to_numeric(group[amount_column], errors="coerce")
                            .fillna(0.0)
                            .sum()
                        )
                    # Get Vendor_Prefix (should be same for all rows in group)
                    vendor_prefix_value = ""
                    if "Vendor_Prefix" in group.columns:
                        vendor_prefixes = (
                            group["Vendor_Prefix"]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .unique()
                            .tolist()
                        )
                        vendor_prefix_value = (
                            vendor_prefixes[0] if vendor_prefixes else ""
                        )
                    grouped_records.append(
                        {
                            "description_key": description_key,
                            "display_description": display_description,
                            "count": count_value,
                            "total_amount": total_amount,
                            "vendor_prefix": vendor_prefix_value,
                        }
                    )

                if not grouped_records:
                    st.info("No unmapped descriptions available for grouping.")

                if master_df.empty:
                    st.info(
                        "Populate the vendor master file to map vendors against unmapped descriptions."
                    )
                else:
                    mappings_path = Path("data/mappings")
                    try:
                        mappings_df = read_parquet_with_fallback(mappings_path)
                    except FileNotFoundError:
                        mappings_df = pd.DataFrame(columns=["DESCRIPTION", "PREFIX"])
                    except Exception as exc:  # pylint: disable=broad-except
                        st.error(f"Unable to read vendor mappings file: {exc}")
                        mappings_df = pd.DataFrame(columns=["DESCRIPTION", "PREFIX"])

                    missing_master_columns = {
                        column
                        for column in ("PREFIX", "CATEGORY", "DEPT")
                        if column not in master_df.columns
                    }
                    if missing_master_columns:
                        st.error(
                            "Vendor master file is missing required columns: "
                            + ", ".join(sorted(missing_master_columns))
                        )
                    else:
                        master_options: list[str] = []
                        master_lookup: dict[str, dict[str, object]] = {}
                        for _, master_row in master_df.iterrows():
                            prefix_value = str(master_row.get("PREFIX", "")).strip()
                            category_value = str(master_row.get("CATEGORY", "")).strip()
                            dept_value = str(master_row.get("DEPT", "")).strip()
                            payment_terms_value = str(
                                master_row.get("PAYMENT TERMS", "")
                            ).strip()

                            label_parts = [
                                value
                                for value in (prefix_value, category_value, dept_value)
                                if value
                            ]
                            if payment_terms_value:
                                label_parts.append(payment_terms_value)
                            if not label_parts:
                                continue
                            option_label = " · ".join(label_parts)
                            master_options.append(option_label)
                            master_lookup[option_label] = master_row.to_dict()

                        if not master_options:
                            st.info(
                                "Vendor master has no usable entries (missing prefix/category/department)."
                            )
                        else:
                            master_options = sorted(set(master_options))
                            editor_index = [
                                str(record["description_key"])
                                for record in grouped_records
                            ]
                            editor_df = pd.DataFrame(
                                {
                                    "Description": [
                                        record["display_description"]
                                        or "(blank description)"
                                        for record in grouped_records
                                    ],
                                    "Transaction_Count": [
                                        record["count"] for record in grouped_records
                                    ],
                                    "Total_CC_Amount": [
                                        record["total_amount"]
                                        for record in grouped_records
                                    ],
                                    "Vendor_Prefix": [
                                        record.get("vendor_prefix", "")
                                        for record in grouped_records
                                    ],
                                    "Vendor": ["Select a vendor"]
                                    * len(grouped_records),
                                    "Apply": [False] * len(grouped_records),
                                },
                                index=editor_index,
                            )

                            with st.form("unmapped_apply_form", clear_on_submit=False):
                                edited_df = st.data_editor(
                                    editor_df,
                                    hide_index=True,
                                    key="unmapped_vendor_editor",
                                    use_container_width=True,
                                    column_config={
                                        "Transaction_Count": st.column_config.NumberColumn(
                                            "Transaction Count", format="%d"
                                        ),
                                        "Total_CC_Amount": st.column_config.NumberColumn(
                                            "Total CC Amount",
                                            format="$%0.2f",
                                            step=0.01,
                                        ),
                                        "Vendor_Prefix": st.column_config.TextColumn(
                                            "Vendor Prefix",
                                            disabled=True,
                                            help="Current vendor prefix for this description group.",
                                        ),
                                        "Vendor": st.column_config.SelectboxColumn(
                                            "Vendor",
                                            options=["Select a vendor"]
                                            + master_options,
                                            required=False,
                                        ),
                                        "Apply": st.column_config.CheckboxColumn(
                                            "Apply",
                                            default=False,
                                            help="Check to update this group.",
                                        ),
                                    },
                                )

                                apply_submitted = st.form_submit_button(
                                    "Apply Selected",
                                    type="primary",
                                    use_container_width=True,
                                )

                            # Initialize variables
                            updates_applied: list[str] = []
                            missing_vendor_descriptions: list[str] = []
                            mapping_updates: list[dict[str, str]] = []

                            # Only process if form was submitted
                            if apply_submitted:
                                if "Category" in reco_df.columns:
                                    reco_category_series = (
                                        reco_df["Category"]
                                        .fillna("")
                                        .astype(str)
                                        .str.upper()
                                    )
                                else:
                                    reco_category_series = pd.Series(
                                        [""] * len(reco_df), index=reco_df.index
                                    )

                                for description_key, row in edited_df.iterrows():
                                    apply_value = bool(row.get("Apply", False))
                                    if not apply_value:
                                        continue

                                    selected_option = row.get(
                                        "Vendor", "Select a vendor"
                                    )
                                    vendor_details = master_lookup.get(selected_option)
                                    if (
                                        not vendor_details
                                        or selected_option == "Select a vendor"
                                    ):
                                        missing_vendor_descriptions.append(
                                            row["Description"]
                                        )
                                        continue

                                    mask = reco_description_series == description_key
                                    if "Category" in reco_df.columns:
                                        mask &= reco_category_series == "UNMAPPED"

                                    match_count = int(mask.sum())
                                    if match_count == 0:
                                        continue

                                    category_value = str(
                                        vendor_details.get("CATEGORY", "")
                                    ).strip()
                                    dept_value = str(
                                        vendor_details.get("DEPT", "")
                                    ).strip()
                                    prefix_value = str(
                                        vendor_details.get("PREFIX", "")
                                    ).strip()
                                    vendor_name_value = str(
                                        vendor_details.get("VENDOR_NAME", "")
                                    ).strip()
                                    payment_terms_value = str(
                                        vendor_details.get("PAYMENT TERMS", "")
                                    ).strip()
                                    effective_category_value = category_value
                                    if (
                                        category_value.upper() == "COMMON"
                                        and dept_value.upper() in ("FBA", "FBM")
                                    ):
                                        effective_category_value = (
                                            f"ONLY {dept_value.upper()}"
                                        )

                                    if "Category" in reco_df.columns:
                                        reco_df.loc[mask, "Category"] = (
                                            effective_category_value
                                            if effective_category_value
                                            else pd.NA
                                        )
                                    if "Dept" in reco_df.columns:
                                        reco_df.loc[mask, "Dept"] = (
                                            dept_value if dept_value else pd.NA
                                        )
                                    if (
                                        "Vendor_Prefix" in reco_df.columns
                                        and prefix_value
                                    ):
                                        reco_df.loc[mask, "Vendor_Prefix"] = (
                                            prefix_value
                                        )
                                    if "Vendor_Name" in reco_df.columns:
                                        reco_df.loc[mask, "Vendor_Name"] = (
                                            vendor_name_value
                                            if vendor_name_value
                                            else pd.NA
                                        )
                                    if "Payment_Terms" in reco_df.columns:
                                        reco_df.loc[mask, "Payment_Terms"] = (
                                            pd.to_numeric(
                                                payment_terms_value, errors="coerce"
                                            )
                                            if pd.notna(payment_terms_value)
                                            and payment_terms_value
                                            else 0
                                        )
                                    if "PAYMENT TERMS" in reco_df.columns:
                                        reco_df.loc[mask, "PAYMENT TERMS"] = (
                                            pd.to_numeric(
                                                payment_terms_value, errors="coerce"
                                            )
                                            if pd.notna(payment_terms_value)
                                            and payment_terms_value
                                            else 0
                                        )

                                    updates_applied.append(
                                        f"{row['Description']} ({match_count} rows)"
                                    )
                                    description_for_mapping = str(
                                        row["Description"]
                                    ).strip()
                                    if description_for_mapping:
                                        mapping_updates.append(
                                            {
                                                "description": description_for_mapping,
                                                "prefix": prefix_value
                                                if prefix_value
                                                else "OTHERS",
                                            }
                                        )

                            if missing_vendor_descriptions:
                                st.warning(
                                    "Vendor selection required for: "
                                    + ", ".join(missing_vendor_descriptions[:5])
                                    + (
                                        ", ..."
                                        if len(missing_vendor_descriptions) > 5
                                        else ""
                                    )
                                )

                            if updates_applied:
                                # Update session state with modified reco_df
                                st.session_state["reco_df"] = reco_df
                                st.session_state["reco_success"] = (
                                    "Updated mappings for: "
                                    + ", ".join(updates_applied[:5])
                                    + (", ..." if len(updates_applied) > 5 else "")
                                )
                                # Don't set pending_active_page since we're already on Reco page
                                if mapping_updates:
                                    mappings_path = Path("data/mappings")
                                    if "DESCRIPTION" not in mappings_df.columns:
                                        mappings_df["DESCRIPTION"] = pd.Series(
                                            dtype=str
                                        )
                                    if "PREFIX" not in mappings_df.columns:
                                        mappings_df["PREFIX"] = pd.Series(dtype=str)
                                    for update in mapping_updates:
                                        description_value = update["description"]
                                        prefix_value = update["prefix"]
                                        description_mask = (
                                            mappings_df["DESCRIPTION"]
                                            .astype(str)
                                            .str.strip()
                                            .str.casefold()
                                            == description_value.casefold()
                                        )
                                        if description_mask.any():
                                            mappings_df.loc[
                                                description_mask, "PREFIX"
                                            ] = prefix_value
                                        else:
                                            new_row_df = pd.DataFrame(
                                                {
                                                    "DESCRIPTION": [description_value],
                                                    "PREFIX": [prefix_value],
                                                }
                                            )
                                            mappings_df = pd.concat(
                                                [mappings_df, new_row_df],
                                                ignore_index=True,
                                            )
                                    try:
                                        write_parquet(mappings_df, mappings_path)
                                    except Exception as exc:  # pylint: disable=broad-except
                                        st.error(
                                            f"Unable to update vendor mappings file: {exc}"
                                        )
                                if hasattr(st, "rerun"):
                                    st.rerun()
                                else:  # pragma: no cover
                                    st.experimental_rerun()

    with tab_others:
        if others_df.empty:
            st.info("No 'NOT AVBL' transactions to review.")
        else:
            st.dataframe(others_df, hide_index=True, use_container_width=True)

    with tab_analysis:
        st.markdown("### Vendor Analysis")

        if mapped_df.empty:
            st.info("No mapped transactions available for analysis.")
        else:
            # Get unique vendors from results_df
            if results_df.empty:
                st.info("No reconciliation results available for analysis.")
            else:
                vendor_options = sorted(
                    results_df["Vendor_Prefix"].dropna().unique().tolist()
                )

                if not vendor_options:
                    st.info("No vendors found in reconciliation results.")
                else:
                    # Create vendor display options with names
                    vendor_display_options = []
                    vendor_display_map = {}
                    for vendor_prefix in vendor_options:
                        vendor_name = (
                            results_df[results_df["Vendor_Prefix"] == vendor_prefix][
                                "Vendor_Name"
                            ].iloc[0]
                            if "Vendor_Name" in results_df.columns
                            else ""
                        )
                        display_text = (
                            f"{vendor_prefix} - {vendor_name}"
                            if vendor_name
                            else vendor_prefix
                        )
                        vendor_display_options.append(display_text)
                        vendor_display_map[display_text] = vendor_prefix

                    # Filter selection - default to first vendor
                    selected_vendor_display = st.selectbox(
                        "Select Vendor for Analysis",
                        options=vendor_display_options,
                        index=0,
                        key="analysis_vendor_select",
                    )

                    # Apply filters - always filter by selected vendor
                    selected_vendor_prefix = vendor_display_map[selected_vendor_display]
                    filtered_results = results_df[
                        results_df["Vendor_Prefix"] == selected_vendor_prefix
                    ].copy()

                    if filtered_results.empty:
                        st.warning("No results found for the selected vendor.")
                    else:
                        vendor_results = filtered_results

                        # Prepare data for tabs
                        vendor_po_data = None
                        vendor_cc_data = None

                        if po_merge_data is not None and not po_merge_data.empty:
                            vendor_po_data = po_merge_data[
                                po_merge_data["Vendor_Prefix"].astype(str).str.upper()
                                == selected_vendor_prefix.upper()
                            ].copy()

                        if (
                            results_base_data is not None
                            and not results_base_data.empty
                        ):
                            vendor_cc_data = results_base_data[
                                results_base_data["Vendor_Prefix"]
                                .astype(str)
                                .str.upper()
                                == selected_vendor_prefix.upper()
                            ].copy()

                        # Create tabs for different sections
                        tab_po, tab_cc, tab_grouped = st.tabs(
                            [
                                "PO Details",
                                "CC Transaction Details",
                                "Grouped Results by Date/Dept/Payment Terms",
                            ]
                        )

                        with tab_po:
                            # PO Details with inline header and totals
                            po_header_col1, po_header_col2 = st.columns([3, 2])
                            with po_header_col1:
                                st.markdown("### PO Details")
                            with po_header_col2:
                                if (
                                    vendor_po_data is not None
                                    and not vendor_po_data.empty
                                ):
                                    # Deduplicate by PO_Number to get accurate count
                                    vendor_po_unique = vendor_po_data.drop_duplicates(
                                        subset=["PO_Number"], keep="first"
                                    )
                                    po_count = len(vendor_po_unique)
                                    # Use Comparison_Amount (CC_Charge_Amount) for total
                                    if "Comparison_Amount" in vendor_po_unique.columns:
                                        po_total = vendor_po_unique[
                                            "Comparison_Amount"
                                        ].sum()
                                    else:
                                        po_total = vendor_po_unique["PO_Amount"].sum()
                                    po_total_col1, po_total_col2 = st.columns(2)
                                    with po_total_col1:
                                        st.metric("PO Count", po_count)
                                    with po_total_col2:
                                        st.metric(
                                            "Total PO (incl CC fee)",
                                            f"${po_total:,.2f}",
                                        )
                                else:
                                    po_total_col1, po_total_col2 = st.columns(2)
                                    with po_total_col1:
                                        st.metric("PO Count", 0)
                                    with po_total_col2:
                                        st.metric("Total PO (incl CC fee)", "$0.00")

                            if vendor_po_data is not None and not vendor_po_data.empty:
                                # Prepare PO data with deduction columns
                                po_table = vendor_po_data.drop_duplicates(
                                    subset=["PO_Number"]
                                ).copy()

                                # Calculate available balance for each PO
                                if "Original_PO_Amount" not in po_table.columns:
                                    po_table["Original_PO_Amount"] = po_table[
                                        "PO_Amount"
                                    ]
                                if "Total_Deductions" not in po_table.columns:
                                    po_table["Total_Deductions"] = 0.0
                                if "Adjusted_PO_Amount" not in po_table.columns:
                                    po_table["Adjusted_PO_Amount"] = po_table[
                                        "Original_PO_Amount"
                                    ]

                                po_table["Available_Balance"] = (
                                    po_table["Original_PO_Amount"]
                                    - po_table["Total_Deductions"]
                                )

                                # Add Deduct column for input
                                po_table["Deduct_Amount"] = 0.0

                                # Prepare display table
                                display_table = po_table[
                                    [
                                        "PO_Date",
                                        "PO_Number",
                                        "Original_PO_Amount",
                                        "Total_Deductions",
                                        "Available_Balance",
                                        "Adjusted_PO_Amount",
                                        "Comparison_Amount",
                                        "CC_Txn_Date",
                                        "Window_Start",
                                        "Window_End",
                                        "Deduct_Amount",
                                    ]
                                ].copy()

                                # Format dates
                                if "PO_Date" in display_table.columns:
                                    display_table["PO_Date"] = display_table[
                                        "PO_Date"
                                    ].dt.strftime("%Y-%m-%d")
                                if "Window_Start" in display_table.columns:
                                    display_table["Window_Start"] = display_table[
                                        "Window_Start"
                                    ].dt.strftime("%Y-%m-%d")
                                if "Window_End" in display_table.columns:
                                    display_table["Window_End"] = display_table[
                                        "Window_End"
                                    ].dt.strftime("%Y-%m-%d")

                                # Rename columns for display
                                display_table = display_table.rename(
                                    columns={
                                        "PO_Date": "PO Date",
                                        "PO_Number": "PO Number",
                                        "Original_PO_Amount": "Original Amount",
                                        "Total_Deductions": "Total Deducted",
                                        "Available_Balance": "Available",
                                        "Adjusted_PO_Amount": "Adjusted Amount",
                                        "Comparison_Amount": "Comparison",
                                        "Deduct_Amount": "Deduct",
                                        "CC_Txn_Date": "CC Date",
                                        "Window_Start": "Window Start",
                                        "Window_End": "Window End",
                                    }
                                )

                                # Create form for inline deductions
                                with st.form("inline_po_deduction_form"):
                                    # Editable dataframe
                                    edited_df = st.data_editor(
                                        display_table,
                                        hide_index=True,
                                        use_container_width=True,
                                        column_config={
                                            "PO Number": st.column_config.TextColumn(
                                                "PO Number",
                                                width="medium",
                                                disabled=True,
                                            ),
                                            "PO Date": st.column_config.TextColumn(
                                                "PO Date", width="small", disabled=True
                                            ),
                                            "Original Amount": st.column_config.NumberColumn(
                                                "Original Amount",
                                                format="$%.2f",
                                                disabled=True,
                                            ),
                                            "Total Deducted": st.column_config.NumberColumn(
                                                "Total Deducted",
                                                format="$%.2f",
                                                disabled=True,
                                            ),
                                            "Available": st.column_config.NumberColumn(
                                                "Available",
                                                format="$%.2f",
                                                disabled=True,
                                            ),
                                            "Adjusted Amount": st.column_config.NumberColumn(
                                                "Adjusted Amount",
                                                format="$%.2f",
                                                disabled=True,
                                            ),
                                            "Comparison": st.column_config.NumberColumn(
                                                "Comparison",
                                                format="$%.2f",
                                                disabled=True,
                                            ),
                                            "Deduct": st.column_config.NumberColumn(
                                                "Deduct",
                                                help="Enter amount to deduct from this PO",
                                                min_value=0.0,
                                                format="$%.2f",
                                                width="small",
                                            ),
                                            "CC Date": st.column_config.TextColumn(
                                                "CC Date", width="small", disabled=True
                                            ),
                                            "Window Start": st.column_config.TextColumn(
                                                "Window Start",
                                                width="small",
                                                disabled=True,
                                            ),
                                            "Window End": st.column_config.TextColumn(
                                                "Window End",
                                                width="small",
                                                disabled=True,
                                            ),
                                        },
                                        key="po_deduction_table",
                                    )

                                    st.markdown("---")

                                    # Reason and submit
                                    col_reason, col_submit = st.columns([3, 1])
                                    with col_reason:
                                        bulk_reason = st.text_input(
                                            "reason_label",
                                            placeholder="Reason for deductions (optional)",
                                            key="inline_deduction_reason",
                                            label_visibility="collapsed",
                                        )
                                    with col_submit:
                                        submit_deductions = st.form_submit_button(
                                            "Apply All Deductions",
                                            type="primary",
                                            use_container_width=True,
                                        )

                                    # Store deduction inputs from edited dataframe
                                    deduction_inputs = {}
                                    if submit_deductions:
                                        for idx, row in edited_df.iterrows():
                                            po_num = row["PO Number"]
                                            deduct_amt = row.get("Deduct", 0.0)
                                            if deduct_amt > 0:
                                                deduction_inputs[po_num] = deduct_amt

                                    # Process deductions when form is submitted
                                    if submit_deductions:
                                        # Get CC Batch ID
                                        cc_batch_id = None

                                        if (
                                            not results_base_data.empty
                                            and "Import_Batch_ID"
                                            in results_base_data.columns
                                        ):
                                            batch_ids = (
                                                results_base_data["Import_Batch_ID"]
                                                .dropna()
                                                .unique()
                                            )
                                            if len(batch_ids) > 0:
                                                cc_batch_id = batch_ids[0]

                                        if not cc_batch_id:
                                            cc_batch_id = st.session_state.get(
                                                "current_reco_batch"
                                            )

                                        if (
                                            not cc_batch_id
                                            and vendor_cc_data is not None
                                            and not vendor_cc_data.empty
                                        ):
                                            if (
                                                "Import_Batch_ID"
                                                in vendor_cc_data.columns
                                            ):
                                                batch_ids = (
                                                    vendor_cc_data["Import_Batch_ID"]
                                                    .dropna()
                                                    .unique()
                                                )
                                                if len(batch_ids) > 0:
                                                    cc_batch_id = batch_ids[0]

                                        if not cc_batch_id:
                                            st.error(
                                                "Cannot determine CC Batch ID. Please ensure you have selected a CC batch."
                                            )
                                        else:
                                            # Process all deductions
                                            success_count = 0
                                            error_count = 0
                                            total_deducted = 0.0
                                            errors = []

                                            for (
                                                po_num,
                                                amount,
                                            ) in deduction_inputs.items():
                                                if amount > 0:
                                                    try:
                                                        po_row = po_table[
                                                            po_table["PO_Number"]
                                                            == po_num
                                                        ].iloc[0]
                                                        original = po_row[
                                                            "Original_PO_Amount"
                                                        ]

                                                        save_po_deduction(
                                                            po_num,
                                                            amount,
                                                            original,
                                                            cc_batch_id,
                                                            bulk_reason,
                                                        )

                                                        success_count += 1
                                                        total_deducted += amount
                                                    except ValueError as ve:
                                                        error_count += 1
                                                        errors.append(
                                                            f"PO {po_num}: {str(ve)}"
                                                        )
                                                    except Exception as exc:
                                                        error_count += 1
                                                        errors.append(
                                                            f"PO {po_num}: {str(exc)}"
                                                        )

                                            if success_count > 0:
                                                st.success(
                                                    f"✅ Successfully applied {success_count} deduction(s)!\n\n"
                                                    f"Total Amount Deducted: ${total_deducted:,.2f}\n"
                                                    f"CC Batch ID: {cc_batch_id}"
                                                )

                                            if error_count > 0:
                                                st.error(
                                                    f"❌ {error_count} deduction(s) failed:"
                                                )
                                                for error in errors:
                                                    st.error(f"  • {error}")

                                            if success_count > 0:
                                                st.info(
                                                    "🔄 Reloading reconciliation..."
                                                )
                                                st.session_state.pop("reco_df", None)
                                                st.session_state.pop(
                                                    "current_reco_batch", None
                                                )
                                                if hasattr(st, "rerun"):
                                                    st.rerun()
                                                else:
                                                    st.experimental_rerun()

                                # Add deduction history
                                st.markdown("---")
                                st.markdown("### 📜 Deduction History")

                                deductions_df = load_po_deductions()
                                if not deductions_df.empty:
                                    # Get PO numbers that belong to this vendor from PO data
                                    try:
                                        po_data = read_parquet_with_fallback(
                                            Path("records/po")
                                        )
                                        vendor_po_numbers = (
                                            po_data[
                                                po_data["Vendor_Prefix"]
                                                .astype(str)
                                                .str.strip()
                                                .str.upper()
                                                == selected_vendor_prefix.upper()
                                            ]["PO_Number"]
                                            .astype(str)
                                            .str.strip()
                                            .unique()
                                            .tolist()
                                        )
                                    except Exception:
                                        vendor_po_numbers = []

                                    # Filter deductions by PO numbers belonging to this vendor
                                    vendor_deductions = deductions_df[
                                        deductions_df["PO_Number"]
                                        .astype(str)
                                        .str.strip()
                                        .isin(vendor_po_numbers)
                                    ].sort_values("Timestamp", ascending=False)

                                    if not vendor_deductions.empty:
                                        display_deductions = vendor_deductions.copy()
                                        display_deductions["Deduction_Amount"] = (
                                            display_deductions["Deduction_Amount"].map(
                                                lambda x: f"${x:,.2f}"
                                            )
                                        )

                                        # Parse CC Batch to show date
                                        if "CC_Batch_ID" in display_deductions.columns:
                                            display_deductions["CC_Date"] = (
                                                display_deductions["CC_Batch_ID"]
                                                .str.extract(r"(\d{8})")[0]
                                                .apply(
                                                    lambda x: f"{x[0:2]}/{x[2:4]}/{x[4:]}"
                                                    if pd.notna(x)
                                                    else ""
                                                )
                                            )

                                        display_cols = [
                                            "PO_Number",
                                            "Deduction_Amount",
                                            "CC_Batch_ID",
                                        ]
                                        if "CC_Date" in display_deductions.columns:
                                            display_cols.append("CC_Date")
                                        display_cols.extend(["Reason", "Timestamp"])

                                        # Only show columns that exist
                                        display_cols = [
                                            col
                                            for col in display_cols
                                            if col in display_deductions.columns
                                        ]

                                        st.dataframe(
                                            display_deductions[display_cols],
                                            hide_index=True,
                                            use_container_width=True,
                                        )
                                    else:
                                        st.info(
                                            f"No deductions for vendor {selected_vendor_prefix}"
                                        )
                                else:
                                    st.info("No deductions have been made yet")
                            else:
                                st.info(
                                    f"No PO records matched for vendor {selected_vendor_prefix}"
                                )

                        with tab_cc:
                            # CC Transaction Details with inline header and totals
                            cc_header_col1, cc_header_col2 = st.columns([3, 2])
                            with cc_header_col1:
                                st.markdown("### CC Transaction Details")
                            with cc_header_col2:
                                if (
                                    vendor_cc_data is not None
                                    and not vendor_cc_data.empty
                                ):
                                    cc_count = vendor_cc_data[
                                        "CC_Reference_ID"
                                    ].nunique()
                                    cc_total = vendor_cc_data["CC_Amt"].sum()
                                    cc_total_col1, cc_total_col2 = st.columns(2)
                                    with cc_total_col1:
                                        st.metric("CC Count", cc_count)
                                    with cc_total_col2:
                                        st.metric("Total", f"${cc_total:,.2f}")
                                else:
                                    cc_total_col1, cc_total_col2 = st.columns(2)
                                    with cc_total_col1:
                                        st.metric("CC Count", 0)
                                    with cc_total_col2:
                                        st.metric("Total", "$0.00")

                            if vendor_cc_data is not None and not vendor_cc_data.empty:
                                # Show CC details
                                cc_display = vendor_cc_data[
                                    [
                                        "CC_Txn_Date",
                                        "CC_Reference_ID",
                                        "CC_Description",
                                        "CC_Amt",
                                        "Dept",
                                        "Payment_Terms",
                                    ]
                                ].copy()
                                cc_display["CC_Amt"] = cc_display["CC_Amt"].apply(
                                    lambda x: f"${x:,.2f}" if pd.notna(x) else ""
                                )
                                st.dataframe(
                                    cc_display,
                                    hide_index=True,
                                    use_container_width=True,
                                )
                            else:
                                st.info(
                                    f"No CC transactions found for vendor {selected_vendor_prefix}"
                                )

                        with tab_grouped:
                            # Grouped Results
                            results_display = vendor_results.copy()
                            results_display["Total_CC_Amount"] = results_display[
                                "Total_CC_Amount"
                            ].map(lambda x: f"${x:,.2f}")
                            results_display["Total_PO_Amount"] = results_display[
                                "Total_PO_Amount"
                            ].map(lambda x: f"${x:,.2f}")
                            st.dataframe(
                                results_display,
                                hide_index=True,
                                use_container_width=True,
                            )


def main() -> None:
    st.set_page_config(
        page_title="Ergode CC Reconciliation System",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS
    apply_custom_css()

    # Sidebar header
    st.sidebar.markdown("# Ergode CC Reco App")
    st.sidebar.markdown("---")

    # Initialize current page in session state
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Masters"

    pending_page = st.session_state.pop("pending_active_page", None)
    if pending_page:
        st.session_state["active_page"] = pending_page

    # Navigation section header
    st.sidebar.markdown("### Navigation")

    # Navigation options with page keys
    nav_options = [
        ("Masters", "Masters"),
        ("PO Data", "PO Data"),
        ("CC Data", "CC Data"),
        ("Reco", "Reco"),
    ]

    # Create navigation buttons
    current_page = st.session_state.get("active_page", "Masters")

    # Create unique keys for each button to ensure proper state tracking
    for label, page_key in nav_options:
        is_selected = current_page == page_key

        # Create button with explicit type parameter
        btn_key = f"nav_btn_{page_key}"
        clicked = st.sidebar.button(
            label,
            key=btn_key,
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        )

        if clicked:
            st.session_state["active_page"] = page_key
            st.rerun()

    page = st.session_state.get("active_page", "Masters")

    if page == "Masters":
        render_masters()
    elif page == "PO Data":
        render_po_data()
    elif page == "CC Data":
        render_cc_data()
    else:
        render_reco()


if __name__ == "__main__":
    main()
