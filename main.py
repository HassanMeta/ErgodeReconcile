from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


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
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    # Optionally remove old CSV file if it exists
    csv_path = file_path.with_suffix(".csv")
    if csv_path.exists() and csv_path != parquet_path:
        try:
            csv_path.unlink()  # Remove old CSV file
        except Exception:
            pass  # Ignore errors when removing old CSV


# Custom CSS for modern dark theme UI
def apply_custom_css():
    """Apply modern dark theme styling to the entire application"""
    st.markdown(
        """
        <style>
        /* Main theme - Dark mode */
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
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

    tab1, tab2 = st.tabs(["Vendor Master", "Description Mapping"])

    master_path = Path("data/master")
    mapping_path = Path("data/mappings")
    master_columns = ["PREFIX", "VENDOR_NAME", "CATEGORY", "DEPT", "PAYMENT TERMS"]
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

        with st.expander("Add Vendor Master Entry", expanded=False):
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
        st.dataframe(master_df, hide_index=True, use_container_width=True)

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
            else:
                selection = st.selectbox(
                    "Select Import Batch to Roll Back",
                    options=["Select a batch"] + batch_options,
                    index=0,
                    key="rollback_batch",
                )

                if selection != "Select a batch":
                    selected_batch = selection.split(" (", maxsplit=1)[0]
                else:
                    selected_batch = None

                rollback_clicked = st.button(
                    "Rollback Selected Batch",
                    disabled=selected_batch is None,
                    type="primary",
                )

                if rollback_clicked and selected_batch:
                    updated_df = existing_df[
                        existing_df["Import_Batch_ID"].astype(str) != selected_batch
                    ]
                    write_parquet(updated_df, po_path)
                    st.session_state["po_upload_success"] = (
                        f"Rolled back batch {selected_batch}"
                    )
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
    upload_counter = st.session_state.get("po_upload_counter", 0)
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel with PO records",
        type=["csv", "xlsx", "xls"],
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
    st.dataframe(incoming_df.head(), hide_index=True, use_container_width=True)

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
    aligned_df["Dept"] = dept_value
    batch_id = datetime.utcnow().strftime("BATCH-%Y%m%d-%H%M%S")
    aligned_df["Import_Batch_ID"] = batch_id

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
                selected_batch = None
                existing_reco_id = None
            else:
                selection = st.selectbox(
                    "Select Import Batch",
                    options=["Select a batch"] + batch_options,
                    index=0,
                    key="cc_manage_batch",
                )

                if selection != "Select a batch":
                    selected_batch = selection.split(" (", maxsplit=1)[0]
                    batch_mask = (
                        existing_df["Import_Batch_ID"].astype(str) == selected_batch
                    )
                    reco_ids = (
                        existing_df.loc[batch_mask, "Reco_ID"]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    existing_reco_id = reco_ids[0] if reco_ids else None
                else:
                    selected_batch = None
                    existing_reco_id = None

            col_rollback, col_reco = st.columns([1, 1], gap="small")
            with col_rollback:
                rollback_clicked = st.button(
                    "Rollback Selected Batch",
                    disabled=selected_batch is None,
                    type="secondary",
                    key="cc_rollback_button",
                    use_container_width=True,
                )
            with col_reco:
                run_reco_clicked = st.button(
                    "Show Reco" if existing_reco_id else "Run Reco",
                    disabled=selected_batch is None,
                    type="primary",
                    key="cc_run_reco_button",
                    use_container_width=True,
                )

            if rollback_clicked and selected_batch:
                updated_df = existing_df[
                    existing_df["Import_Batch_ID"].astype(str) != selected_batch
                ]
                write_parquet(updated_df, cc_path)
                st.session_state["cc_upload_success"] = (
                    f"Rolled back batch {selected_batch}"
                )
                if hasattr(st, "rerun"):
                    st.rerun()
                else:  # pragma: no cover
                    st.experimental_rerun()

            if run_reco_clicked and selected_batch:
                if existing_reco_id:
                    st.session_state["selected_reco_id"] = existing_reco_id
                    st.session_state["pending_active_page"] = "Reco"
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:  # pragma: no cover
                        st.experimental_rerun()
                    return

                reco_dir = Path("records/reco")
                reco_dir.mkdir(parents=True, exist_ok=True)

                batch_df = existing_df[
                    existing_df["Import_Batch_ID"].astype(str) == selected_batch
                ].copy()

                if batch_df.empty:
                    st.error("No transactions found for the selected batch.")
                else:
                    try:
                        mappings_df = read_parquet_with_fallback(Path("data/mappings"))
                        master_df = read_parquet_with_fallback(Path("data/master"))
                    except Exception as exc:  # pylint: disable=broad-except
                        st.error(f"Unable to load mapping data: {exc}")
                    else:
                        mappings_df["DESCRIPTION"] = (
                            mappings_df["DESCRIPTION"].astype(str).str.strip()
                        )
                        mappings_df["_norm_description"] = mappings_df[
                            "DESCRIPTION"
                        ].str.upper()

                        batch_df["_norm_description"] = (
                            batch_df["CC_Description"]
                            .astype(str)
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
                            master_trim["CATEGORY"].to_dict()
                            if "CATEGORY" in master_trim.columns
                            else {}
                        )
                        prefix_dept_map = (
                            master_trim["DEPT"].to_dict()
                            if "DEPT" in master_trim.columns
                            else {}
                        )
                        prefix_terms_map = (
                            master_trim.get(
                                "PAYMENT TERMS", pd.Series(dtype=str)
                            ).to_dict()
                            if "PAYMENT TERMS" in master_trim.columns
                            else {}
                        )
                        prefix_vendor_name_map = (
                            master_trim["VENDOR_NAME"].to_dict()
                            if "VENDOR_NAME" in master_trim.columns
                            else {}
                        )

                        reco_df["Vendor_Prefix"] = reco_df["PREFIX"].fillna("")
                        reco_df["Vendor_Name"] = reco_df["Vendor_Prefix"].map(
                            prefix_vendor_name_map
                        )
                        reco_df["Category"] = reco_df["Vendor_Prefix"].map(
                            prefix_category_map
                        )
                        reco_df.loc[reco_df["Vendor_Prefix"] == "", "Category"] = (
                            "Unmapped"
                        )
                        reco_df["Category"] = reco_df["Category"].fillna("Unmapped")

                        reco_df["Dept"] = reco_df["Vendor_Prefix"].map(prefix_dept_map)
                        reco_df["Payment_Terms"] = reco_df["Vendor_Prefix"].map(
                            prefix_terms_map
                        )
                        reco_df.loc[
                            reco_df["Vendor_Prefix"] == "",
                            ["Dept", "Payment_Terms", "Vendor_Name"],
                        ] = pd.NA

                        # Extract last 4 digits from CC_Number for reco ID
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

                        # Get the most common CC number's last 4 digits from batch_df
                        if "CC_Number" in batch_df.columns:
                            cc_numbers = batch_df["CC_Number"].dropna()
                            if not cc_numbers.empty:
                                cc_number_counts = cc_numbers.value_counts()
                                most_common_cc = (
                                    cc_number_counts.index[0]
                                    if len(cc_number_counts) > 0
                                    else None
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

                        base_reco_id = datetime.utcnow().strftime("RECO-%Y%m%d-%H%M%S")
                        reco_id = f"{base_reco_id}-{cc_last4}"
                        reco_df["Reco_ID"] = reco_id

                        export_columns = [
                            "Reco_ID",
                            "Import_Batch_ID",
                            "CC_Reference_ID",
                            "CC_Txn_Date",
                            "CC_Description",
                            "CC_Amt",
                            "Vendor_Prefix",
                            "Vendor_Name",
                            "Category",
                            "Dept",
                            "Payment_Terms",
                        ]
                        # Ensure CC_Number is not included in export columns
                        if "CC_Number" in export_columns:
                            export_columns.remove("CC_Number")

                        reco_output = reco_df[export_columns]
                        reco_path = reco_dir / f"{reco_id}"
                        write_parquet(reco_output, reco_path)

                        existing_df.loc[
                            existing_df["Import_Batch_ID"].astype(str)
                            == selected_batch,
                            "Reco_ID",
                        ] = reco_id
                        write_parquet(existing_df, cc_path)

                        st.session_state["reco_success"] = (
                            f"Reco file {reco_id} created from batch {selected_batch}."
                        )
                        st.session_state["selected_reco_id"] = reco_id
                        st.session_state["pending_active_page"] = "Reco"

                        if hasattr(st, "rerun"):
                            st.rerun()
                        else:  # pragma: no cover
                            st.experimental_rerun()

        st.subheader("Existing Credit Card Records")
        st.caption(f"{len(filtered_df)} of {len(existing_df)} records shown")
        st.dataframe(filtered_df, hide_index=True, use_container_width=True)

    # Use dynamic key based on upload counter to force reset after upload
    upload_counter = st.session_state.get("cc_upload_counter", 0)
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel with credit card records",
        type=["csv", "xlsx", "xls"],
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
    st.dataframe(incoming_df.head(), hide_index=True, use_container_width=True)

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

    # Validate CC_Number
    if "CC_Number" in aligned_df.columns:
        aligned_df["CC_Number"] = aligned_df["CC_Number"].astype(str).str.strip()
        if (
            aligned_df["CC_Number"].isna().any()
            or (aligned_df["CC_Number"] == "").any()
        ):
            st.error("All rows must have a CC_Number value.")
            return
    else:
        st.error("CC_Number field is required but not found in the mapped data.")
        return

    aligned_df["CC_Reference_ID"] = (
        aligned_df["CC_Reference_ID"].astype(str).str.strip()
    )
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

    aligned_df["CC_Txn_Date"] = pd.to_datetime(
        aligned_df["CC_Txn_Date"], errors="coerce"
    )
    if aligned_df["CC_Txn_Date"].isna().any():
        st.error(
            "Unable to parse CC_Txn_Date for all rows. Please ensure the dates are valid."
        )
        return
    unique_dates = aligned_df["CC_Txn_Date"].dt.date.dropna().unique()
    if len(unique_dates) > 1:
        formatted_dates = ", ".join(sorted({date.isoformat() for date in unique_dates}))
        st.error(
            "Import aborted. All rows in a single upload must share the same CC_Txn_Date. "
            "Please adjust the file to contain only one transaction date. "
            f"Detected dates: {formatted_dates}"
        )
        return
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
    incoming_references = aligned_df["CC_Reference_ID"].tolist()
    duplicate_existing_refs = sorted(
        set(incoming_references).intersection(existing_references)
    )
    if duplicate_existing_refs:
        preview = ", ".join(duplicate_existing_refs[:10])
        suffix = "..." if len(duplicate_existing_refs) > 10 else ""
        st.error(
            "Import aborted. The following CC_Reference_ID values already exist: "
            + preview
            + suffix
        )
        return

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

    # Get the most common CC number's last 4 digits (or first if all unique)
    if "CC_Number" in aligned_df.columns:
        cc_numbers = aligned_df["CC_Number"].dropna()
        if not cc_numbers.empty:
            # Get the most common CC number, or first if all are unique
            cc_number_counts = cc_numbers.value_counts()
            most_common_cc = (
                cc_number_counts.index[0] if len(cc_number_counts) > 0 else None
            )
            cc_last4 = (
                get_cc_last4(most_common_cc) if most_common_cc is not None else "0000"
            )
        else:
            cc_last4 = "0000"
    else:
        cc_last4 = "0000"

    base_batch_id = datetime.utcnow().strftime("CCBATCH-%Y%m%d-%H%M%S")
    batch_id = f"{base_batch_id}-{cc_last4}"
    aligned_df["Import_Batch_ID"] = batch_id
    aligned_df["Reco_ID"] = pd.NA

    combined_df = pd.concat(
        [existing_df, aligned_df],
        ignore_index=True,
    )
    write_parquet(combined_df, cc_path)

    st.session_state["cc_upload_success"] = (
        f"New credit card records appended (batch {batch_id})."
    )
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
    # st.write(
    #     "Run reconciliation workflows and surface key exceptions here. "
    #     "Add summaries, downloadable reports, and action buttons."
    # )

    success_message = st.session_state.pop("reco_success", None)
    if success_message:
        st.success(success_message)

    reco_dir = Path("records/reco")
    reco_dir.mkdir(parents=True, exist_ok=True)

    reco_files = sorted(
        reco_dir.glob("*.parquet"), key=lambda path: path.stat().st_mtime, reverse=True
    )

    if not reco_files:
        st.info("No reconciliation files available yet.")
        return

    available_reco_ids = [path.stem for path in reco_files]

    default_reco_id = st.session_state.get("selected_reco_id")
    if default_reco_id not in available_reco_ids:
        default_index = 0
    else:
        default_index = available_reco_ids.index(default_reco_id)

    selected_reco_id = st.selectbox(
        "Select Reco File",
        options=available_reco_ids,
        index=default_index,
        key="reco_file_selector",
    )

    st.session_state["selected_reco_id"] = selected_reco_id

    selected_path_parquet = reco_dir / f"{selected_reco_id}.parquet"
    selected_path_csv = reco_dir / f"{selected_reco_id}.csv"

    try:
        if selected_path_parquet.exists():
            reco_df = pd.read_parquet(selected_path_parquet, engine="pyarrow")
        elif selected_path_csv.exists():
            reco_df = pd.read_csv(selected_path_csv)
            # Convert to Parquet for future use
            write_parquet(reco_df, reco_dir / f"{selected_reco_id}")
        else:
            st.error(f"Reco file {selected_reco_id} not found.")
            return
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Unable to read reco file {selected_reco_id}: {exc}")
        return

    header_col_title, header_col_button = st.columns([8, 1], gap="small")
    with header_col_title:
        st.subheader(f"Reco File: {selected_reco_id}")
    with header_col_button:
        rollback_clicked = st.button(
            "Rollback",
            type="secondary",
            key="reco_rollback_button",
            use_container_width=True,
        )
    if rollback_clicked:
        try:
            # Delete both Parquet and CSV versions if they exist
            if selected_path_parquet.exists():
                selected_path_parquet.unlink()
            if selected_path_csv.exists():
                selected_path_csv.unlink()
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Unable to delete reco file: {exc}")
        else:
            cc_path = Path("records/cc")
            try:
                cc_df = read_parquet_with_fallback(cc_path)
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Unable to update CC records: {exc}")
            else:
                if "Reco_ID" in cc_df.columns:
                    cc_df.loc[
                        cc_df["Reco_ID"].astype(str) == selected_reco_id,
                        "Reco_ID",
                    ] = pd.NA
                    write_parquet(cc_df, cc_path)

            st.session_state["selected_reco_id"] = None
            st.session_state["reco_success"] = (
                f"Reco file {selected_reco_id} has been rolled back."
            )
            st.session_state["pending_active_page"] = "CC Data"
            if hasattr(st, "rerun"):
                st.rerun()
            else:  # pragma: no cover
                st.experimental_rerun()
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
                "Transaction_Count",
                "Total_PO_Amount",
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
            results_base["CC_Txn_Date"], dayfirst=True, errors="coerce"
        )
        payment_terms_norm = (
            results_base["Payment_Terms"].astype(str).str.strip().str.lower()
        )
        results_base["Window_Start"] = results_base["CC_Txn_Date_dt"] - pd.Timedelta(
            days=grace_days
        )
        results_base["Window_End"] = results_base["CC_Txn_Date_dt"] + pd.Timedelta(
            days=grace_days
        )
        net10_mask = payment_terms_norm == "net 10"
        results_base.loc[net10_mask, "Window_Start"] = results_base.loc[
            net10_mask, "CC_Txn_Date_dt"
        ] - pd.Timedelta(days=10 + grace_days)
        results_base.loc[net10_mask, "Window_End"] = results_base.loc[
            net10_mask, "CC_Txn_Date_dt"
        ]
        prepay_mask = payment_terms_norm == "pre_payment"
        results_base.loc[prepay_mask, "Window_Start"] = results_base.loc[
            prepay_mask, "CC_Txn_Date_dt"
        ] - pd.Timedelta(days=grace_days)
        results_base.loc[prepay_mask, "Window_End"] = results_base.loc[
            prepay_mask, "CC_Txn_Date_dt"
        ] + pd.Timedelta(days=grace_days)
        results_df = (
            results_base.groupby(results_group_cols)
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
            po_df["PO_Date"] = pd.to_datetime(
                po_df["PO_Date"], dayfirst=True, errors="coerce"
            )
            po_summary = pd.DataFrame(columns=results_group_cols + ["PO_Amount"])
            window_cols = results_base[
                results_group_cols + ["Window_Start", "Window_End", "CC_Txn_Date_dt"]
            ].copy()
            window_cols = window_cols.dropna(
                subset=["CC_Txn_Date_dt", "Window_Start", "Window_End"]
            )
            if not window_cols.empty and not po_df.empty:
                po_merge = window_cols.merge(
                    po_df[["Vendor_Prefix", "PO_Date", "PO_Amount", "PO_Number"]],
                    on="Vendor_Prefix",
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
                    po_summary = (
                        po_merge_data.groupby(results_group_cols, dropna=False)[
                            "PO_Amount"
                        ]
                        .sum()
                        .reset_index()
                    )
                else:
                    po_summary = pd.DataFrame(
                        columns=results_group_cols + ["PO_Amount"]
                    )
                    po_merge_data = pd.DataFrame()
        except Exception as exc:  # pylint: disable=broad-except
            st.warning(f"Unable to load PO summary: {exc}")
            po_summary = pd.DataFrame(columns=results_group_cols + ["PO_Amount"])
            po_merge_data = pd.DataFrame()

        results_df = results_df.merge(
            po_summary,
            on=results_group_cols,
            how="left",
        ).rename(columns={"PO_Amount": "Total_PO_Amount"})
        results_df["Total_PO_Amount"] = results_df["Total_PO_Amount"].fillna(0.0)
        results_df["Flag"] = np.where(
            results_df["Total_CC_Amount"] > results_df["Total_PO_Amount"],
            "Red Flag",
            "Green Flag",
        )

        # Add Vendor_Name from master using Vendor_Prefix
        master_path = Path("data/master")
        try:
            master_df = read_parquet_with_fallback(master_path)
            if "PREFIX" in master_df.columns and "VENDOR_NAME" in master_df.columns:
                master_df["PREFIX"] = (
                    master_df["PREFIX"].astype(str).str.strip().str.upper()
                )
                master_df = master_df.dropna(subset=["PREFIX"]).drop_duplicates(
                    subset=["PREFIX"]
                )
                prefix_to_vendor_name = dict(
                    zip(master_df["PREFIX"], master_df["VENDOR_NAME"])
                )
                results_df["Vendor_Name"] = results_df["Vendor_Prefix"].map(
                    prefix_to_vendor_name
                )
                results_df["Vendor_Name"] = results_df["Vendor_Name"].fillna("")
            else:
                results_df["Vendor_Name"] = ""
        except Exception:  # pylint: disable=broad-except
            results_df["Vendor_Name"] = ""

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
            # Reorder columns to show Vendor_Name after Vendor_Prefix
            if "Vendor_Name" in results_display.columns:
                cols = list(results_display.columns)
                if "Vendor_Prefix" in cols and "Vendor_Name" in cols:
                    prefix_idx = cols.index("Vendor_Prefix")
                    name_idx = cols.index("Vendor_Name")
                    # Remove Vendor_Name from current position and insert after Vendor_Prefix
                    cols.pop(name_idx)
                    cols.insert(prefix_idx + 1, "Vendor_Name")
                    results_display = results_display[cols]
            results_display["Total_CC_Amount"] = results_display["Total_CC_Amount"].map(
                lambda x: f"${x:,.2f}"
            )
            if "Total_PO_Amount" in results_display.columns:
                results_display["Total_PO_Amount"] = results_display[
                    "Total_PO_Amount"
                ].map(lambda x: f"${x:,.2f}")
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
                    working_df["Amount"] = pd.to_numeric(
                        common_df[amt_column], errors="coerce"
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

                        return pd.Series(
                            {
                                "Description": description_value,
                                "Amount": total_amount,
                                "Transaction_Count": transaction_count,
                                "Dept": "",
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

                display_df = display_df[
                    ["Description", "Amount", "Transaction_Count", "Dept"]
                ]

                display_df["Amount"] = display_df["Amount"].apply(
                    lambda value: round(float(value), 2)
                    if pd.notna(value) and value != ""
                    else value
                )

                existing_dept_choices: dict[str, str] = st.session_state.setdefault(
                    "common_dept_choices", {}
                )

                # Auto-assign FBM if total amount < 100
                def assign_dept(row):
                    desc_key = str(row["Description"]).strip().upper()
                    # Check if user has already selected a dept for this description
                    if desc_key in existing_dept_choices:
                        return existing_dept_choices[desc_key]
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
                            auto_processed_descriptions.append(description_value)
                            st.session_state["common_apply_flags"][description_norm] = (
                                False
                            )

                        if auto_processed_descriptions:
                            try:
                                write_parquet(reco_df, selected_path_parquet)
                                # Also save CSV for compatibility
                                reco_df.to_csv(selected_path_csv, index=False)
                            except Exception as exc:  # pylint: disable=broad-except
                                st.error(f"Unable to update reconciliation file: {exc}")
                            else:
                                summary_descs = ", ".join(
                                    auto_processed_descriptions[:5]
                                )
                                if len(auto_processed_descriptions) > 5:
                                    summary_descs += ", ..."
                                st.success(
                                    f"Auto-processed {len(auto_processed_descriptions)} description(s) with sum < $100 as FBM: "
                                    + summary_descs
                                )
                                # Reload reco_df after processing
                                reco_df = read_parquet_with_fallback(
                                    selected_path_parquet
                                )
                                # Reload common_df to reflect changes
                                display_df = reco_df.drop(
                                    columns=["Import_Batch_ID"], errors="ignore"
                                )
                                category_series = (
                                    display_df["Category"].astype(str).str.upper()
                                )
                                common_df = display_df[category_series == "COMMON"]
                                # Recalculate display_df after removing auto-processed items
                                if not common_df.empty:
                                    working_df = common_df.copy()
                                    working_df["Amount"] = pd.to_numeric(
                                        common_df[amt_column], errors="coerce"
                                    )
                                    if desc_column:
                                        working_df["_norm_desc"] = (
                                            working_df[desc_column]
                                            .astype(str)
                                            .str.strip()
                                            .str.upper()
                                        )
                                    else:
                                        working_df["_norm_desc"] = ""
                                    display_df = (
                                        working_df.groupby("_norm_desc", as_index=False)
                                        .apply(agg_desc_group)
                                        .reset_index(drop=True)
                                    )
                                    if "_norm_desc" in display_df.columns:
                                        display_df = display_df.drop(
                                            columns=["_norm_desc"]
                                        )
                                    if "Transaction_Count" not in display_df.columns:
                                        display_df["Transaction_Count"] = 1
                                    display_df = display_df[
                                        [
                                            "Description",
                                            "Amount",
                                            "Transaction_Count",
                                            "Dept",
                                        ]
                                    ]
                                    display_df["Amount"] = display_df["Amount"].apply(
                                        lambda value: round(float(value), 2)
                                        if pd.notna(value) and value != ""
                                        else value
                                    )
                                else:
                                    display_df = pd.DataFrame(
                                        columns=[
                                            "Description",
                                            "Amount",
                                            "Transaction_Count",
                                            "Dept",
                                        ]
                                    )
                                # Rerun to refresh the page
                                st.rerun()

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
                                try:
                                    write_parquet(
                                        reco_df, reco_dir / f"{selected_reco_id}"
                                    )
                                except Exception as exc:  # pylint: disable=broad-except
                                    st.error(
                                        f"Unable to update reconciliation file: {exc}"
                                    )
                                else:
                                    summary_descs = ", ".join(
                                        processed_descriptions[:5]
                                    )
                                    if len(processed_descriptions) > 5:
                                        summary_descs += ", ..."
                                    st.session_state["reco_success"] = (
                                        f"Updated {len(processed_descriptions)} description group(s): "
                                        + summary_descs
                                    )
                                    st.session_state["selected_reco_id"] = (
                                        selected_reco_id
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
                    grouped_records.append(
                        {
                            "description_key": description_key,
                            "display_description": display_description,
                            "count": count_value,
                            "total_amount": total_amount,
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
                                            payment_terms_value
                                            if payment_terms_value
                                            else pd.NA
                                        )
                                    if "PAYMENT TERMS" in reco_df.columns:
                                        reco_df.loc[mask, "PAYMENT TERMS"] = (
                                            payment_terms_value
                                            if payment_terms_value
                                            else pd.NA
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
                                try:
                                    write_parquet(
                                        reco_df, reco_dir / f"{selected_reco_id}"
                                    )
                                except Exception as exc:  # pylint: disable=broad-except
                                    st.error(
                                        f"Unable to update reconciliation file: {exc}"
                                    )
                                else:
                                    st.session_state["reco_success"] = (
                                        "Updated mappings for: "
                                        + ", ".join(updates_applied[:5])
                                        + (", ..." if len(updates_applied) > 5 else "")
                                    )
                                    st.session_state["selected_reco_id"] = (
                                        selected_reco_id
                                    )
                                    # Don't set pending_active_page since we're already on Reco page
                                    if mapping_updates:
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
                                                        "DESCRIPTION": [
                                                            description_value
                                                        ],
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
                                    po_count = len(vendor_po_data)
                                    po_total = vendor_po_data["PO_Amount"].sum()
                                    po_total_col1, po_total_col2 = st.columns(2)
                                    with po_total_col1:
                                        st.metric("PO Count", po_count)
                                    with po_total_col2:
                                        st.metric("Total", f"${po_total:,.2f}")
                                else:
                                    po_total_col1, po_total_col2 = st.columns(2)
                                    with po_total_col1:
                                        st.metric("PO Count", 0)
                                    with po_total_col2:
                                        st.metric("Total", "$0.00")

                            if vendor_po_data is not None and not vendor_po_data.empty:
                                # Show PO details
                                po_display = vendor_po_data[
                                    [
                                        "PO_Date",
                                        "PO_Number",
                                        "PO_Amount",
                                        "CC_Txn_Date",
                                        "Window_Start",
                                        "Window_End",
                                    ]
                                ].copy()
                                po_display["PO_Date"] = po_display[
                                    "PO_Date"
                                ].dt.strftime("%Y-%m-%d")
                                po_display["Window_Start"] = po_display[
                                    "Window_Start"
                                ].dt.strftime("%Y-%m-%d")
                                po_display["Window_End"] = po_display[
                                    "Window_End"
                                ].dt.strftime("%Y-%m-%d")
                                st.dataframe(
                                    po_display,
                                    hide_index=True,
                                    use_container_width=True,
                                )
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
