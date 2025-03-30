# src/logging/manager.py
import pandas as pd
import datetime
import os
import config # Import config to access YOLO_CLASS_NAMES

def initialize_log():
    """Initializes an empty list to store log entries."""
    return []

def add_log_entry(data_log, log_data):
    """Appends a log entry dictionary (prepared by GameState) to the data_log list."""
    # log_data dictionary now contains all necessary keys and values prepared by GameState
    data_log.append(log_data.copy()) # Append a copy to avoid modification issues
    return log_data # Return the added entry for potential immediate use

# --- Removed prepare_stable_grid_log function ---

def save_log_to_csv(data_log, filename):
    """Saves the collected log data to a CSV file."""
    if not data_log:
        print("\nINFO: No data entries were logged to save.")
        return

    print(f"\nINFO: Saving {len(data_log)} logged entries to '{filename}'...")
    df = pd.DataFrame(data_log)

    # Define desired column order dynamically
    # Base columns (ensure keys match those created in GameState)
    base_cols = ['timestamp', 'frame_num', 'EventType', 'current_round', 'current_stage',
                 'confirmed_balance_val', 'balance_change_val', 'outcome_str',
                 'accumulated_win_val', 'raw_balance_text',
                 'stable_grid_found', 'stable_grid_frame_num'] # Added grid info columns

    # Symbol columns (dynamic based on config)
    symbol_cols = [f'symbol_{name.replace(" ", "_")}' for name in config.YOLO_CLASS_NAMES]
    total_symbol_col = ['total_symbols']

    # Combine column order
    # Putting grid info near the event type, then symbols at the end
    cols_order = ['timestamp', 'frame_num', 'EventType',
                  'current_round', 'current_stage',
                  'confirmed_balance_val', 'balance_change_val', 'outcome_str',
                  'accumulated_win_val', 'raw_balance_text',
                  'stable_grid_found', 'stable_grid_frame_num',
                  'total_symbols'] + symbol_cols

    # Reindex DataFrame - ensures all columns defined in cols_order exist
    df = df.reindex(columns=cols_order)

    # --- Fill NaN/None values in specific columns ---

    # Ensure boolean column defaults to False if missing or NaN
    if 'stable_grid_found' in df.columns:
         df['stable_grid_found'] = df['stable_grid_found'].fillna(False).astype(bool)
    else: # Should not happen if col is in cols_order, but safety check
         df['stable_grid_found'] = False

    # Fill symbol counts and total with 0 where missing/NaN
    # (This covers both entries where grid wasn't found AND entries created after retry failed)
    fill_zeros = {'total_symbols': 0}
    fill_zeros.update({col: 0 for col in symbol_cols if col in df.columns})
    df.fillna(fill_zeros, inplace=True)

    # Fill stable_grid_frame_num with 0 or -1 where missing/NaN (optional, otherwise becomes empty string in CSV)
    df.fillna({'stable_grid_frame_num': 0}, inplace=True)

    # -----------------------------------------

    # Rename columns for better readability in CSV header
    column_rename_map = {
        'timestamp': 'Timestamp',
        'frame_num': 'EventFrame', # Renamed for clarity (frame event was logged)
        'EventType': 'EventType',
        'current_round': 'RoundValue',
        'current_stage': 'Stage',
        'confirmed_balance_val': 'ConfirmedBalance',
        'balance_change_val': 'BalanceChange',
        'outcome_str': 'Outcome',
        'accumulated_win_val': 'AccumulatedWin',
        'raw_balance_text': 'RawBalance',
        'stable_grid_found': 'StableGridFound',
        'stable_grid_frame_num': 'StableGridFrame', # Frame where grid was stable
        'total_symbols': 'TotalSymbols'
        # Individual symbol columns already have Symbol_Xxx naming scheme
    }
    df.rename(columns=column_rename_map, inplace=True)

    try:
        # Use float_format for consistent formatting of numeric columns
        df.to_csv(filename, index=False, float_format='%.2f')
        print(f"INFO: Data log successfully saved.")

        # Print summary to console
        print("\n--- Final Log Summary ---")
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        # Only print if dataframe is not empty
        if not df.empty:
            # Select columns to display in summary (optional)
            display_cols = [col for col in column_rename_map.values() if col in df.columns] # Show renamed columns
            if 'TotalSymbols' in df.columns and df['TotalSymbols'].any(): # Add symbol cols only if any were found
                 symbol_display_cols = [col for col in df.columns if col.startswith('symbol_')]
                 display_cols.extend(symbol_display_cols)

            print(df[display_cols].tail(20)) # Print last 20 entries
        else:
            print("(No data logged)")
        print("-" * 60)

    except Exception as e:
        print(f"\nERROR: Failed to save data log to CSV '{filename}': {e}")
        # Optionally print the raw data if saving failed
        # print("\n--- Log Data (NOT SAVED) ---")
        # for entry in data_log:
        #     print(entry)
        # print("-" * 60)