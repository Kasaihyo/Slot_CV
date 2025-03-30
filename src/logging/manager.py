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
    # log_data dictionary already contains all necessary keys and values
    # including frame_num, timestamp, round, stage, balance, outcome, symbols, etc.
    data_log.append(log_data.copy()) # Append a copy to avoid modification issues
    return log_data # Return the added entry for potential immediate use

# --- New function to prepare stable grid log ---
def prepare_stable_grid_log(frame_num, symbol_counts, total_symbols, current_round, current_stage):
    """Creates a dictionary specifically for logging a stable grid event."""
    log_entry = {
        'frame_num': frame_num,
        'timestamp': datetime.datetime.now().isoformat(),
        'EventType': 'STABLE_GRID',
        'current_round': current_round, # Context
        'current_stage': current_stage, # Context
        'total_symbols': total_symbols
        # Include other base columns with None or default values if desired for consistency
        # 'confirmed_balance_val': None,
        # 'balance_change_val': 0.0,
        # 'outcome_str': 'GRID',
        # 'accumulated_win_val': None,
        # 'raw_balance_text': None
    }
    # Add individual symbol counts
    log_entry.update({f'symbol_{name.replace(" ", "_")}': count for name, count in symbol_counts.items()})
    return log_entry
# ------------------------------------------

def save_log_to_csv(data_log, filename):
    """Saves the collected log data to a CSV file."""
    if not data_log:
        print("\nINFO: No data entries were logged to save.")
        return

    print(f"\nINFO: Saving {len(data_log)} logged entries to '{filename}'...")
    df = pd.DataFrame(data_log)

    # Define desired column order dynamically
    # Base columns (ensure keys match those created in GameState and prepare_stable_grid_log)
    base_cols = ['timestamp', 'frame_num', 'EventType', 'current_round', 'current_stage',
                 'confirmed_balance_val', 'balance_change_val', 'outcome_str',
                 'accumulated_win_val', 'raw_balance_text']

    # Symbol columns
    symbol_cols = [f'symbol_{name.replace(" ", "_")}' for name in config.YOLO_CLASS_NAMES]
    total_symbol_col = ['total_symbols']

    # Combine column order
    cols_order = base_cols + total_symbol_col + symbol_cols

    # Reindex DataFrame - this ensures all columns exist, filling missing ones with NaN
    df = df.reindex(columns=cols_order)

    # --- Fill NaN values in specific columns ---
    # Fill symbol counts with 0 for non-STABLE_GRID events
    fill_zeros = {'total_symbols': 0}
    fill_zeros.update({col: 0 for col in symbol_cols})
    df.fillna(fill_zeros, inplace=True)
    # Optionally fill other columns if needed
    # df.fillna({'BalanceChange': 0.0}, inplace=True)
    # -----------------------------------------

    # Rename columns for better readability in CSV header
    column_rename_map = {
        'timestamp': 'Timestamp',
        'frame_num': 'Frame',
        'current_round': 'RoundValue',
        'current_stage': 'Stage',
        'confirmed_balance_val': 'ConfirmedBalance',
        'balance_change_val': 'BalanceChange',
        'outcome_str': 'Outcome',
        'accumulated_win_val': 'AccumulatedWin',
        'raw_balance_text': 'RawBalance',
        'total_symbols': 'TotalSymbols',
        'EventType': 'EventType' # Add rename for EventType
        # Individual symbol columns already have a decent naming scheme (Symbol_Xxx_Yyy)
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
            print(df.tail(20)) # Print last 20 entries as a sample
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