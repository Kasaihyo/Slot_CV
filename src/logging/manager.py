# src/logging/manager.py
import pandas as pd
import datetime
import os

def initialize_log():
    """Initializes an empty list to store log entries."""
    return []

def add_log_entry(data_log, frame_num, current_round, current_stage,
                  confirmed_balance_val, raw_balance_text, balance_change_val,
                  outcome_str, accumulated_win_val):
    """Creates and appends a log entry dictionary to the data_log list."""
    timestamp = datetime.datetime.now()

    # Format values for logging (especially floats)
    balance_log_str = f"{confirmed_balance_val:.2f}" if confirmed_balance_val is not None else None
    balance_change_log_str = f"{balance_change_val:.2f}"
    accumulated_win_log_str = f"{accumulated_win_val:.2f}"

    log_entry = {
        'Timestamp': timestamp,
        'Frame': frame_num,
        'RoundValue': current_round,
        'Stage': current_stage,
        'CorrectedBalance': balance_log_str,
        'BalanceChange': balance_change_log_str,
        'Outcome': outcome_str,
        'RawBalanceValue': raw_balance_text,
        'AccumulatedWin': accumulated_win_log_str
    }
    data_log.append(log_entry)
    return log_entry # Return the created entry for potential immediate use (like printing)


def save_log_to_csv(data_log, filename):
    """Saves the collected log data to a CSV file."""
    if not data_log:
        print("\nINFO: No data entries were logged to save.")
        return

    print(f"\nINFO: Saving {len(data_log)} logged entries to '{filename}'...")
    df = pd.DataFrame(data_log)

    # Define desired column order
    cols_order = ['Timestamp', 'Frame', 'RoundValue', 'Stage', 'CorrectedBalance',
                  'BalanceChange', 'Outcome', 'RawBalanceValue', 'AccumulatedWin']

    # Reindex DataFrame, keeping only existing columns from the desired list
    # This prevents errors if a column somehow wasn't generated
    df = df.reindex(columns=[col for col in cols_order if col in df.columns])

    try:
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