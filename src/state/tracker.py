# src/state/tracker.py
import config # For confirmation frame counts, decimal correction etc.
import datetime # Need for timestamp in log data

class StabilityTracker:
    """Tracks a value over frames to confirm stability."""
    def __init__(self, confirmation_frames):
        self.confirmation_frames = confirmation_frames
        self.last_confirmed_value = None
        self.potential_value = None
        self.confirmation_counter = 0

    def update(self, current_value):
        """
        Updates the tracker with the current value detected in a frame.

        Returns:
            tuple: (confirmed_value, has_changed)
                   confirmed_value: The value once it meets stability criteria, otherwise None.
                   has_changed: True if the confirmed value just changed this update, False otherwise.
        """
        confirmed_value = None
        has_changed = False

        # Handle empty/None current value - resets potential
        if not current_value:
            self.potential_value = None
            self.confirmation_counter = 0
            return None, False

        # If current is same as last confirmed, do nothing (already stable)
        if current_value == self.last_confirmed_value:
            self.potential_value = None
            self.confirmation_counter = 0
            return None, False

        # If current matches potential, increment counter
        if current_value == self.potential_value:
            self.confirmation_counter += 1
        # If current is new, start tracking it as potential
        else:
            self.potential_value = current_value
            self.confirmation_counter = 1

        # Check if potential value reached confirmation threshold
        if self.confirmation_counter >= self.confirmation_frames:
            if self.potential_value != self.last_confirmed_value:
                self.last_confirmed_value = self.potential_value
                confirmed_value = self.last_confirmed_value
                has_changed = True
            # Reset potential tracking after confirmation
            self.potential_value = None
            self.confirmation_counter = 0

        return confirmed_value, has_changed

    def get_confirmed(self):
        """Returns the last confirmed stable value."""
        return self.last_confirmed_value

    def get_potential(self):
        """Returns the current potential value being tracked."""
        return self.potential_value

    def get_counter(self):
        """Returns the current confirmation count for the potential value."""
        return self.confirmation_counter


class GameState:
    """Manages the overall tracked state of the game."""
    def __init__(self):
        self.balance_tracker = StabilityTracker(config.CONFIRMATION_FRAMES_OCR)
        self.round_tracker = StabilityTracker(config.CONFIRMATION_FRAMES_OCR)
        self.stage_tracker = StabilityTracker(config.CONFIRMATION_FRAMES_STAGE)

        self.last_confirmed_balance_numeric = None # The numeric balance value
        self.last_confirmed_round = None
        self.last_confirmed_stage = None
        self.previous_confirmed_stage = None # Track stage just before the current one

        self.last_round_start_balance = None # Balance at the start of the current round
        self.accumulated_win_in_round = 0.00

    def update_from_result(self, result_data):
        """
        Updates the game state based on processed data from a single frame.

        Args:
            result_data (dict): The dictionary received from the FrameProcessor.

        Returns:
            tuple: (round_changed, stage_changed_mid_round)
                   round_changed (bool): True if a new round was confirmed in this update.
                   stage_changed_mid_round (bool): True if the stage changed while a round was active.
        """
        if result_data.get('skipped', True):
            return False, False # No data to process

        # --- Update Trackers ---
        raw_balance = result_data.get('cleaned_balance')
        raw_round = result_data.get('cleaned_round')
        detected_stage = result_data.get('detected_stage') # Stage after confidence check

        confirmed_balance_text, balance_text_changed = self.balance_tracker.update(raw_balance)
        confirmed_round, round_changed = self.round_tracker.update(raw_round)
        confirmed_stage, stage_changed = self.stage_tracker.update(detected_stage)

        round_was_ongoing = self.last_confirmed_round is not None
        stage_changed_mid_round = False

        # --- Handle Confirmed Balance ---
        if balance_text_changed and confirmed_balance_text:
            try:
                # Use float() to handle potential decimals from OCR
                balance_value = float(confirmed_balance_text)
                if config.DECIMAL_CORRECTION and config.DECIMAL_CORRECTION != 0:
                    self.last_confirmed_balance_numeric = balance_value / config.DECIMAL_CORRECTION
                else:
                    self.last_confirmed_balance_numeric = float(balance_value) # Store as float
            except (ValueError, TypeError):
                print(f"Warning: Could not convert confirmed balance '{confirmed_balance_text}' to number.")
                self.last_confirmed_balance_numeric = None # Invalidate numeric balance on error

        # --- Handle Confirmed Stage ---
        if stage_changed and confirmed_stage:
            self.previous_confirmed_stage = self.last_confirmed_stage # Store old before update
            self.last_confirmed_stage = confirmed_stage
            # Check if stage change happened *during* an ongoing round
            if round_was_ongoing and self.previous_confirmed_stage is not None:
                 stage_changed_mid_round = True

        # --- Handle Confirmed Round Change ---
        if round_changed and confirmed_round:
            self.last_confirmed_round = confirmed_round
            self.previous_confirmed_stage = self.last_confirmed_stage # Sync previous stage at round start

            # Calculate balance change and update round start balance
            if self.last_confirmed_balance_numeric is not None:
                if self.last_round_start_balance is not None:
                    balance_change = self.last_confirmed_balance_numeric - self.last_round_start_balance
                    # Reset accumulated win *using* the change from the *previous* round
                    self.accumulated_win_in_round = balance_change
                else:
                    # First round detected or balance error recovery
                    self.accumulated_win_in_round = 0.00
                # Set the start balance for the *new* round
                self.last_round_start_balance = self.last_confirmed_balance_numeric
            else:
                # Balance couldn't be read at round change
                self.last_round_start_balance = None
                self.accumulated_win_in_round = 0.00 # Reset on error

        # If not a round change, but balance changed, update accumulated win
        elif balance_text_changed and round_was_ongoing and self.last_round_start_balance is not None and self.last_confirmed_balance_numeric is not None:
             # This condition might not be needed if we only log round changes,
             # but keeping it allows tracking win accumulation *during* a round if needed later.
             self.accumulated_win_in_round = self.last_confirmed_balance_numeric - self.last_round_start_balance

        return round_changed, stage_changed_mid_round


    def get_log_data(self, frame_num):
        """Prepares data for logging based on the current confirmed state."""
        balance_change = 0.00
        outcome = "N/A"

        if self.last_confirmed_balance_numeric is not None:
            if self.last_round_start_balance is not None:
                 # Calculate the change relative to the start of the *current* round being logged
                 # Note: accumulated_win_in_round already holds this value based on the *previous* round's end
                 balance_change = self.accumulated_win_in_round # Use the calculated accumulated win
                 outcome = f"{balance_change:.2f}"
            else:
                 # First round or balance error
                 balance_change = 0.00
                 outcome = "START" # Or "BAL_ERR" if appropriate? Needs context from update logic
                 # Check if balance is None but text exists
                 if self.balance_tracker.get_confirmed() is not None:
                     outcome = "START"
                 else:
                     outcome = "BAL_ERR"

        # Data needed by the logger
        log_entry = {
            'frame_num': frame_num,
            'timestamp': datetime.datetime.now().isoformat(), # Add timestamp
            'current_round': self.last_confirmed_round,
            'current_stage': self.last_confirmed_stage,
            'confirmed_balance_val': self.last_confirmed_balance_numeric,
            'raw_balance_text': self.balance_tracker.get_confirmed(),
            'balance_change_val': balance_change,
            'outcome_str': outcome,
            'accumulated_win_val': self.accumulated_win_in_round, # Log the win accumulated up to this point
            'EventType': 'ROUND_CHANGE' # Add event type
        }
        return log_entry

    def get_log_data_stage_change(self, frame_num):
         """Prepares data for logging specifically for a mid-round stage change."""
         # No balance change event for stage change itself
         balance_change_val = 0.00
         outcome_str = "STAGE_CHANGE"

         log_entry = {
            'frame_num': frame_num,
            'timestamp': datetime.datetime.now().isoformat(), # Add timestamp
            'current_round': self.last_confirmed_round, # The round it happened in
            'current_stage': self.last_confirmed_stage, # The *new* stage
            'confirmed_balance_val': self.last_confirmed_balance_numeric, # Current balance
            'raw_balance_text': self.balance_tracker.get_confirmed(),
            'balance_change_val': balance_change_val,
            'outcome_str': outcome_str,
            'accumulated_win_val': self.accumulated_win_in_round, # Log win accumulated *so far*
            'EventType': 'STAGE_CHANGE' # Add event type
         }
         return log_entry

    def get_display_status(self):
        """Generates status strings for display overlay."""
        balance_str = "N/A"
        if self.last_confirmed_balance_numeric is not None:
            balance_str = f"{self.last_confirmed_balance_numeric:.2f}"
        elif self.balance_tracker.get_confirmed():
            balance_str = f"RAW:{self.balance_tracker.get_confirmed()}"

        stage_str = "N/A"
        if self.last_confirmed_stage:
            stage_str = self.last_confirmed_stage
        elif self.stage_tracker.get_potential():
            stage_str = f"{self.stage_tracker.get_potential()}?({self.stage_tracker.get_counter()})"

        round_str = self.last_confirmed_round if self.last_confirmed_round else "N/A"

        status_line1 = f"R: {round_str}  S: {stage_str}  B: {balance_str}"

        potentials = []
        if self.round_tracker.get_potential():
            potentials.append(f"R?{self.round_tracker.get_potential()}({self.round_tracker.get_counter()})")
        if self.balance_tracker.get_potential():
             # Only show potential if numeric conversion failed or not yet confirmed
             if self.last_confirmed_balance_numeric is None or self.balance_tracker.potential_value != self.balance_tracker.last_confirmed_value:
                 potentials.append(f"B?{self.balance_tracker.get_potential()}({self.balance_tracker.get_counter()})")
        # Stage potential already included in stage_str

        if potentials:
            status_line1 += " | OCR?: " + " ".join(potentials)

        return status_line1