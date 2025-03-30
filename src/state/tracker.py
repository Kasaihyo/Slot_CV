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
        self.last_round_balance_change = 0.0 # Initialize variable to store the change of the *previous* round

        # --- Frame Number Tracking for Retry --- (UPDATED)
        self.current_round_start_frame = None
        self.current_stage_start_frame = None
        self.previous_round_start_frame = None # Added
        self.previous_stage_start_frame = None # Added
        self.last_event_frame_num = 0 # Track the frame number of the last processed event
        # -----------------------------------------

        # --- Attributes to store the last stable grid data ---
        self._last_stable_grid_frame_num = None
        self._last_stable_grid_counts = None
        self._last_stable_grid_total = None
        # ------------------------------------------------------

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

        # Store frame number for context
        current_frame_num = result_data['frame_num']
        self.last_event_frame_num = current_frame_num

        # --- Check for stable grid BEFORE updating trackers ---
        # Store the grid data if detected, overwriting any previous unlogged grid
        if result_data.get('grid_event') == 'stable':
            self._last_stable_grid_frame_num = result_data.get('frame_num')
            self._last_stable_grid_counts = result_data.get('symbol_counts')
            self._last_stable_grid_total = result_data.get('total_symbols')
            # print(f"DEBUG: Stored stable grid data from frame {self._last_stable_grid_frame_num}") # Optional debug print

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

        # --- Handle Confirmed Stage --- (Update start frame)
        if stage_changed and confirmed_stage:
            self.previous_confirmed_stage = self.last_confirmed_stage
            self.last_confirmed_stage = confirmed_stage
            self.previous_stage_start_frame = self.current_stage_start_frame # Store previous start frame
            self.current_stage_start_frame = current_frame_num # Record start frame for new stage
            if round_was_ongoing and self.previous_confirmed_stage is not None:
                 stage_changed_mid_round = True

        # --- Handle Confirmed Round Change --- (Update start frame)
        if round_changed and confirmed_round:
            self.last_confirmed_round = confirmed_round
            self.previous_confirmed_stage = self.last_confirmed_stage
            self.previous_round_start_frame = self.current_round_start_frame # Store previous start frame
            self.current_round_start_frame = current_frame_num # Record start frame for new round

            # If stage didn't *just* change, align stage start frame with round start
            if not stage_changed:
                # We also need to store the previous stage start frame here if it aligns
                self.previous_stage_start_frame = self.current_stage_start_frame
                self.current_stage_start_frame = current_frame_num

            # Calculate balance change over the round that just ended
            balance_change = 0.0 # Default
            if self.last_confirmed_balance_numeric is not None:
                if self.last_round_start_balance is not None:
                    balance_change = self.last_confirmed_balance_numeric - self.last_round_start_balance
                    self.accumulated_win_in_round = 0.00 # Reset for the new round START
                else:
                    # First round detected or balance error recovery
                    self.accumulated_win_in_round = 0.00
                    balance_change = 0.0 # No change known for first round
                # Set the start balance for the *new* round
                self.last_round_start_balance = self.last_confirmed_balance_numeric
            else:
                # Balance couldn't be read at round change
                self.last_round_start_balance = None
                self.accumulated_win_in_round = 0.00 # Reset on error
                balance_change = 0.0

            # Store the calculated balance change for the round that just ended
            self.last_round_balance_change = balance_change

        # If not a round change, but balance changed, update accumulated win FOR THE CURRENT round
        elif balance_text_changed and round_was_ongoing and self.last_round_start_balance is not None and self.last_confirmed_balance_numeric is not None:
             self.accumulated_win_in_round = self.last_confirmed_balance_numeric - self.last_round_start_balance
             # This accumulation happens *during* the round, potentially reflecting intermediate wins/losses before the next stage/round.

        return round_changed, stage_changed_mid_round

    # --- Method to get frame range for previous state --- (UPDATED)
    def get_previous_state_frame_range(self, event_type):
        """Gets the start and end frame numbers for the state that just ended.

        Args:
            event_type (str): 'ROUND_CHANGE' or 'STAGE_CHANGE' indicating which state ended.

        Returns:
            tuple: (start_frame, end_frame) or (None, None) if info is unavailable.
                   end_frame is the frame number where the change was detected.
        """
        end_frame = self.last_event_frame_num
        start_frame = None

        if event_type == 'ROUND_CHANGE':
            start_frame = self.previous_round_start_frame # Use the stored previous value

        elif event_type == 'STAGE_CHANGE':
            start_frame = self.previous_stage_start_frame # Use the stored previous value

        if start_frame is None or end_frame is None or start_frame >= end_frame:
            print(f"Warning: Cannot determine valid frame range for previous {event_type}. Start: {start_frame}, End: {end_frame}")
            return None, None

        # Return start frame of the *previous* state and the frame the change was detected
        # Add 1 to start_frame? If start_frame is when the *previous* state *started*, we want to process from the next frame onwards?
        # Let's return the exact start frame for now. The processor can handle the range.
        return start_frame, end_frame
    # -----------------------------------------------------

    def get_log_data(self, frame_num):
        """Prepares data for logging a ROUND CHANGE, including associated stable grid data if found."""
        balance_change = 0.00
        outcome = "N/A"

        # --- Use the stored balance change for the completed round ---
        balance_change = self.last_round_balance_change
        outcome = f"{balance_change:.2f}"
        # Simple check: if the change is 0.0 AND the balance wasn't readable at start, flag as error maybe?
        # Or rely on the user interpreting 0.0 change with None balance appropriately.
        # Let's keep it simple for now.

        # --- Retrieve and reset stable grid data ---
        grid_found = self._last_stable_grid_counts is not None
        grid_frame = self._last_stable_grid_frame_num
        grid_total = self._last_stable_grid_total
        grid_counts = self._last_stable_grid_counts

        if grid_found:
            # print(f"DEBUG: Retrieving stable grid data (Frame: {grid_frame}) for ROUND CHANGE log (Frame: {frame_num})") # Optional debug
            self._last_stable_grid_frame_num = None
            self._last_stable_grid_counts = None
            self._last_stable_grid_total = None
        # -----------------------------------------

        # Prepare base log entry
        log_entry = {
            'frame_num': frame_num, # Frame num when the change was *confirmed*
            'timestamp': datetime.datetime.now().isoformat(), # Add timestamp
            'current_round': self.last_confirmed_round, # The new round number
            'current_stage': self.last_confirmed_stage, # Stage at the start of the new round
            'confirmed_balance_val': self.last_confirmed_balance_numeric, # Balance at the start of the new round
            'raw_balance_text': self.balance_tracker.get_confirmed(),
            'balance_change_val': balance_change, # Use the stored change over the *previous* round
            'outcome_str': outcome,
            'accumulated_win_val': 0.0, # Reset for the new round log entry
            'EventType': 'ROUND_CHANGE',
        }

        # --- Conditionally add grid data --- #
        if grid_found:
            # print(f"DEBUG: Including stable grid data (Frame: {grid_frame}) for ROUND CHANGE log (Frame: {frame_num})") # Optional debug
            log_entry['stable_grid_found'] = True
            log_entry['stable_grid_frame_num'] = grid_frame
            log_entry['total_symbols'] = grid_total # Add total symbols
            if grid_counts:
                 log_entry.update({f'symbol_{name.replace(" ", "_")}': count for name, count in grid_counts.items()})
            # Reset stored grid data only after successfully adding it
            self._last_stable_grid_frame_num = None
            self._last_stable_grid_counts = None
            self._last_stable_grid_total = None
        # --- Else: Grid keys are NOT added if grid_found is False initially --- #

        return log_entry

    def get_log_data_stage_change(self, frame_num):
         """Prepares data for logging a mid-round STAGE CHANGE, including associated stable grid data if found."""
         balance_change_val = 0.00 # Stage changes don't have an associated balance change event in this logic
         outcome_str = "STAGE_CHANGE"

         # --- Retrieve and reset stable grid data ---
         grid_found = self._last_stable_grid_counts is not None
         grid_frame = self._last_stable_grid_frame_num
         grid_total = self._last_stable_grid_total
         grid_counts = self._last_stable_grid_counts

         if grid_found:
            # print(f"DEBUG: Retrieving stable grid data (Frame: {grid_frame}) for STAGE CHANGE log (Frame: {frame_num})") # Optional debug
            self._last_stable_grid_frame_num = None
            self._last_stable_grid_counts = None
            self._last_stable_grid_total = None
         # -----------------------------------------

         # Prepare base log entry
         log_entry = {
            'frame_num': frame_num, # Frame num when the change was *confirmed*
            'timestamp': datetime.datetime.now().isoformat(), # Add timestamp
            'current_round': self.last_confirmed_round, # The round it happened in
            'current_stage': self.last_confirmed_stage, # The *new* stage
            'confirmed_balance_val': self.last_confirmed_balance_numeric, # Current balance when stage changed
            'raw_balance_text': self.balance_tracker.get_confirmed(),
            'balance_change_val': balance_change_val,
            'outcome_str': outcome_str,
            'accumulated_win_val': self.accumulated_win_in_round, # Log win accumulated *so far* in the current round
            'EventType': 'STAGE_CHANGE', # Add event type
         }

         # --- Conditionally add grid data --- #
         if grid_found:
            # print(f"DEBUG: Including stable grid data (Frame: {grid_frame}) for STAGE CHANGE log (Frame: {frame_num})") # Optional debug
            log_entry['stable_grid_found'] = True
            log_entry['stable_grid_frame_num'] = grid_frame
            log_entry['total_symbols'] = grid_total # Add total symbols
            if grid_counts:
                 log_entry.update({f'symbol_{name.replace(" ", "_")}': count for name, count in grid_counts.items()})
            # Reset stored grid data only after successfully adding it
            self._last_stable_grid_frame_num = None
            self._last_stable_grid_counts = None
            self._last_stable_grid_total = None
         # --- Else: Grid keys are NOT added if grid_found is False initially --- #

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