import tkinter as tk
from tkinter import messagebox, filedialog
import time
import cv2
import numpy as np
import joblib
import os

# --- Sudoku Solving Logic (Unchanged) ---

def find_empty(board):
    """Finds an empty cell (represented by 0) in the board."""
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return (r, c)  # row, col
    return None

def is_valid(board, num, pos):
    """Checks if placing 'num' at 'pos' (row, col) is valid."""
    row, col = pos
    for c_check in range(9):
        if board[row][c_check] == num and col != c_check:
            return False
    for r_check in range(9):
        if board[r_check][col] == num and row != r_check:
            return False
    box_x = col // 3
    box_y = row // 3
    for r_check in range(box_y * 3, box_y * 3 + 3):
        for c_check in range(box_x * 3, box_x * 3 + 3):
            if board[r_check][c_check] == num and (r_check, c_check) != pos:
                return False
    return True

# --- GUI Class ---

class SudokuGUI:
    def __init__(self, master):
        """Initializes the Sudoku GUI with KNN model."""
        self.master = master
        master.title("Visual Sudoku Solver (KNN)")
        master.config(bg="#f0f0f0")

        self.cells = {}
        self.board = [[0 for _ in range(9)] for _ in range(9)]
        self.animation_speed = 0.05
        self.solving = False
        self.model = None
        self.initial_cells = set()
        self.edit_mode = False  # New: Track edit mode state

        # --- Load KNN Model ---
        model_path = 'KNN.sav'
        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print(f"KNN Model loaded successfully from {model_path}")
                if not hasattr(self.model, 'predict_proba'):
                    print("Warning: Loaded KNN model does not support predict_proba. Confidence check disabled.")
            else:
                messagebox.showerror("Model Error", f"{model_path} file not found. Please ensure it's in the correct directory.")
                self.model = None
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load KNN model: {str(e)}")
            self.model = None

        # Create debug directory for cell images
        self.debug_dir = "debug_cells"
        if not os.path.exists(self.debug_dir):
            try:
                os.makedirs(self.debug_dir)
                print(f"Debug directory created: {self.debug_dir}")
            except OSError as e:
                print(f"Error creating debug directory {self.debug_dir}: {e}")
                self.debug_dir = None

        # Frame for the grid
        grid_frame = tk.Frame(master, bg="#cccccc", bd=3)
        grid_frame.pack(pady=10, padx=20)

        # Create the 9x9 grid
        for r in range(9):
            for c in range(9):
                bg_color = "#ffffff"
                if (r // 3 + c // 3) % 2 == 0:
                    bg_color = "#e8e8e8"
                cell = tk.Entry(
                    grid_frame, width=3, font=('Arial', 18, 'bold'), justify='center',
                    bd=1, relief=tk.SOLID, bg=bg_color, fg="#333333",
                    highlightthickness=0, insertbackground="#333333",
                    disabledbackground=bg_color, disabledforeground="#000080"
                )
                padx_val = (1, 1); pady_val = (1, 1)
                if c % 3 == 2 and c != 8: padx_val = (1, 3)
                if r % 3 == 2 and r != 8: pady_val = (1, 3)
                cell.grid(row=r, column=c, padx=padx_val, pady=pady_val, sticky="nsew")
                cell.bind('<KeyRelease>', lambda e, row=r, col=c: self.validate_input(e, row, col))
                self.cells[(r, c)] = cell

        # Frame for controls
        control_frame = tk.Frame(master, bg="#f0f0f0")
        control_frame.pack(pady=5, fill=tk.X, padx=20)

        # --- Buttons ---
        button_subframe = tk.Frame(control_frame, bg="#f0f0f0")
        button_subframe.pack(pady=5)
        tk.Button(
            button_subframe, text="Load Image", command=self.load_from_image,
            font=('Arial', 12, 'bold'), bg="#2196F3", fg="white",
            padx=10, pady=5, relief=tk.RAISED, bd=2
        ).grid(row=0, column=0, padx=10)
        tk.Button(
            button_subframe, text="Solve", command=self.solve_gui,
            font=('Arial', 12, 'bold'), bg="#4CAF50", fg="white",
            padx=10, pady=5, relief=tk.RAISED, bd=2
        ).grid(row=0, column=1, padx=10)
        tk.Button(
            button_subframe, text="Clear", command=self.clear_board,
            font=('Arial', 12, 'bold'), bg="#f44336", fg="white",
            padx=10, pady=5, relief=tk.RAISED, bd=2
        ).grid(row=0, column=2, padx=10)
        tk.Button(
            button_subframe, text="Toggle Edit Mode", command=self.toggle_edit_mode,
            font=('Arial', 12, 'bold'), bg="#FF9800", fg="white",
            padx=10, pady=5, relief=tk.RAISED, bd=2
        ).grid(row=0, column=3, padx=10)

        # --- Speed Slider ---
        speed_frame = tk.Frame(control_frame, bg="#f0f0f0")
        speed_frame.pack(pady=5)
        tk.Label(speed_frame, text="Animation Speed:", bg="#f0f0f0", font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        self.speed_scale = tk.Scale(
            speed_frame, from_=0.001, to=0.5, resolution=0.01, orient=tk.HORIZONTAL,
            length=200, command=self.update_speed, bg="#f0f0f0", highlightthickness=0
        )
        self.speed_scale.set(self.animation_speed)
        self.speed_scale.pack(side=tk.LEFT)

    def update_speed(self, val):
        self.animation_speed = float(val)

    def toggle_edit_mode(self):
        """Toggles edit mode to allow/disallow editing of loaded digits."""
        self.edit_mode = not self.edit_mode
        mode_text = "ON" if self.edit_mode else "OFF"
        print(f"Edit Mode: {mode_text}")
        for r in range(9):
            for c in range(9):
                cell = self.cells[(r, c)]
                if self.edit_mode:
                    cell.config(state='normal', fg="#333333")
                else:
                    if (r, c) in self.initial_cells and self.board[r][c] != 0:
                        cell.config(state='disabled', disabledforeground="#000080")
                    else:
                        cell.config(state='normal', fg="#333333")

    def validate_input(self, event, row, col):
        # Prevent editing during solving
        if self.solving:
            entry = self.cells[(row, col)]
            entry.delete(0, tk.END)
            if self.board[row][col] != 0: entry.insert(0, str(self.board[row][col]))
            return

        # Allow editing of initial cells only in edit mode
        if not self.edit_mode and (row, col) in self.initial_cells:
            entry = self.cells[(row, col)]
            entry.delete(0, tk.END)
            if self.board[row][col] != 0: entry.insert(0, str(self.board[row][col]))
            return

        entry = self.cells[(row, col)]
        value = entry.get()
        original_board_val = self.board[row][col]

        if not (value.isdigit() and len(value) == 1 and value != '0') and value != "":
            entry.delete(0, tk.END)
            self.board[row][col] = 0
            entry.config(fg="#333333")
        elif value != "":
            num = int(value)
            temp_val = self.board[row][col]
            self.board[row][col] = 0
            if not is_valid(self.board, num, (row, col)):
                entry.config(fg="red")
                self.board[row][col] = temp_val
            else:
                entry.config(fg="#333333")
                self.board[row][col] = num
        else:
            self.board[row][col] = 0
            entry.config(fg="#333333")

    def load_from_image(self):
        if self.solving: messagebox.showwarning("Warning", "Cannot load image while solving."); return
        if self.model is None: messagebox.showerror("Model Error", "KNN model is not loaded."); return

        file_path = filedialog.askopenfilename(
            title="Select Sudoku Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path: return

        try:
            self.clear_board(clear_initial=True)
            recognized_grid = self._extract_grid_from_image(file_path)
            if recognized_grid is None: return

            self.set_board_from_recognition(recognized_grid)

            print("--- Recognized Grid ---")
            for row_idx, row_data in enumerate(recognized_grid): print(f"Row {row_idx}: {row_data}")
            print(f"Initial cells identified: {self.initial_cells}")
            print("-----------------------")
            messagebox.showinfo("Success", "Image processed. Check console for recognized grid.")

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred during image processing: {str(e)}")
            print(f"Image processing failed: {e}")

    def _sort_contour_points(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        return rect

    def _extract_grid_from_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None: messagebox.showerror("Error", f"Could not read image file: {image_path}"); return None

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: messagebox.showerror("Error", "No contours found."); return None
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            grid_contour_approx = None
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4 and cv2.contourArea(contour) > (image.shape[0] * image.shape[1] * 0.05):
                    grid_contour_approx = approx; break
            if grid_contour_approx is None: messagebox.showerror("Error", "Could not find Sudoku grid contour."); return None

            pts = grid_contour_approx.reshape(4, 2).astype(np.float32)
            rect = self._sort_contour_points(pts)
            target_size = 450
            dst_pts = np.array([[0, 0], [target_size - 1, 0], [target_size - 1, target_size - 1], [0, target_size - 1]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(rect, dst_pts)
            warped_thresh = cv2.warpPerspective(thresh, matrix, (target_size, target_size))

            cell_size = target_size // 9
            grid = np.zeros((9, 9), dtype=int)
            self.initial_cells.clear()

            for i in range(9):
                for j in range(9):
                    y_start, y_end = i * cell_size, (i + 1) * cell_size
                    x_start, x_end = j * cell_size, (j + 1) * cell_size
                    cell_thresh_roi = warped_thresh[y_start:y_end, x_start:x_end]
                    digit = self._recognize_digit(cell_thresh_roi, i, j)
                    grid[i][j] = digit
                    if digit != 0:
                        self.initial_cells.add((i, j))
            return grid

        except Exception as e:
            messagebox.showerror("Extraction Error", f"Error during grid extraction: {e}")
            print(f"Grid extraction failed: {e}")
            return None

    def _recognize_digit(self, cell_thresh_roi, row, col):
        """Recognizes the digit using KNN, matching MNIST training preprocessing."""
        digit = 0
        max_proba = 0.0
        final_image_for_knn = np.zeros((28, 28), dtype=np.uint8)
        reason_empty = "Start"

        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            opened_cell = cv2.morphologyEx(cell_thresh_roi, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(opened_cell.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                min_area_threshold = (cell_thresh_roi.shape[0] * cell_thresh_roi.shape[1]) * 0.05

                if area > min_area_threshold:
                    reason_empty = "Contour area OK"
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    aspect_ratio = w / float(h) if h > 0 else 0
                    min_aspect_ratio = 0.2
                    max_aspect_ratio = 5.0
                    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                        reason_empty = f"Bad aspect ratio ({aspect_ratio:.2f})"
                        digit = 0; max_proba = 1.0
                    else:
                        reason_empty = "Aspect ratio OK"
                        margin = 2
                        digit_roi = opened_cell[max(0, y - margin):min(opened_cell.shape[0], y + h + margin),
                                                max(0, x - margin):min(opened_cell.shape[1], x + w + margin)]

                        if digit_roi.size > 0:
                            interpolation = cv2.INTER_AREA if h > 28 or w > 28 else cv2.INTER_LINEAR
                            final_image_for_knn = cv2.resize(digit_roi, (28, 28), interpolation=interpolation).astype(np.uint8)
                        else:
                            reason_empty = "Empty ROI extracted"; digit = 0; max_proba = 1.0

                        if digit == 0 and reason_empty == "Aspect ratio OK":
                            if self.model:
                                normalized = final_image_for_knn.astype('float32') / 255.0
                                flattened = normalized.flatten().reshape(1, -1)

                                non_zero_pixels = cv2.countNonZero(final_image_for_knn)
                                min_pixel_threshold = 290

                                if non_zero_pixels < min_pixel_threshold:
                                    digit = 0
                                    reason_empty = f"Too few pixels ({non_zero_pixels} < {min_pixel_threshold})"
                                    max_proba = 1.0
                                else:
                                    predicted = self.model.predict(flattened)[0]
                                    if hasattr(self.model, 'predict_proba'):
                                        proba = self.model.predict_proba(flattened)
                                        max_proba = np.max(proba)
                                        min_confidence_threshold = 0.80

                                        if max_proba < min_confidence_threshold:
                                            digit = 0
                                            reason_empty = f"Low confidence ({max_proba:.2f} < {min_confidence_threshold})"
                                            max_proba = 1.0 - max_proba
                                        else:
                                            digit = int(predicted)
                                            reason_empty = "Confident prediction"
                                    else:
                                        digit = int(predicted)
                                        max_proba = 1.0
                                        reason_empty = "Predicted (no proba)"

                else:
                    reason_empty = f"Contour area small ({area:.0f} < {min_area_threshold:.0f})"
                    digit = 0; max_proba = 1.0
            else:
                reason_empty = "No contour found"; digit = 0; max_proba = 1.0

            if self.debug_dir:
                try:
                    safe_reason = ''.join(c if c.isalnum() or c in ['_','-','.'] else '_' for c in reason_empty)
                    reason_tag = f"_REASON_{safe_reason}" if digit == 0 else ""
                    save_path = os.path.join(self.debug_dir, f"cell_{row}_{col}_pred_{digit}_conf_{max_proba:.2f}{reason_tag}.jpg")
                    cv2.imwrite(save_path, final_image_for_knn)
                except Exception as save_err:
                    print(f"Error saving debug image for cell ({row},{col}): {save_err}")

        except Exception as e:
            print(f"Error in digit recognition for cell ({row},{col}): {e}")
            digit = 0; max_proba = 0.0

        if digit == 0 and reason_empty != "Start":
            pass
        elif reason_empty.startswith("Too few") or reason_empty.startswith("Low conf") or reason_empty.startswith("Contour area small") or reason_empty.startswith("Bad aspect") or reason_empty.startswith("No contour") or reason_empty.startswith("Empty ROI"):
            digit = 0

        return digit

    def set_board_from_recognition(self, recognized_grid):
        for r in range(9):
            for c in range(9):
                num = recognized_grid[r][c]
                cell = self.cells[(r, c)]
                cell.config(state='normal')
                cell.delete(0, tk.END)
                self.board[r][c] = num
                if num != 0:
                    cell.insert(0, str(num))
                    if self.edit_mode:
                        cell.config(state='normal', fg="#333333", font=('Arial', 18, 'bold'))
                    else:
                        if (r, c) in self.initial_cells:
                            cell.config(state='disabled', disabledforeground="#000080", font=('Arial', 18, 'bold'))
                        else:
                            print(f"Warning: Cell ({r},{c}) recognized {num} but not in initial_cells.")
                            cell.config(state='normal', fg="#333333", font=('Arial', 18, 'bold'))
                else:
                    cell.config(state='normal', fg="#333333", font=('Arial', 18, 'bold'))
                bg_color = "#e8e8e8" if (r // 3 + c // 3) % 2 != 0 else "#ffffff"
                cell.config(bg=bg_color)

    def get_board_from_gui(self):
        gui_initial_cells = set(self.initial_cells)
        valid_board = True
        temp_board = [[0 for _ in range(9)] for _ in range(9)]

        for r in range(9):
            for c in range(9):
                if not self.edit_mode and (r, c) in self.initial_cells:
                    temp_board[r][c] = self.board[r][c]; continue
                val = self.cells[(r, c)].get()
                if val.isdigit() and len(val) == 1 and val != '0':
                    num = int(val); temp_board[r][c] = num
                    gui_initial_cells.add((r, c))
                else:
                    temp_board[r][c] = 0
                    if (r,c) in gui_initial_cells and (r,c) not in self.initial_cells:
                        gui_initial_cells.remove((r,c))

        validation_board = [row[:] for row in temp_board]
        for r in range(9):
            for c in range(9):
                num = validation_board[r][c]
                if num != 0:
                    validation_board[r][c] = 0
                    if not is_valid(validation_board, num, (r, c)):
                        self.cells[(r, c)].config(fg='red')
                        messagebox.showerror("Input Error", f"Invalid number {num} at row {r+1}, col {c+1} conflicts.")
                        valid_board = False
                    validation_board[r][c] = num

        if not valid_board: return None, None

        self.board = temp_board
        for r in range(9):
            for c in range(9):
                if (r,c) in gui_initial_cells: self.cells[(r,c)].config(fg='#000080')
                elif (r,c) not in self.initial_cells: self.cells[(r,c)].config(fg='#333333')
        return self.board, gui_initial_cells

    def update_gui_cell(self, row, col, value, color_state):
        cell = self.cells[(row, col)]
        base_bg = "#e8e8e8" if (row // 3 + col // 3) % 2 != 0 else "#ffffff"
        cell.config(state='normal')
        cell.delete(0, tk.END)
        if value != 0: cell.insert(0, str(value))

        if color_state == "solving": cell.config(bg="#fffacd", fg="#daa520")
        elif color_state == "placing": cell.config(bg="#90ee90", fg="#006400")
        elif color_state == "backtracking": cell.config(bg="#ffcccb", fg="#dc143c")
        elif color_state == "solved": cell.config(bg=base_bg, fg="#006400")
        else: cell.config(bg=base_bg, fg="#333333")

        if color_state in ["solved", "placing"] and not self.edit_mode:
            cell.config(state='disabled')
        else:
            cell.config(state='normal')

        self.master.update_idletasks()
        time.sleep(self.animation_speed)

    def solve_gui(self):
        if self.solving: print("Solver is already running."); return

        current_board_state, run_initial_cells = self.get_board_from_gui()
        if current_board_state is None: self.solving = False; return

        self.board = [row[:] for row in current_board_state]
        self.solving = True

        for r in range(9):
            for c in range(9):
                cell = self.cells[(r,c)]
                bg_color = "#e8e8e8" if (r // 3 + c // 3) % 2 != 0 else "#ffffff"
                cell.config(bg=bg_color)
                if not self.edit_mode and (r, c) in run_initial_cells:
                    cell.config(state='normal')
                    cell.delete(0, tk.END)
                    cell.insert(0, str(self.board[r][c]))
                    cell.config(state='disabled', disabledforeground="#000080", font=('Arial', 18, 'bold'))
                else:
                    cell.config(state='normal', fg="#333333", font=('Arial', 18, 'bold'))
                    cell.delete(0, tk.END)

        def solve():
            find = find_empty(self.board)
            if not find: return True
            row, col = find
            self.update_gui_cell(row, col, 0, "solving")
            for num in range(1, 10):
                if is_valid(self.board, num, (row, col)):
                    self.board[row][col] = num
                    self.update_gui_cell(row, col, num, "placing")
                    if solve():
                        self.update_gui_cell(row, col, num, "solved")
                        return True
                    self.board[row][col] = 0
                    self.update_gui_cell(row, col, 0, "backtracking")
            self.update_gui_cell(row, col, 0, "reset")
            return False

        solve_successful = solve()

        if solve_successful: messagebox.showinfo("Sudoku Solver", "Puzzle Solved Successfully!")
        else: messagebox.showwarning("Sudoku Solver", "No solution exists."); self.clear_board(clear_initial=False)

        self.solving = False

    def clear_board(self, clear_initial=True):
        if self.solving: print("Cannot clear while solving."); return

        cells_to_keep = set(self.initial_cells) if not clear_initial else set()
        if clear_initial: self.initial_cells.clear()

        for r in range(9):
            for c in range(9):
                cell = self.cells[(r, c)]
                bg_color = "#e8e8e8" if (r // 3 + c // 3) % 2 != 0 else "#ffffff"
                cell.config(bg=bg_color)
                if not self.edit_mode and (r, c) in cells_to_keep:
                    cell.config(state='normal')
                    cell.delete(0, tk.END)
                    if self.board[r][c] != 0: cell.insert(0, str(self.board[r][c]))
                    cell.config(state='disabled', disabledforeground="#000080")
                else:
                    if not ((r,c) in self.initial_cells and not clear_initial):
                        self.board[r][c] = 0
                    cell.config(state='normal', fg="#333333", font=('Arial', 18, 'bold'))
                    cell.delete(0, tk.END)

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SudokuGUI(root)
    root.mainloop()