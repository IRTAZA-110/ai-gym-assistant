# main.py

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from pose_detector import PoseDetector
import exercise_logic as ex
import time
import csv
import os
from datetime import datetime
import threading


class GymAssistantApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.username = self.get_username()
        if not self.username:
            self.destroy()
            return

        # --- APP SETUP & STYLING ---
        self.title(f"AI Gym Assistant - Welcome, {self.username}!")
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        self.FONT_TITLE = ctk.CTkFont(size=24, weight="bold")
        self.FONT_LARGE = ctk.CTkFont(size=60, weight="bold")
        self.FONT_MEDIUM = ctk.CTkFont(size=20)
        self.FONT_SMALL = ctk.CTkFont(size=16)
        self.COLOR_SUCCESS, self.COLOR_INFO, self.COLOR_WARNING = "#2ECC71", "#3498DB", "#E74C3C"

        # --- EXERCISE SETUP ---
        self.exercise_logic_map = {
            "Bicep Curl": ex.bicep_curl_counter,
            "Shoulder Press": ex.shoulder_press_counter,
            "Side Raise": ex.side_raise_counter,
            "Overhead Clap": ex.overhead_clap_counter,
            "Jumping Jack": ex.jumping_jack_counter,
        }
        self.EXERCISES = list(self.exercise_logic_map.keys())
        self.current_exercise = self.EXERCISES[0]

        # --- STATE VARIABLES ---
        self.rep_counter, self.rep_goal = 0, 10
        self.set_counter, self.set_goal = 0, 3
        self.stage, self.feedback, self.feedback_type = "down", "Start", "info"
        self.app_state, self.rest_duration, self.rest_timer_start = "counting", 30, 0

        # --- THREADING SETUP ---
        self.detector = PoseDetector(complexity=0)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.latest_frame = None
        self.data_lock = threading.Lock()
        self.stop_event = threading.Event()

        # --- GUI LAYOUT ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._create_sidebar()
        self._create_main_panel()

        # --- START THE APP & BACKGROUND THREAD ---
        self._start_video_thread()
        self.update_gui()

    def _start_video_thread(self):
        self.video_thread = threading.Thread(target=self._video_processing_loop)
        self.video_thread.daemon = True
        self.video_thread.start()

    def _video_processing_loop(self):
        while not self.stop_event.is_set():
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            annotated_frame = self.detector.find_pose(frame, draw=True)
            lm_list = self.detector.find_landmarks(annotated_frame)

            with self.data_lock:
                if self.app_state == "counting" and len(lm_list) != 0:
                    logic_func = self.exercise_logic_map[self.current_exercise]
                    previous_rep_count = self.rep_counter
                    self.stage, self.rep_counter, self.feedback, self.feedback_type = logic_func(self.detector,
                                                                                                 self.stage,
                                                                                                 self.rep_counter)

                    if self.rep_counter > previous_rep_count and self.rep_counter >= self.rep_goal:
                        self.set_counter += 1
                        if self.set_counter >= self.set_goal:
                            self.app_state = 'finished'
                        else:
                            self.app_state = 'resting'
                            self.rest_timer_start = time.time()
                            self.rep_counter = 0

                self.latest_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        self.cap.release()

    def update_gui(self):
        if self.app_state == 'resting':
            self.process_resting_state()
        elif self.app_state == 'finished' or self.app_state == 'saved':
            self.process_finished_state()
        else:
            self.overlay_label.grid_forget()
            self.video_label.grid(row=0, column=0, sticky="nsew")

        with self.data_lock:
            frame = self.latest_frame
            rep_counter, set_counter = self.rep_counter, self.set_counter
            stage, feedback, feedback_type = self.stage, self.feedback, self.feedback_type

        if frame is not None:
            img = Image.fromarray(frame)
            img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(frame.shape[1], frame.shape[0]))
            self.video_label.configure(image=img_tk)

        self.reps_value.configure(text=str(rep_counter))
        self.sets_value.configure(text=str(set_counter))
        self.stage_value.configure(text=stage.upper())

        color = self.COLOR_INFO
        if feedback_type == "success":
            color = self.COLOR_SUCCESS
        elif feedback_type == "warning":
            color = self.COLOR_WARNING
        self.feedback_text.configure(text=feedback, text_color=color)

        progress = min(rep_counter / self.rep_goal, 1.0) if self.rep_goal > 0 else 0
        self.progress_bar.set(progress)

        self.after(30, self.update_gui)

    def get_username(self):
        dialog = ctk.CTkInputDialog(text="Enter your username:", title="Welcome!")
        return dialog.get_input()

    def _create_sidebar(self):
        sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=0)
        sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        sidebar_frame.grid_rowconfigure(6, weight=1)
        ctk.CTkLabel(sidebar_frame, text="AI Fitness Coach", font=self.FONT_TITLE).pack(pady=20)
        ctk.CTkLabel(sidebar_frame, text="Select Exercise", font=self.FONT_MEDIUM).pack(pady=(10, 5))
        self.exercise_menu = ctk.CTkOptionMenu(sidebar_frame, values=self.EXERCISES, command=self.on_exercise_change)
        self.exercise_menu.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(sidebar_frame, text="Set Rep Goal", font=self.FONT_MEDIUM).pack(pady=(10, 5))
        self.rep_goal_entry = ctk.CTkEntry(sidebar_frame, placeholder_text=f"Reps per set: {self.rep_goal}")
        self.rep_goal_entry.pack(pady=10, padx=20, fill="x")
        self.rep_goal_entry.bind("<Return>", self.set_new_goals)
        ctk.CTkLabel(sidebar_frame, text="Set Total Sets", font=self.FONT_MEDIUM).pack(pady=(10, 5))
        self.set_goal_entry = ctk.CTkEntry(sidebar_frame, placeholder_text=f"Total sets: {self.set_goal}")
        self.set_goal_entry.pack(pady=10, padx=20, fill="x")
        self.set_goal_entry.bind("<Return>", self.set_new_goals)
        self.reset_button = ctk.CTkButton(sidebar_frame, text="Reset Workout", command=self.reset_workout)
        self.reset_button.pack(pady=20, padx=20, fill="x")
        self.quit_button = ctk.CTkButton(sidebar_frame, text="Quit", fg_color="#C0392B", hover_color="#E74C3C",
                                         command=self.on_closing)
        self.quit_button.pack(side="bottom", pady=20, padx=20, fill="x")

    def _create_main_panel(self):
        main_panel = ctk.CTkFrame(self, fg_color="transparent")
        main_panel.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        main_panel.grid_columnconfigure(0, weight=1)
        main_panel.grid_rowconfigure(2, weight=1)
        stats_frame = ctk.CTkFrame(main_panel)
        stats_frame.grid(row=0, column=0, pady=(0, 10), sticky="ew")
        stats_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.reps_value = self._create_stat_box(stats_frame, "REPS", self.rep_counter, 0)
        self.sets_value = self._create_stat_box(stats_frame, "SETS", self.set_counter, 1)
        self.stage_value = self._create_stat_box(stats_frame, "STAGE", self.stage.upper(), 2)
        self.progress_bar = ctk.CTkProgressBar(main_panel, orientation="horizontal")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, pady=10, sticky="ew")
        self.video_container = ctk.CTkFrame(main_panel, fg_color="black")
        self.video_container.grid(row=2, column=0, sticky="nsew")
        self.video_container.grid_columnconfigure(0, weight=1)
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_label = ctk.CTkLabel(self.video_container, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.overlay_label = ctk.CTkLabel(self.video_container, text="", font=self.FONT_LARGE, fg_color="#2B2B2B",
                                          text_color=self.COLOR_SUCCESS)
        self.feedback_text = ctk.CTkLabel(main_panel, text=self.feedback, font=self.FONT_MEDIUM, wraplength=700)
        self.feedback_text.grid(row=3, column=0, pady=10)

    def _create_stat_box(self, parent, title, initial_value, col):
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=0, column=col)
        ctk.CTkLabel(box, text=title, font=self.FONT_MEDIUM).pack()
        value_label = ctk.CTkLabel(box, text=str(initial_value), font=self.FONT_LARGE)
        value_label.pack()
        return value_label

    def set_new_goals(self, event=None):
        try:
            new_rep_goal = int(self.rep_goal_entry.get()) if self.rep_goal_entry.get() else self.rep_goal
            new_set_goal = int(self.set_goal_entry.get()) if self.set_goal_entry.get() else self.set_goal
            if new_rep_goal > 0 and new_set_goal > 0:
                with self.data_lock:
                    self.rep_goal, self.set_goal = new_rep_goal, new_set_goal
                self.reset_workout()
        except ValueError:
            with self.data_lock:
                self.feedback = "Please enter valid numbers for goals."

    # --- CORRECTED METHODS ---
    def on_exercise_change(self, new_exercise):
        """
        Called when a new exercise is selected.
        This no longer uses a lock, preventing the deadlock.
        """
        self.reset_workout(new_exercise=new_exercise)

    def reset_workout(self, new_exercise=None):
        """
        Resets all workout state variables. Made fully thread-safe.
        Now optionally accepts a new exercise to switch to.
        """
        with self.data_lock:
            if new_exercise:
                self.current_exercise = new_exercise
            self.rep_counter = 0
            self.set_counter = 0
            self.app_state = "counting"
            self.stage = "down"
            self.feedback = "Let's begin!"
            self.feedback_type = "info"

        # GUI updates can happen outside the lock
        if hasattr(self, 'progress_bar'):
            self.progress_bar.set(0)

    # --- REST OF THE METHODS (UNCHANGED BUT INCLUDED FOR COMPLETENESS) ---
    def process_resting_state(self):
        self.video_label.grid_forget()
        self.overlay_label.grid(row=0, column=0, sticky="nsew")
        elapsed = time.time() - self.rest_timer_start
        remaining = max(0, self.rest_duration - elapsed)
        self.overlay_label.configure(text=f"REST\n{int(remaining)}s")
        self.feedback_text.configure(text=f"Next set in {int(remaining)}s.")
        if remaining <= 0:
            with self.data_lock:
                self.app_state = "counting"
                self.feedback = "Let's go!"

    def process_finished_state(self):
        if self.app_state == 'finished':
            self._save_workout_history()
            with self.data_lock:
                self.app_state = "saved"
        self.video_label.grid_forget()
        self.overlay_label.grid(row=0, column=0, sticky="nsew")
        self.overlay_label.configure(text="Workout\nComplete!")
        self.feedback_text.configure(text="Great job! Select a new exercise or reset.")
        self.progress_bar.set(1.0)

    def _save_workout_history(self):
        filename = "workout_history.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Username", "Exercise", "Total_Sets", "Reps_Per_Set"])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, self.username, self.current_exercise, self.set_goal, self.rep_goal])
        print(f"Workout for {self.username} saved to {filename}")

    def on_closing(self):
        print("Closing application...")
        self.stop_event.set()
        # No need to join(), daemon thread will exit with main app
        self.destroy()


if __name__ == "__main__":
    app = GymAssistantApp()
    if app.winfo_exists():
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        apitti
        p.mainloop()