"""
================================================================================
 Dateiname   :    YoloTSRVideoPlayer.py
 Projekt     :    ROS Turtlebot3 Object Detection GUI
 Firma       :    Inclusive Gaming/ Hochschule Anhalt
 Version     :    1.0
 Erstellt am :    19.05.2025
 Letzte Änderung : 19.05.2025

 Beschreibung:
 ------------------------------------------------------------------------------
 Die GUI-Enwicklungsplattform wird genutzt um Algorithmen zu testen und 
 die Objekterkennung von z.B. Verkehrsschildern zu verbessern. Dies soll es 
 den Studenten ermöglichen die TSR schnell per GUI zu verstehen, zu evaluieren 
 und weiter zu optimieren.

 Der code nutzt aktuell ein Yolo Model. Es können aber auch andere Algorithmen 
 verwendet werden. Siehe Methode FrameProcess().

 Änderungen:
 ------------------------------------------------------------------------------
 |   Datum    |   Autor        |   Version   |   Beschreibung der Änderung      |
 |------------|----------------|-------------|----------------------------------|
 | 19.05.2025 | IG             | [1.0]       | Erste Version                    |
 | TT.MM.JJJJ | [Ihr Name]     | [1.1]       | [Änderung hier beschreiben]      |
================================================================================
"""



import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import os
import time
# from ultralytics import YOLO
import torch
from nanodet.util import cfg, load_config, Logger, mkdir,load_model_weight
from nanodet.model.arch import build_model
from nanodet.data.transform import Pipeline
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("HS-Anhalt ROS TSR Yolo Tool - Inclusive Gaming")
        self.root.geometry("800x600")

        # Video variables
        self.cap = None
        self.video_path = None
        self.out_path = None
        self.is_playing = False
        self.current_frame = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.save_enabled = False
        self.source_type = "file"  # Default source type is file

        # Output video writer
        self.out_writer = None
        # Camera variables
        self.selected_camera_index = 0  # Default camera index
        # self.camera_list = self.detect_cameras()  # Detect available cameras
        
        # Create GUI elements
        self.create_widgets()

        # YOLO model variables
        self.model_path = os.path.join('.', 'Model', 'best.pt')  # Default model path
        self.model = None
        self.load_nanodet_model('nanodet/config/my_dataset.yml', 'workspace/nanodet-plus-m-1.5x_320/model_best/model_best.ckpt', device="cpu")  # Load NanoDet model
        #self.load_yolo_model(self.model_path)  # Load YOLO model with error handling

        # Bind key press event for taking a snapshot
        self.root.bind("s", lambda event: self.take_snapshot())  # Bind "s" key to take_snapshot

        # Bind resizing event to handle window size changes
        self.root.bind("<Configure>", self.on_resize)

        # Create a snapshots folder if it does not exist
        os.makedirs("snapshots", exist_ok=True)

    def create_widgets(self):
        # Frame for buttons
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, pady=10)

        # Source Selection: File or Camera
        source_frame = tk.Frame(self.root)
        source_frame.pack(side=tk.TOP, pady=10)

        self.source_var = tk.StringVar(value="file")
        self.file_radio = ttk.Radiobutton(source_frame, text="Video File", variable=self.source_var, value="file", command=self.update_source)
        self.file_radio.pack(side=tk.LEFT, padx=5)

        self.camera_radio = ttk.Radiobutton(source_frame, text="Camera", variable=self.source_var, value="camera", command=self.update_source)
        self.camera_radio.pack(side=tk.LEFT, padx=5)

        # Camera selection dropdown
        self.camera_label = ttk.Label(source_frame, text="Camera:")
        self.camera_label.pack(side=tk.LEFT, padx=5)

        # self.camera_combobox = ttk.Combobox(source_frame, state="readonly")
        # self.camera_combobox['values'] = [f"Camera {i}" for i in range(len(self.camera_list))]
        # self.camera_combobox.current(0)  # Select the first camera by default
        # self.camera_combobox.bind("<<ComboboxSelected>>", self.select_camera)
        # self.camera_combobox.pack(side=tk.LEFT, padx=5)

        # Toggle Play/Pause Button
        self.play_pause_button = ttk.Button(control_frame, text="Play", command=self.toggle_play_pause)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)

        # Stop Button
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_video)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Forward and Reverse Buttons
        self.forward_button = ttk.Button(control_frame, text=">>", command=self.forward)
        self.forward_button.pack(side=tk.LEFT, padx=5)

        self.reverse_button = ttk.Button(control_frame, text="<<", command=self.reverse)
        self.reverse_button.pack(side=tk.LEFT, padx=5)

        # Snapshot Button
        self.snapshot_button = ttk.Button(control_frame, text="Snapshot [s]", command=self.take_snapshot)
        self.snapshot_button.pack(side=tk.LEFT, padx=5)

        # Change YOLO Model Button
        self.change_model_button = ttk.Button(control_frame, text="Change YOLO Model", command=self.change_model_path)
        self.change_model_button.pack(side=tk.LEFT, padx=5)

        # Slider for seeking position
        self.progress_slider = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek_video)
        self.progress_slider.pack(fill=tk.X, pady=5)

        # Video display canvas
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # File selection buttons
        file_frame = tk.Frame(self.root)
        file_frame.pack(side=tk.TOP, pady=10)

        self.open_button = ttk.Button(file_frame, text="Open Video", command=self.open_video)
        self.open_button.pack(side=tk.LEFT, padx=5)

        # Checkbox for enabling/disabling output file generation
        self.save_var = tk.BooleanVar()
        self.save_checkbox = ttk.Checkbutton(file_frame, text="Enable Output File Generation", variable=self.save_var, command=self.toggle_save)
        self.save_checkbox.pack(side=tk.LEFT, padx=5)

        # Checkbox for enabling/disabling frame processing
        self.process_var = tk.BooleanVar()
        self.process_checkbox = ttk.Checkbutton(file_frame, text="Enable Frame Processing", variable=self.process_var)
        self.process_checkbox.pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def detect_cameras(self):
        """ Dynamically detect available cameras connected to the system with error handling. """
        available_cameras = []
        index = 0

        while True:
            try:
                cap = cv2.VideoCapture(index)
                if cap is None or not cap.isOpened():
                    break  # Stop if the camera index is invalid or unavailable
                available_cameras.append(index)
                cap.release()
                index += 1
            except Exception as e:
                # Handle any unexpected errors during camera detection
                print(f"Error accessing camera index {index}: {e}")
                break

        return available_cameras

    def load_nanodet_model(self ,config_path, weight_path,  device="cpu"):
       
        self.device = torch.device(device)
        load_config(cfg, config_path)
        logger = Logger(local_rank=0, use_tensorboard=False)
        model = build_model(cfg.model)#.to(self.device)

        # Load checkpoint
        ckpt = torch.load(weight_path, map_location=device)
        load_model_weight(model, ckpt, logger)

        model.to(self.device).eval()

        # Setup pipeline for preprocessing
        pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        
        # Save for later inference
        self.nanodet_model = model
        self.nanodet_pipeline = pipeline
        self.nanodet_logger = logger
        self.nanodet_cfg = cfg

    def nanodet_inference(self, frame):
        # Prepare meta dictionary like the demo
        meta = {
            "img": frame,
            "raw_img": frame.copy(),
            "img_info": {
                "height": frame.shape[0],
                "width": frame.shape[1],
                "id": 0,
                "file_name": None,
            },
        }

        # Preprocess
        meta = self.nanodet_pipeline(None, meta, self.nanodet_cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)

        # Inference
        with torch.no_grad():
            results = self.nanodet_model.inference(meta)

        return meta, results


    # def load_yolo_model(self, model_path):
    #     """ Load the YOLO model with error handling. """
    #     try:
    #         self.model = YOLO(model_path)
    #         self.status_var.set(f"Model loaded successfully: {model_path}")
    #     except Exception as e:
    #         self.model = None
    #         messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
    #         self.status_var.set("Failed to load YOLO model")

    def change_model_path(self):
        """ Allow the user to change the YOLO model path. """
        new_model_path = filedialog.askopenfilename(title="Select YOLO Model File", filetypes=[("YOLO Model Files", "*.pt")])
        if new_model_path:
            self.model_path = new_model_path
            self.load_yolo_model(self.model_path)

    def select_camera(self, event=None):
        """ Update the selected camera index based on the dropdown selection. """
        self.selected_camera_index = self.camera_combobox.current()

    def update_source(self):
        """ Update source type based on radio button selection. """
        self.source_type = self.source_var.get()
        if self.source_type == "camera":
            # Disable file selection buttons if camera is selected
            self.open_button.config(state=tk.DISABLED)
            self.progress_slider.config(state=tk.DISABLED)
            # Use the first camera by default
            self.open_camera(self.selected_camera_index)
        else:
            # Enable file selection buttons if file is selected
            self.open_button.config(state=tk.NORMAL)
            self.progress_slider.config(state=tk.NORMAL)
            if self.cap:
                self.cap.release()  # Release the camera if switching to file mode

    def open_camera(self, camera_index):
        """ Open the selected camera by index. """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Failed to open the camera at index {camera_index}.")
            self.cap = None

    def open_video(self):
        if self.source_type == "file":
            self.video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
            if self.video_path:
                self.cap = cv2.VideoCapture(self.video_path)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.progress_slider.config(to=self.total_frames)
            else:
                messagebox.showerror("Error", "Failed to open video file.")
        elif self.source_type == "camera":
            self.open_camera(self.selected_camera_index)

    def save_processed_video(self):
        self.out_path = filedialog.asksaveasfilename(title="Save Processed Video", defaultextension=".avi", filetypes=[("AVI Files", "*.avi")])
        if self.out_path and self.cap:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Initialize the video writer with the output path
            self.out_writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (frame_width, frame_height))
        else:
            self.out_path = None
            messagebox.showerror("Error", "Failed to set output file.")
            self.save_var.set(False)  # Reset the checkbox if setting output fails

    def toggle_save(self):
        if self.save_var.get():
            # If saving is enabled, prompt user to set an output file
            self.save_processed_video()
            # If no valid file path was set, disable the checkbox
            if not self.out_path:
                self.save_var.set(False)
        else:
            # If the checkbox is unchecked, disable file saving
            self.out_writer = None

        # Disable seeking if output generation is enabled
        if self.save_var.get():
            self.forward_button.config(state=tk.DISABLED)
            self.reverse_button.config(state=tk.DISABLED)
            self.progress_slider.config(state=tk.DISABLED)
        else:
            # Enable seeking controls if output generation is disabled
            self.forward_button.config(state=tk.NORMAL)
            self.reverse_button.config(state=tk.NORMAL)
            self.progress_slider.config(state=tk.NORMAL)

    def toggle_play_pause(self):
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()

    def play_video(self):
        if not self.cap:
            messagebox.showwarning("Warning", "Please open a video file or camera first.")
            return
        if self.save_var.get() and not self.out_writer:
            messagebox.showwarning("Warning", "Please set an output file before playing.")
            return

        self.is_playing = True
        self.play_pause_button.config(text="Pause")
        self.video_playback_loop()  # Use a loop function for video playback

    def pause_video(self):
        self.is_playing = False
        self.play_pause_button.config(text="Play")

    def stop_video(self):
        self.is_playing = False
        if self.cap and self.source_type == "file":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.progress_slider.set(0)
        self.play_pause_button.config(text="Play")

    def forward(self):
        if not self.save_var.get() and self.cap:
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, min(current_pos + 30, self.total_frames))
            self.display_current_frame()

    def reverse(self):
        if not self.save_var.get() and self.cap:
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(current_pos - 30, 0))
            self.display_current_frame()

    def seek_video(self, value):
        if not self.save_var.get() and self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))
            self.display_current_frame()

    def video_playback_loop(self):
        """ Continuous video playback loop without recursion. """
        while self.is_playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.is_playing = False
                self.play_pause_button.config(text="Play")
                break

            # Process the frame if the checkbox is enabled
            processed_frame = self.FrameProcess(frame)

            # Save the current frame for snapshot functionality
            self.current_frame = processed_frame.copy()

            # Display the frame
            self.display_frame(processed_frame)

            # Save processed frame if output is enabled
            if self.save_var.get() and self.out_writer:
                self.out_writer.write(processed_frame)

            # Update slider for file playback
            if self.source_type == "file":
                current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.progress_slider.set(current_pos)

            # Update GUI to handle events
            self.root.update()

    def display_current_frame(self):
        """ Display the current frame for seeking without recursion. """
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                processed_frame = self.FrameProcess(frame)
                self.current_frame = processed_frame.copy()
                self.display_frame(processed_frame)

    def display_frame(self, frame):
        # Resize the frame to match the canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        resized_frame = cv2.resize(frame, (canvas_width, canvas_height))

        # Convert to RGB and display
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Display on canvas
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def on_resize(self, event):
        if self.is_playing and self.cap:
            # Force an update of the displayed frame when the window is resized
            self.display_current_frame()

    # def FrameProcess(self, frame):
    #     # Check if frame processing is enabled
    #     if self.process_var.get() and self.model:
    #         # Add "Processed" text to the frame
    #         cv2.putText(frame, 'P', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
    #         #run Yolo Model or another Model
    #         start_time = time.time()
    #         results = self.model(frame, conf = 0.70,imgsz=320, device="cpu") # confidence at least 0.7
    #         #results = self.model(frame, conf = 0.80,imgsz=320) # confidence at least 0.8
    #         end_time = time.time()
    #         elapsed_time = (end_time - start_time)*1000
    #         print(f"Yolo duration time: {elapsed_time:.4f} msecond")
    #         frame = results[0].plot()
    #     return frame



    def FrameProcess(self, frame):
        if self.process_var.get() and hasattr(self, "nanodet_model"):
            start_time = time.time()
            meta, results = self.nanodet_inference(frame)
            result_frame = self.nanodet_model.head.show_result(
                meta["raw_img"][0], results[0], self.nanodet_cfg.class_names, score_thres=0.35, show=False
            )
            elapsed_time = (time.time() - start_time) * 1000
            print(f"NanoDet inference time: {elapsed_time:.2f} ms")
            return result_frame
        return frame


    def take_snapshot(self):
        if self.current_frame is not None:
            # Save the current frame as a JPEG in the "snapshots" folder
            snapshot_filename = f"snapshots/snapshot_{int(time.time())}.jpg"
            cv2.imwrite(snapshot_filename, self.current_frame)
            #messagebox.showinfo("Snapshot Saved", f"Snapshot saved as {snapshot_filename}")
        else:
            messagebox.showwarning("No Frame", "No frame available to snapshot.")

if __name__ == "__main__":
    cv2.setNumThreads(0)  # Disable OpenCV multi-threading
    root = tk.Tk()
    player = VideoPlayer(root)

    root.mainloop()
