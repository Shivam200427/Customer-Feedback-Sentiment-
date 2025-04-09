print("started")


# app.py
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import threading
import time
import os
import json
import shutil
from datetime import datetime
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor
import uuid
import wave
import pyaudio





import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os





app = Flask(__name__)

class EnhancedEmotionCaptureSystem:
    def __init__(self):
        # Create directories for storing data
        self.data_dir = "emotion_data"
        self.recordings_dir = "recordings"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # JSON file paths
        self.current_data_file = os.path.join(self.data_dir, "current_emotions.json")
        self.history_data_file = os.path.join(self.data_dir, "emotion_history.json")
        self.sessions_file = os.path.join(self.data_dir, "sessions.json")
        
        # Camera and processing variables
        self.camera = None
        self.is_running = False
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.processed_frame = None
        
        # Thread management
        self.camera_thread = None
        self.processing_thread = None
        self.recording_thread = None
        self.audio_thread = None
        self.should_run = False
        self.is_recording = False
        
        # Emotion data
        self.emotion_data = {"faces": {}, "lock": threading.Lock()}
        self.last_save_time = 0
        
        # Thread Pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Configuration
        self.detection_interval = 0.1  # Seconds between emotion detections
        self.save_interval = 10.0      # Auto-save every 10 seconds
        self.max_width = 640           # Resize frames for better performance
        
        # Face detection optimizations
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # For tracking and smoothing emotions
        self.emotion_history = {}
        self.smoothing_window_size = 5
        
        # Recording variables
        self.video_writer = None
        self.recording_frames = []
        self.audio_frames = []
        self.session_id = None
        self.session_start_time = None
        self.recorded_frames = []
        self.review_text = ""
        
        # Audio settings
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()
        self.audio_stream = None

    def start_camera(self):
        """Initialize and start the camera in a separate thread"""
        if self.is_running:
            return True
            
        self.camera = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.camera.isOpened():
            print("Failed to open camera")
            return False
            
        self.should_run = True
        self.is_running = True
        
        # Reset emotion data
        self.emotion_data = {"faces": {}, "lock": threading.Lock()}
        self.emotion_history = {}
        self.last_save_time = time.time()
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self._camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("Camera and processing started")
        return True

    def stop_camera(self):
        """Stop the camera and processing threads"""
        self.should_run = False
        
        # Stop recording if it's running
        if self.is_recording:
            self.stop_recording()
        
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
            
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            
        if self.camera:
            self.camera.release()
            self.camera = None
            
        self.is_running = False
        
        # Save final emotion data
        self.save_emotion_data()
        print("Camera and processing stopped")
        return True

    def start_recording(self, review_text=""):
        """Start recording video and audio"""
        if not self.is_running or self.is_recording:
            return False
            
        self.is_recording = True
        self.session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now()
        self.review_text = review_text
        self.recorded_frames = []
        self.audio_frames = []
        
        # Get video properties
        fps = self.camera.get(cv2.CAP_PROP_FPS)
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        self.recording_dir = os.path.join(self.recordings_dir, self.session_id)
        os.makedirs(self.recording_dir, exist_ok=True)
        
        video_path = os.path.join(self.recording_dir, f"recording.mp4")
        self.video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        # Start audio recording
        self.audio_stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Start audio thread
        self.audio_thread = threading.Thread(target=self._audio_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        print(f"Recording started with session ID: {self.session_id}")
        return True

    def stop_recording(self):
        """Stop recording and save files"""
        if not self.is_recording:
            return False
            
        self.is_recording = False
        
        # Wait for threads to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
            
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
            
        # Close video writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        # Close audio stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
            
        # Save audio to WAV file
        audio_path = os.path.join(self.recording_dir, f"audio.wav")
        with wave.open(audio_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.audio_frames))
            
        # Save session info
        session_end_time = datetime.now()
        duration = (session_end_time - self.session_start_time).total_seconds()
        
        session_info = {
            "session_id": self.session_id,
            "start_time": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": session_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "video_path": f"recordings/{self.session_id}/recording.mp4",
            "audio_path": f"recordings/{self.session_id}/audio.wav",
            "review": self.review_text,
            "emotion_summary": self.compute_average_emotions()
        }
        
        # Save to session file
        json_path = os.path.join(self.recording_dir, f"session_info.json")
        with open(json_path, 'w') as f:
            json.dump(session_info, f, indent=4)
            
        # Update sessions history
        sessions = []
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r') as f:
                    sessions = json.load(f)
            except json.JSONDecodeError:
                sessions = []
                
        sessions.append(session_info)
        
        with open(self.sessions_file, 'w') as f:
            json.dump(sessions, f, indent=4)
            
        print(f"Recording stopped and saved to {self.recording_dir}")
        return session_info

    def _camera_loop(self):
        """Camera capture loop running in a separate thread"""
        last_frame_time = time.time()
        frame_count = 0
        
        while self.should_run and self.camera:
            success, frame = self.camera.read()
            
            if not success:
                time.sleep(0.01)
                continue
                
            # Flip horizontally for natural mirror view
            frame = cv2.flip(frame, 1)
            
            # Update current frame with lock
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            if current_time - last_frame_time >= 1.0:
                fps = frame_count / (current_time - last_frame_time)
                print(f"Camera FPS: {fps:.2f}")
                frame_count = 0
                last_frame_time = current_time
                
            # Auto-save emotion data
            if current_time - self.last_save_time >= self.save_interval:
                self.save_emotion_data()
                self.last_save_time = current_time
                
        print("Camera loop exited")

    def _processing_loop(self):
        """Processing loop running in a separate thread"""
        last_detection_time = time.time()
        
        while self.should_run:
            current_time = time.time()
            
            # Get current frame with lock
            with self.frame_lock:
                if self.current_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.current_frame.copy()
            
            # Process for emotion detection at intervals
            if current_time - last_detection_time >= self.detection_interval:
                # Create a resized version for processing
                if frame.shape[1] > self.max_width:
                    scale = self.max_width / frame.shape[1]
                    small_frame = cv2.resize(frame, (self.max_width, int(frame.shape[0] * scale)))
                else:
                    small_frame = frame.copy()
                
                # Process frame
                processed_frame = self._process_frame(frame, small_frame)
                
                # Update processed frame with lock
                with self.frame_lock:
                    self.processed_frame = processed_frame
                    
                last_detection_time = current_time
            
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
            
        print("Processing loop exited")

    def _recording_loop(self):
        """Video recording loop running in a separate thread"""
        while self.is_recording and self.should_run:
            with self.frame_lock:
                if self.current_frame is not None:
                    # Save frame to video
                    if self.video_writer:
                        self.video_writer.write(self.current_frame)
                    
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
            
        print("Recording loop exited")

    def _audio_loop(self):
        """Audio recording loop running in a separate thread"""
        while self.is_recording and self.should_run:
            if self.audio_stream:
                # Read audio data
                data = self.audio_stream.read(self.chunk, exception_on_overflow=False)
                self.audio_frames.append(data)
                
        print("Audio loop exited")

    def _process_frame(self, frame, small_frame):
        """Process a frame for emotion detection"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Optimize face detection with scale factor and minimum neighbors
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each detected face
        face_futures = []
        for i, (x, y, w, h) in enumerate(faces):
            # Scale coordinates back to original frame size
            if small_frame.shape != frame.shape:
                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]
                x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Submit to thread pool
            future = self.executor.submit(self._analyze_face_emotion, face_img, i)
            face_futures.append((future, (x, y, w, h)))
        
        # Draw overlay with detected emotions
        for future, (x, y, w, h) in face_futures:
            result = future.result()
            if result:
                # Store emotion data
                with self.emotion_data["lock"]:
                    self.emotion_data["faces"][result["face_id"]] = result["emotions"]
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Show dominant emotion
                emotion_text = f"{result['dominant_emotion']}: {result['dominant_confidence']:.1f}%"
                cv2.putText(frame, emotion_text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw average emotions on frame
        self._draw_emotion_overlay(frame)
        
        # Add recording indicator if recording
        if self.is_recording:
            # Draw red circle indicator
            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (50, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw timer
            if self.session_start_time:
                elapsed = (datetime.now() - self.session_start_time).total_seconds()
                time_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                cv2.putText(frame, time_str, (50, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def _analyze_face_emotion(self, face_img, face_id):
        """Analyze emotions in a face image using DeepFace"""
        try:
            if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
                return None
                
            # Preprocess face image
            face_img = cv2.resize(face_img, (224, 224))
                
            # Analyze with DeepFace
            result = DeepFace.analyze(
                face_img, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True
            )
            
            if result and len(result) > 0 and 'emotion' in result[0]:
                emotions = result[0]['emotion']
                dominant_emotion = max(emotions, key=emotions.get)
                dominant_confidence = emotions[dominant_emotion]
                
                # Apply smoothing if we have history
                if face_id in self.emotion_history:
                    # Add current to history
                    self.emotion_history[face_id].append(emotions)
                    
                    # Keep history within window size
                    if len(self.emotion_history[face_id]) > self.smoothing_window_size:
                        self.emotion_history[face_id].pop(0)
                    
                    # Calculate smoothed emotions
                    smoothed_emotions = self._smooth_emotions(face_id)
                    dominant_emotion = max(smoothed_emotions, key=smoothed_emotions.get)
                    dominant_confidence = smoothed_emotions[dominant_emotion]
                    
                    return {
                        "face_id": face_id,
                        "dominant_emotion": dominant_emotion,
                        "dominant_confidence": dominant_confidence,
                        "emotions": smoothed_emotions
                    }
                else:
                    # First detection for this face
                    self.emotion_history[face_id] = [emotions]
                    
                    return {
                        "face_id": face_id,
                        "dominant_emotion": dominant_emotion,
                        "dominant_confidence": dominant_confidence,
                        "emotions": emotions
                    }
                    
        except Exception as e:
            print(f"Error analyzing face emotion: {e}")
            
        return None

    def _smooth_emotions(self, face_id):
        """Apply temporal smoothing to emotions for stability"""
        if face_id not in self.emotion_history:
            return {}
            
        # Initialize with all emotion types
        all_emotions = set()
        for history_item in self.emotion_history[face_id]:
            all_emotions.update(history_item.keys())
            
        # Compute average for each emotion
        smoothed = {emotion: 0.0 for emotion in all_emotions}
        history_length = len(self.emotion_history[face_id])
        
        for history_item in self.emotion_history[face_id]:
            for emotion in all_emotions:
                smoothed[emotion] += history_item.get(emotion, 0.0) / history_length
                
        return smoothed

    def _draw_emotion_overlay(self, frame):
        """Draw average emotions overlay on frame"""
        avg_emotions = self.compute_average_emotions()
        
        if not avg_emotions:
            return frame
            
        # Draw background for emotion text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw emotion text
        y_offset = 30
        cv2.putText(frame, "Average Emotions:", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        for emotion, value in sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True):
            cv2.putText(frame, f"{emotion}: {value:.1f}%", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 20
            
        return frame

    def get_frame(self):
        """Get the latest processed frame"""
        with self.frame_lock:
            if self.processed_frame is not None:
                # Encode the frame as JPEG
                ret, jpeg = cv2.imencode('.jpg', self.processed_frame)
                if ret:
                    return jpeg.tobytes()
            elif self.current_frame is not None:
                # If no processed frame, use current frame
                ret, jpeg = cv2.imencode('.jpg', self.current_frame)
                if ret:
                    return jpeg.tobytes()
                    
        return None

    def compute_average_emotions(self):
        """Compute average emotions across all detected faces"""
        with self.emotion_data["lock"]:
            faces = self.emotion_data["faces"].copy()
        
        if not faces:
            return {}
        
        # Combine emotions from all faces
        all_emotions = {}
        for face_emotions in faces.values():
            for emotion, value in face_emotions.items():
                if emotion in all_emotions:
                    all_emotions[emotion] += value
                else:
                    all_emotions[emotion] = value
        
        # Calculate averages
        total_faces = len(faces)
        avg_emotions = {emotion: value/total_faces for emotion, value in all_emotions.items()}
        
        return avg_emotions

    def save_emotion_data(self):
        """Save emotion data to JSON files"""
        try:
            avg_emotions = self.compute_average_emotions()
            
            if avg_emotions:
                # Create timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Current emotions data
                current_data = {
                    "timestamp": timestamp,
                    "avg_emotions": avg_emotions,
                    "total_faces": len(self.emotion_data["faces"])
                }
                
                with open(self.current_data_file, 'w') as f:
                    json.dump(current_data, f, indent=4)
                
                # Historical data
                history_data = []
                if os.path.exists(self.history_data_file):
                    try:
                        with open(self.history_data_file, 'r') as f:
                            history_data = json.load(f)
                    except json.JSONDecodeError:
                        history_data = []
                
                history_data.append(current_data)
                
                with open(self.history_data_file, 'w') as f:
                    json.dump(history_data, f, indent=4)
                
                print(f"Emotion data auto-saved at {timestamp}")
                return current_data
            
            return None
        except Exception as e:
            print(f"Error saving emotion data: {e}")
            return None

    def get_session_count(self):
        """Get the number of recorded sessions"""
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r') as f:
                    sessions = json.load(f)
                return len(sessions)
            except:
                pass
        return 0











# ====================================================================================================================================================================================

    # Add this method to the EnhancedEmotionCaptureSystem class
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio file to text using Google Speech Recognition"""
        # Create a directory for transcripts if it doesn't exist
        transcripts_dir = os.path.join(self.recording_dir, "transcripts")
        os.makedirs(transcripts_dir, exist_ok=True)
        
        transcript_file = os.path.join(transcripts_dir, "transcript.txt")
        full_transcript = ""
        
        try:
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Load audio file with pydub for chunking
            sound = AudioSegment.from_wav(audio_file_path)
            
            # Split audio where silence is 500ms or more and get chunks
            chunks = split_on_silence(
                sound,
                min_silence_len=500,  # minimum silence length in ms
                silence_thresh=sound.dBFS-14,  # silence threshold
                keep_silence=500  # keep 500ms of leading/trailing silence
            )
            
            # If the audio file is very short, use the whole file
            if len(chunks) == 0:
                chunks = [sound]
            
            print(f"Audio will be processed in {len(chunks)} chunks")
            
            # Process each chunk with recognition
            for i, chunk in enumerate(chunks):
                # Export chunk as a temporary wav file
                chunk_filename = os.path.join(transcripts_dir, f"chunk{i}.wav")
                chunk.export(chunk_filename, format="wav")
                
                # Recognize speech in the chunk
                with sr.AudioFile(chunk_filename) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data)
                        full_transcript += text + " "
                        print(f"Chunk {i}: {text}")
                    except sr.UnknownValueError:
                        print(f"Chunk {i}: Speech not recognized")
                    except sr.RequestError as e:
                        print(f"Chunk {i}: Could not request results; {e}")
                
                # Remove temporary chunk file
                os.remove(chunk_filename)
            
            # Save full transcript
            with open(transcript_file, 'w') as f:
                f.write(full_transcript.strip())
            
            return full_transcript.strip()
            
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            
            # Save error message
            with open(transcript_file, 'w') as f:
                f.write(f"Transcription error: {str(e)}")
                
            return f"Transcription error: {str(e)}"

    # Modify the stop_recording method in EnhancedEmotionCaptureSystem class
    # Update the method by adding the transcript processing section
    def stop_recording(self):
        """Stop recording and save files"""
        if not self.is_recording:
            return False
            
        self.is_recording = False
        
        # Wait for threads to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
            
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
            
        # Close video writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        # Close audio stream
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
            
        # Save audio to WAV file
        audio_path = os.path.join(self.recording_dir, "audio.wav")
        with wave.open(audio_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.audio_frames))
        
        # Create session info with basic data first
        session_end_time = datetime.now()
        duration = (session_end_time - self.session_start_time).total_seconds()
        
        session_info = {
            "session_id": self.session_id,
            "start_time": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": session_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "video_path": f"recordings/{self.session_id}/recording.mp4",
            "audio_path": f"recordings/{self.session_id}/audio.wav",
            "review": self.review_text,
            "emotion_summary": self.compute_average_emotions(),
            "transcript": "Processing...",  # Placeholder
            "transcript_path": f"recordings/{self.session_id}/transcripts/transcript.txt"
        }
        
        # Save initial session info (without transcript)
        json_path = os.path.join(self.recording_dir, "session_info.json")
        with open(json_path, 'w') as f:
            json.dump(session_info, f, indent=4)
        
        # Start transcription in a separate thread to avoid blocking
        def transcribe_and_update():
            try:
                print("Starting audio transcription...")
                transcript = self.transcribe_audio(audio_path)
                print(f"Transcription complete: {transcript[:100]}...")
                
                # Update session info with transcript
                session_info["transcript"] = transcript
                
                # Save updated session info
                with open(json_path, 'w') as f:
                    json.dump(session_info, f, indent=4)
                    
                # Update sessions history
                sessions = []
                if os.path.exists(self.sessions_file):
                    try:
                        with open(self.sessions_file, 'r') as f:
                            sessions = json.load(f)
                    except json.JSONDecodeError:
                        sessions = []
                        
                # Find and update the session in history
                for i, session in enumerate(sessions):
                    if session["session_id"] == self.session_id:
                        sessions[i] = session_info
                        break
                
                with open(self.sessions_file, 'w') as f:
                    json.dump(sessions, f, indent=4)
                    
            except Exception as e:
                print(f"Error in transcription process: {e}")
        
        # Start transcription in background
        threading.Thread(target=transcribe_and_update, daemon=True).start()
            
        print(f"Recording stopped and saved to {self.recording_dir}")
        return session_info
# ====================================================================================================================================================================================




















# Create system instance
emotion_system = EnhancedEmotionCaptureSystem()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while emotion_system.is_running:
            frame = emotion_system.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.01)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera endpoint"""
    success = emotion_system.start_camera()
    return jsonify({"success": success})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera endpoint"""
    success = emotion_system.stop_camera()
    return jsonify({"success": success})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start recording endpoint"""
    data = request.json
    review_text = data.get('review', '')
    success = emotion_system.start_recording(review_text)
    return jsonify({
        "success": success, 
        "session_id": emotion_system.session_id if success else None
    })

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Stop recording endpoint"""
    session_info = emotion_system.stop_recording()
    return jsonify({
        "success": session_info is not None,
        "session_info": session_info
    })

@app.route('/get_emotions', methods=['GET'])
def get_emotions():
    """Get current emotions data"""
    avg_emotions = emotion_system.compute_average_emotions()
    
    # Calculate session stats
    history_data = []
    if os.path.exists(emotion_system.history_data_file):
        try:
            with open(emotion_system.history_data_file, 'r') as f:
                history_data = json.load(f)
        except:
            history_data = []
    
    # Return emotions and stats
    return jsonify({
        "emotions": avg_emotions,
        "faces_detected": len(emotion_system.emotion_data["faces"]),
        "session_duration": time.time() - emotion_system.last_save_time if emotion_system.is_running else 0,
        "samples_collected": len(history_data),
        "is_recording": emotion_system.is_recording,
        "recorded_sessions": emotion_system.get_session_count()
    })

@app.route('/save_emotions', methods=['POST'])
def save_emotions():
    """Manual save of current emotion data"""
    result = emotion_system.save_emotion_data()
    return jsonify({"success": result is not None, "data": result})

@app.route('/get_sessions', methods=['GET'])
def get_sessions():
    """Get recorded sessions"""
    if os.path.exists(emotion_system.sessions_file):
        try:
            with open(emotion_system.sessions_file, 'r') as f:
                sessions = json.load(f)
            return jsonify({"success": True, "sessions": sessions})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    return jsonify({"success": True, "sessions": []})

# ====================================================================================================================================================================================

# Add a new endpoint to access transcripts
@app.route('/get_transcript/<session_id>', methods=['GET'])
def get_transcript(session_id):
    """Get transcript for a specific session"""
    transcript_path = os.path.join(emotion_system.recordings_dir, session_id, "transcripts", "transcript.txt")
    
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, 'r') as f:
                transcript = f.read()
            return jsonify({"success": True, "transcript": transcript})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    
    return jsonify({"success": False, "error": "Transcript not found"})

# ====================================================================================================================================================================================


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')