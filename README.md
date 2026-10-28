# Customer Feedback Sentiment Analysis

A Flask-based customer feedback analysis application that combines real-time face and emotion detection, audio recording, speech transcription, and sentiment analysis in a browser-based interface.

## Features

- Real-time webcam streaming with OpenCV.
- Face detection using OpenCV Haar cascades.
- Facial emotion analysis using DeepFace.
- Temporal emotion smoothing for more stable results.
- Average emotion tracking across detected faces.
- Video and audio recording for feedback sessions.
- Automatic session metadata and emotion-history storage.
- Speech-to-text transcription using Google Speech Recognition.
- Sentiment classification using CardiffNLP's RoBERTa sentiment model.
- Positive, neutral, and negative sentiment results with confidence scores.
- Flask endpoints for camera control, recording, emotions, sessions, transcripts, and sentiment.

## Tech Stack

- **Backend:** Python, Flask
- **Computer vision:** OpenCV, DeepFace
- **Machine learning:** Hugging Face Transformers, CardiffNLP RoBERTa
- **Audio processing:** PyAudio, Pydub, SpeechRecognition
- **Frontend:** HTML templates served by Flask
- **Storage:** JSON files, WAV audio recordings, MP4 video recordings

## Project Structure

```text
Customer-Feedback-Sentiment-/
├── app.py                 # Flask application and analysis pipeline
├── templates/             # HTML templates for the web interface
├── labels.csv             # Dataset or label data
├── req.txt                # Python dependencies
├── emotion_data/          # Generated emotion, session, and transcript data
└── recordings/            # Generated video, audio, and transcript files
```

The `emotion_data/` and `recordings/` directories are created automatically when the application runs.

## Requirements

- Python 3.9 or later
- A working webcam
- A working microphone
- FFmpeg available on your system for audio processing
- Git

Some dependencies, including TensorFlow, PyTorch, DeepFace, and Transformers, may require additional platform-specific setup. PyAudio can also require PortAudio development libraries on Linux and macOS.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Shivam200427/Customer-Feedback-Sentiment-.git
   cd Customer-Feedback-Sentiment-
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   **Windows:**

   ```bash
   .venv\\Scripts\\activate
   ```

   **macOS/Linux:**

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r req.txt
   ```

4. Make sure your webcam and microphone are connected and accessible.

## Running the Application

Start the Flask server with:

```bash
python app.py
```

Then open the local address displayed by Flask, typically:

```text
http://127.0.0.1:5000
```

On the first run, DeepFace and Hugging Face may download model files. This can take some time and requires an internet connection.

## Typical Workflow

1. Open the application in a browser.
2. Start the camera.
3. Allow camera and microphone permissions.
4. Observe detected faces and emotion labels in the live video feed.
5. Add optional customer feedback text.
6. Start and stop a recording session.
7. The application saves the video, audio, session information, emotion summary, and transcript.
8. Retrieve the latest sentiment analysis from the generated transcript.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Render the main application page |
| `GET` | `/video_feed` | Stream processed camera frames |
| `POST` | `/start_camera` | Start camera capture and emotion processing |
| `POST` | `/stop_camera` | Stop camera capture and save emotion data |
| `POST` | `/start_recording` | Start a video/audio feedback session |
| `POST` | `/stop_recording` | Stop recording and save session information |
| `GET` | `/get_emotions` | Return current emotions and session statistics |
| `POST` | `/save_emotions` | Manually save current emotion data |
| `GET` | `/get_sessions` | Return recorded session history |
| `GET` | `/get_transcript/<session_id>` | Return a session transcript |
| `GET` | `/get_sentiment` | Analyze the latest transcript sentiment |

## Generated Data

The application generates local files such as:

- `emotion_data/current_emotions.json`
- `emotion_data/emotion_history.json`
- `emotion_data/sessions.json`
- `emotion_data/latest_transcript.json`
- `emotion_data/sentiment_analysis.json`
- `recordings/<session_id>/recording.mp4`
- `recordings/<session_id>/audio.wav`
- `recordings/<session_id>/transcripts/transcript.txt`

These files can contain camera recordings, audio, transcripts, and sensitive customer information. Protect them appropriately and avoid committing generated or confidential data to source control.

## Important Notes

- The application accesses the local camera and microphone through Python, not directly through the browser.
- The current implementation runs Flask with `debug=True`; disable debug mode before deploying to production.
- Speech transcription uses Google's speech-recognition service and may require internet access.
- Emotion and sentiment predictions are probabilistic and may be inaccurate or affected by lighting, audio quality, language, and context.
- Do not use automated emotion or sentiment predictions as the sole basis for decisions about individuals.
- Review applicable privacy, consent, biometric-data, and data-retention requirements before using this project with real customer data.

## Troubleshooting

### Camera cannot be opened

- Confirm that no other application is using the webcam.
- Check the camera index in `cv2.VideoCapture(0)`.
- Verify that the operating system has granted camera permissions.

### Audio recording fails

- Confirm that a microphone is connected and selected as the default input device.
- Install the system audio dependencies required by PyAudio.
- Confirm that FFmpeg is installed and available on your `PATH`.

### Model downloads or imports fail

- Ensure the virtual environment is activated.
- Reinstall dependencies from `req.txt`.
- Check that your Python version is compatible with the installed TensorFlow and PyTorch versions.
- Run the application once with an active internet connection so model files can be downloaded.

## License

No license has been specified for this repository yet.
