# Face Detection + Emotion Mapping (Streamlit)

Professional local Streamlit application for:
- Live camera face detection
- Session-based face mapping (`Face-1`, `Face-2`, ...)
- Emotion labeling on video stream
- UI summary for emotions (`Happy`, `Sad`, `Tension`, `Ill`, `Afraid`, `Surprised`, `Neutral`)

## Important
- This app uses temporary per-session face IDs.
- It does **not** identify real-world person names/identity.
- Emotion estimation is rule-based (OpenCV features: face/eyes/smile), optimized for local stability and no heavy ML runtime dependency.

## Project Structure
- `app.py` : Streamlit application entry point
- `tracker.py` : Session face-ID tracker
- `requirements.txt` : Python dependencies

## Setup

```bash
cd /Users/Omkar_Python/facingdetection
python3 -m pip install -r requirements.txt
```

## Run

```bash
streamlit run /Users/Omkar_Python/facingdetection/app.py
```

If `streamlit` command is missing:

```bash
python3 -m streamlit run /Users/Omkar_Python/facingdetection/app.py
```

## Usage
1. Open sidebar.
2. Choose `Camera Index`.
3. Set `Min Emotion Confidence` and `Max FPS`.
4. Click `Start`.
5. Watch face boxes and emotion labels on stream.
6. Click `Stop` to stop camera.

## Troubleshooting
- If camera fails to open, close other apps using webcam and click Start again.
- If emotion labels fluctuate, raise `Min Emotion Confidence`.
- If app is slow, reduce `Max FPS`.
