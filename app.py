import streamlit as st
import whisper_timestamped as whisper
import tempfile

# App Header
st.title("Speech-to-Text Transcription with Whisper")

# Upload video file
video_file = st.file_uploader("Upload your video file", type=["mp4", "mov", "avi"])

# Process the uploaded video and transcribe speech to text
if video_file is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        video_path = tmp_file.name

    # Button to start transcription
    if st.button("Transcribe Speech to Text"):
        with st.spinner("Transcribing... Please wait."):
            # Load the audio from the video file
            audio = whisper.load_audio(video_path)

            # Load Whisper model
            model = whisper.load_model("medium", device="cpu")

            # Transcribe the audio
            result = whisper.transcribe(model, audio)

            # Extract and display segments
            segments_data = []
            for segment in result['segments']:
                segments_data.append({
                    'text': segment['text'],
                    'start': segment['start'],
                    'end': segment['end']
                })

            # Display results
            st.subheader("Transcription Result (Segment Level)")
            for segment in segments_data:
                st.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")

        st.success("Transcription completed!")
