import streamlit as st
import pandas as pd
import whisper_timestamped as whisper
import tempfile
import os

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

    st.write(f"Uploaded video saved as a temporary file: {os.path.basename(video_path)}")

    # Transcription Button
    if st.button("Transcribe Speech to Text"):
        with st.spinner("Transcribing... This may take a few minutes depending on the length of the video."):
            # Load the audio from the video file
            audio = whisper.load_audio(video_path)

            # Load the Whisper model
            model = whisper.load_model("medium", device="cpu")

            # Transcribe the audio
            result = whisper.transcribe(model, audio)

            # Extract word-level data from the segments
            word_texts = []
            word_starts = []
            word_ends = []
            word_confidences = []

            for segment in result['segments']:
                for word in segment['words']:
                    word_texts.append(word['text'])
                    word_starts.append(word['start'])
                    word_ends.append(word['end'])
                    word_confidences.append(word['confidence'])

            # Create DataFrames for segments and words
            segments_data = []
            for segment in result['segments']:
                segments_data.append({
                    'text': segment['text'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'confidence': segment['confidence']
                })

            words_data = {
                'text': word_texts,
                'start': word_starts,
                'end': word_ends,
                'confidence': word_confidences
            }

            segments_df = pd.DataFrame(segments_data)
            words_df = pd.DataFrame(words_data)

            # Display the transcription results directly in the app
            st.subheader("Transcription Segments")
            st.dataframe(segments_df)

            st.subheader("Word-Level Transcription")
            st.dataframe(words_df)

        st.success("Transcription completed!")
