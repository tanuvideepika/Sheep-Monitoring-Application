# # app.py

# import streamlit as st
# import os
# from processor import process_video

# st.title("🐑 Sheep Monitoring & Movement Analysis")
# st.markdown("Upload a video and get a tracked output with movement paths.")

# uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

# if uploaded_video is not None:
#     with open(os.path.join("uploads", uploaded_video.name), "wb") as f:
#         f.write(uploaded_video.getbuffer())

#     input_path = os.path.join("uploads", uploaded_video.name)
#     output_path = os.path.join("outputs", f"processed_{uploaded_video.name}")

#     st.info("Processing video. Please wait...")

#     progress_bar = st.progress(0)

#     def update_progress(p):
#         progress_bar.progress(min(p, 1.0))  # Clamp to 1.0 max

#     process_video(input_path, output_path, update_progress)

#     st.success("Processing complete! 🎉")
#     st.video(output_path)

#     with open(output_path, "rb") as f:
#         st.download_button("📥 Download Processed Video", f, file_name=f"tracked_{uploaded_video.name}")


import streamlit as st
import os
import uuid
import cv2
from processor import process_with_id_only, process_with_tracking

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

st.title("🐑 Sheep Monitoring Application")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_file:
    unique_filename = str(uuid.uuid4()) + ".mp4"
    input_video_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("Video uploaded successfully!")

    option = st.selectbox("Choose analysis mode", ["Sheep Detection", "Sheep Tracking"])

    if st.button("Process Video"):
        # st.warning("⚠️ Only the first 20 seconds of video will be processed.")

        with st.spinner("Processing video..."):
            if option == "Sheep Detection":
                output_path = process_with_id_only(input_video_path, OUTPUT_FOLDER)
            else:
                output_path = process_with_tracking(input_video_path, OUTPUT_FOLDER)

        st.success("Processing complete!")

        # st.video(output_path)
        with open(output_path, "rb") as video_file:
            st.video(video_file.read())

        with open(output_path, "rb") as f:
            st.download_button("Download Processed Video", f, file_name=os.path.basename(output_path), mime="video/mp4")
