# # =========================================================
# 📦 Import Libraries
# =========================================================
import os
import io
import requests
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
from google import genai  # ✅ Google Gemini API
from google.genai import types
from io import BytesIO
import speech_recognition as sr
from gtts import gTTS
import tempfile
import base64
import wave
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# =========================================================
# 🔑 Load API Keys
# =========================================================
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not groq_api_key:
    st.error("❌ No GROQ_API_KEY found in .env")
    st.stop()
if not hf_token:
    st.error("❌ No HF_TOKEN found in .env")
    st.stop()
if not gemini_api_key:
    st.error("❌ No GEMINI_API_KEY found in .env")
    st.stop()

# =========================================================
# 🤖 Initialize Clients
# =========================================================
hf_client = InferenceClient(provider="nebius", api_key=hf_token)
video_client = InferenceClient(provider="replicate", api_key=hf_token)
classification_client = InferenceClient(provider="hf-inference", api_key=hf_token)
gemini_client = genai.Client(api_key=gemini_api_key)

# =========================================================
# 🎨 Streamlit Page Setup
# =========================================================
st.set_page_config(page_title="AI Tools Suite", page_icon="💬")
st.title("💬 Chat + 🖼 Image + 🎥 Video + 🏷 Classification AI")

# =========================================================
# 📌 Sidebar Navigation
# =========================================================
with st.sidebar:
    st.header("Menu")

    app_mode = st.selectbox(
        "Choose a tool:",
        ("Chat AI Assistant", "Text-to-Image Generator", "Text-to-Video Generator", "Image Classification"),
        key="tool_selector",
    )

    st.header("Chat Settings")

    if app_mode == "Chat AI Assistant":
        model_options = ("llama3-8b-8192", "gemini-2.5-flash")
        default_model = "llama3-8b-8192"
    elif app_mode == "Text-to-Image Generator":
        model_options = ("gemini-2.5-flash",)
        default_model = "gemini-2.5-flash"
    elif app_mode == "Text-to-Video Generator":
        model_options = ("Wan-AI/Wan2.2-TI2V-5B",)
        default_model = "Wan-AI/Wan2.2-TI2V-5B"
    elif app_mode == "Image Classification":
        model_options = ("Falconsai/nsfw_image_detection",)
        default_model = "Falconsai/nsfw_image_detection"

    chat_model = st.selectbox("Chat Model:", model_options, index=model_options.index(default_model))

    if app_mode == "Chat AI Assistant":
        st.session_state["continuous_voice_chat"] = st.checkbox("🎤 Voice Chat Mode", value=False)
        transcription_engine = st.radio("Transcription Engine:", ("Google SpeechRecognition", "Groq Whisper API"))

    if st.button("Clear Chat / Data"):
        st.session_state.clear()

# =========================================================
# 💬 Chat AI Assistant
# =========================================================
if app_mode == "Chat AI Assistant":
    st.subheader("💬 Chat AI Assistant")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = None
    audio_file = "user_input.wav"

    # 🎤 Voice Input
    if st.session_state["continuous_voice_chat"]:
        st.info("🎙 Speak into your microphone...")

        def audio_callback(frame: av.AudioFrame):
            audio = frame.to_ndarray().flatten().astype("int16")
            with wave.open(audio_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio.tobytes())
            return frame

        webrtc_streamer(
            key="speech",
            mode=WebRtcMode.SENDONLY,
            audio_frame_callback=audio_callback,
            media_stream_constraints={"audio": True, "video": False},
        )

        if st.button("Transcribe Voice Input"):
            if transcription_engine == "Google SpeechRecognition":
                try:
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(audio_file) as source:
                        audio_data = recognizer.record(source)
                        user_input = recognizer.recognize_google(audio_data)
                        st.success(f"🗣 You said: {user_input}")
                except Exception as e:
                    st.error(f"Speech recognition failed: {e}")

            elif transcription_engine == "Groq Whisper API":
                try:
                    with open(audio_file, "rb") as f:
                        response = requests.post(
                            "https://api.groq.com/openai/v1/audio/transcriptions",
                            headers={"Authorization": f"Bearer {groq_api_key}"},
                            files={"file": f},
                            data={"model": "whisper-large-v3"},
                        )
                        user_input = response.json()["text"]
                        st.success(f"🗣 You said: {user_input}")
                except Exception as e:
                    st.error(f"Whisper API failed: {e}")

    # 📝 Text Input
    if not st.session_state["continuous_voice_chat"]:
        user_input = st.chat_input("Type your message...")

    # Handle Chat
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        messages = [{"role": "system", "content": "Ask me anything!"}]
        messages.extend(st.session_state["messages"])

        try:
            if chat_model == "gemini-2.5-flash":
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                config = types.GenerateContentConfig(tools=[grounding_tool])
                combined_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                chat_session = gemini_client.chats.create(model="gemini-2.5-flash", config=config)
                response = chat_session.send_message(combined_prompt)

                ai_reply = "\n".join(
                    [part.text for part in response.candidates[0].content.parts if part.text]
                )
            else:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
                    json={"model": chat_model, "messages": messages},
                )
                ai_reply = response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            ai_reply = f"Error: {e}"

        with st.chat_message("assistant"):
            st.markdown(ai_reply)

        # 🔊 Voice Output
        if st.session_state["continuous_voice_chat"]:
            try:
                tts = gTTS(text=ai_reply, lang="en")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tts.save(tmp_file.name)
                    audio_file_path = tmp_file.name
                audio_html = f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{base64.b64encode(open(audio_file_path, 'rb').read()).decode()}" type="audio/mp3">
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Voice output error: {e}")

        st.session_state["messages"].append({"role": "assistant", "content": ai_reply})

# =========================================================
# 🖼 Text-to-Image
# =========================================================
elif app_mode == "Text-to-Image Generator":
    st.subheader("🖼 Text-to-Image Generator")
    image_prompt = st.text_input("Enter image description:")
    if st.button("Generate Image") and image_prompt:
        with st.spinner("Generating image..."):
            try:
                full_prompt = f"{image_prompt}. Ultra-realistic, cinematic lighting, 4K resolution."
                gemini_image_response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash-preview-image-generation",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
                )
                img_data = None
                caption_text = None
                for part in gemini_image_response.candidates[0].content.parts:
                    if part.text: caption_text = part.text
                    elif part.inline_data: img_data = BytesIO(part.inline_data.data)
                if img_data:
                    full_res_img = Image.open(img_data)
                    st.image(full_res_img, caption=caption_text or image_prompt)
            except Exception as e:
                st.error(f"Image generation failed: {e}")

# =========================================================
# 🎥 Text-to-Video
# =========================================================
elif app_mode == "Text-to-Video Generator":
    st.subheader("🎥 Text-to-Video Generator")
    video_prompt = st.text_input("Enter video description:")
    if st.button("Generate Video") and video_prompt:
        with st.spinner("Generating video..."):
            try:
                video_data = video_client.text_to_video(prompt=video_prompt, model="Wan-AI/Wan2.2-TI2V-5B")
                video_bytes = io.BytesIO(video_data)
                st.video(video_bytes)
            except Exception as e:
                st.error(f"Video generation failed: {e}")

# =========================================================
# 🏷 Image Classification
# =========================================================
elif app_mode == "Image Classification":
    st.subheader("🏷 Image Classification")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image and st.button("Classify Image"):
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image")
        with st.spinner("Classifying..."):
            try:
                result = classification_client.image_classification(uploaded_image.read(), model="Falconsai/nsfw_image_detection")
                for item in result:
                    st.write(f"**{item['label']}**: {item['score']:.4f}")
            except Exception as e:
                st.error(f"Classification failed: {e}")
✅ requirements.txt
Add:

nginx
Copy code
streamlit
requests
python-dotenv
huggingface_hub
Pillow
google-genai
SpeechRecognition
gTTS
streamlit-webrtc
🎯 Summary
Uses browser mic (streamlit-webrtc).

Transcription toggle: Google SpeechRecognition OR Groq Whisper API.

Fixed WAV corruption error.

Chat, Image, Video, Classification all still work.

Do you also want me to add automatic transcription after recording (no button press) so it feels like real-time voice chat?













ChatGPT can make mistakes. Check important info. See Cookie Preferences.




# =========================================================
# 🖼 Text-to-Image Generator (Gemini - 4K Request)
# =========================================================
elif app_mode == "Text-to-Image Generator":
    st.subheader("🖼 Text-to-Image Generator")

    image_prompt = st.text_input(
        "Enter image description:",
        placeholder="e.g., A 3D-rendered pig with wings wearing a top hat flying over a futuristic green city",
        key="image_prompt",
    )

    # Trigger generation if:
    # 1) User clicks "Generate Image" OR
    # 2) User pressed Enter (detected by prompt change)
    generate_trigger = st.button("Generate Image") or (
        image_prompt and st.session_state.get("last_prompt") != image_prompt
    )

    if generate_trigger:
        if not image_prompt.strip():
            st.warning("Please enter a prompt for image generation.")
        else:
            st.session_state["last_prompt"] = image_prompt
            with st.spinner("Generating image..."):
                try:
                    # Ask Gemini for a native 4K image
                    full_prompt = f"{image_prompt}. Ultra-realistic, cinematic lighting, EXACT resolution 3840x2160."
                    gemini_image_response = gemini_client.models.generate_content(
                        model="gemini-2.0-flash-preview-image-generation",
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["TEXT", "IMAGE"]
                        ),
                    )

                    img_data = None
                    caption_text = None
                    for part in gemini_image_response.candidates[0].content.parts:
                        if part.text is not None:
                            caption_text = part.text
                        elif part.inline_data is not None:
                            img_data = BytesIO(part.inline_data.data)

                    if img_data:
                        full_res_img = Image.open(img_data)
                        st.session_state["generated_img"] = full_res_img
                        if caption_text:
                            st.session_state["generated_caption"] = caption_text
                    else:
                        st.error("No image data received from Gemini.")
                except Exception as e:
                    st.error(f"Image generation failed: {e}")

    # Show preview + download
    if "generated_img" in st.session_state:
        caption = st.session_state.get(
            "generated_caption", st.session_state.get("last_prompt", "Generated Image")
        )
        preview_img = st.session_state["generated_img"].copy()
        preview_img.thumbnail((800, 800))  # Keep small preview for UI speed
        st.image(preview_img, caption=caption)

        img_bytes = BytesIO()
        st.session_state["generated_img"].save(img_bytes, format="PNG")
        img_bytes.seek(0)
        st.download_button(
            label="📥 Download 4K Image",
            data=img_bytes,
            file_name="generated_image_4k.png",
            mime="image/png",
        )

# =========================================================
# 🎥 Text-to-Video Generator
# =========================================================
elif app_mode == "Text-to-Video Generator":
    st.subheader("🎥 Text-to-Video Generator")
    video_prompt = st.text_input(
        "Enter video description:",
        placeholder="e.g., A young man walking on the street",
    )
    if st.button("Generate Video"):
        if not video_prompt.strip():
            st.warning("Please enter a prompt for video generation.")
        else:
            with st.spinner("Generating video... (this may take 30-60 seconds)"):
                try:
                    video_data = video_client.text_to_video(
                        prompt=video_prompt, model="Wan-AI/Wan2.2-TI2V-5B"
                    )
                    video_bytes = io.BytesIO(video_data)
                    st.session_state["generated_video"] = video_bytes
                except Exception as e:
                    st.error(f"Video generation failed: {e}")
    if "generated_video" in st.session_state:
        st.video(st.session_state["generated_video"])
        st.download_button(
            label="📥 Download Video",
            data=st.session_state["generated_video"],
            file_name="generated_video.mp4",
            mime="video/mp4",
        )

# =========================================================
# 🏷 Image Classification
# =========================================================
elif app_mode == "Image Classification":
    st.subheader("🏷 Image Classification")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Classify Image"):
            with st.spinner("Classifying image..."):
                try:
                    result = classification_client.image_classification(
                        uploaded_image.read(), model="Falconsai/nsfw_image_detection"
                    )
                    st.success("✅ Classification Complete")
                    for item in result:
                        st.write(f"**{item['label']}**: {item['score']:.4f}")
                except Exception as e:
                    st.error(f"Image classification failed: {e}")

# =========================================================
# 📜 Footer
# =========================================================
st.markdown("---")
st.caption("Made with ❤️ by Anit Saha")


