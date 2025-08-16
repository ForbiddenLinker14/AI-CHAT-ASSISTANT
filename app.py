# =========================================================
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
from google.genai import types  # ✅ For Grounding tools
from io import BytesIO
import speech_recognition as sr
from gtts import gTTS
import tempfile
import base64

# 🎤 Browser audio recorder
from st_audiorec import st_audiorec

# =========================================================
# 🔑 Load API Keys from .env
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

    chat_model = st.selectbox(
        "Chat Model:", model_options, index=model_options.index(default_model), key="chat_model_selector"
    )

    if app_mode == "Chat AI Assistant":
        enable_code_execution = st.checkbox("⚡ Enable Code Execution", value=False)
        enable_voice_chat = st.checkbox("🎤 Voice Chat Mode", value=False)
        st.session_state["continuous_voice_chat"] = enable_voice_chat

    if st.button("Clear Chat / Data"):
        st.session_state.clear()

# =========================================================
# 💬 Chat AI Assistant (with Continuous Voice)
# =========================================================
if app_mode == "Chat AI Assistant":
    st.subheader("💬 Chat AI Assistant")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = None

    # 🎤 Voice chat mode
    if st.session_state.get("continuous_voice_chat", False):
        st.info("🎤 Record your voice below:")
        audio_bytes = st_audiorec()
        if audio_bytes is not None:
            recognizer = sr.Recognizer()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file.flush()
                with sr.AudioFile(tmp_file.name) as source:
                    audio_data = recognizer.record(source)
                    try:
                        user_input = recognizer.recognize_google(audio_data)
                        st.success(f"🗣 You said: {user_input}")
                    except sr.UnknownValueError:
                        st.warning("❌ Could not understand, please try again.")
                        st.rerun()
                    except sr.RequestError as e:
                        st.error(f"❌ Speech recognition error: {e}")
                        st.rerun()
    else:
        # 📝 Text input fallback
        user_input = st.chat_input("Type your message...")

    if user_input:
        # Save user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare messages
        messages = [{"role": "system", "content": "Ask me anything!"}]
        messages.extend(st.session_state["messages"])

        try:
            if chat_model == "gemini-2.5-flash":
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                tools_list = [grounding_tool]
                if st.session_state.get("enable_code_execution", False):
                    tools_list.append(types.Tool(code_execution=types.ToolCodeExecution()))

                config = types.GenerateContentConfig(tools=tools_list)
                combined_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

                chat_session = gemini_client.chats.create(model="gemini-2.5-flash", config=config)
                response = chat_session.send_message(combined_prompt)

                ai_reply_parts = []
                for part in response.candidates[0].content.parts:
                    if part.text:
                        ai_reply_parts.append(part.text)
                ai_reply = "\n".join(ai_reply_parts)
            else:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"},
                    json={"model": chat_model, "messages": messages},
                )
                ai_reply = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            ai_reply = f"Error: {e}"

        # Show AI reply
        with st.chat_message("assistant"):
            st.markdown(ai_reply)

        # 🔊 Voice output
        if st.session_state.get("continuous_voice_chat", False):
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

        # Save AI reply
        st.session_state["messages"].append({"role": "assistant", "content": ai_reply})

        # 🚀 Auto-ready for next voice input
        if st.session_state.get("continuous_voice_chat", False):
            st.rerun()

# =========================================================
# 🖼 Text-to-Image Generator
# =========================================================
elif app_mode == "Text-to-Image Generator":
    st.subheader("🖼 Text-to-Image Generator")
    image_prompt = st.text_input("Enter image description:")
    if st.button("Generate Image"):
        if not image_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Generating..."):
                try:
                    full_prompt = f"{image_prompt}. Ultra-realistic, cinematic lighting, resolution 3840x2160."
                    gemini_image_response = gemini_client.models.generate_content(
                        model="gemini-2.0-flash-preview-image-generation",
                        contents=full_prompt,
                        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
                    )
                    img_data = None
                    caption_text = None
                    for part in gemini_image_response.candidates[0].content.parts:
                        if part.text:
                            caption_text = part.text
                        elif part.inline_data:
                            img_data = BytesIO(part.inline_data.data)

                    if img_data:
                        img = Image.open(img_data)
                        st.image(img, caption=caption_text or image_prompt)
                        img_bytes = BytesIO()
                        img.save(img_bytes, format="PNG")
                        st.download_button("📥 Download", img_bytes.getvalue(), "generated.png", "image/png")
                except Exception as e:
                    st.error(f"Image generation failed: {e}")

# =========================================================
# 🎥 Text-to-Video Generator
# =========================================================
elif app_mode == "Text-to-Video Generator":
    st.subheader("🎥 Text-to-Video Generator")
    video_prompt = st.text_input("Enter video description:")
    if st.button("Generate Video"):
        with st.spinner("Generating..."):
            try:
                video_data = video_client.text_to_video(prompt=video_prompt, model="Wan-AI/Wan2.2-TI2V-5B")
                video_bytes = io.BytesIO(video_data)
                st.video(video_bytes)
                st.download_button("📥 Download Video", video_bytes, "video.mp4", "video/mp4")
            except Exception as e:
                st.error(f"Video generation failed: {e}")

# =========================================================
# 🏷 Image Classification
# =========================================================
elif app_mode == "Image Classification":
    st.subheader("🏷 Image Classification")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded", use_container_width=True)
        if st.button("Classify"):
            with st.spinner("Classifying..."):
                try:
                    result = classification_client.image_classification(
                        uploaded_image.read(), model="Falconsai/nsfw_image_detection"
                    )
                    st.success("✅ Done")
                    for item in result:
                        st.write(f"**{item['label']}**: {item['score']:.4f}")
                except Exception as e:
                    st.error(f"Classification failed: {e}")

# =========================================================
# 📜 Footer
# =========================================================
st.markdown("---")
st.caption("Made with ❤️ by Anit Saha")
