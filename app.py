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
from io import BytesIO  # ✅ Needed for image byte handling
from PIL import Image  # ✅ Pillow for image display & saving
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
import time
import requests
import streamlit.components.v1 as components

# =========================================================
# 🔑 Load API Keys from .env
# =========================================================
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
gemini_api_key = os.getenv("GEMINI_API_KEY")  # ✅ Gemini API Key

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
hf_client = InferenceClient(provider="nebius", api_key=hf_token)  # Image Generation
video_client = InferenceClient(
    provider="replicate", api_key=hf_token
)  # Video Generation
classification_client = InferenceClient(
    provider="hf-inference", api_key=hf_token
)  # Image Classification
gemini_client = genai.Client(api_key=gemini_api_key)  # ✅ Gemini Client

# =========================================================
# 🎨 Streamlit Page Setup & Sky Blue Theme
# =========================================================

# =========================================================
# 📜 Inject Popunder Ad Script
# =========================================================
components.html(
    """
    <script type="text/javascript" src="https://pl27448593.profitableratecpm.com/a5/35/0f/a5350f98f88d27271cdd55daad15e888.js"></script>
    """,
    height=0,   # keeps it invisible
    width=0
)

st.set_page_config(page_title="AI Tools Suite", page_icon="💬")
st.title("💬 Chat + 🖼 Image + 🎥 Video + 🏷 Classification AI")

# ===================== CUSTOM THEME =====================
st.markdown(
    """
<style>
.stApp {background: linear-gradient(135deg, #000428, #004e92, #00aaff); color: #FAFAFA !important;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #000428, #004e92, #00aaff); color: #FAFAFA !important;}
[data-testid="stSidebar"] * {color: #FAFAFA !important;}
html, body, [class*="css"] {color: #FAFAFA !important;}
.stChatMessage {background: linear-gradient(135deg, rgba(0,4,40,0.4), rgba(0,78,146,0.4)); border: 1px solid #80dfff; border-radius: 10px; padding: 10px;}
.stButton>button {background: linear-gradient(90deg, #000428, #004e92, #00aaff); color: #FFFFFF !important; border: none; border-radius: 8px; font-weight: bold;}
.stButton>button:hover {background: linear-gradient(90deg, #00aaff, #004e92, #000428);}
.stTextInput>div>div>input, .stTextArea textarea {background: linear-gradient(to right, rgba(0,0,0,0.2), rgba(0,0,0,0.1)); color: #FAFAFA !important; border-radius: 10px; border: 1px solid #80dfff; padding: 8px;}
.stTextInput>div>div>input::placeholder, .stTextArea textarea::placeholder {color: #B0C4DE !important;}
.stTextInput, .stTextArea {background: linear-gradient(135deg, #000428, #004e92, #00aaff); padding: 5px; border-radius: 12px;}
.stPromptBox {background: linear-gradient(135deg, rgba(0,4,40,0.3), rgba(0,78,146,0.3)); padding: 10px; border-radius: 10px; border: 1px solid #80dfff;}
hr {border-top: 1px solid #80dfff;}
header[data-testid="stHeader"] {background: linear-gradient(90deg, #000428, #004e92, #00aaff) !important; color: #FAFAFA !important;}
footer {visibility: hidden;}
[data-testid="stBottomBlockContainer"] {background: linear-gradient(135deg, #000428, #004e92, #00aaff) !important; max-width: 100% !important; padding: 0.5rem 1rem 0.5rem !important;}
[data-testid="stChatInput"] > div {max-width: 700px; width: 100%; margin-left: auto; margin-right: auto;}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 📌 Sidebar Navigation (Dropdown & Chat Settings)
# =========================================================
with st.sidebar:
    st.header("Menu")

    app_mode = st.selectbox(
        "Choose a tool:",
        (
            "Chat AI Assistant",
            "Text-to-Image Generator",
            "Text-to-Video Generator",
            "Image Classification",
        ),
        key="tool_selector",
    )

    st.header("Chat Settings")

    if app_mode == "Chat AI Assistant":
        model_options = (
            "llama3-8b-8192",
            "gemini-2.5-flash",
            "other-model-1",
            "other-model-2",
        )
        default_model = "llama3-8b-8192"

    elif app_mode == "Text-to-Image Generator":
        model_options = ("gemini-2.5-flash", "other-model-1", "other-model-2")
        default_model = "gemini-2.5-flash"

    elif app_mode == "Text-to-Video Generator":
        model_options = ("Wan-AI/Wan2.2-TI2V-5B", "other-model-1", "other-model-2")
        default_model = "Wan-AI/Wan2.2-TI2V-5B"

    elif app_mode == "Image Classification":
        model_options = (
            "Falconsai/nsfw_image_detection",
            "other-model-1",
            "other-model-2",
        )
        default_model = "Falconsai/nsfw_image_detection"

    chat_model = st.selectbox(
        "Chat Model:",
        model_options,
        index=model_options.index(default_model),
        key="chat_model_selector",
    )

    st.markdown(
        '<div style="background: linear-gradient(135deg, #000428, #004e92, #00aaff); '
        'padding: 10px; border-radius: 10px; color: #FAFAFA;">'
        "<strong>System Prompt:</strong> Ask me anything!</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear Chat / Data"):
        # Clear all tools' session state
        st.session_state.clear()

# =========================================================
# 📜 Inline Banner Ad (728x90)
# =========================================================
components.html(
    """
    <script type="text/javascript">
        atOptions = {
            'key' : 'ce19aabaaaceb5654105a6dfac8719ec',
            'format' : 'iframe',
            'height' : 90,
            'width' : 728,
            'params' : {}
        };
    </script>
    <script type="text/javascript" src="//www.highperformanceformat.com/ce19aabaaaceb5654105a6dfac8719ec/invoke.js"></script>
    """,
    height=100,  # enough to fit 90px height
    width=740,  # enough to fit 728px width
    scrolling=False,
)

# =========================================================
# 📜 Second Banner Ad (468x60)
# =========================================================
components.html(
    """
    <script type="text/javascript">
        atOptions = {
            'key' : '68d5886f8f1b26a3bfd5b9f21f29b548',
            'format' : 'iframe',
            'height' : 60,
            'width' : 468,
            'params' : {}
        };
    </script>
    <script type="text/javascript" src="//www.highperformanceformat.com/68d5886f8f1b26a3bfd5b9f21f29b548/invoke.js"></script>
    """,
    height=70,  # slightly bigger to fit 60px
    width=480,  # slightly bigger to fit 468px
    scrolling=False,
)

# =========================================================
# 💬 Chat AI Assistant (Continuous Voice + Autoplay Update)
# =========================================================
import base64  # Make sure this is at the top of your script

if app_mode == "Chat AI Assistant":
    st.subheader("💬 Chat AI Assistant")

    # Session state setup
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "continuous_voice_chat" not in st.session_state:
        st.session_state["continuous_voice_chat"] = False

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    enable_code_execution = False
    enable_voice_chat = False

    # Model-specific options
    if chat_model == "gemini-2.5-flash":
        enable_code_execution = st.checkbox(
            "⚡ Enable Code Execution for this query", value=False
        )
        enable_voice_chat = st.checkbox("🎤 Continuous Voice Chat Mode", value=False)
        st.session_state["continuous_voice_chat"] = enable_voice_chat

    user_input = None

    # 🎤 Continuous voice input loop
    if st.session_state["continuous_voice_chat"]:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                # Optional beep sound before listening
                import sys

                sys.stdout.write("\a")
                sys.stdout.flush()

                st.info("🎙 Listening... Speak now.")
                audio_data = recognizer.listen(source)
                user_input = recognizer.recognize_google(audio_data)
                st.success(f"🗣 You said: {user_input}")

            except sr.UnknownValueError:
                st.warning("❌ Could not understand, please try again.")
                st.rerun()  # Retry listening immediately

            except sr.RequestError as e:
                st.error(f"❌ Speech recognition error: {e}")
                st.rerun()  # Retry listening immediately

    # 📝 Text input fallback when voice mode is off
    if not st.session_state["continuous_voice_chat"]:
        user_input = st.chat_input("Type your message...")

    if user_input:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare messages for model
        messages = [{"role": "system", "content": "Ask me anything!"}]
        messages.extend(st.session_state["messages"])

        try:
            if chat_model == "gemini-2.5-flash":
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                tools_list = [grounding_tool]
                if enable_code_execution:
                    tools_list.append(
                        types.Tool(code_execution=types.ToolCodeExecution())
                    )

                config = types.GenerateContentConfig(tools=tools_list)
                combined_prompt = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in messages]
                )

                chat_session = gemini_client.chats.create(
                    model="gemini-2.5-flash", config=config
                )
                response = chat_session.send_message(combined_prompt)

                ai_reply_parts = []
                for part in response.candidates[0].content.parts:
                    if part.text and not part.text.strip().lower().startswith(
                        "result:"
                    ):
                        ai_reply_parts.append(part.text)
                    if part.executable_code and enable_code_execution:
                        st.code(part.executable_code.code, language="python")

                ai_reply = "\n".join(ai_reply_parts)

            else:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": chat_model, "messages": messages},
                )
                ai_reply = response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            ai_reply = f"Error: {e}"

        # Display AI reply
        with st.chat_message("assistant"):
            st.markdown(ai_reply)

        # 🔊 Voice output in continuous mode with autoplay
        if st.session_state["continuous_voice_chat"]:
            try:
                tts = gTTS(text=ai_reply, lang="en")
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp3"
                ) as tmp_file:
                    tts.save(tmp_file.name)
                    audio_file_path = tmp_file.name

                # Autoplay audio in browser using HTML
                audio_html = f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{base64.b64encode(open(audio_file_path, 'rb').read()).decode()}" type="audio/mp3">
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Voice output error: {e}")

        # Save AI message
        st.session_state["messages"].append({"role": "assistant", "content": ai_reply})

        # 🚀 Automatically listen for next voice input
        if st.session_state["continuous_voice_chat"]:
            st.rerun()


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

# =========================================================
# 📜 Footer + Adsterra Ads
# =========================================================

# ✅ Direct Link Ad (iframe banner style)
st.markdown(
    """
    <div style="text-align:center; margin-top:20px;">
        <iframe src="https://www.profitableratecpm.com/anj0v0tyj?key=810c2a66cc9787bb094ec1fba2ea32fe" 
                width="100%" height="90" frameborder="0" scrolling="no">
        </iframe>
    </div>
    """,
    unsafe_allow_html=True,
)

# ✅ Native Banner Ad (in place of scrollbar ad)
components.html(
    """
    <script async="async" data-cfasync="false" 
        src="//pl27450014.profitableratecpm.com/d4cd9731abd5eab614191a36354ee7bc/invoke.js">
    </script>
    <div id="container-d4cd9731abd5eab614191a36354ee7bc"></div>
    """,
    height=300,  # Adjust depending on ad size
    width=800,   # Adjust to fit your layout
    scrolling=False,
)

# =========================================================
# 📜 Third Banner Ad (468x60)
# =========================================================
components.html(
    """
    <script type="text/javascript">
	atOptions = {
		'key' : '72e674d3fcf49ab599755d0eec4f9191',
		'format' : 'iframe',
		'height' : 250,
		'width' : 300,
		'params' : {}
	};
</script>
<script type="text/javascript" src="//www.highperformanceformat.com/72e674d3fcf49ab599755d0eec4f9191/invoke.js"></script>
    """,
    height=260,   # a bit more than 250px
    width=310,  # slightly bigger to fit 468px
    scrolling=False,
)

# # ✅ Scrollbar Ads (Injected JS safely with components.html)
# components.html(
#     """
#     <script type='text/javascript'
#         src='//pl27448332.profitableratecpm.com/79/ff/a4/79ffa4ff1e9a9e5d88238e900ccc5a23.js'>
#     </script>
#     """,
#     height=0,  # keeps it hidden, script still runs
# )
# # ✅ Extra Direct Link Ad (iframe banner style)
# st.markdown(
#     """
#     <div style="text-align:center; margin-top:20px;">
#         <iframe src="https://www.profitableratecpm.com/nejb9w4xw?key=474171a0992eb5419784974a369a7871"
#                 width="100%" height="90" frameborder="0" scrolling="no">
#         </iframe>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )
# # =========================================================
# # 📜 Inject Popunder Ad (into <head>)
# # =========================================================
# components.html(
#     """
#     <script type='text/javascript'>
#         var script = document.createElement("script");
#         script.src = "//pl27448593.profitableratecpm.com/a5/35/0f/a5350f98f88d27271cdd55daad15e888.js";
#         document.head.appendChild(script);
#     </script>
#     """,
#     height=0,  # keep container invisible
# )
