import base64
import html
import io
import os
import re
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests
import streamlit as st
from openai import OpenAI
from PIL import Image


DEFAULT_MODEL = "gpt-image-2"
STYLE_ANALYSIS_MODEL = "gpt-5.4"
STYLE_ANALYSIS_PROMPT = (
    "Describe the style, font and colorization of the text on this image. "
    "Don't add any other suggestions or questions into the output."
)
STYLE_INSTRUCTION_PREFIX = (
    "When replacing the text, consider this description on original text. "
    "Use it to accurately recreate the same style and color of text as in the original:"
)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def is_authorized(secret_key: str) -> bool:
    expected_key = os.getenv("APP_SECRET_KEY", get_secret("APP_SECRET_KEY"))
    return bool(expected_key) and secret_key.strip() == expected_key


def extract_youtube_video_id(url: str) -> Optional[str]:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().replace("www.", "")

    if host in {"youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        match = re.match(r"^/(shorts|embed|live)/([^/?#]+)", parsed.path)
        if match:
            return match.group(2)

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
        return video_id or None

    return None


def detect_link_platform(url: str) -> str:
    host = urlparse(url.strip()).netloc.lower().replace("www.", "")

    if host in {"youtube.com", "m.youtube.com", "youtu.be"}:
        return "youtube"

    if host in {"instagram.com", "m.instagram.com"} or host.endswith(".instagram.com"):
        return "instagram"

    return "unsupported"


def fetch_youtube_thumbnail(video_url: str) -> Tuple[Image.Image, str]:
    video_id = extract_youtube_video_id(video_url)
    if not video_id:
        raise ValueError("That does not look like a valid YouTube video link.")

    thumbnail_urls = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/sddefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    ]

    last_error = None
    for thumbnail_url in thumbnail_urls:
        try:
            response = requests.get(thumbnail_url, headers=REQUEST_HEADERS, timeout=15)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")

            # YouTube can return a tiny placeholder for unavailable max-res art.
            if image.width > 200 and image.height > 100:
                return image, thumbnail_url
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not download a thumbnail for this video. {last_error}")


def extract_meta_image_url(page_html: str) -> Optional[str]:
    patterns = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
        r'"display_url"\s*:\s*"([^"]+)"',
        r'"thumbnail_src"\s*:\s*"([^"]+)"',
    ]

    for pattern in patterns:
        match = re.search(pattern, page_html, flags=re.IGNORECASE)
        if match:
            image_url = html.unescape(match.group(1))
            return image_url.encode("utf-8").decode("unicode_escape")

    return None


def fetch_instagram_thumbnail(instagram_url: str) -> Tuple[Image.Image, str]:
    response = requests.get(instagram_url, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()

    image_url = extract_meta_image_url(response.text)
    if not image_url:
        raise RuntimeError(
            "Could not find a public preview image for this Instagram link. "
            "Some Instagram posts, Reels, private accounts, age-restricted posts, "
            "or login-gated pages do not expose a thumbnail to Streamlit."
        )

    image_response = requests.get(image_url, headers=REQUEST_HEADERS, timeout=20)
    image_response.raise_for_status()

    image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
    if image.width <= 100 or image.height <= 100:
        raise RuntimeError("Instagram returned a preview image that is too small to edit.")

    return image, image_url


def fetch_social_thumbnail(url: str) -> Tuple[Image.Image, str, str]:
    platform = detect_link_platform(url)

    if platform == "youtube":
        image, source = fetch_youtube_thumbnail(url)
        return image, source, "YouTube"

    if platform == "instagram":
        image, source = fetch_instagram_thumbnail(url)
        return image, source, "Instagram"

    raise ValueError("Paste a YouTube or Instagram link, or use the image upload option.")


def image_to_png_file(image: Image.Image) -> io.BytesIO:
    image_file = io.BytesIO()
    image.save(image_file, format="PNG")
    image_file.seek(0)
    image_file.name = "input_image.png"
    return image_file


def analyze_text_style_with_openai(
    openai_api_key: str,
    input_image: Image.Image,
) -> str:
    client = OpenAI(api_key=openai_api_key)

    image_file = image_to_png_file(input_image)
    image_base64 = base64.b64encode(image_file.getvalue()).decode("utf-8")

    response = client.responses.create(
        model=STYLE_ANALYSIS_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": STYLE_ANALYSIS_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_base64}",
                        "detail": "high",
                    },
                ],
            }
        ],
    )

    description = response.output_text.strip()
    if not description:
        raise RuntimeError("GPT-5.4 returned an empty text-style description.")

    return f"{STYLE_INSTRUCTION_PREFIX}\n\n{description}"


def edit_image_with_openai(
    openai_api_key: str,
    model_name: str,
    input_image: Image.Image,
    edit_mode: str,
    replacement_text: str,
    extra_instruction: str,
    output_size: str,
) -> bytes:
    if edit_mode == "Remove text":
        prompt = f"""
Edit the provided image.

Remove all visible text from the image. Keep everything else intact, including
the original layout, composition, colors, lighting, background, people, objects,
and overall style. Fill the removed text areas naturally so the image looks like
the text was never there.

Additional instructions from the user:
{extra_instruction or "None"}
""".strip()
    else:
        prompt = f"""
Edit the provided image.

Replace the visible text in the image with this exact text:
{replacement_text}

Keep the original layout, composition, colors, lighting, background, people,
objects, and overall thumbnail style as close as possible. Only change text
that should be replaced. Make the new text readable and naturally integrated.

Additional instructions from the user:
{extra_instruction or "None"}
""".strip()

    client = OpenAI(api_key=openai_api_key)

    result = client.images.edit(
        model=model_name,
        image=image_to_png_file(input_image),
        prompt=prompt,
        size=output_size,
    )

    if not result.data:
        raise RuntimeError("OpenAI returned a response, but no image was included.")

    image_base64 = result.data[0].b64_json
    if not image_base64:
        raise RuntimeError("OpenAI did not return image bytes. Try again or adjust the prompt.")

    return base64.b64decode(image_base64)


st.set_page_config(page_title="OpenAI Thumbnail Text Replacer", page_icon="image")

st.title("OpenAI Thumbnail Text Replacer")
st.caption("Use a YouTube or Instagram thumbnail, or upload an image, then replace its text with OpenAI Image.")

with st.sidebar:
    st.header("Access")
    app_secret_key = st.text_input(
        "Secret Key",
        type="password",
        help="Share this app key only with people you want to let use your OpenAI quota.",
    )

    st.header("OpenAI settings")
    model_name = st.text_input("Image model", value=DEFAULT_MODEL)
    output_size = st.selectbox(
        "Output size",
        ["auto", "1024x1024", "1536x1024", "1024x1536"],
        index=0,
    )

openai_api_key = os.getenv("OPENAI_API_KEY", get_secret("OPENAI_API_KEY"))
configured_secret = os.getenv("APP_SECRET_KEY", get_secret("APP_SECRET_KEY"))
authorized = is_authorized(app_secret_key)

if not configured_secret:
    st.error("The app owner has not configured APP_SECRET_KEY in Streamlit secrets yet.")
    st.stop()

if not openai_api_key:
    st.error("The app owner has not configured OPENAI_API_KEY in Streamlit secrets yet.")
    st.stop()

if not authorized:
    st.info("Enter the shared Secret Key to use this app.")
    st.stop()

input_mode = st.radio(
    "Choose input",
    ["Social link", "Upload image"],
    horizontal=True,
)

source_image = None
source_label = None

if input_mode == "Social link":
    social_url = st.text_input(
        "YouTube or Instagram URL",
        placeholder="https://www.youtube.com/watch?v=... or https://www.instagram.com/p/...",
    )
    if social_url:
        try:
            source_image, source_label, platform_name = fetch_social_thumbnail(social_url)
            st.success(f"{platform_name} thumbnail loaded.")
        except Exception as exc:
            st.error(str(exc))
else:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file:
        source_image = Image.open(uploaded_file).convert("RGB")
        source_label = uploaded_file.name

if source_image:
    st.subheader("Input image")
    st.image(source_image, caption=source_label, use_container_width=True)

analyze_style = st.button(
    "Analyze text style",
    disabled=source_image is None,
    help="Use GPT-5.4 to describe the original text styling.",
)

if analyze_style:
    with st.spinner("Analyzing the original text style..."):
        try:
            st.session_state["extra_instruction"] = analyze_text_style_with_openai(
                openai_api_key,
                source_image,
            )
            st.success("Text-style description added to Optional extra instructions.")
        except Exception as exc:
            st.error(str(exc))

edit_mode = st.radio(
    "Output",
    ["Replace text", "Remove text"],
    horizontal=True,
)

replacement_text = ""
extra_instruction = ""

if edit_mode == "Replace text":
    replacement_text = st.text_area(
        "Replacement text",
        placeholder="Type the exact text you want OpenAI to place on the image...",
        height=100,
    )

extra_instruction_placeholder = (
    "Example: Use bold white uppercase text with a black outline."
    if edit_mode == "Replace text"
    else "Example: Keep Sadhguru's signature in the bottom left corner."
)

extra_instruction = st.text_area(
    "Optional extra instructions",
    placeholder=extra_instruction_placeholder,
    height=80,
    key="extra_instruction",
)

generate = st.button(edit_mode, type="primary", disabled=not source_image)

if generate:
    if edit_mode == "Replace text" and not replacement_text.strip():
        st.error("Add the replacement text first.")
    else:
        with st.spinner("Sending image to OpenAI..."):
            try:
                output_bytes = edit_image_with_openai(
                    openai_api_key=openai_api_key,
                    model_name=model_name.strip(),
                    input_image=source_image,
                    edit_mode=edit_mode,
                    replacement_text=replacement_text.strip(),
                    extra_instruction=extra_instruction.strip(),
                    output_size=output_size,
                )

                st.subheader("Output image")
                st.image(output_bytes, use_container_width=True)

                st.download_button(
                    "Download edited image",
                    data=output_bytes,
                    file_name="openai_edited_thumbnail.png",
                    mime="image/png",
                )
            except Exception as exc:
                st.error(str(exc))
