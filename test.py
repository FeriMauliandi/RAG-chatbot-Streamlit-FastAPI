import base64
from IPython.display import Image, display
from langchain.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-image")

response = model.invoke("Generate a photorealistic image of a cuddly cat wearing a hat.")

def _get_image_base64(response: AIMessage) -> None:
    image_block = next(
        block
        for block in response.content
        if isinstance(block, dict) and block.get("image_url")
    )
    return image_block["image_url"].get("url").split(",")[-1]

image_base64 = _get_image_base64(response)
display(Image(data=base64.b64decode(image_base64), width=300))