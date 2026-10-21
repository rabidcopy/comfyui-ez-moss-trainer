import logging

import aiohttp
from PIL import Image

from comfy_api.latest import io

from .vlm import chat, encode_image, resolve_model

OUTPUT_RULES = ("Write for MOSS-Image, a text-to-image model with an LLM text encoder that follows long, precise natural language. "
                "Output only the final prompt: one paragraph of plain natural language, no title, preamble, lists, markdown, tags or commentary. "
                "Quote any text that should appear in the image exactly, and say where it goes.")

PRESETS = {
    "enhance prompt": "You turn short image ideas into detailed text-to-image prompts. Keep every element and the intent of the user's prompt and never contradict it. "
                      "Add concrete, fitting details: the subject's appearance, clothing, pose and expression, the setting, composition and camera angle, "
                      "lighting, colors, mood, and the medium or art style. If reference images are given, use them as the visual reference for what the user describes.",
    "character from references": "The reference images show one character. First work out the character's permanent visual identity precisely: face, hair style and color, "
                                 "eye color, skin, body type, species features, and their signature outfit, accessories and colors. "
                                 "Then write a prompt that places this exact character into the user's scene, describing their identity in full so the character is recognizable "
                                 "without the images. With no user prompt, write a full-body portrait of the character on a simple background.",
    "style from references": "The reference images share an art style. Work out that style precisely: medium and technique, line work, shading and rendering, color palette, "
                             "lighting, texture, level of detail, composition tendencies, mood and era. Then write the user's prompt as an image in that style, "
                             "with the style described in full so it carries without the images. Do not copy the subjects of the reference images unless the user asks for them.",
    "character + style from references": "The reference images show a character and an art style, possibly in the same images. Work out the character's permanent visual identity "
                                         "(face, hair, eyes, skin, body, species features, signature outfit and colors) and the art style (medium, line work, shading, palette, "
                                         "lighting, texture, mood). Then write the user's scene with that character rendered in that style, describing both in full.",
    "recreate image": "Write a prompt that would recreate the reference image as closely as possible: subjects and their appearance, pose, clothing, objects, setting, "
                      "composition and framing, camera angle, lighting, colors, art style and any visible text quoted exactly. "
                      "If the user gives a prompt, it lists changes: apply them and keep everything else the same.",
}


def tensor_to_pil(image):
    return Image.fromarray((image[..., :3].clamp(0, 1) * 255).round().byte().cpu().numpy())


class EZMOSSPromptWriter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EZMOSSPromptWriter",
            display_name="EZ MOSS Prompt Writer (OpenAI-compatible)",
            category="model/conditioning/moss_image",
            description="Writes or enhances a MOSS-Image prompt with a vision LLM behind an OpenAI-compatible API, optionally using reference images "
                        "for a character, an art style, or an image to recreate.",
            inputs=[
                io.String.Input("prompt", multiline=True, default="", tooltip="Your prompt or idea. Can be empty for the reference image presets."),
                io.Autogrow.Input("images", template=io.Autogrow.TemplatePrefix(io.Image.Input("image", optional=True), prefix="image", min=0, max=16), optional=True),
                io.String.Input("base_url", default="http://127.0.0.1:1234/v1",
                                tooltip="OpenAI-compatible API root. LM Studio: http://127.0.0.1:1234/v1, Ollama: http://127.0.0.1:11434/v1, llama.cpp: http://127.0.0.1:8080/v1"),
                io.String.Input("model", default="", tooltip="Model id. Use the 'list models' button to pick one, empty uses the first model the server lists. Needs a vision model when images are connected."),
                io.String.Input("api_key", default="", tooltip="Leave empty to use the OPENAI_API_KEY environment variable. A key typed here is saved in the workflow and image metadata."),
                io.Combo.Input("preset", options=list(PRESETS), default="enhance prompt"),
                io.String.Input("system_prompt", multiline=True, default="", tooltip="Replaces the preset's instructions when not empty."),
                io.String.Input("extra_instructions", multiline=True, default="",
                                tooltip="Appended to the active system prompt (the preset, or system_prompt when set), e.g. \"Call the character 'kiri'. Keep it under 150 words.\""),
                io.Int.Input("max_image_size", default=1024, min=256, max=4096, step=64, tooltip="Longest side of each image sent to the model."),
                io.Int.Input("max_tokens", default=2048, min=16, max=32768, tooltip="Reply token limit. Thinking models spend tokens reasoning before the answer, so keep this high."),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05),
                io.Int.Input("timeout", default=180, min=5, max=3600, tooltip="Seconds per request."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                             tooltip="Change it to get a new prompt for the same inputs. Sent to the API as its seed, which most servers honor."),
            ],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    async def execute(cls, prompt, base_url, model, api_key, preset, system_prompt, extra_instructions, max_image_size, max_tokens, temperature, timeout, seed,
                      images: io.Autogrow.Type = None) -> io.NodeOutput:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset}")
        system = (system_prompt.strip() or PRESETS[preset] + "\n\n" + OUTPUT_RULES)
        if extra_instructions.strip():
            system += "\n\n" + extra_instructions.strip()

        frames = []
        for name in sorted(images or {}, key=lambda name: int(name.removeprefix("image"))):
            if images[name] is not None:
                frames += [tensor_to_pil(frame) for frame in images[name]]
        content = [{"type": "text", "text": f"User prompt: {prompt.strip() or '(none)'}" + (f"\n\n{len(frames)} reference image(s) follow." if frames else "")}]
        for i, frame in enumerate(frames, 1):
            content += [{"type": "text", "text": f"Image {i}:"}, {"type": "image_url", "image_url": {"url": encode_image(frame, max_image_size)}}]

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            model = await resolve_model(session, base_url, model, api_key)
            logging.info(f"[EZ Prompt Writer] Using {model} at {base_url}, preset '{preset if not system_prompt.strip() else 'custom'}'"
                         + (" + extra instructions" if extra_instructions.strip() else "") + f", {len(frames)} image(s)")
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed % 2 ** 31,
            }
            text = await chat(session, base_url, api_key, payload, "prompt", "[EZ Prompt Writer]")
        logging.info(f"[EZ Prompt Writer] {text}")
        return io.NodeOutput(text)
