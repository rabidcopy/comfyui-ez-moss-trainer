import logging

import aiohttp
from PIL import Image

from comfy_api.latest import io

from .vlm import chat, encode_image, resolve_model, warn_inline_key

OUTPUT_RULES = ("Write for MOSS-Image, a text-to-image model with an LLM text encoder that follows long, precise natural language. "
                "Output only the final prompt: one paragraph of plain natural language, no title, preamble, lists, markdown, tags or commentary. "
                "Quote any text that should appear in the image exactly, and say where it goes.")

H3_OUTPUT_RULES = ("Output only the final prompt: the labeled sections from the system prompt written as 'name:' on its own line followed by its text, sections separated by "
                   "blank lines. With connected pictures that is exactly subject_definitions, summary, retention_analysis and detailed_description in that order; with none, "
                   "only detailed_description. "
                   "Plain text: no markdown, bullets, headings or commentary. Keep every <Picture N> and <Subject N> label consistent across sections. "
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
    "minimax h3 (single image)": (
        "You rewrite the user's request into a MiniMax-H3-style prompt for a model that generates one still image from text and reference pictures. "
        "Keep H3's reference-label format: every connected picture is labeled <Picture 1>, <Picture 2>, ... in the order given, and you use those exact labels, "
        "consistently, without inventing labels for content the user does not supply.\n\n"
        "Decide each picture's role from the user's prompt:\n"
        "- Composition anchor or keyframe, but ONLY when the user's prompt explicitly ties the target to that picture's own composition. "
        "Examples: 'add a hat to the character in the second image', 'put the item from the first image on the clipboard held by the character in the second image', "
        "'replace the background of the first image'. There the description is built on that picture: what it shows, with the user's change applied.\n"
        "- Character reference: the picture only defines how someone or something looks: identity, face, hair, eyes, skin, body, species features, signature outfit and colors.\n"
        "- Style reference: the picture only defines the art style: medium and technique, line work, shading and rendering, color palette, lighting, texture, level of detail, era, mood.\n"
        "One picture may fill several roles, and one subject may come from several pictures. When the user's prompt does not explicitly anchor the target to a picture's "
        "composition, treat the picture as a character or style reference only and give the target image a fresh composition, pose, camera angle and setting; "
        "never recreate a reference's framing, pose or scene layout just because it shows the character or style.\n\n"
        "Output exactly four sections, in this order:\n\n"
        "subject_definitions:\n"
        "One line per piece of referenced content actually used in the target image:\n"
        "- <Subject N> is the ...: reusable content, such as the character in <Picture 2> with her defining features, the watercolor style of <Picture 3>, or the prop "
        "in <Picture 1>. Cite each source picture inside the subject's definition. Combine sources when several pictures define one subject ('<Subject 1> is the woman "
        "whose appearance comes from <Picture 1> and whose style comes from <Picture 2>').\n"
        "- <Picture N> gets its own line ONLY when that picture is a composition anchor, keyframe or edit target, e.g. '<Picture 2> is the base image of the edit, showing "
        "a woman seated beside a cafe window.' Otherwise never give a picture a standalone entry; it lives inside its <Subject N> definition.\n\n"
        "summary:\n"
        "One short paragraph. Start with a square-bracketed task-type prefix describing what each picture contributes, combined with ' + ' when several apply:\n"
        "- reference generation: a picture guides the character, scene or style only, without anchoring composition.\n"
        "- keyframe completion: a picture's full composition is preserved as the target image (or as one part of it).\n"
        "- image edit: the target is a picture with the user's change applied.\n"
        "Then describe the target image and its main reference relationships, using only labels defined above.\n\n"
        "retention_analysis:\n"
        "One line per defined label, describing how its referenced content is preserved in the target image, using these fixed markers:\n"
        "- fully_preserved: the defined role is preserved fully.\n"
        "- partially_preserved: the content is still used but some defined characteristics are changed or only partly retained.\n"
        "- attribute_transfer: its characteristics are transferred to a different subject.\n"
        "- weak_reference: only broad similarity in style, category or atmosphere is retained.\n"
        "Format: '<Subject 1> (target image): fully_preserved - the long dark hair, blue cardigan and silver necklace are retained.' When a picture is a composition anchor "
        "or edit target, use '<Picture 2> (composition anchor): partially_preserved - its framing and subject placement are retained, with the user's hat change applied.' "
        "Do not treat details newly added at the user's request as losses of fidelity; state them alongside what is preserved.\n\n"
        "detailed_description:\n"
        "The main body. Start with one or two sentences establishing the overall style of the target image (medium, rendering, lighting, palette), then describe the complete "
        "single image in one continuous passage: composition and framing (shot type, camera angle), subject appearance, pose, expression and clothing, key props and their "
        "placement, the environment, background details and lighting, colors and texture, in concrete visual language rather than abstract words like 'cinematic' or 'beautiful'. "
        "Typically 250-400 words. Insert <Subject N> and <Picture N> labels where their characteristics actually appear. When a picture is a composition anchor, say which part "
        "of the image follows it ('the framing and composition of the cafe scene follow <Picture 2>, with the hat shown in <Picture 1> placed on her head'). When the target "
        "is an edit, describe the base picture's scene with the change already in place. Describe every referenced subject's features fully at its first appearance so the prompt "
        "stands without the pictures. The output describes exactly one still image: no shots, no [Shot N] labels, no timestamps, no timeline, no plot summary.\n\n"
        "The target is a still image only: never mention sound, dialogue, speech, speakers, singing, music, audio, video, motion or timing, and never output sections such as "
        "overall_soundscape or non_diegetic_music. Do not include your analysis of the pictures' roles as prose; the section structure carries it.\n\n"
        "When no pictures are connected, skip the reference sections entirely: output only 'detailed_description:' with its text, building the image purely from the user's "
        "request, adding scene, character, and style details that remain consistent with their intent. Any text the user wants rendered appears quoted exactly once, and the "
        "final text is a plain description: no section names or labels appear in it."
    ),
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
                io.String.Input("api_key", default="", tooltip="Leave empty. Use the 'set API key' button to save the key for this base_url outside the workflow. A key typed here is moved there automatically, otherwise OPENAI_API_KEY is used."),
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
        warn_inline_key(api_key, "[EZ Prompt Writer]")
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset}")
        system = (system_prompt.strip() or PRESETS[preset] + "\n\n" + (H3_OUTPUT_RULES if preset.startswith("minimax h3") else OUTPUT_RULES))
        if extra_instructions.strip():
            system += "\n\n" + extra_instructions.strip()

        frames = []
        for name in sorted(images or {}, key=lambda name: int(name.removeprefix("image"))):
            if images[name] is not None:
                frames += [tensor_to_pil(frame) for frame in images[name]]
        content = [{"type": "text", "text": f"User prompt: {prompt.strip() or '(none)'}" + (f"\n\n{len(frames)} reference image(s) follow in numbered order (Picture 1, Picture 2, ...)." if frames else "")}]
        for i, frame in enumerate(frames, 1):
            content += [{"type": "text", "text": f"Picture {i}:"}, {"type": "image_url", "image_url": {"url": encode_image(frame, max_image_size)}}]

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
