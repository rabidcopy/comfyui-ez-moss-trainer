from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .captioner import EZCaptionDataset
from .nodes import EZMOSSTrainLoRA

WEB_DIRECTORY = "./web"


class EZMOSSTrainerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [EZMOSSTrainLoRA, EZCaptionDataset]


async def comfy_entrypoint() -> EZMOSSTrainerExtension:
    return EZMOSSTrainerExtension()
