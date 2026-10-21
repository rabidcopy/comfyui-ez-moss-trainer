import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "ez_moss.captioner",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "EZCaptionDataset") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const widget = (name) => this.widgets.find((w) => w.name === name);
            this.addWidget("button", "list models", null, async () => {
                const res = await api.fetchApi("/ez_moss/caption_models", {
                    method: "POST",
                    body: JSON.stringify({ base_url: widget("base_url").value, api_key: widget("api_key").value }),
                });
                const data = await res.json();
                if (!res.ok || !data.models?.length) {
                    alert(`Could not list models at ${widget("base_url").value}: ${data.error ?? "the server returned no models"}`);
                    return;
                }
                new LiteGraph.ContextMenu(data.models, {
                    event: app.canvas.last_mouse_event,
                    callback: (value) => {
                        widget("model").value = value;
                        app.graph.setDirtyCanvas(true);
                    },
                });
            });
            return r;
        };
    },
});
