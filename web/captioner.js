import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// button callbacks don't get the click event in every frontend version, so remember where the pointer went down
let lastPointer = null;
window.addEventListener("pointerdown", (e) => (lastPointer = e), true);

app.registerExtension({
    name: "ez_moss.captioner",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "EZCaptionDataset") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const widget = (name) => this.widgets.find((w) => w.name === name);
            this.addWidget("button", "list models", null, async () => {
                const event = lastPointer;
                const res = await api.fetchApi("/ez_moss/caption_models", {
                    method: "POST",
                    body: JSON.stringify({ base_url: widget("base_url").value, api_key: widget("api_key").value }),
                });
                const data = await res.json();
                if (!res.ok || !data.models?.length) {
                    alert(`Could not list models at ${widget("base_url").value}: ${data.error ?? "the server returned no models"}`);
                    return;
                }
                // text already in the model field filters the list, so long provider catalogs stay usable
                const filter = widget("model").value.trim().toLowerCase();
                const models = data.models.filter((m) => m.toLowerCase().includes(filter));
                if (!models.length) {
                    alert(`No models match "${filter}". Clear the model field to list all ${data.models.length} models.`);
                    return;
                }
                new LiteGraph.ContextMenu(models, {
                    event,
                    title: filter ? `${models.length} of ${data.models.length} models matching "${filter}"` : `${models.length} models`,
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
