import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// the last training preview, drawn by every trainer node
let latest = null;
const views = new Set();

function draw(view) {
    if (!latest) return;
    view.img.src = latest.image;
    view.label.textContent = `step ${latest.step}`;
}

api.addEventListener("ez_moss.preview", ({ detail }) => {
    latest = detail;
    views.forEach(draw);
});

app.registerExtension({
    name: "ez_moss.preview",
    async setup() {
        const res = await api.fetchApi("/ez_moss/preview");
        const data = await res.json();
        if (data.image) {
            latest = data;
            views.forEach(draw);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "EZMOSSTrainLoRA") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            this.addWidget("button", "sample now", null, async () => {
                const value = (name) => this.widgets.find((w) => w.name === `preview_${name}`).value;
                const settings = Object.fromEntries(["width", "height", "steps", "cfg", "sampler", "scheduler", "lora", "lora_strength"].map((k) => [k, value(k)]));
                await api.fetchApi("/ez_moss/sample", { method: "POST", body: JSON.stringify(settings) });
            });

            const root = document.createElement("div");
            Object.assign(root.style, { display: "flex", flexDirection: "column", alignItems: "center", gap: "4px", height: "100%", minHeight: "0" });
            const img = document.createElement("img");
            Object.assign(img.style, { width: "100%", flex: "1", minHeight: "0", objectFit: "contain" });
            const label = document.createElement("div");
            Object.assign(label.style, { font: "12px sans-serif", color: "var(--input-text, #ddd)" });
            label.textContent = "no preview yet";
            root.append(img, label);
            this.addDOMWidget("training_preview", "div", root, { serialize: false, getMinHeight: () => 320 });

            const view = { img, label };
            views.add(view);
            draw(view);
            const onRemoved = this.onRemoved;
            this.onRemoved = function () {
                views.delete(view);
                return onRemoved?.apply(this, arguments);
            };
            return r;
        };
    },
});
