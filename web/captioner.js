import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// button callbacks don't get the click event in every frontend version, so remember where the pointer went down
let lastPointer = null;
window.addEventListener("pointerdown", (e) => (lastPointer = e), true);

function showModelPicker(models, current, x, y, onPick) {
    document.getElementById("ez-moss-model-picker")?.remove();
    const root = document.createElement("div");
    root.id = "ez-moss-model-picker";
    Object.assign(root.style, {
        position: "fixed", zIndex: 10000, width: "360px", maxHeight: "60vh", display: "flex", flexDirection: "column",
        background: "var(--comfy-menu-bg, #202020)", color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #555)",
        borderRadius: "6px", boxShadow: "0 4px 16px rgba(0,0,0,0.5)", font: "13px sans-serif",
    });
    const search = document.createElement("input");
    search.placeholder = `Search ${models.length} models...`;
    search.value = current;
    Object.assign(search.style, {
        margin: "6px", padding: "6px 8px", background: "var(--comfy-input-bg, #111)", color: "inherit",
        border: "1px solid var(--border-color, #555)", borderRadius: "4px", outline: "none",
    });
    const list = document.createElement("div");
    Object.assign(list.style, { overflowY: "auto", padding: "0 0 6px 0" });
    root.append(search, list);

    const close = () => {
        root.remove();
        window.removeEventListener("pointerdown", outside, true);
    };
    const outside = (e) => { if (!root.contains(e.target)) close(); };
    const render = () => {
        const terms = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
        const shown = models.filter((m) => terms.every((t) => m.toLowerCase().includes(t)));
        list.replaceChildren(...shown.slice(0, 500).map((m) => {
            const row = document.createElement("div");
            row.textContent = m;
            Object.assign(row.style, { padding: "4px 10px", cursor: "pointer", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" });
            row.onmouseenter = () => (row.style.background = "var(--comfy-input-bg, #333)");
            row.onmouseleave = () => (row.style.background = "");
            row.onclick = () => { onPick(m); close(); };
            return row;
        }));
        if (!shown.length) list.textContent = "  No matching models";
    };
    search.oninput = render;
    search.onkeydown = (e) => {
        e.stopPropagation();
        if (e.key === "Escape") close();
        if (e.key === "Enter" && list.firstChild?.onclick) list.firstChild.onclick();
    };
    render();
    document.body.append(root);
    // keep the popup inside the window
    const rect = root.getBoundingClientRect();
    root.style.left = `${Math.max(8, Math.min(x, window.innerWidth - rect.width - 8))}px`;
    root.style.top = `${Math.max(8, Math.min(y, window.innerHeight - rect.height - 8))}px`;
    setTimeout(() => window.addEventListener("pointerdown", outside, true));
    search.focus();
    search.select();
}

app.registerExtension({
    name: "ez_moss.captioner",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["EZCaptionDataset", "EZMOSSPromptWriter"].includes(nodeData.name)) return;
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
                const x = event?.clientX ?? window.innerWidth / 2 - 180;
                const y = event?.clientY ?? window.innerHeight / 4;
                showModelPicker(data.models, widget("model").value, x, y, (value) => {
                    widget("model").value = value;
                    app.graph.setDirtyCanvas(true);
                });
            });
            return r;
        };
    },
});
