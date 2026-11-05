import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// training progress of the current run, drawn by every trainer node
let latest = null; // {run, step, ema, losses, validation, validation_steps, samples}
let shown = -1; // sample index the flippers display, -1 follows the newest
const views = new Set();

function sampleUrl(index) {
    // run is part of the cache key, a new run can reuse an index with another file
    return `/ez_moss/sample/image?index=${index}&run=${encodeURIComponent(latest.run)}`;
}

function apply(data, full = false) {
    // a new queued run resets the flipper and the graph
    if (latest && latest.run !== data.run) shown = -1;
    latest ||= { run: data.run, step: data.step, ema: data.ema, losses: [], validation: [], validation_steps: [], samples: [] };
    latest.run = data.run;
    latest.step = data.step;
    latest.ema = data.ema;
    latest.validation = data.validation || latest.validation;
    latest.validation_steps = data.validation_steps || latest.validation_steps;
    if (full) {
        latest.losses = data.losses;
    } else if (typeof data.loss_start === "number") {
        latest.losses = latest.losses.slice(0, data.loss_start).concat(data.losses);
    }
    if (data.samples) {
        if (data.samples.length !== (latest.samples?.length ?? -1)) shown = data.samples.length - 1;
        latest.samples = data.samples;
    }
    views.forEach(draw);
}

function viewIndex() {
    const samples = latest?.samples || [];
    if (!samples.length) return -1;
    return Math.max(0, Math.min(shown < 0 ? samples.length - 1 : shown, samples.length - 1));
}

function draw(view) {
    if (!latest) return;
    const samples = latest.samples || [];
    const index = viewIndex();
    view.prev.disabled = index <= 0;
    view.next.disabled = index < 0 || index >= samples.length - 1;
    view.label.textContent = index >= 0 ? `step ${samples[index].step}  ${index + 1}/${samples.length}` : "no preview yet";
    if (index >= 0) {
        const url = sampleUrl(index);
        if (view.img.dataset.url !== url) {
            view.img.dataset.url = url;
            view.img.src = url;
        }
    } else {
        view.img.removeAttribute("src");
        delete view.img.dataset.url;
    }
    drawGraph(view.canvas, latest);
}

function drawGraph(canvas, data) {
    const width = canvas.clientWidth;
    if (!width) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = 80 * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, 80);

    const losses = data.losses || [];
    const valSteps = data.validation_steps || [];
    const vals = data.validation || [];
    const lastStep = Math.max(data.step || 0, losses.length, valSteps[valSteps.length - 1] || 1);
    if (!lastStep) return;

    let min = Infinity, max = -Infinity;
    for (const v of losses) { min = Math.min(min, v); max = Math.max(max, v); }
    for (const v of vals) { min = Math.min(min, v); max = Math.max(max, v); }
    if (!isFinite(min)) { min = 0; max = 1; }
    const pad = (max - min) * 0.1 || Math.abs(max) * 0.1 || 0.1;
    min -= pad;
    max += pad;
    const x = (step) => step / lastStep * (width - 2);
    const y = (v) => 80 - (v - min) / (max - min) * 80;

    // training loss, bucket-averaged so a long run stays a thin line
    const stride = Math.max(1, Math.ceil(losses.length / (width / 4)));
    ctx.strokeStyle = "rgba(148, 163, 184, 0.55)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < losses.length; i += stride) {
        let sum = 0, n = 0;
        const end = Math.min(i + stride, losses.length);
        for (let j = i; j < end; j++) { sum += losses[j]; n++; }
        const step = end;
        const sx = x(step), sy = y(sum / n);
        if (started) ctx.lineTo(sx, sy); else { ctx.moveTo(sx, sy); started = true; }
    }
    if (started) ctx.stroke();

    // validation, fixed images and noise so its points compare step to step
    ctx.fillStyle = "#eab308";
    for (let i = 0; i < valSteps.length; i++) {
        ctx.beginPath();
        ctx.arc(x(valSteps[i]), y(vals[i]), 2.5, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
    ctx.font = "10px sans-serif";
    ctx.fillText("loss", 4, 11);
}

api.addEventListener("ez_moss.progress", ({ detail }) => {
    apply(detail);
});

app.registerExtension({
    name: "ez_moss.preview",
    async setup() {
        const res = await api.fetchApi("/ez_moss/progress");
        if (res.ok) apply(await res.json(), true);
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
            Object.assign(root.style, { display: "flex", flexDirection: "column", alignItems: "stretch", gap: "4px", height: "100%", minHeight: "0" });
            const img = document.createElement("img");
            Object.assign(img.style, { width: "100%", flex: "1", minHeight: "0", objectFit: "contain" });
            const canvas = document.createElement("canvas");
            Object.assign(canvas.style, { width: "100%", height: "80px" });
            const controls = document.createElement("div");
            Object.assign(controls.style, { display: "flex", alignItems: "center", gap: "8px" });
            const mkButton = (icon) => {
                const b = document.createElement("button");
                b.textContent = icon;
                Object.assign(b.style, { font: "11px sans-serif", padding: "2px 10px", cursor: "pointer" });
                return b;
            };
            const prev = mkButton("◀");
            const next = mkButton("▶");
            const label = document.createElement("div");
            Object.assign(label.style, { flex: "1", font: "12px sans-serif", color: "var(--input-text, #ddd)", textAlign: "center" });
            label.textContent = "no preview yet";
            controls.append(prev, label, next);
            root.append(img, canvas, controls);
            this.addDOMWidget("training_preview", "div", root, { serialize: false, getMinHeight: () => 440 });

            const flip = (delta) => {
                if (!latest?.samples?.length) return;
                shown = Math.max(0, viewIndex() + delta);
                views.forEach(draw);
            };
            prev.onclick = () => flip(-1);
            next.onclick = () => flip(1);

            const view = { img, label, canvas, prev, next };
            views.add(view);
            draw(view);
            // canvas.clientWidth is 0 until the widget is laid out, and changes with the node's size
            new ResizeObserver(() => draw(view)).observe(canvas);
            const onRemoved = this.onRemoved;
            this.onRemoved = function () {
                views.delete(view);
                return onRemoved?.apply(this, arguments);
            };
            return r;
        };
    },
});
