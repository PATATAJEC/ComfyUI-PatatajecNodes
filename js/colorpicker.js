// web/colorpicker.js
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

function clamp255(n) {
    n = Number(n);
    if (isNaN(n)) return 127;
    return Math.max(0, Math.min(255, Math.round(n)));
}
function rgbToHex(r, g, b) {
    return (
        "#" +
        clamp255(r).toString(16).padStart(2, "0") +
        clamp255(g).toString(16).padStart(2, "0") +
        clamp255(b).toString(16).padStart(2, "0")
    ).toUpperCase();
}
function parseAnyToHex(str) {
    if (!str) return "#808080";
    let s = String(str).trim();

    // JSON [r,g,b]
    try {
        const arr = JSON.parse(s);
        if (Array.isArray(arr) && arr.length >= 3)
            return rgbToHex(arr[0], arr[1], arr[2]);
    } catch (_) {}

    // CSV r,g,b
    const parts = s.replace(";", ",").split(",").map((p) => p.trim()).filter(Boolean);
    if (parts.length >= 3) {
        return rgbToHex(parts[0], parts[1], parts[2]);
    }

    // HEX
    if (s[0] !== "#") s = "#" + s;
    if (/^#[0-9a-fA-F]{6}$/.test(s)) return s.toUpperCase();

    return "#808080";
}

app.registerExtension({
    name: "Patatajec.ColorPicker",
    async init() {
        // Rejestrujemy nowy typ widgetu dla INPUT_TYPES: ("COLOR", {...})
        ComfyWidgets.COLOR = (node, inputName, inputData, app_) => {
            const defHex = parseAnyToHex(inputData?.default ?? "#808080");

            // Pole tekstowe HEX (pozwala też wkleić CSV/JSON — zamienimy na HEX)
            const w = node.addWidget("text", inputName, defHex, (val) => {
                const hex = parseAnyToHex(val);
                if (hex !== w.value) {
                    w.value = hex;
                    node.setDirtyCanvas(true);
                }
            }, { multiline: false });

            // Przycisk z nativem <input type="color">
            const btn = node.addWidget("button", "🎨 pick", "pick", () => {
                const input = document.createElement("input");
                input.type = "color";
                input.value = parseAnyToHex(w.value);
                input.style.position = "fixed";
                input.style.left = "-1000px";
                document.body.appendChild(input);
                input.addEventListener("input", () => {
                    w.value = input.value.toUpperCase();
                    node.setDirtyCanvas(true);
                }, { passive: true });
                input.addEventListener("change", () => {
                    document.body.removeChild(input);
                }, { once: true });
                input.click();
            });
            btn.serialize = false; // nie zapisujemy przycisku do sceny

            return { widget: w };
        };
    },
});