import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "CustomNodes.VideoCounter",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VideoCounter") {
            // Dodaj przycisk "Update Inputs" i logikę dynamicznego dodawania wejść
            nodeType.prototype.onNodeCreated = function () {
                this.addWidget("button", "Update inputs", null, () => {
                    if (!this.inputs) {
                        this.inputs = [];
                    }

                    // Pobierz wartość input_count
                    const inputCountWidget = this.widgets.find(w => w.name === "input_count");
                    const targetNumberOfInputs = inputCountWidget ? inputCountWidget.value : 1;

                    // Ustal, ile wejść już istnieje
                    const existingInputs = this.inputs.length;

                    // Oblicz, ile nowych wejść trzeba dodać
                    const inputsToAdd = targetNumberOfInputs - existingInputs;

                    if (inputsToAdd === 0) return; // Nic nie rób, jeśli liczba wejść jest już poprawna

                    if (inputsToAdd < 0) {
                        // Usuń nadmiarowe wejścia
                        for (let i = existingInputs - 1; i >= targetNumberOfInputs; i--) {
                            this.removeInput(i);
                        }
                    } else {
                        // Dodaj nowe wejścia
                        for (let i = existingInputs; i < targetNumberOfInputs; i++) {
                            const inputName = `v${i + 1}_fcount`; // Indeksowanie od v1_fcount
                            this.addInput(inputName, "INT");
                        }
                    }
                });
            };
        }
    },
});