import mido
from mido import MidiFile, tempo2bpm
from collections import defaultdict

class MidiAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "path/to/your/midi/file.mid"}),
                "ppq": ("INT", {"default": 192})  # Dodanie parametru ppq
            }
        }

    RETURN_TYPES = ("FLOAT", "LIST", "LIST")  # Zmienione zwracane typy na FLOAT, LIST, LIST
    FUNCTION = "analyze_midi"
    CATEGORY = "MIDI"

    def analyze_midi(self, file_path, ppq):
        try:
            midi_file = MidiFile(file_path)
        except Exception as e:
            raise Exception(f"Error opening MIDI file: {e}")

        # Inicjalizacja zmiennych do śledzenia czasu
        current_time_ticks = 0.0
        tempo_info = []
        notes_info = defaultdict(list)
        tempo = 500000  # Domyślne tempo MIDI (120 BPM)

        debug_info = []  # Lista do przechowywania debugowych informacji

        for message in midi_file:
            # Dodaj delta czasu w tickach do bieżącego czasu
            current_time_ticks += message.time

            # Debugowanie wartości
            debug_info.append(f"Message: {message}, Time in Ticks: {current_time_ticks}, Time in Seconds: {current_time_ticks}")

            if message.type == 'set_tempo':
                bpm = tempo2bpm(message.tempo)
                tempo_info.append({"time": current_time_ticks, "tempo": bpm})
                tempo = message.tempo  # Aktualizuj obecne tempo
            elif message.type == 'note_on' and message.velocity > 0:
                notes_info[message.note].append({
                    "start_time": current_time_ticks,
                    "velocity": message.velocity
                })
            elif message.type == 'note_off':
                for note_data in reversed(notes_info[message.note]):
                    if "end_time" not in note_data:
                        note_data["end_time"] = current_time_ticks
                        break
            elif message.type == 'note_on' and message.velocity == 0:
                # Traktuj kompresję jako note_off
                for note_data in reversed(notes_info[message.note]):
                    if "end_time" not in note_data:
                        note_data["end_time"] = current_time_ticks
                        break

        # Przekonwertuj defaultdict na listę dla łatwiejszego przetwarzania
        notes_list = []
        for note_number, note_data_list in notes_info.items():
            for note_data in note_data_list:
                if "end_time" not in note_data:
                    # Jeśli end_time nie jest ustawione, użyj ostatniej znanej wartości czasu
                    note_data["end_time"] = current_time_ticks
                notes_list.append({
                    "note_number": note_number,
                    **note_data
                })

        # Zwracaj tempo_info jako pierwszą wartość, notes_list jako drugą wartość i debug_info jako trzecią wartość
        return (tempo, notes_list, debug_info)

# Definicja NODE_CLASS dla ComfyUI
MIDI_ANALYZER_NODE_CLASS_MAPPINGS = {
    "MidiAnalyzer": MidiAnalyzer,
}

# Dodanie definicji NODE_DISPLAY_NAME_MAPPINGS
MIDI_ANALYZER_NODE_DISPLAY_NAME_MAPPINGS = {
    "MidiAnalyzer": "Midi Analyzer",
}