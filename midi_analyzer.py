# midi_analyzer/midi_analyzer.py
import mido
from mido import MidiFile, tempo2bpm

class MidiAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "path/to/your/midi/file.mid"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "LIST")
    FUNCTION = "analyze_midi"
    CATEGORY = "MIDI"

    def analyze_midi(self, file_path):
        midi = MidiFile(file_path)
        
        tempo_info = []
        notes_info = []

        for track in midi.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo_bpm = mido.tempo2bpm(msg.tempo)
                    tempo_info.append({"time": msg.time, "tempo": tempo_bpm})
                elif msg.type in ['note_on', 'note_off']:
                    notes_info.append({
                        "type": msg.type,
                        "channel": msg.channel,
                        "note_number": msg.note,
                        "velocity": msg.velocity,
                        "time": msg.time
                    })

        return (tempo_info, notes_info)

# Definicja NODE_CLASS dla ComfyUI
MIDI_ANALYZER_NODE_CLASS_MAPPINGS = {
    "MidiAnalyzer": MidiAnalyzer,
}

# Dodanie definicji NODE_DISPLAY_NAME_MAPPINGS
MIDI_ANALYZER_NODE_DISPLAY_NAME_MAPPINGS = {
    "MidiAnalyzer": "Midi Analyzer",
}

    