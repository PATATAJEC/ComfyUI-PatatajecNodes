import mido
from mido import MidiFile, tempo2bpm
import logging  # Dodano import modułu logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)  # Można zmienić na logging.WARNING, aby wyłączyć debugowanie

class MidiReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "D:\\INPUTS\\120_3notes.mid"})                
            }
        }

    RETURN_TYPES = ("MIDI Data", "STRING")  # Zwraca dane MIDI i listę unikalnych nut
    RETURN_NAMES = ("MIDI Data", "unique_notes")  # Nazwy wyjść
    FUNCTION = "read_midi"
    CATEGORY = "MIDI"

    def read_midi(self, file_path):
        try:
            midi_file = MidiFile(file_path)
        except Exception as e:
            logging.error(f"Error opening MIDI file: {e}")  # Zmieniono print na logging.error
            raise Exception(f"Error opening MIDI file: {e}")

        # Inicjalizacja zmiennych do śledzenia czasu i unikalnych nut
        current_time = 0.0
        midi_data = []  # Lista do przechowywania danych MIDI
        unique_notes = set()  # Zbiór unikalnych numerów nut

        for message in midi_file:
            current_time += message.time

            # Tworzymy słownik z informacjami o wiadomości
            message_data = {
                "type": message.type,  # Typ wiadomości (np. 'note_on', 'note_off', 'meta')
                "note": getattr(message, "note", None),  # Numer nuty (jeśli dotyczy)
                "velocity": getattr(message, "velocity", None),  # Prędkość (jeśli dotyczy)
                "time": message.time,  # Czas do następnej wiadomości
                "clip_time": current_time,  # Czas od początku utworu
                "meta": message.is_meta,  # Czy to wiadomość meta (np. 'track_name', 'set_tempo')
                "meta_type": getattr(message, "type", None) if message.is_meta else None,  # Typ wiadomości meta
            }

            # Dodaj numer nuty do zbioru, jeśli to wiadomość note_on
            if message.type == "note_on":
                unique_notes.add(message.note)

            # Dodajemy dane do listy
            midi_data.append(message_data)

        # Konwersja zbioru unikalnych nut na posortowaną listę
        unique_notes_list = sorted(unique_notes)
        unique_notes_str = ",".join(map(str, unique_notes_list))

        logging.info("Debug: MIDI data processed successfully")  # Zmieniono print na logging.info
        return (midi_data, unique_notes_str)

# Zmiana nazw zmiennych mapowań
MIDI_ANALYZER_NODE_CLASS_MAPPINGS = {
    "MidiReader": MidiReader,
}

MIDI_ANALYZER_NODE_DISPLAY_NAME_MAPPINGS = {
    "MidiReader": "Midi Reader",
}