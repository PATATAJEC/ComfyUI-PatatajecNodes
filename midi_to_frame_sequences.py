import math
import logging  # Dodano import modułu logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)  # Można zmienić na logging.WARNING, aby wyłączyć debugowanie

class MidiToFrameSequences:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_data": ("MIDI Data",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "note_priority": ("STRING", {"default": ""}),
                "allowed_notes": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("KEYFRAMES_BY_NOTES", "NOTES_BY_KEYFRAMES", "LIST", "LIST", "INT")
    RETURN_NAMES = ("keyframes_by_notes", "notes_by_keyframes", "allowed_notes", "end_frame", "allowed_note_count")
    FUNCTION = "generate_sequences"
    CATEGORY = "MIDI"

    def generate_sequences(self, midi_data, fps, note_priority, allowed_notes):
        logging.info("Debug: MIDI Data received: %s", midi_data)  # Zmieniono print na logging.info
        keyframes_by_notes = {}
        end_frame = None

        note_priority_list = [int(note.strip()) for note in note_priority.split(",")] if note_priority else []
        allowed_notes_list = [int(note.strip()) for note in allowed_notes.split(",")] if allowed_notes else []
        allowed_notes_set = set(allowed_notes_list)

        if not allowed_notes_set:
            logging.info("Debug: Allowed notes is empty. Skipping processing.")  # Zmieniono print na logging.info
            return ({}, [], allowed_notes_list, [-1], 0)

        frame_events = {}

        for entry in midi_data:
            logging.info("Debug: Processing entry: %s", entry)  # Zmieniono print na logging.info

            if entry["type"] == "end_of_track":
                clip_time = entry["clip_time"]
                end_frame = [math.ceil(clip_time * fps)]  # Zaokrąglanie W GÓRĘ
                logging.info(f"Debug: End of track at frame {end_frame[0]}")  # Zmieniono print na logging.info
                continue

            if entry["meta"]:
                logging.info("Debug: Skipping meta message: %s", entry)  # Zmieniono print na logging.info
                continue

            if entry["type"] == "note_on":
                note_number = entry["note"]
                clip_time = entry["clip_time"]

                if note_number not in allowed_notes_set:
                    logging.info(f"Debug: Skipping note {note_number} (not allowed)")  # Zmieniono print na logging.info
                    continue

                frame_number = math.ceil(clip_time * fps)  # Zaokrąglanie W GÓRĘ
                logging.info(f"Debug: Note {note_number} at frame {frame_number} (ceil)")  # Zmieniono print na logging.info

                if note_number not in keyframes_by_notes:
                    keyframes_by_notes[note_number] = []

                if frame_number in frame_events:
                    highest_priority_note = None
                    for note in frame_events[frame_number] + [note_number]:
                        if note in note_priority_list:
                            if highest_priority_note is None or note_priority_list.index(note) < note_priority_list.index(highest_priority_note):
                                highest_priority_note = note

                    if highest_priority_note != note_number:
                        logging.info(f"Debug: Skipping note {note_number} (lower priority)")  # Zmieniono print na logging.info
                        continue

                    for note in frame_events[frame_number]:
                        if note in keyframes_by_notes:
                            if [frame_number] in keyframes_by_notes[note]:
                                keyframes_by_notes[note].remove([frame_number])
                                logging.info(f"Debug: Removed note {note} from frame {frame_number} (lower priority)")  # Zmieniono print na logging.info

                    frame_events[frame_number] = [note_number]
                else:
                    frame_events[frame_number] = [note_number]

                keyframes_by_notes[note_number].append([frame_number])

        notes_by_keyframes = []
        for note_number, frames in keyframes_by_notes.items():
            for frame in frames:
                notes_by_keyframes.append((frame[0], note_number))

        notes_by_keyframes.sort(key=lambda x: x[0])

        sorted_keyframes_by_notes = {}
        for frame, note_number in notes_by_keyframes:
            if note_number not in sorted_keyframes_by_notes:
                sorted_keyframes_by_notes[note_number] = []
            sorted_keyframes_by_notes[note_number].append([frame])

        allowed_note_count = len(allowed_notes_set)

        logging.info("Debug: Final sorted keyframes by notes: %s", sorted_keyframes_by_notes)  # Zmieniono print na logging.info
        logging.info("Debug: Notes by keyframes: %s", notes_by_keyframes)  # Zmieniono print na logging.info
        logging.info("Debug: Allowed notes: %s", allowed_notes_list)  # Zmieniono print na logging.info
        logging.info("Debug: End of track frame: %s", end_frame)  # Zmieniono print na logging.info
        logging.info("Debug: Allowed note count: %s", allowed_note_count)  # Zmieniono print na logging.info

        return (sorted_keyframes_by_notes, notes_by_keyframes, allowed_notes_list, end_frame, allowed_note_count)

MIDI_TO_FRAME_SEQUENCES_NODE_CLASS_MAPPINGS = {
    "MidiToFrameSequences": MidiToFrameSequences,
}

MIDI_TO_FRAME_SEQUENCES_NODE_DISPLAY_NAME_MAPPINGS = {
    "MidiToFrameSequences": "MIDI to Frame Sequences",
}