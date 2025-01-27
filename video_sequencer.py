import logging
import math

class VideoSequencer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "notes_by_keyframes": ("NOTES_BY_KEYFRAMES",),
                "allowed_notes": ("LIST",),
                "end_frame": ("LIST",),
                "video_frame_list": ("LIST",),
                "reset_mode": (["reset", "cumulative"], {"default": "reset"}),
                "overflow_mode": (["freeze", "palindrome", "loop"], {"default": "freeze"}),
            }
        }

    RETURN_TYPES = ("FRAME_SEQUENCE", "LIST", "LIST", "STRING")
    RETURN_NAMES = ("sequence_structure", "notes_to_vx_debug", "keyframe_difference_debug", "frame_sequence")
    FUNCTION = "generate_sequence"
    CATEGORY = "Video Processing"

    def __init__(self):
        pass

    def calculate_frame(self, counter, frame_count, overflow_mode):
        if overflow_mode == "freeze":
            return min(counter, frame_count - 1)
        elif overflow_mode == "palindrome":
            cycle_length = 2 * (frame_count - 1)
            position_in_cycle = counter % cycle_length
            return frame_count - 1 - abs(position_in_cycle - (frame_count - 1))
        elif overflow_mode == "loop":
            return counter % frame_count

    def generate_sequence(self, notes_by_keyframes, allowed_notes, end_frame, video_frame_list, reset_mode, overflow_mode, **kwargs):
        logging.info("Debug: Generating video sequence in mode: %s", reset_mode)
        frame_sequence = []
        current_frame = 0
        video_frame_counter = 1
        note_index = 0

        if not end_frame or end_frame[0] <= 0:
            logging.error("Error: end_frame must be greater than 0")
            return ({"sequence_structure": []}, [], {}, "")

        # Oblicz zakresy dla każdego wideo
        cumulative_offsets = {}
        current_offset = 0
        for i, note in enumerate(allowed_notes):
            video_key = f"v{i + 1}"
            frame_count = video_frame_list[video_key][0][0]
            cumulative_offsets[video_key] = {
                "start": current_offset,
                "end": current_offset + frame_count - 1,
                "current": 0,
                "total_frames": frame_count
            }
            current_offset += frame_count

        # Mapowanie nut na wideo
        notes_to_vx_debug = []
        video_mapping = {}
        for i, note in enumerate(allowed_notes):
            video_key = f"v{i + 1}"
            video_mapping[note] = video_key
            notes_to_vx_debug.append({f"note_{note}": [[video_key]]})

        current_note = None
        current_video = None
        keyframe_difference_debug = {}
        previous_frame = 0
        img_in_batch_values = []

        while current_frame < end_frame[0]:
            if note_index < len(notes_by_keyframes) and current_frame >= notes_by_keyframes[note_index][0]:
                new_note = notes_by_keyframes[note_index][1]
                new_video = video_mapping.get(new_note, None)

                if reset_mode == "reset" and new_video is not None:
                    cumulative_offsets[new_video]["current"] = 0

                if current_note is not None:
                    frame_difference = current_frame - previous_frame
                    keyframe_difference_debug.setdefault(str(current_note), []).append([frame_difference])

                previous_frame = current_frame
                current_note = new_note
                current_video = new_video
                note_index += 1

            img_in_batch = None
            if current_video:
                video_data = cumulative_offsets[current_video]
                frame_count = video_data["total_frames"]
                
                if reset_mode == "reset":
                    counter = video_data["current"]
                    adjusted_counter = self.calculate_frame(counter, frame_count, overflow_mode)
                    img_in_batch = video_data["start"] + adjusted_counter
                    video_data["current"] += 1
                else:
                    global_counter = current_frame
                    adjusted_counter = self.calculate_frame(global_counter, frame_count, overflow_mode)
                    img_in_batch = video_data["start"] + adjusted_counter

            frame_sequence.append({
                "frame": video_frame_counter,
                "img": current_frame,
                "note_number": current_note,
                "video": current_video,
                "img_in_batch": img_in_batch,
            })

            img_in_batch_values.append(str(img_in_batch) if img_in_batch is not None else "None")
            current_frame += 1
            video_frame_counter += 1

        if current_note is not None:
            frame_difference = end_frame[0] - previous_frame
            keyframe_difference_debug.setdefault(str(current_note), []).append([frame_difference])

        frame_sequence_str = ", ".join(img_in_batch_values)
        logging.info("Debug: Final sequence structure: %s", frame_sequence)
        return ({"sequence_structure": frame_sequence}, notes_to_vx_debug, keyframe_difference_debug, frame_sequence_str)

VIDEO_SEQUENCER_NODE_CLASS_MAPPINGS = {
    "VideoSequencer": VideoSequencer,
}

VIDEO_SEQUENCER_NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoSequencer": "Video Sequencer",
}