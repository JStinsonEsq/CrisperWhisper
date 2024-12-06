from typing import List, Dict, Any
from pyannote.audio import Pipeline
from datasets import Audio
import math

class AudioSegmenter:
    """
    A class that performs voice activity detection (VAD) on a long audio file and 
    returns audio segments as dictionaries containing "array" and "sampling_rate"
    keys, matching the structure of the `Audio` feature from the datasets library.

    Each returned segment can be passed directly to Whisper for transcription.
    """

    def __init__(self, 
                 vad_model: str = "pyannote/voice-activity-detection",
                 face_hugger_token: str = "",
                 max_segment_length: float = 30.0):
        """
        Initialize the AudioSegmenter.

        Parameters:
        -----------
        vad_model: str
            Pretrained VAD model from pyannote. Default is "pyannote/voice-activity-detection".
        max_segment_length: float
            The maximum length for any returned audio segment in seconds.
        """
        self.pipeline = Pipeline.from_pretrained(vad_model, use_auth_token=face_hugger_token)
        self.max_segment_length = max_segment_length

    def segment_audio(self, audio_file_path: str) -> List[Dict[str, Any]]:
        """
        Segment the provided audio file into smaller segments based on voice activity detection.
        Ensures that no segment exceeds max_segment_length in duration. Returns a list of 
        segment dictionaries as decoded by the `Audio` feature from the datasets library.

        Parameters:
        -----------
        audio_file_path: str
            Path to the audio file to segment.

        Returns:
        --------
        segments: List[Dict[str, Any]]
            A list of dictionaries, each containing:
            {
                "array": np.ndarray,
                "sampling_rate": int
            }
            representing the audio segment data and its sampling rate.
        """
        # First, decode the entire audio file using the datasets Audio feature
        audio_feature = Audio()
        decoded = audio_feature.decode_example({"path": audio_file_path, "bytes": None})
        full_audio_array = decoded["array"]
        sampling_rate = decoded["sampling_rate"]

        # Run the VAD pipeline
        vad_result = self.pipeline({"audio": audio_file_path})
        
        segments = []
        
        # Iterate over the speech segments detected by the VAD
        for segment in vad_result.get_timeline().support():
            start = segment.start
            end = segment.end
            
            # Split long segments into smaller ones if necessary
            for sub_segment_start, sub_segment_end in self._split_segment(start, end):
                # Convert times to sample indices
                start_sample = int(math.floor(sub_segment_start * sampling_rate))
                end_sample   = int(math.floor(sub_segment_end * sampling_rate))
                
                segment_data = {
                    "array": full_audio_array[start_sample:end_sample],
                    "sampling_rate": sampling_rate
                }
                segments.append(segment_data)
        
        return segments

    def _split_segment(self, start: float, end: float):
        """
        Split a long segment into smaller segments no longer than max_segment_length.

        Parameters:
        -----------
        start: float
            The start time of the long segment.
        end: float
            The end time of the long segment.

        Returns:
        --------
        sub_segments: List[(float, float)]
            A list of (start_time, end_time) tuples after splitting.
        """
        sub_segments = []
        current_start = start
        while current_start < end:
            current_end = min(current_start + self.max_segment_length, end)
            sub_segments.append((current_start, current_end))
            current_start = current_end
        return sub_segments
