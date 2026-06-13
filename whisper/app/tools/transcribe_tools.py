import whisper
from typing import List, Tuple, Dict, Any, Union, Optional
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import numpy as np
import torch
from gc import collect
import soundfile as sf
from importlib.metadata import version
import os
from nemo.collections.asr.models import SortformerEncLabelModel
from collections import defaultdict, Counter
import re

from app.tools.vad import silero_gate_each_channel_then_merge_mono
MODEL_DIR = "./models"
HF_CACHE_DIR = os.path.join(MODEL_DIR, "hf_cache")
DIARIZE_DIR = os.path.join(MODEL_DIR, "diarization")
DIARIZE_FILE = os.path.join(DIARIZE_DIR, "model.nemo")

class Model:
    def __init__(self, model_transcribe_name: str, model_diarize_name: str):
        use_cuda = os.getenv("USE_CUDA", "false").lower() == "true"
        print(f"User wants to set device cuda: {use_cuda}")
        if use_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Device is {self.device}")
        
        self.model_transcribe_name = model_transcribe_name
        self.model_diarize_name = model_diarize_name
        self.engine_name = "openai-whisper"
        self.engine_version = version("openai-whisper")
        self.model_transcribe = None
        self.model_diarize = None
        self.language = os.getenv("LANGUAGE")

    def _prepare_dirs(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(HF_CACHE_DIR, exist_ok=True)
        os.makedirs(DIARIZE_DIR, exist_ok=True)

        print(f"Models directory: {os.path.abspath(MODEL_DIR)}")
        print(f"HF cache directory: {os.path.abspath(HF_CACHE_DIR)}")
        print(f"Diarization directory: {os.path.abspath(DIARIZE_DIR)}")

    def _load_or_download_diarization_model(self):
        if os.path.exists(DIARIZE_FILE):
            print(f"Loading local diarization model from: {DIARIZE_FILE}")
            model = SortformerEncLabelModel.restore_from(DIARIZE_FILE)
            return model.eval()

        print("Local diarization model not found — downloading from the internet...")
        model = SortformerEncLabelModel.from_pretrained(self.model_diarize_name)

        print(f"Saving diarization model to: {DIARIZE_FILE}")
        model.save_to(DIARIZE_FILE)

        print("Reloading diarization model from local file...")
        model = SortformerEncLabelModel.restore_from(DIARIZE_FILE)
        return model.eval()

    def load_models(self):
        self._prepare_dirs()

        os.environ["HF_HOME"] = os.path.abspath(HF_CACHE_DIR)

        print("Loading Whisper model...")
        self.model_transcribe = whisper.load_model(
            self.model_transcribe_name,
            device=self.device,
            download_root=MODEL_DIR
        ).eval()

        print("Loading diarization model...")
        self.model_diarize = self._load_or_download_diarization_model()

        self.model_diarize.sortformer_modules.chunk_len = 340
        self.model_diarize.sortformer_modules.chunk_right_context = 40
        self.model_diarize.sortformer_modules.fifo_len = 40
        self.model_diarize.sortformer_modules.spkcache_update_period = 300

    def _find_segments(self, kept_ranges: List[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
        end_of_seg = 30_000
        seg = 0
        name_of_seg = f"seg_{seg}"
        dict_of_segments = {name_of_seg: []}

        for kept_range in kept_ranges:
            name_of_seg = f"seg_{seg}"

            if kept_range[1] < end_of_seg:
                dict_of_segments[name_of_seg].append(kept_range)
            else:
                if len(dict_of_segments[name_of_seg]) == 0:
                    del dict_of_segments[name_of_seg]

                end_of_seg = kept_range[0] + 30_000
                seg = seg + 1
                name_of_seg = f"seg_{seg}"
                dict_of_segments[name_of_seg] = []
                dict_of_segments[name_of_seg].append(kept_range)

        return dict_of_segments
    
    def _audiosegment_to_np(self, seg: AudioSegment) -> np.ndarray:
        seg = seg.set_channels(1).set_frame_rate(16000)
        samples = np.array(seg.get_array_of_samples())
        scale = float(1 << (8 * seg.sample_width - 1))  
        return samples.astype(np.float32) / scale
    
    def _cut_silence_intervals(
        self,
        audio_path: Union[str, Path],
        *,
        min_silence_len: int = 250,
        silence_thresh: int = -50,
        keep_silence: int = 500):
        
        audio_path = Path(audio_path)
        audio = AudioSegment.from_file(str(audio_path))
        dur = len(audio)  # ms

        nonsilent = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
        )

        if not nonsilent:
            segments_audio = []
            segments = {}
            return segments_audio, segments

        kept_ranges = []
        for s, e in nonsilent:
            s2 = max(0, s - keep_silence)
            e2 = min(dur, e + keep_silence)
            kept_ranges.append((s2, e2))

        segments = self._find_segments(kept_ranges)  
        print(segments)

        def _seg_key(name):
            try:
                return int(name.split("_")[1])
            except Exception:
                return 10**9
        
        segments_audio = []
        for seg_name in sorted(segments.keys(), key=_seg_key): 
            seg_audio = AudioSegment.empty()
            ranges_of_one_segment = segments[seg_name]
            
            start_of_segment = ranges_of_one_segment[0][0]
            end_of_segment = ranges_of_one_segment[len(ranges_of_one_segment)-1][1]

            seg_audio = audio[start_of_segment: end_of_segment]
            segments_audio.append(self._audiosegment_to_np(seg_audio))

        return segments_audio, segments
    
    def _find_speakers(self, path_to_audio: List[Union[str, Path]]):
        
        def _group_diar_segments(predicted_segments):
            speaker_dict = defaultdict(list)

            for seg in predicted_segments[0]:
                start, end, speaker = seg.split()

                start = float(start)
                end = float(end)

                speaker_dict[speaker].append((start, end))

            return [speaker_dict[s] for s in sorted(speaker_dict.keys())]

        result_from_all_channels = []
        
        for saved_file in path_to_audio:
            with torch.inference_mode():
                predicted_segments = self.model_diarize.diarize(audio=[saved_file], batch_size=1)
            result = _group_diar_segments(predicted_segments)
            result_from_all_channels.extend(result)

        return result_from_all_channels
    
    def _find_speaekr_per_word(self, whole_segments: List[Dict[str, Any]], speakers_ranges: List[List[Tuple[float, float]]], diarized: bool) -> List[Dict[str, Any]]:
        type_of_division = None
        if diarized:
            type_of_division = 'speaker'
        else:
            type_of_division = 'channel'
        
        for segment in whole_segments:

            for word in segment['words']:
                start = int(word['start'] * 1_000)  
                end = int(word['end'] * 1_000)

                to_subtract = 75 
                to_add = 250
                
                word_ms_range = set(list(range(end - to_subtract, end + to_add)))

                for i, speaker_range in enumerate(speakers_ranges):

                    sum_for_one_speaker = 0
                    for start_range, end_range in speaker_range:
                        if not isinstance(start_range, int) and not isinstance(end_range, int):
                            start_range = int(start_range * 1_000)
                            end_range = int(end_range * 1_000)

                        one_range_of_channel = set(list(range(start_range, end_range)))
                        n_of_common_ms = len(word_ms_range & one_range_of_channel)
                        sum_for_one_speaker += n_of_common_ms

                    word[f'{type_of_division}_{i}_score'] =  sum_for_one_speaker / (to_subtract + to_add)
                
            for i in range(len(segment['words']) - 1, -1, -1):
                word = segment['words'][i] 

                list_of_scores = []
                for j in range(0, len(speakers_ranges)):
                    list_of_scores.append(word[f'{type_of_division}_{j}_score'])

                if len(list_of_scores) > 1 and all(s == 0 for s in list_of_scores):
                    del segment['words'][i] 

            for i, word in enumerate(segment['words']): 

                list_of_scores = []
                for j in range(0, len(speakers_ranges)):
                    list_of_scores.append(word[f'{type_of_division}_{j}_score'])

                if len(list_of_scores) > 1 and len(set(list_of_scores)) == 1 and list_of_scores[0] != 0 and i != 0:
                    word[type_of_division] = segment['words'][i-1][type_of_division]
                else:
                    word[type_of_division] = list_of_scores.index(max(list_of_scores))

        return whole_segments


    def _generate_transcriptions_of_segments(self, segments_audio: List[np.ndarray], use_context: bool) -> List[Dict[str, Any]]:
        initial_prompt = ""
        results = []
        
        for i, segment_audio in enumerate(segments_audio):

            print(f"Generating text for the segment {i+1} z {len(segments_audio)}.")

            if np.all(segment_audio == 0):
                result = {
                    "text": "",
                    "segments": [
                        {
                            "id": 0,
                            "seek": 0,
                            "start": 0.0,
                            "end": 0.0,
                            "text": "",
                            "tokens": [],
                            "temperature": 0.0,
                            "avg_logprob": 0.0,
                            "compression_ratio": 0.0,
                            "no_speech_prob": 1.0,
                            "words": []
                        }
                    ],
                    "language": "nn"
                }

                results.append(result)
                initial_prompt = ""
                continue
            
            if initial_prompt != "":
                with torch.inference_mode():
                    result = self.model_transcribe.transcribe(segment_audio, word_timestamps=True, initial_prompt=initial_prompt, language=self.language, temperature=0)
                print("The segment was generated with a prompt")
                print(result['text'])
            else:
                with torch.inference_mode():
                    result = self.model_transcribe.transcribe(segment_audio, word_timestamps=True, language=self.language,  temperature=0)
                print(result['text'])
                print("The segment was generated WITHOUT a prompt!")

            results.append(result)

            if len(result['segments']) != 0:

                text = result["text"]
                clean_text = re.sub(r"[^\w\s]", "", text.lower())
                words = clean_text.split()
                word_counts = Counter(words)
                has_repetition = any(count >= 10 for count in word_counts.values())

                if result['segments'][0]["avg_logprob"] > -1 and result['segments'][0]["no_speech_prob"] < 0.5 and result['segments'][0]["compression_ratio"] < 2.0 and has_repetition is not True and use_context:
                    initial_prompt = result["text"]
                else:
                    initial_prompt = ""

                print(f'temperature: {result["segments"][0]["temperature"]}, avg_logprob: {result["segments"][0]["avg_logprob"]}, no_speech_prob: {result["segments"][0]["no_speech_prob"]}, compression_ratio: {result["segments"][0]["compression_ratio"]}')
            else:
                initial_prompt = ""

            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                collect()
            except Exception:
                pass

        return results
    
    def _fix_time_stamps(self, results: List[Dict[str, Any]], segments: Dict[str, List[Tuple[int, int]]]) -> List[Dict[str, Any]]:
        whole_segments = []

        for i, key in enumerate(list(segments.keys())):
            
            start_of_segment = segments[key][0][0]
            result = results[i]
            
            for segment in result['segments']:
                segment['start'] = float(segment['start'])
                segment['end'] = float(segment['end'])
                del segment['seek']

                min_time = []
                max_time = []
                    
                if len(segment['words']) == 0:
                    continue
                
                for word in segment['words']:
                    word['start'] = float(word['start']) + start_of_segment / 1_000
                    word['end'] = float(word['end']) + start_of_segment / 1_000
                    word['probability'] = float(word['probability'])

                    min_time.append(word['start'])
                    max_time.append(word['end'])

                segment['start'] = min(min_time)
                segment['end'] = max(max_time)
                segment['language'] = result['language']
                whole_segments.append(segment)
        
        return whole_segments
    
    def _filter_good_quality_segments(self, whole_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        for i in range(len(whole_segments) - 1, -1, -1):
            if whole_segments[i]['avg_logprob'] <= -1 or whole_segments[i]['compression_ratio'] >= 2.0:
                del whole_segments[i]

        return whole_segments
    
    def _fix_segments_id(self, whole_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        
        for i, segment in enumerate(whole_segments):
            segment['id'] = i

        return whole_segments
    
    def _generate_whole_text(self, whole_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        whole_text = ""

        for segment in whole_segments:
            whole_text = whole_text + segment['text']

        return whole_text

    def _generate_json(self, whole_segments: List[Dict[str, Any]], whole_text: str) -> Dict[str, Any]:
        final_dict = {"text": whole_text,
                      "segments": whole_segments}
        
        return final_dict

    def transribe(self, path_to_audio: Union[str, Path], use_context: bool = False, filter: bool = True):
        try:
            out_path, channels_ranges, audios_per_channels = silero_gate_each_channel_then_merge_mono(path_to_audio)
            segments_audio, segments = self._cut_silence_intervals(out_path)
            speakers_ranges = self._find_speakers(audios_per_channels)

            results = self._generate_transcriptions_of_segments(segments_audio, use_context)
            whole_segments = self._fix_time_stamps(results, segments)
            whole_segments = self._find_speaekr_per_word(whole_segments, speakers_ranges, True) # przydzielenie słów na podstawie diaryzacji
            whole_segments = self._find_speaekr_per_word(whole_segments, channels_ranges, False) # przydzielenie słów na podstawie kanałów
            
            if filter:
                whole_segments = self._filter_good_quality_segments(whole_segments)

            whole_segments = self._fix_segments_id(whole_segments)
            whole_text = self._generate_whole_text(whole_segments)
            final_result = self._generate_json(whole_segments, whole_text)

            return final_result 
        
        except Exception:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                collect()
            except Exception:
                pass
            
            raise