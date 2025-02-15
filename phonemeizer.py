import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import soundfile as sf
from transformers import pipeline
from pydub import AudioSegment
import tempfile
import os
import subprocess
import tkinter as tk
from tkinter import filedialog
import random
import re
import traceback

class PhonemeAnalyzer:
    def __init__(self):
        self.colors = sns.color_palette('colorblind', n_colors=10)
        self.colors.extend(sns.color_palette('husl', n_colors=30))
        
        try:
            self.transcriber = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                chunk_length_s=30,
                return_timestamps=True,
                generate_kwargs={"language": "en"}  # Force English transcription
            )
        except Exception as e:
            print(f"Error initializing transcriber: {str(e)}")
            raise
    
    def select_file(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select MP3 File",
            filetypes=[("MP3 files", "*.mp3")]
        )
        return file_path
    
    def convert_to_wav(self, mp3_path):
        audio = AudioSegment.from_mp3(mp3_path)
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio.export(temp_wav.name, format='wav')
        return temp_wav.name
    
    def get_phonemes_espeak(self, text):
        try:
            cmd = ['espeak', '-q', '--ipa', '-v', 'en-us', text]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout.strip().split()
        except FileNotFoundError:
            print("Error: espeak not found. Please install it using: brew install espeak")
            return None
        
    def detect_speech_activity(self, y, sr):
        # Calculate RMS energy with finer resolution
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
        
        # Use slightly higher threshold (30th percentile)
        threshold = np.percentile(rms, 30) * 0.3
        speech_frames = rms > threshold
        times = librosa.times_like(rms, sr=sr, hop_length=128)
        
        speech_segments = []
        in_speech = False
        start_time = 0
        min_silence_duration = 0.1  # Minimum silence between segments
        last_speech_end = 0

        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                # Only start new segment if we've had enough silence
                if times[i] - last_speech_end >= min_silence_duration:
                    start_time = times[i]
                    in_speech = True
            elif not is_speech and in_speech:
                # Use shorter minimum duration (0.02s)
                if times[i] - start_time >= 0.02:
                    speech_segments.append((start_time, times[i]))
                    last_speech_end = times[i]
                in_speech = False

        if in_speech and times[-1] - start_time >= 0.02:
            speech_segments.append((start_time, times[-1]))

        print(f"Speech segments detected: {speech_segments}")  # Debugging

        return speech_segments
    
    def extract_words_from_audio(self, wav_path):
        y, sr = librosa.load(wav_path)
        self.audio_duration = librosa.get_duration(y=y, sr=sr)
        
        try:
            speech_segments = self.detect_speech_activity(y, sr)
            print("\nDEBUG: Speech detection complete")
            print(f"DEBUG: Found segments: {speech_segments}")
            
            if not speech_segments:
                print("No speech segments detected")
                return []
            
            # Get total speech duration
            total_speech_duration = sum(end - start for start, end in speech_segments)
            print(f"DEBUG: Total speech duration: {total_speech_duration}")
            
            result = self.transcriber(wav_path, return_timestamps=True)
            print("\nDEBUG: Transcription complete")
            print(f"DEBUG: Raw result: {result}")
            
            words = []
            word_segments = []
            
            if 'chunks' in result:
                for chunk_idx, chunk in enumerate(result['chunks']):
                    print(f"\nDEBUG: Processing chunk {chunk_idx}")
                    if not chunk['text'].strip():
                        print("DEBUG: Empty chunk, skipping")
                        continue
                        
                    phrase_text = chunk['text'].strip()
                    split_words = [w.strip('.,!?') for w in phrase_text.split()]
                    split_words = [w for w in split_words if w]
                    
                    print(f"DEBUG: Split words: {split_words}")
                    
                    if not split_words:
                        print("DEBUG: No valid words after splitting")
                        continue
                    
                    # Calculate time per word
                    time_per_word = total_speech_duration / len(split_words)
                    print(f"DEBUG: Time per word: {time_per_word}")
                    
                    # Process each word
                    for word_idx, word in enumerate(split_words):
                        print(f"\nDEBUG: Processing word {word_idx}: {word}")
                        
                        try:
                            # Assign word timings within detected speech segments
                            for segment_start, segment_end in speech_segments:
                                segment_duration = segment_end - segment_start
                                words_in_segment = [w for w in words if segment_start <= w['start'] < segment_end]

                                if not words_in_segment:
                                    continue  # Skip segments with no detected words

                                time_per_word = segment_duration / len(words_in_segment)

                                for i, word in enumerate(words_in_segment):
                                    word['start'] = segment_start + (i * time_per_word)
                                    word['end'] = min(word['start'] + time_per_word, segment_end)
                            
                            print(f"DEBUG: Calculated timings - Start: {word_start:.3f}, End: {word_end:.3f}")
                            
                            if word_end <= word_start:
                                print(f"DEBUG: Invalid timing for word {word}")
                                continue
                                
                            word_entry = {
                                "text": word,
                                "start": word_start,
                                "end": word_end
                            }
                            
                            words.append(word_entry)
                            word_segments.append((word_start, word_end))
                            print(f"Word: {word} | Start: {word_start:.3f}, End: {word_end:.3f}")
                            
                        except Exception as e:
                            print(f"DEBUG: Error processing word {word}: {str(e)}")
                            traceback.print_exc()
            
            print("\nDEBUG: Word processing complete")
            print(f"DEBUG: Total words processed: {len(words)}")
            
            self.word_segments = word_segments
            return words
            
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            traceback.print_exc()
            return []

    
    def extract_words_from_audio(self, wav_path):
        y, sr = librosa.load(wav_path)
        self.audio_duration = librosa.get_duration(y=y, sr=sr)
        
        speech_segments = self.detect_speech_activity(y, sr)
        if not speech_segments:
            print("No speech segments detected")
            return []
        
        print(f"DEBUG: Speech segments found: {speech_segments}")
        words = []
        
        try:
            result = self.transcriber(wav_path, return_timestamps=True)
            print("Raw transcriber result:", result)
            
            if 'chunks' in result:
                for chunk in result['chunks']:
                    phrase_text = chunk['text'].strip()
                    split_words = phrase_text.split()
                    print(f"\nProcessing words: {split_words}")
                    
                    if not split_words:
                        continue
                    
                    # Get segment timing
                    segment_start, segment_end = speech_segments[0]
                    segment_duration = segment_end - segment_start
                    
                    # Calculate word timings
                    word_duration = segment_duration / len(split_words)
                    
                    # Process each word
                    for i, word in enumerate(split_words):
                        try:
                            # Calculate word timing
                            word_start = segment_start + (i * word_duration)
                            word_end = word_start + word_duration
                            
                            # Ensure we don't exceed segment boundary
                            if word_end > segment_end:
                                word_end = segment_end
                            
                            # Create word entry
                            word_entry = {
                                "text": word.strip('.,!?'),
                                "start": word_start,
                                "end": word_end
                            }
                            
                            words.append(word_entry)
                            print(f"Word: {word} | Start: {word_start:.3f}, End: {word_end:.3f}")
                            
                        except Exception as e:
                            print(f"Error processing individual word: {str(e)}")
                            continue
            
            return words
            
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            traceback.print_exc()
            return words  # Return any words we've managed to process


    
    def process_audio(self, mode="#both"):
        try:
            mp3_path = self.select_file()
            if not mp3_path:
                return

            print("\nDEBUG: Starting audio processing")
            wav_path = self.convert_to_wav(mp3_path)
            y, sr = librosa.load(wav_path)

            print("DEBUG: Extracting words")
            words = self.extract_words_from_audio(wav_path)
            print(f"DEBUG: Extracted {len(words)} words: {words}")

            word_segments = [(word['start'], word['end']) for word in words]
            print(f"DEBUG: Generated word segments: {word_segments}")

            print("DEBUG: Getting speech segments")
            self.speech_segments = self.detect_speech_activity(y, sr)
            print(f"DEBUG: Speech segments: {self.speech_segments}")

            print("DEBUG: Generating phoneme timings")
            phonemes = self.create_phoneme_timings(words)
            print(f"DEBUG: Generated {len(phonemes)} phonemes")

            print("DEBUG: Creating visualization")
            plot = self.create_visualization(y, sr, phonemes, word_segments, words, mode)

            if plot is None:
                print("ERROR: Visualization failed, returning early.")
                return

            plot.show()
            os.unlink(wav_path)

        except Exception as e:
            print(f"ERROR in process_audio: {str(e)}")
            traceback.print_exc()





    def create_phoneme_timings(self, words):
        print("\nDEBUG: Starting phoneme timing creation")
        phonemes = []
        
        for i, word in enumerate(words):
            try:
                print(f"\nDEBUG: Processing word {i}: {word}")
                
                word_text = word["text"].strip('.,!?')
                if not word_text:
                    print(f"DEBUG: Empty word text after stripping punctuation")
                    continue
                    
                start = word["start"]
                end = word["end"]
                print(f"DEBUG: Word timing - Start: {start}, End: {end}")
                
                print(f"DEBUG: Getting phonemes for word: {word_text}")
                phoneme_string = self.get_phonemes_espeak(word_text)
                print(f"DEBUG: Raw phoneme string: {phoneme_string}")
                
                if phoneme_string is None:
                    print(f"DEBUG: No phoneme string returned for {word_text}")
                    continue
                    
                if not phoneme_string:
                    print(f"DEBUG: Empty phoneme string for {word_text}")
                    continue
                
                if not isinstance(phoneme_string, list):
                    print(f"DEBUG: Unexpected phoneme_string type: {type(phoneme_string)}")
                    phoneme_string = [str(phoneme_string)]
                
                # Use regex to extract phonemes
                # Use regex to extract complete IPA symbols
                phoneme_list = []
                for ps in phoneme_string:
                    # Pattern to match common English IPA symbols including digraphs
                    ipa_pattern = r'tʃ|dʒ|eɪ|aɪ|ɔɪ|aʊ|oʊ|θ|ð|ʃ|ʒ|ŋ|[ɑɔæʌəɛɪʊɜː]|[ptkmn]|[bdgw]|[lfvs]|[rjh]|[ˈˌ]'
                    matches = re.findall(ipa_pattern, str(ps))
                    phoneme_list.extend([m for m in matches if m.strip()])
                
                print(f"DEBUG: Extracted phoneme list: {phoneme_list}")
                
                if not phoneme_list:
                    print(f"DEBUG: No valid phonemes found for {word_text}")
                    continue
                
                duration = end - start
                if duration <= 0:
                    print(f"DEBUG: Invalid duration for {word_text}: {duration}")
                    continue
                    
                phoneme_duration = duration / len(phoneme_list)
                print(f"DEBUG: Phoneme duration: {phoneme_duration}")
                
                for j, phoneme in enumerate(phoneme_list):
                    phoneme_start = start + (j * phoneme_duration)
                    phoneme_end = phoneme_start + phoneme_duration
                    
                    # Ensure we don't exceed word boundaries
                    if phoneme_end > end:
                        phoneme_end = end
                        
                    phoneme_entry = {
                        "phoneme": phoneme,
                        "start": phoneme_start,
                        "end": phoneme_end,
                        "word": word_text
                    }
                    phonemes.append(phoneme_entry)
                    print(f"DEBUG: Added phoneme: {phoneme} ({phoneme_start:.3f} - {phoneme_end:.3f})")
                    
            except Exception as e:
                print(f"ERROR processing word {word.get('text', '')}: {str(e)}")
                traceback.print_exc()
                continue
        
        print(f"\nDEBUG: Completed phoneme timing creation. Total phonemes: {len(phonemes)}")
        return phonemes




    def create_visualization(self, y, sr, phonemes, word_segments, words, mode="#both"):
        plt.figure(figsize=(12, 6))

        # Ensure words is correctly formatted
        if not isinstance(words, list) or not all(isinstance(w, dict) and 'text' in w for w in words):
            print("ERROR: 'words' is not in the expected format. Debugging output:", words)
            return

        # Get total duration of the waveform in seconds
        total_duration = librosa.get_duration(y=y, sr=sr)

        # Generate time axis for waveform
        time_axis = np.linspace(0, total_duration, len(y))
        plt.plot(time_axis, y, alpha=0.6, label="Waveform")

        if mode in ["#word", "#both"]:
            # Apply word overlays & labels
            for word_start, word_end in word_segments:
                for segment_start, segment_end in self.speech_segments:
                    if segment_start <= word_start <= segment_end:
                        start = word_start / total_duration
                        end = word_end / total_duration
                        color = random.choice(self.colors)

                        # Word overlay (full height)
                        plt.gca().add_patch(
                            Rectangle((start * total_duration, -1),  
                                    (end - start) * total_duration,
                                    2, color=color, alpha=0.3)
                        )

                        # Get the actual word text safely
                        word_text = next((w.get('text', "UNKNOWN") for w in words 
                                        if isinstance(w, dict) and abs(w.get('start', -99) - word_start) < 0.01), 
                                        "UNKNOWN")

                        # Word label centered inside the patch
                        plt.text((start + end) / 2 * total_duration, 1.25, 
                                word_text, fontsize=10, ha='center', va='center', color='black')

                        if mode == "#both":
                            # Thinner dashed extension lines for word boundaries
                            plt.plot([start * total_duration, start * total_duration], 
                                    [-1, 1.0], color='black', linestyle='dashed', linewidth=0.5)
                            plt.plot([end * total_duration, end * total_duration], 
                                    [-1, 1.0], color='black', linestyle='dashed', linewidth=0.5)

                        break  # Stop checking once a valid speech segment is found

        if mode in ["#phoneme", "#both"]:
            # Apply phoneme overlays & labels
            for phoneme in phonemes:
                for segment_start, segment_end in self.speech_segments:
                    if segment_start <= phoneme["start"] <= segment_end:
                        start = phoneme["start"] / total_duration
                        end = phoneme["end"] / total_duration
                        color = random.choice(self.colors)

                        # Phoneme overlay (full height)
                        plt.gca().add_patch(
                            Rectangle((start * total_duration, -1),  
                                    (end - start) * total_duration,
                                    2, color=color, alpha=0.5)
                        )

                        # IPA phoneme label centered ABOVE the patch
                        plt.text((start + end) / 2 * total_duration, 1.05,  # Raised to 1.3
                                phoneme["phoneme"], fontsize=9, ha='center', va='center', color='black')
                        break  # Stop checking once a valid speech segment is found

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.ylim(-1.2, 1.5)  # Increase space above waveform
        plt.grid(True)
        plt.legend()
        return plt






if __name__ == "__main__":
    analyzer = PhonemeAnalyzer()
    analyzer.process_audio(mode="#phonemex")
