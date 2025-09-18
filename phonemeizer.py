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
            self.word_transcriber = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                return_timestamps="word",
                generate_kwargs={"language": "en"}
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
            # Use a unique separator for phonemes
            separator = "_"
            cmd = ['espeak', '-q', '--ipa', f'--sep={separator}', '-v', 'en-us', text]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Split the output by the separator
            phonemes = result.stdout.strip().split(separator)
            # Remove any empty strings that may result from the split
            return [p for p in phonemes if p]

        except FileNotFoundError:
            print("Error: espeak not found. Please install it using: sudo apt-get install espeak")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error executing espeak: {e}")
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
            # Although we detect speech activity, we'll rely on Whisper's timestamps primarily.
            # This detection can be used for other purposes, like filtering out non-speech audio if needed.
            self.speech_segments = self.detect_speech_activity(y, sr)
            print("\nDEBUG: Speech detection complete")
            print(f"DEBUG: Found segments: {self.speech_segments}")

            if not self.speech_segments:
                print("No speech segments detected")
                return []

            result = self.word_transcriber(wav_path)
            print("\nDEBUG: Transcription complete")
            print(f"DEBUG: Raw result: {result}")

            words = []
            if 'chunks' in result:
                for chunk in result['chunks']:
                    word_text = chunk['text'].strip('.,!?')
                    if not word_text:
                        continue
                    
                    timestamp = chunk['timestamp']
                    word_entry = {
                        "text": word_text,
                        "start": timestamp[0],
                        "end": timestamp[1]
                    }
                    
                    if word_entry["end"] is None:
                        print(f"DEBUG: Skipping word with null end time: {word_entry['text']}")
                        continue

                    words.append(word_entry)
                    print(f"Word: {word_entry['text']} | Start: {word_entry['start']:.3f}, End: {word_entry['end']:.3f}")
            
            print("\nDEBUG: Word processing complete")
            print(f"DEBUG: Total words processed: {len(words)}")
            
            # For visualization purposes
            self.word_segments = [(w['start'], w['end']) for w in words]
            return words
            
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            traceback.print_exc()
            return []


    
    def process_audio(self, mp3_path=None, mode="#both"):
        try:
            if mp3_path is None:
                mp3_path = self.select_file()
                if not mp3_path:
                    return

            print("\nDEBUG: Starting audio processing")
            wav_path = self.convert_to_wav(mp3_path)
            y, sr = librosa.load(wav_path)

            print("DEBUG: Extracting words")
            words = self.extract_words_from_audio(wav_path)
            print(f"DEBUG: Extracted {len(words)} words: {words}")

            word_segments = self.word_segments
            print(f"DEBUG: Generated word segments: {word_segments}")

            print("DEBUG: Generating phoneme timings")
            phonemes = self.create_phoneme_timings(words)
            print(f"DEBUG: Generated {len(phonemes)} phonemes")

            print("DEBUG: Creating visualization")
            plot = self.create_visualization(y, sr, phonemes, word_segments, words, mode)

            if plot is None:
                print("ERROR: Visualization failed, returning early.")
                return

            output_path = "output.png"
            plot.savefig(output_path)
            print(f"Visualization saved to {output_path}")
            plot.close()  # Close the plot to free memory
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
                
                phoneme_list = phoneme_string
                print(f"DEBUG: Phoneme list: {phoneme_list}")
                
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

        # Generate and plot the spectrogram
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')

        y_min, y_max = plt.gca().get_ylim()
        rect_height = y_max - y_min

        if mode in ["#word", "#both"]:
            # Apply word overlays & labels
            for word_start, word_end in word_segments:
                color = random.choice(self.colors)

                # Word overlay (full height)
                plt.gca().add_patch(
                    Rectangle((word_start, y_min),
                            word_end - word_start,
                            rect_height, color=color, alpha=0.3)
                )

                # Get the actual word text safely
                word_text = next((w.get('text', "UNKNOWN") for w in words
                                if isinstance(w, dict) and abs(w.get('start', -99) - word_start) < 0.01),
                                "UNKNOWN")

                # Word label centered above the patch
                plt.text((word_start + word_end) / 2, y_max * 1.3,
                        word_text, fontsize=10, ha='center', va='center', color='black')

                if mode == "#both":
                    # Thinner dashed extension lines for word boundaries
                    plt.plot([word_start, word_start],
                            [y_min, y_max], color='black', linestyle='dashed', linewidth=0.5)
                    plt.plot([word_end, word_end],
                            [y_min, y_max], color='black', linestyle='dashed', linewidth=0.5)

        if mode in ["#phoneme", "#both"]:
            # Apply phoneme overlays & labels
            for phoneme in phonemes:
                color = random.choice(self.colors)

                # Phoneme overlay (full height)
                plt.gca().add_patch(
                    Rectangle((phoneme["start"], y_min),
                            phoneme["end"] - phoneme["start"],
                            rect_height, color=color, alpha=0.5)
                )

                # IPA phoneme label centered ABOVE the patch
                plt.text((phoneme["start"] + phoneme["end"]) / 2, y_max * 1.1,
                        phoneme["phoneme"], fontsize=9, ha='center', va='center', color='black')

        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(y_min, y_max * 1.4)
        plt.grid(True)
        return plt






if __name__ == "__main__":
    analyzer = PhonemeAnalyzer()
    analyzer.process_audio(mp3_path="ladies-make.mp3", mode="#both")
