import io
import os
import time
from pydub import AudioSegment
import librosa
import numpy as np

# Audio constraints
TARGET_SAMPLE_RATE = 16000
CHANNELS = 1
# Enrollment: 15s  |  Login: 10s  |  Minimum accepted: 3s
MIN_DURATION_SECONDS = 3.0

def convert_audio_to_wav(input_path: str, output_path: str) -> bool:
    """
    Converts any incoming webm/mp4/m4a audio to 16kHz Mono WAV.
    """
    try:
        audio = AudioSegment.from_file(input_path)
        # Force 16kHz and mono
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(CHANNELS)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        return False

def convert_audio_bytes_to_wav_bytes(audio_bytes: bytes) -> bytes:
    """
    Converts audio bytes to 16kHz Mono WAV bytes natively in memory without ffmpeg subprocess.
    """
    try:
        import soundfile as sf
        import librosa
        
        # Parse the raw bytes into a soundfile object
        # The frontend sends webm (Opus), so soundfile needs pysoundfile / libsndfile with OGG/Opus support.
        # If it fails, we fall back to pydub as a slow safe path
        input_stream = io.BytesIO(audio_bytes)
        
        try:
            y, sr = sf.read(input_stream)
            # Ensure Mono by averaging channels if needed
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
                
            # Resample strictly to 16kHz
            if sr != TARGET_SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
                
            # Write out to new wav bytes
            output_stream = io.BytesIO()
            sf.write(output_stream, y, TARGET_SAMPLE_RATE, format='WAV', subtype='PCM_16')
            return output_stream.getvalue()
            
        except Exception as e:
            # Fallback to slow Pydub if soundfile lacks codec support on this OS
            input_stream.seek(0)
            from pydub import AudioSegment
            audio = AudioSegment.from_file(input_stream)
            audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(CHANNELS)
            
            output_stream = io.BytesIO()
            audio.export(output_stream, format="wav")
            return output_stream.getvalue()
            
    except Exception as e:
        print(f"Error converting audio bytes: {str(e)}")
        return b""

def check_liveness_heuristic(audio_path: str) -> dict:
    """
    Basic heuristic-based liveness/replay detection from file.
    """
    try:
        from scipy.io import wavfile
        sr, y = wavfile.read(audio_path)
        return _compute_liveness(sr, y)
    except Exception as e:
        print(f"Liveness check error: {str(e)}")
        return {"is_live": False, "score": 0.0, "reason": "Audio processing failed."}

def check_liveness_heuristic_bytes(wav_bytes: bytes) -> dict:
    """
    Enhanced heuristic-based liveness/replay detection from memory.
    """
    try:
        import soundfile as sf
        stream = io.BytesIO(wav_bytes)
        y, sr = sf.read(stream)
        return _compute_liveness(sr, y)
    except Exception as e:
        print(f"Liveness check error (bytes): {str(e)}")
        return {"is_live": False, "score": 0.0, "reason": "Audio processing failed."}

def _compute_liveness(sr, y) -> dict:
    # Convert to float and Mono
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
        
    if len(y.shape) > 1:
         y = np.mean(y, axis=1)
        
    # Check duration - Replay attacks are often hastily cut clips
    duration = len(y) / sr
    if duration < MIN_DURATION_SECONDS:
        return {"is_live": False, "score": 0.0, "reason": f"Audio too short ({duration:.1f}s). Need {MIN_DURATION_SECONDS}s minimum."}
        
    # Volume check (Prevent silent or barely audible clips)
    rms_volume = np.sqrt(np.mean(y**2))
    if rms_volume < 0.01:
        return {"is_live": False, "score": 0.0, "reason": "Audio is too quiet/silent."}
    
    # Calculate Zero Crossing Rate purely in NumPy (extremely fast)
    zcr = np.mean(np.abs(np.diff(np.sign(y)))) / 2.0
    
    # Simple SNR estimation
    signal_power = np.mean(y ** 2)
    noise_power = np.var(y)
    snr = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))
    
    is_live = True
    reason = "Pass"
    
    # Heuristics for spoofing/replay (strict ranges typical for human speech on laptop microhone)
    if zcr < 0.01 or zcr > 0.40:
        is_live = False
        reason = f"Unnatural voice frequency patterns (ZCR: {zcr:.4f})"
        
    # Replay attacks (from speakers) usually warp the noise floor. 
    # Exact SNR bounds may require tuning per mic, so keep it somewhat generous.
    if snr < -20.0 or snr > 50.0:
        is_live = False
        reason = f"Abnormal signal-to-noise ratio: {snr:.1f} dB"

    # Score calculation (1.0 is highest)
    # Penalize extremely high ZCR (excessive hiss) or extremely low (rumble)
    penalty = abs(zcr - 0.15) * 2.0 
    score = max(0.0, min(1.0, float(1.0 - penalty))) 
    
    return {
        "is_live": is_live,
        "score": score,
        "reason": reason,
        "metrics": {
            "zcr": float(zcr),
            "snr": float(snr),
            "rms": float(rms_volume)
        }
    }
