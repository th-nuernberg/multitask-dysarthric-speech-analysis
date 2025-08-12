# Copyright 2025 Technische Hochschule Nürnberg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Author: Dominik Wagner, Technische Hochschule Nürnberg
import io
import logging

import numpy as np
import librosa
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


def extract_speech_segments(audio_input, sr, vad_segments, offset_ms=20):
    """
    Extract speech segments from an audio array based on VAD results.

    Parameters:
        audio_input (numpy.ndarray): The audio signal.
        sr (int): The sample rate of the audio.
        vad_segments (torch.Tensor): A tensor with start and end times of speech segments (in seconds).
        offset_ms (float): Offset in milliseconds to extend each segment.

    Returns:
        numpy.ndarray: A NumPy array containing only the speech segments.
    """
    offset_samples = int((offset_ms / 1000) * sr)
    speech_segments = []
    for start, end in vad_segments.cpu().numpy():
        start_sample = max(0, int((start * sr) - offset_samples))
        end_sample = min(len(audio_input), int((end * sr) + offset_samples))
        speech_segments.append(audio_input[start_sample:end_sample])
    return np.concatenate(speech_segments) if speech_segments else np.array([])


def load_vad_model(root_path: str):
    from speechbrain.inference.VAD import VAD

    logger.info("Loading VAD model...")
    vad_model = VAD.from_hparams(
        source="speechbrain/vad-crdnn-libriparty",
        savedir=f"{root_path}/pretrained_models/vad-crdnn-libriparty",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    return vad_model


def run_vad(audio_file: str, vad_model, target_sr=16_000):
    boundaries = vad_model.get_speech_segments(audio_file).cpu().detach()
    waveform, sample_rate = sf.read(audio_file)
    if target_sr is not None and sample_rate != target_sr:
        waveform = librosa.resample(
            waveform.T, orig_sr=sample_rate, target_sr=target_sr
        )
        waveform = librosa.to_mono(waveform) if len(waveform.shape) > 1 else waveform
    waveform = extract_speech_segments(waveform, sample_rate, boundaries, offset_ms=20)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    wav_bytes = buffer.getvalue()
    buffer.close()
    return wav_bytes, waveform.shape[0]
