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
import evaluate

INSTRUCTION = {
    "asr": "Transcribe the audio clip into text. ",
    "asr_intelligibility": "Transcribe the audio clip into text, and then assess its intelligibility. ",
    "asr_naturalness": "Transcribe the audio clip into text, and then assess its naturalness. ",
    "asr_etiology": "Transcribe the audio clip into text, and then assess its etiology. ",
    "asr_category": "Transcribe the audio clip into text, and then assess its speech category. ",
    "intelligibility": "Assess the intelligibility of the audio clip. ",
    "etiology": "Assess the etiology of the audio clip. ",
    "naturalness": "Assess the naturalness of the audio clip. ",
    "category": "Assess the speech category of the audio clip. ",
    "consonants": "Assess the consonant precision of the audio clip. ",
    "asr_consonants": "Transcribe the audio clip into text, and then assess its consonant precision. ",
    "stress": "Assess the stress of the audio clip. ",
    "asr_stress": "Transcribe the audio clip into text, and then assess the stress of the speaker's voice. ",
    "harsh": "Assess the harshness of the audio clip. ",
    "asr_harsh": "Transcribe the audio clip into text, and then assess the harshness of the speaker's voice. ",
    "monoloudness": "Assess the monoloudness of the speaker's voice. ",
    "asr_monoloudness": "Transcribe the audio clip into text, and then assess the monoloudness of the speaker's voice. ",
}

ZERO_SHOT_INSTRUCTION = {
    "asr": "Based on the attached audio, generate a comprehensive text transcription of the spoken content. ",
    "intelligibility": "Assess the intelligibility of the attached audio on an integer scale between 1 and 7. Return only your assessment i.e. a number between 1 and 7 without any other text. ",
    "etiology": "Assess the etiology of the attached audio on an integer scale between 1 and 7. Return only your assessment i.e. a number between 1 and 7 without any other text. ",
    "naturalness": "Assess the naturalness of the attached audio on an integer scale between 1 and 7. Return only your assessment i.e. a number between 1 and 7 without any other text. ",
    "category": "Assess the type of speech of the attached audio on an integer scale between 1 and 7. Return only your assessment i.e. a number between 1 and 7 without any other text. ",
    "consonants": "Assess the consonant precision of the attached audio on an integer scale between 1 and 7. Return only your assessment i.e. a number between 1 and 7 without any other text. ",
    "stress": "Assess the stress of the attached audio on an integer scale between 1 and 7. Return only your assessment i.e. a number between 1 and 7 without any other text. ",
    "harsh": "Assess the harshness of the voice in the attached audio on an integer scale between 1 and 7. Return only your assessment i.e. a number between 1 and 7 without any other text. ",
    "monoloudness": "Assess the monoloudness of the speaker's voice in the attached audio on an integer scale between 1 and 7. Return only your assessment i.e. a number between 1 and 7 without any other text. ",
}

AUX_REG_COLS = [
    "intelligbility",
    "naturalness",
    "imprecise_consonants",
    "monoloudness",
    "harsh_voice",
]

TASK2CLFCOL_MAP = {
    "asr": "words",
    "asr_intelligibility": ("intelligbility", "intelligibility"),
    "asr_naturalness": "naturalness",
    "asr_etiology": "etiology",
    "asr_category": "category",
    "intelligibility": ("intelligbility", "intelligibility"),
    "etiology": "etiology",
    "naturalness": "naturalness",
    "category": "category",
    "consonants": "imprecise_consonants",
    "asr_consonants": "imprecise_consonants",
    "stress": "reduced_stress",
    "asr_stress": "reduced_stress",
    "harsh": "harsh_voice",
    "asr_harsh": "harsh_voice",
    "monoloudness": "monoloudness",
    "asr_monoloudness": "monoloudness",
}

CLF_TASK_WEIGHTS = {
    "asr": 1.0,
    "asr_intelligibility": 1.0,
    "asr_naturalness": 1.0,
    "asr_etiology": 1.0,
    "asr_category": 1.0,
    "asr_harsh": 1.0,
    "asr_stress": 1.0,
    "asr_consonants": 1.0,
    "asr_monoloudness": 1.0,
    "intelligibility": 1.0,
    "etiology": 1.0,
    "naturalness": 1.0,
    "category": 1.0,
    "consonants": 1.0,
    "stress": 1.0,
    "harsh": 1.0,
    "monoloudness": 1.0,
}

CLF_TASK2NUMCLASSES_MAP = {
    "asr": 2,
    "asr_intelligibility": 7,
    "asr_naturalness": 7,
    "asr_etiology": 5,
    "asr_category": 4,
    "asr_harsh": 7,
    "asr_stress": 7,
    "asr_consonants": 7,
    "asr_monoloudness": 7,
    "intelligibility": 7,
    "etiology": 5,
    "naturalness": 7,
    "category": 4,
    "consonants": 7,
    "stress": 7,
    "harsh": 7,
    "monoloudness": 7,
}

ETIOLOGY_MAP = {
    "ALS": 0,
    "Cerebral Palsy": 1,
    "Down Syndrome": 2,
    "Parkinson's Disease": 3,
    "Stroke": 4,
}

CATEGORY_MAP = {
    "Novel Sentences": 0,
    "Spontaneous Speech Prompts": 1,
    "Digital Assistant Commands": 2,
    "Non-spontaneous Speech Prompts": 3,
}

MULTI_SEARCH_PATTERNS = [r"<\|intell\|>+", r"<\|nat\|>+", r"<\|etio\|>+", r"<\|cat\|>+"]

ANSWER_SUFFIX = "<|end|><|endoftext|>"
CLF_SPECIAL_TOKEN = "<|clf|>"
WER_METRIC = evaluate.load("wer")
_IGNORE_INDEX = -100

COLUMNS_TO_KEEP = [
    "utt_id",
    "wav",
    "intelligbility",
    "intelligibility",
    "naturalness",
    "words",
    "etiology",
    "category",
    "imprecise_consonants",
    "reduced_stress",
    "harsh_voice",
    "monoloudness",
    "reg_embed",
]
