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
from dataclasses import dataclass
from typing import Optional, Dict

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CustomCausalLMOutputWithPast(CausalLMOutputWithPast):
    clf_logits: Optional[Dict[str, torch.FloatTensor]] = None
