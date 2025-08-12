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
import wandb
import logging
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class CosSimLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and state.is_world_process_zero:
            model = kwargs["model"]
            if hasattr(model, "latest_cos_sim"):
                try:
                    sim = model.latest_cos_sim.item()
                    logger.info(f"[{state.global_step=}] Cosine sim: {sim:.4f}")
                    wandb.log(
                        {"gradient_cosine_similarity": sim}, step=state.global_step
                    )
                except Exception as e:
                    logger.error(f"Unable to log metric: {e}")


class KLDLossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and state.is_world_process_zero:
            model = kwargs["model"]
            if hasattr(model, "latest_kld_consistency_loss"):
                try:
                    loss = model.latest_kld_consistency_loss.item()
                    logger.info(
                        f"[{state.global_step=}] KLD consistency loss: {loss:.4f}"
                    )
                    wandb.log({"kld_consistency_loss": loss}, step=state.global_step)
                except Exception as e:
                    logger.error(f"Unable to log metric: {e}")


class AuxClfLossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and state.is_world_process_zero:
            model = kwargs["model"]
            if hasattr(model, "latest_aux_clf_loss"):
                try:
                    loss = model.latest_aux_clf_loss.item()
                    logger.info(
                        f"[{state.global_step=}] Aux classifier loss: {loss:.4f}"
                    )
                    wandb.log({"aux_clf_loss": loss}, step=state.global_step)
                except Exception as e:
                    logger.error(f"Unable to log metric: {e}")


class AuxRegLossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and state.is_world_process_zero:
            model = kwargs["model"]
            if hasattr(model, "latest_aux_reg_loss"):
                try:
                    loss = model.latest_aux_reg_loss.item()
                    logger.info(
                        f"[{state.global_step=}] Aux regression loss: {loss:.4f}"
                    )
                    wandb.log({"aux_reg_loss": loss}, step=state.global_step)
                except Exception as e:
                    logger.error(f"Unable to log metric: {e}")
