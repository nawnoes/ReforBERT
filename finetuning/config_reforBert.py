# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" reforBERT model configuration """
import logging
from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

REFORBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "reforBert-base": "./config/reforBert-config.json",
}

class ReforBertConfig(PretrainedConfig):
    pretrained_config_archive_map = REFORBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "reforbert"

    def __init__(
        self,
        num_labels = 2,
        vocab_size=8007,  # vocab 크기
        embedding_size=768,   # 임베딩 사이즈
        max_seq_len = 512,  # 최대 입력 길이
        depth = 12,  # reformer depth
        heads = 8,  # reformer heads
        device = 'cpu',
        causal=True,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.heads = heads
        self.causal = causal
        self.num_labels = num_labels
        self.device = device
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

