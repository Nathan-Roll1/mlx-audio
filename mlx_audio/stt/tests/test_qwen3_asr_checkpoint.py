"""Weight-format regression tests using a small model with the real architecture."""

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_audio.stt.models.qwen3_asr.config import ModelConfig
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel


class TestQwen3ASRCheckpoint(unittest.TestCase):
    def test_native_transformers_strict_load_and_layouts(self):
        config = ModelConfig.from_dict(
            {
                "audio_config": {
                    "d_model": 8,
                    "encoder_attention_heads": 2,
                    "encoder_layers": 1,
                    "encoder_ffn_dim": 16,
                    "downsample_hidden_size": 4,
                    "output_dim": 8,
                    "max_position_embeddings": 13,
                },
                "text_config": {
                    "vocab_size": 16,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "rope_parameters": {"rope_type": "default", "rope_theta": 8192},
                },
            }
        )
        self.assertEqual(config.audio_config.max_source_positions, 13)
        self.assertEqual(config.text_config.rope_theta, 8192)
        original = dict(tree_flatten(Qwen3ASRModel(config).parameters()))
        native, thinker = {}, {}
        for name, array in original.items():
            if "conv2d" in name and name.endswith(".weight"):
                array = array.transpose(0, 3, 1, 2)
            thinker["thinker." + name] = array
            if name.startswith("audio_tower.proj"):
                key = "model.multi_modal_projector.linear_" + name.removeprefix(
                    "audio_tower.proj"
                )
            elif name.startswith("model."):
                key = "model.language_model." + name.removeprefix("model.")
            else:
                key = "model." + name
            native[key] = array

        for state in (native, thinker, original):
            mapped = Qwen3ASRModel.sanitize(state)
            target = Qwen3ASRModel(config)
            target.load_weights(list(mapped.items()), strict=True)
            actual = dict(tree_flatten(target.parameters()))
            self.assertEqual(actual.keys(), original.keys())
            for name in original:
                self.assertTrue(
                    mx.array_equal(actual[name], original[name]).item(), name
                )
