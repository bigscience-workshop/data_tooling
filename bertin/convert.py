#!/usr/bin/env python
import tempfile

import jax
from jax import numpy as jnp
from transformers import AutoTokenizer, FlaxRobertaForMaskedLM, RobertaForMaskedLM


def to_f32(t):
    return jax.tree_map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t
    )


def main():
    # Saving extra files from config.json and tokenizer.json files
    tokenizer = AutoTokenizer.from_pretrained("./")
    tokenizer.save_pretrained("./")

    # Temporary saving bfloat16 Flax model into float32
    tmp = tempfile.mkdtemp()
    flax_model = FlaxRobertaForMaskedLM.from_pretrained("./")
    flax_model.params = to_f32(flax_model.params)
    flax_model.save_pretrained(tmp)
    # Converting float32 Flax to PyTorch
    model = RobertaForMaskedLM.from_pretrained(tmp, from_flax=True)
    model.save_pretrained("./", save_config=False)


if __name__ == "__main__":
    main()
