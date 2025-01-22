"""
Playing around with transformers in jax.

Following along this tutorial:
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html
"""
from jaxtyping import Float, Int, Bool, PyTree
import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange, einsum

type B = int
type N = int
type d_in = int
type h = int
type seq_len = int


def attention(
    x: Float[jax.Array, "N seq_len d_in"],
    w_query: Float[jax.Array, "d_in h"],
    w_key: Float[jax.Array, "d_in h"],
    w_value: Float[jax.Array, "d_in d_in"],
    mask: Bool[jax.Array, "seq_len seq_len"] | None = None,
) -> tuple[Float[jax.Array, "N d_in"], Float[jax.Array, "N N"]]:
    d_in = x.shape[1]
    q = x @ w_query
    k = x @ w_key
    v = x @ w_value

    attention_logits = q @ k.T / jnp.sqrt(d_in)
    attention_matrix = jax.nn.softmax(
        attention_logits, where=mask if mask is not None else None
    )
    y_i = attention_matrix @ v
    return y_i, attention_matrix


def multi_head_attention(
    x: Float[jax.Array, "N seq_len d_in"],
    w_query: Float[jax.Array, "d_in h"],
    w_key: Float[jax.Array, "d_in h"],
    w_value: Float[jax.Array, "d_in d_in"],
    num_heads: int,
    mask: Bool[jax.Array, "seq_len seq_len"] | None = None,
) -> tuple[Float[jax.Array, "N d_in"], Float[jax.Array, "N N"]]:
    d_in = x.shape[-1]
    q = x @ w_query
    k = x @ w_key
    v = x @ w_value

    q = q.reshape(q.shape[0], q.shape[1], num_heads, -1)
    k = k.reshape(k.shape[0], k.shape[1], num_heads, -1)
    v = v.reshape(v.shape[0], v.shape[1], num_heads, -1)

    attention_logits = q @ k.T / jnp.sqrt(d_in)
    attention_matrix = jax.nn.softmax(
        attention_logits, where=mask if mask is not None else None
    )
    y_i = attention_matrix @ v
    return y_i, attention_matrix


# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    assert (
        mask.ndim >= 2
    ), "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiheadAttention(nn.Module):
    embed_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads (h)

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional

        self.qkv_proj = nn.Dense(
            3 * self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
        )
        self.o_proj = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(
        self,
        x: Float[jax.Array, "B seq_len d_in"],
        mask: Bool[jax.Array, "seq_len seq_len"] | None = None,
    ):
        batch_size, seq_length, embed_dim = x.shape
        if mask is not None:
            mask = expand_mask(mask)
        # qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        # qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        # qkv = qkv.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        # q, k, v = jnp.array_split(qkv, 3, axis=-1)
        # Determine value outputs
        values, attention_matrix = attention(x, q, k, v, mask=mask)
        values = values.transpose(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)


def main():
    from datasets import load_dataset
    from transformers import PreTrainedTokenizerFast

    xs = [
        [0, 2, 4, 6, 8],
        [3, 5, 7, 9, 11],
        [1, 3, 5, 7, 9],
        [2, 4, 6, 8, 10],
    ]

    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    # tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")

    # seq_len, d_k = 3, 2
    # main_rng, rand1 = jax.random.split(main_rng)
    # qkv = jax.random.normal(rand1, (3, seq_len, d_k))
    # q, k, v = qkv[0], qkv[1], qkv[2]
    # values, attention = attention(q, k, v)

    main_rng, x_rng = jax.random.split(main_rng)
    x = jax.random.normal(x_rng, (3, 16, 128))
    # Create attention
    mh_attn = MultiheadAttention(embed_dim=128, num_heads=4)
    # Initialize parameters of attention with random key and inputs
    main_rng, init_rng = jax.random.split(main_rng)
    params = mh_attn.init(init_rng, x)["params"]
    # Apply attention with parameters on the inputs
    out, attn = mh_attn.apply({"params": params}, x)
    print("Out", out.shape, "Attention", attn.shape)


if __name__ == "__main__":
    main()
