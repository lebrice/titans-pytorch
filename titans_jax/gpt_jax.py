"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping

import chex
import jax
import jax.nn
import jax.numpy as jnp
import optax
import torch

# from typeguard import typechecked as type_checker
from beartype import beartype as type_checker
from einops import pack, rearrange
from flax import nnx
from jaxtyping import Float, Int, jaxtyped

type seq_len = int
type hidden_dim = int
type batch_size = int


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


@jaxtyped(typechecker=type_checker)
class CausalSelfAttention(nnx.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        bias: bool,
        dropout: float,
        block_size: int,
        kernel_init: jax.nn.initializers.Initializer,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nnx.Linear(
            n_embd, 3 * n_embd, use_bias=bias, kernel_init=kernel_init, rngs=rngs
        )
        # output projection
        self.c_proj = nnx.Linear(
            n_embd, n_embd, use_bias=bias, kernel_init=kernel_init, rngs=rngs
        )
        # regularization
        self.attn_dropout = nnx.Dropout(dropout, rngs=rngs)
        self.resid_dropout = nnx.Dropout(dropout, rngs=rngs)
        self.n_heads = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.block_size = block_size
        self.mask = nnx.Variable(
            jnp.tril(
                jnp.ones((self.block_size, self.block_size), dtype=jnp.bool)
            ).reshape(1, 1, self.block_size, self.block_size)
        )

    def __call__(
        self, x: Float[jax.Array, "B seq_length {self.n_embd}"]
    ) -> Float[jax.Array, "B seq_length {self.n_embd}"]:
        (
            B,
            T,
            C,
        ) = (
            x.shape
        )  # batch size, sequence length (T), embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        combined_out = self.c_attn(x)
        # NOTE: differs from torch where we provide the size of each split, here we give the number of splits.
        q, k, v = jnp.split(combined_out, 3, axis=2)

        split_heads = functools.partial(
            rearrange,
            pattern="B seq_len (n_heads hs) -> B n_heads seq_len hs",
            n_heads=self.n_heads,
        )

        k = split_heads(k)  # (B, nh, T, hs)
        q = split_heads(q)  # (B, nh, T, hs)
        v = split_heads(v)  # (B, nh, T, hs)
        hs = k.shape[-1]
        scale = 1.0 / math.sqrt(hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # y = jax.nn.dot_product_attention(
        #     q, k, v, mask=None, is_causal=True, # todo: doesn't seem to have a dropout parameter.
        # )
        k_T = rearrange(k, "B N T H -> B N H T")
        att = (q @ k_T) * scale
        # NOTE: the `where` argument of softmax already does the masking we want.
        # att = jnp.where(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = jax.nn.softmax(att, axis=-1, where=self.mask[:, :, :T, :T])
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = rearrange(y, "B N T H -> B T (N H)")

        # efficient attention using Flash Attention CUDA kernels
        # y = torch.nn.functional.scaled_dot_product_attention(
        #     q,
        #     k,
        #     v,
        #     attn_mask=None,
        #     dropout_p=self.dropout if self.training else 0,
        #     is_causal=True,
        # )
        # else:
        #     # manual implementation of attention
        #     att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #     att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        #     att = F.softmax(att, dim=-1)
        #     att = self.attn_dropout(att)
        #     y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = (
        #     y.transpose(1, 2).contiguous().view(B, T, C)
        # )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


@jaxtyped(typechecker=type_checker)
class MLP(nnx.Module):
    def __init__(
        self,
        n_embd: int,
        bias: bool,
        dropout: float,
        linear_weight_init: jax.nn.initializers.Initializer,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.c_fc = nnx.Linear(
            n_embd, 4 * n_embd, use_bias=bias, kernel_init=linear_weight_init, rngs=rngs
        )
        self.gelu = nnx.gelu
        self.c_proj = nnx.Linear(
            4 * n_embd, n_embd, use_bias=bias, kernel_init=linear_weight_init, rngs=rngs
        )
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self, x: Float[jax.Array, "B seq_len {self.n_embd}"]
    ) -> Float[jax.Array, "B seq_len {self.n_embd}"]:
        x = self.c_fc(x)  # [B seq_len n_embd*4]
        x = self.gelu(x)
        x = self.c_proj(x)  # [B seq_len n_embd]
        x = self.dropout(x)
        return x


@jaxtyped(typechecker=type_checker)
class Block(nnx.Module):
    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        bias: bool,
        dropout: float,
        block_size: int,
        rngs: nnx.Rngs,
        linear_weight_init: jax.nn.initializers.Initializer,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.ln_1 = nnx.LayerNorm(num_features=n_embd, use_bias=bias, rngs=rngs)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_heads,
            bias=bias,
            dropout=dropout,
            block_size=block_size,
            rngs=rngs,
            kernel_init=linear_weight_init,
        )
        self.ln_2 = nnx.LayerNorm(num_features=n_embd, use_bias=bias, rngs=rngs)
        self.mlp = MLP(
            n_embd=n_embd,
            bias=bias,
            dropout=dropout,
            linear_weight_init=linear_weight_init,
            rngs=rngs,
        )

    def __call__(
        self, x: Float[jax.Array, "B seq_len {self.n_embd}"]
    ) -> Float[jax.Array, "B seq_len {self.n_embd}"]:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@jaxtyped(typechecker=type_checker)
class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer_wte = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.n_embd,
            embedding_init=jax.nn.initializers.normal(stddev=0.02),
            rngs=rngs,
        )
        self.transformer_wpe = nnx.Embed(
            num_embeddings=config.block_size,
            features=config.n_embd,
            embedding_init=jax.nn.initializers.normal(stddev=0.02),
            rngs=rngs,
        )
        self.transformer_drop = nnx.Dropout(config.dropout, rngs=rngs)
        self.transformer_h = [
            Block(
                block_size=config.block_size,
                n_heads=config.n_head,
                n_embd=config.n_embd,
                dropout=config.dropout,
                bias=config.bias,
                rngs=rngs,
                linear_weight_init=jax.nn.initializers.normal(stddev=0.02),
            )
            for _ in range(config.n_layer)
        ]
        self.transformer_ln_f = nnx.LayerNorm(
            config.n_embd, use_bias=config.bias, rngs=rngs
        )

        self.lm_head = nnx.Linear(
            config.n_embd,
            config.vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=0.02),
            rngs=rngs,
        )
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer_wte.embedding = (
        #     self.lm_head.kernel.T  # note: extra transpose which is different from torch.
        # )  # https://paperswithcode.com/method/weight-tying

        # todo: init all weights
        # self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith("c_proj.weight"):
        #         torch.nn.init.normal_(
        #             p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
        #         )

        # report number of parameters

    def parameters(self) -> Iterable[nnx.Param[jax.Array]]:
        yield from self.state_dict().values()

    def named_parameters(self) -> Iterable[tuple[str, nnx.Param]]:
        yield from self.state_dict().items()

    def state_dict(self) -> dict[str, nnx.Param]:
        return {
            ".".join(map(str, keys)): v
            for keys, v in nnx.variables(self, nnx.Param).flat_state().items()
            if v.value is not None
        }  # type: ignore

    def get_num_params(self, non_embedding=True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        # nnx.display(params)

        n_params = sum(p.size for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer_wpe.embedding.size
        assert isinstance(n_params, int)
        return n_params

    @jaxtyped(typechecker=type_checker)
    def __call__(
        self,
        idx: Int[jax.Array, "B seq_len"],
        targets: Int[jax.Array, "B seq_len"] | None = None,
    ):
        b, t = idx.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = jnp.arange(0, t, dtype=int)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer_wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer_wpe(pos)  # position embeddings of shape (t, n_embd)
        chex.assert_shape(tok_emb, (b, t, self.config.n_embd))
        chex.assert_shape(pos_emb, (t, self.config.n_embd))
        x = self.transformer_drop(
            tok_emb + rearrange(pos_emb, "T n_embd -> 1 T n_embd")
        )
        for block in self.transformer_h:
            x = block(x)
        x = self.transformer_ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            targets = rearrange(targets, "B seq_len -> (B seq_len)")
            loss = optax.softmax_cross_entropy_with_integer_labels(
                rearrange(logits, "B seq_len vocab_size -> (B seq_len) vocab_size"),
                targets,
                # targets.reshape(-1),
                # where=targets != -1,  # TODO: Re-enable this!
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer_wpe.embedding = self.transformer_wpe.embedding[:block_size]
        for block in self.transformer_h:
            if block.attn.mask is not None:
                block.attn.mask = block.attn.mask[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(
        cls,
        model_type: Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        override_args: dict[Literal["dropout"], float] | None = None,
    ):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args: dict[str, Any] = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config, rngs=nnx.Rngs(0))

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        # todo: Copy the parameters from torch to jax somehow
        # model_def, params = nnx.split(model)

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type,
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        optimizer = nnx.Optimizer(
            self,
            optax.adamw(
                learning_rate, b1=betas[0], b2=betas[1], weight_decay=weight_decay
            ),
        )

        # Create AdamW optimizer and use the fused version if it is available
        # fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        # use_fused = fused_available and device_type == "cuda"
        # extra_args = dict(fused=True) if use_fused else dict()
        # optimizer = torch.optim.AdamW(
        #     optim_groups, lr=learning_rate, betas=betas, **extra_args
        # )
        # print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: float, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    # @torch.no_grad()
    def generate(
        self,
        idx: Int[jax.Array, "B seq_len"],
        max_new_tokens: int,
        key: chex.PRNGKey,
        temperature: float = 1.0,
        top_k=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        # todo: use jax.lax.while_loop to make this more efficient and jit-able, as done here:
        # https://github.com/google/flax/blob/main/examples/lm1b/temperature_sampler.py#L27

        for i in range(max_new_tokens):
            key_i = jax.random.fold_in(key, i)
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.shape[1] <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
                # v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = jnp.where(logits < v[:, [-1]], -jnp.inf, logits)
                # logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            # probs = jax.nn.softmax(logits, axis=-1)
            # sample from the distribution
            idx_next = jax.random.categorical(key=key_i, logits=logits)
            # idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx, _ = pack((idx, idx_next), "B *")
            # idx = jnp.concatenate((idx, idx_next), axis=1)

        return idx


type NestedDict[K, V] = dict[K, V | NestedDict[K, V]]
type NestedMapping[K, V] = Mapping[K, V | NestedMapping[K, V]]


def flatten[V](p: NestedMapping[str, V]) -> dict[str, V]:
    return dict(flatten_iter(p))


def flatten_iter[V](
    p: NestedDict[str, V], label: str | None = None
) -> Iterable[tuple[str, V]]:
    if isinstance(p, dict):
        for k, v in p.items():
            yield from flatten_iter(v, k if label is None else f"{label}.{k}")
    else:
        yield (label, p)


def chexify[C: Callable | nnx.Module](obj: C) -> C:
    return chex.chexify(obj)  # type: ignore


def main():
    gpt = GPT(GPTConfig(), rngs=nnx.Rngs(0))
    print(f"number of parameters: {gpt.get_num_params() / 1e6:.2f}M")
    # gpt = nnx.jit(gpt)
    # gpt = chexify(gpt)
    x = jax.random.randint(jax.random.key(0), (1, 16), 0, gpt.config.vocab_size)
    logits, loss = jax.jit(gpt)(x[..., :-1], targets=x[..., 1:])
    assert loss is not None
    print(logits)
    print(loss)
    print(f"{loss.shape=}")
    print(f"{logits.shape=}")
    result = gpt.generate(x, 10, key=jax.random.key(0))
    chex.block_until_chexify_assertions_complete()
    print(f"{x=}")
    print(f"{result=}")


if __name__ == "__main__":
    main()
