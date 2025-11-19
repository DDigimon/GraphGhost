from collections import defaultdict
from contextlib import contextmanager
from functools import partial
import token
from typing import Callable, Dict, List, Optional, Tuple, Union, Literal, NamedTuple

import torch
from torch import nn
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
import json
import os

from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder, load_trained_transcoder, load_transcoder_set
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens.utils import (
    USE_DEFAULT_VALUE,
    init_kaiming_normal_,
    init_kaiming_uniform_,
    init_xavier_normal_,
    init_xavier_uniform_,
    
)
import transformer_lens.utils as utils
import logging
from jaxtyping import Float

SingleLoss = Float[torch.Tensor, ""]  # Type alias for a single element tensor
LossPerToken = Float[torch.Tensor, "batch pos-1"]
Loss = Union[SingleLoss, LossPerToken]


def model_name_func(name):
    if name == 'gemma-2B':
        return 'google/gemma-2-2b'
    if name == 'gemma-1B':
        return 'google/gemma-3-1b'
    if name == 'Qwen-0.5B':
        return "Qwen/Qwen2.5-0.5B"
    if name == 'Qwen3-0.6B':
        return "Qwen/Qwen3-0.6B"
    if name == 'Qwen-1.5B':
        return "Qwen/Qwen2.5-1.5B"
    if name == 'Llama-1B':
        return 'meta-llama/Llama-3.2-1B'
    if name == 'Llama-3B':
        return 'meta-llama/Llama-3.2-3B'
    if name == 'Llama-8B':
        return 'meta-llama/Llama-3.1-8B'
    if name == 'Llama-8BI':
        return 'meta-llama/Llama-3.1-8B-Instruct'
    if name == 'ds-qwen-1.5B':
        return 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    if name == 'ds-llama':
        return 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

class ReplacementMLP(nn.Module):
    """Wrapper for a TransformerLens MLP layer that adds in extra hooks"""

    def __init__(self, old_mlp: nn.Module):
        super().__init__()
        self.old_mlp = old_mlp
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, x):
        # print(x.shape)
        x = self.hook_in(x)
        mlp_out = self.old_mlp(x)
        return self.hook_out(mlp_out)


class ReplacementUnembed(nn.Module):
    """Wrapper for a TransformerLens Unembed layer that adds in extra hooks"""

    def __init__(self, old_unembed: nn.Module):
        super().__init__()
        self.old_unembed = old_unembed
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    @property
    def W_U(self):
        return self.old_unembed.W_U

    @property
    def b_U(self):
        return self.old_unembed.b_U

    def forward(self, x):
        x = self.hook_pre(x)
        x = self.old_unembed(x)
        return self.hook_post(x)


class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"hello": 0, "world": 1, "!": 2, "<unk>": 3}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in text.lower().split()]
    
    def decode(self, tokens):
        return " ".join([self.inv_vocab.get(tok, "<unk>") for tok in tokens])
    
    def vocab_size(self):
        return len(self.vocab)
    
class Output(NamedTuple):
    """Output Named Tuple.

    Named tuple object for if we want to output both logits and loss.
    """

    logits: Float[torch.Tensor, "batch pos d_vocab"]
    loss: Loss


class ReplacementModel(HookedTransformer):
    d_transcoder: int
    transcoders: nn.ModuleList
    feature_input_hook: str
    feature_output_hook: str
    skip_transcoder: bool
    scan: Optional[Union[str, List[str]]]
    # main_device: torch.device = torch.device("cuda:2")

    @classmethod
    def from_config(
        cls,
        config: HookedTransformerConfig,
        transcoders: Dict[int, SingleLayerTranscoder],
        feature_input_hook: str = "mlp.hook_in",
        feature_output_hook: str = "mlp.hook_out",
        scan: Optional[str] = None,
        **kwargs,
    ) -> "ReplacementModel":
        """Create a ReplacementModel from a given HookedTransformerConfig and dict of transcoders

        Args:
            config (HookedTransformerConfig): the config of the HookedTransformer that this
                ReplacmentModel will inherit from
            transcoders (Dict[int, nn.Module]): A dict that maps from layer -> Transcoder
            feature_input_hook (str, optional): The hookpoint of the model that transcoders
                hook into. Defaults to "mlp.hook_in".

        Returns:
            ReplacementModel: The loaded ReplacementModel
        """
        model = cls(config, **kwargs)
        model._configure_replacement_model(
            transcoders, feature_input_hook, feature_output_hook, scan
        )
        return model


    @classmethod
    def from_pretrained_and_transcoders(
        cls,
        model_name: str,
        transcoders: Dict[int, SingleLayerTranscoder],
        feature_input_hook: str = "mlp.hook_in",
        feature_output_hook: str = "mlp.hook_out",
        scan: str = None,
        **kwargs,
    ) -> "ReplacementModel":
        """Create a ReplacementModel from the name of HookedTransformer and dict of transcoders

        Args:
            model_name (str): the name of the pretrained HookedTransformer that this
                ReplacmentModel will inherit from
            transcoders (Dict[int, nn.Module]): A dict that maps from layer -> Transcoder
            feature_input_hook (str, optional): The hookpoint of the model that transcoders
                hook into for inputs. Defaults to "mlp.hook_in".
            feature_output_hook (str, optional): The hookpoint of the model that transcoders
                hook into for outputs. Defaults to "mlp.hook_out".

        Returns:
            ReplacementModel: The loaded ReplacementModel
        """
        model = super().from_pretrained(
            model_name,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            **kwargs,
        )
        model._configure_replacement_model(
            transcoders, feature_input_hook, feature_output_hook, scan
        )
        return model
    

        
    @classmethod
    def from_self_pretrained_and_transcoders(
        cls,
        
        model_path: str,
        model_name: str,
        transcoders_path: str,
        cfg: HookedTransformerConfig = None,
        base_cfg: HookedTransformerConfig = None,
        feature_input_hook: str = "mlp.hook_in",
        feature_output_hook: str = "mlp.hook_out",
        scan: str = None,
        **kwargs,
    ) -> "ReplacementModel":
        if 'sync' not in model_name:
            name = model_name
            model_name = model_name_func(name)
            if 'ds' in name:
                tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    trust_remote_code=True,
                )
                
                if 'qwen-1.5B' in name:
                    backbone_model = "Qwen/Qwen2.5-1.5B"    
                if 'llama' in name:
                    backbone_model = "meta-llama/Llama-3.1-8B"  
                model = super().from_pretrained(backbone_model, hf_model=hf_model, tokenizer=tok, dtype = torch.bfloat16, cache_dir = model_path, move_to_device = False)
            else:
                model = super().from_pretrained(model_name, dtype = torch.bfloat16, cache_dir = model_path, move_to_device = False)
        else:
            model = ReplacementModel(base_cfg).to('cpu')
            cls.config = cfg
            state_dict = torch.load(model_path, map_location='cpu')  
            model.load_state_dict(state_dict)
        cfg.n_layers = model.cfg.n_layers
        transcoder_sets = load_trained_transcoder(configs=cfg, transcoder_config_file=transcoders_path)

        model._configure_replacement_model(
            transcoder_sets, feature_input_hook, feature_output_hook, scan
        )

        return model


    @classmethod
    def from_self_defined_models(
        cls,
        model_name: str,
        transcoder_path: str,
        device: str,
        config: Optional[HookedTransformerConfig] = None,
        dtype: Optional[torch.dtype] = torch.float32,
        **kwargs,
    ) -> "ReplacementModel":
        transcoders, feature_input_hook, feature_output_hook, scan = load_transcoder_set(
            transcoder_path, device=device, dtype=dtype
        )

        return cls.from_pretrained_and_transcoders(
            model_name,
            transcoders,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        transcoder_set: str,
        device: Optional[torch.device] = torch.device("cuda"),
        dtype: Optional[torch.dtype] = torch.float32,
        **kwargs,
    ) -> "ReplacementModel":
        """Create a ReplacementModel from the name of HookedTransformer and dict of transcoders

        Args:
            model_name (str): the name of the pretrained HookedTransformer that this
                ReplacmentModel will inherit from
            transcoder_set (str): Either a predefined transcoder set name, or a config file
                defining where to load them from
            device (torch.device, Optional): the device onto which to load the transcoders
                and HookedTransformer.

        Returns:
            ReplacementModel: The loaded ReplacementModel
        """
        transcoders, feature_input_hook, feature_output_hook, scan = load_transcoder_set(
            transcoder_set, device=device, dtype=dtype
        )

        return cls.from_pretrained_and_transcoders(
            model_name,
            transcoders,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            device=device,
            dtype=dtype,
            **kwargs,
        )
    
    
    ### newly added
    def set_device_map(self, device_map: Dict[int, torch.device]):

        assert len(device_map) == self.cfg.n_layers, "device_map need equal to layer num"
        n_layers = self.cfg.n_layers
        self.device_map = dict(device_map)
        first_dev = self.device_map[0]
        last_dev  = self.device_map[n_layers - 1]

        for attr in ["embed", "pos_embed"]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                getattr(self, attr).to(first_dev)
                self.cfg.device = first_dev
        if hasattr(self, "ln_final") and self.ln_final is not None:
            self.ln_final.to(last_dev)
        if hasattr(self, "unembed") and self.unembed is not None:
            self.unembed.to(last_dev)

        for i, block in enumerate(self.blocks):
            block.to(self.device_map[i])

        if hasattr(self, "transcoders"):
            for i, tr in enumerate(self.transcoders):
                tr.to(self.device_map[i])




    def forward_sharded(
        self,
        input,
        return_type: str | None = "logits",
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: torch.Tensor | None = None,
        shortformer_pos_embed: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,  # [batch, pos]
        stop_at_layer: int | None = None,
        past_kv_cache: "HookedTransformerKeyValueCache | None" = None,
    ):
        def _device_for_layer(layer_idx: int, fallback: torch.device):
            # if hasattr(self, "device_map") and self.device_map is not None:
            return self.device_map[layer_idx]
            # try:
            #     return devices.get_device_for_block_index(layer_idx, self.cfg)
            # except Exception:
            #     return fallback

        def _module_device(m, default: torch.device):
            try:
                p = next(m.parameters())
                return p.device
            except StopIteration:
                return default
            except Exception:
                return default

        emb_dev = next(self.embed.parameters()).device 

        with utils.LocallyOverridenDefaults(
            self, 
            prepend_bos=prepend_bos, 
            padding_side=padding_side,
        ):
            if start_at_layer is None:
                if isinstance(input, torch.Tensor) and input.dtype in (torch.int64, torch.int32, torch.int16):
                    # 索引规则允许 CPU 索引 -> 安全
                    if input.device.type != "cpu":
                        input = input.cpu()
                residual, tokens, shortformer_pos_embed, attention_mask = self.input_to_embed(
                    input,
                    prepend_bos=prepend_bos,
                    padding_side=padding_side,
                    attention_mask=attention_mask,
                    past_kv_cache=past_kv_cache,
                )
                start_at_layer = 0
            else:
                assert isinstance(input, torch.Tensor), "When start_at_layer is set, input must be residual Tensor"
                residual = input
            blocks_and_idxs = list(zip(range(self.cfg.n_layers), self.blocks))
            for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:
                dev_i = _device_for_layer(i, residual.device)
                # print(f"    dev_i:{dev_i}")
                if residual.device != dev_i:
                    residual = residual.to(dev_i, non_blocking=True)
                if shortformer_pos_embed is not None and shortformer_pos_embed.device != dev_i:
                    shortformer_pos_embed = shortformer_pos_embed.to(dev_i, non_blocking=True)

                pkv_entry = past_kv_cache[i] if past_kv_cache is not None else None

                residual = block(
                    residual,
                    past_kv_cache_entry=pkv_entry,
                    shortformer_pos_embed=shortformer_pos_embed,
                    attention_mask=attention_mask,
                )  # [batch, pos, d_model]
            if stop_at_layer is not None:
                return residual
            if self.cfg.normalization_type is not None:
                if hasattr(self, "ln_final") and self.ln_final is not None:
                    dev_ln = _module_device(self.ln_final, residual.device)
                    if residual.device != dev_ln:
                        residual = residual.to(dev_ln, non_blocking=True)
                    residual = self.ln_final(residual)

            if return_type is None:
                return None

            dev_unembed = _module_device(self.unembed, residual.device)
            if residual.device != dev_unembed:
                residual = residual.to(dev_unembed, non_blocking=True)
            logits = self.unembed(residual)  # [batch, pos, d_vocab]

            if getattr(self.cfg, "output_logits_soft_cap", 0.0) and self.cfg.output_logits_soft_cap > 0.0:
                logits = self.cfg.output_logits_soft_cap * torch.tanh(
                    logits / self.cfg.output_logits_soft_cap
                )

            if return_type == "logits":
                return logits

            assert tokens is not None, "tokens must be passed if return_type is 'loss' or 'both'"
            loss = self.loss_fn(logits, tokens, attention_mask, per_token=loss_per_token)

            if return_type == "loss":
                return loss
            elif return_type == "both":
                try:
                    return Output(logits, loss)
                except NameError:
                    return (logits, loss)
            else:
                logging.warning(f"Invalid return_type passed in: {return_type}")
                return None    
            
    def _configure_replacement_model(
        self,
        transcoders: Dict[int, SingleLayerTranscoder],
        feature_input_hook: str,
        feature_output_hook: str,
        scan: Optional[Union[str, List[str]]],
    ):
        for transcoder in transcoders.values():
            # print(self.cfg.device)
            # transcoder.to(self.cfg.device, self.cfg.dtype) # newly deleted
            transcoder.to(dtype=self.cfg.dtype)
            # for p in transcoder.parameters():
            #     p.data = p.data.to(dtype=self.cfg.dtype)
            # transcoder = torch.nn.utils.to_empty(transcoder, device=self.cfg.device, dtype=self.cfg.dtype)
        # print(self.cfg.n_layers, len(transcoders))
        # self.cfg.n_layers = len(transcoders)
        self.add_module(
            "transcoders",
            nn.ModuleList([transcoders[i] for i in range(self.cfg.n_layers)]),
        )
        self.d_transcoder = transcoder.d_transcoder
        self.feature_input_hook = feature_input_hook
        self.original_feature_output_hook = feature_output_hook
        self.feature_output_hook = feature_output_hook + ".hook_out_grad"
        self.skip_transcoder = transcoder.W_skip is not None
        self.scan = scan

        for block in self.blocks:
            block.mlp = ReplacementMLP(block.mlp)

        self.unembed = ReplacementUnembed(self.unembed)

        self._configure_gradient_flow()
        # self._deduplicate_attention_buffers()s
        self.setup()

    def _configure_gradient_flow(self):
        for layer, transcoder in enumerate(self.transcoders):
            self._configure_skip_connection(self.blocks[layer], transcoder)

        def stop_gradient(acts, hook):
            return acts.detach()
            # return acts

        for block in self.blocks:
            block.attn.hook_pattern.add_hook(stop_gradient, is_permanent=True)
            block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)
            block.ln2.hook_scale.add_hook(stop_gradient, is_permanent=True)
            if hasattr(block, "ln1_post"):
                block.ln1_post.hook_scale.add_hook(stop_gradient, is_permanent=True)
            if hasattr(block, "ln2_post"):
                block.ln2_post.hook_scale.add_hook(stop_gradient, is_permanent=True)
            self.ln_final.hook_scale.add_hook(stop_gradient, is_permanent=True)

        for param in self.parameters():
            param.requires_grad = False

        def enable_gradient(acts, hook):
            acts.requires_grad = True
            return acts

        self.hook_embed.add_hook(enable_gradient, is_permanent=True)

    def _configure_skip_connection(self, block, transcoder):
        cached = {}

        def cache_activations(acts, hook):
            cached["acts"] = acts

        def add_skip_connection(acts: torch.Tensor, hook: HookPoint, grad_hook: HookPoint):
            # We add grad_hook because we need a way to hook into the gradients of the output
            # of this function. If we put the backwards hook here at hook, the grads will be 0
            # because we detached acts.
            skip_input_activation = cached.pop("acts")
            if transcoder.W_skip is not None:
                skip = transcoder.compute_skip(skip_input_activation)
            else:
                skip = skip_input_activation * 0
            return grad_hook(skip + (acts - skip).detach())

        # add feature input hook
        output_hook_parts = self.feature_input_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.add_hook(cache_activations, is_permanent=True)

        # add feature output hook and special grad hook
        output_hook_parts = self.original_feature_output_hook.split(".")
        subblock = block
        for part in output_hook_parts:
            subblock = getattr(subblock, part)
        subblock.hook_out_grad = HookPoint()
        subblock.add_hook(
            partial(add_skip_connection, grad_hook=subblock.hook_out_grad),
            is_permanent=True,
        )

    def _deduplicate_attention_buffers(self):
        """
        Share attention buffers across layers to save memory.

        TransformerLens makes separate copies of the same masks and RoPE
        embeddings for each layer - This just keeps one copy
        of each and shares it across all layers.
        """

        attn_masks = {}

        for block in self.blocks:
            attn_masks[block.attn.attn_type] = block.attn.mask
            attn_masks["rotary_sin"] = block.attn.rotary_sin
            attn_masks["rotary_cos"] = block.attn.rotary_cos

        for block in self.blocks:
            block.attn.mask = attn_masks[block.attn.attn_type]
            block.attn.rotary_sin = attn_masks["rotary_sin"]
            block.attn.rotary_cos = attn_masks["rotary_cos"]

    def _get_activation_caching_hooks(
        self,
        zero_bos: bool = False,
        sparse: bool = False,
        apply_activation_function: bool = True,
    ) -> Tuple[List, List[Tuple[str, Callable]]]:
        activation_matrix = [None] * self.cfg.n_layers

        def cache_activations(acts, hook, layer, zero_bos):
            # print(self.transcoders[layer]
            #     .encode(acts, apply_activation_function=apply_activation_function))
            transcoder_acts = (
                self.transcoders[layer]
                .encode(acts, apply_activation_function=apply_activation_function)
                .detach()
                .squeeze(0)
            )
            
            
            if zero_bos:
                transcoder_acts[0] = 0
            # newly added
            # if sparse:
            #     activation_matrix[layer] = transcoder_acts.to_sparse()
            # else:
            #     activation_matrix[layer] = transcoder_acts
            transcoder_acts = transcoder_acts.to('cpu')
            activation_matrix[layer] = transcoder_acts.to_sparse() if sparse else transcoder_acts
            
        activation_hooks = [
            (
                f"blocks.{layer}.{self.feature_input_hook}",
                partial(cache_activations, layer=layer, zero_bos=zero_bos),
            )
            for layer in range(self.cfg.n_layers)
        ]
        return activation_matrix, activation_hooks

    def get_activations(
        self,
        inputs: Union[str, torch.Tensor],
        sparse: bool = False,
        zero_bos: bool = False,
        apply_activation_function: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the transcoder activations for a given prompt

        Args:
            inputs (Union[str, torch.Tensor]): The inputs you want to get activations over
            sparse (bool, optional): Whether to return a sparse tensor of activations.
                Useful if d_transcoder is large. Defaults to False.
            zero_bos (bool, optional): Whether to zero out activations / errors at the 0th
                position (<BOS>). Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the model logits on the inputs and the
                associated activation cache
        """

        activation_cache, activation_hooks = self._get_activation_caching_hooks(
            sparse=sparse,
            zero_bos=zero_bos,
            apply_activation_function=apply_activation_function,
        )
        # print(inputs)
        with torch.inference_mode(), self.hooks(activation_hooks):
            logits = self(inputs)
        activation_cache = torch.stack(activation_cache)
        if sparse:
            activation_cache = activation_cache.coalesce()
        return logits, activation_cache

    @contextmanager
    def zero_softcap(self):
        current_softcap = self.cfg.output_logits_soft_cap
        try:
            self.cfg.output_logits_soft_cap = 0.0
            yield
        finally:
            self.cfg.output_logits_soft_cap = current_softcap

    @torch.no_grad()
    def setup_attribution(
        self,
        inputs: Union[str, torch.Tensor],
        sparse: bool = False,
        zero_bos: bool = True,
    ):
        """Precomputes the transcoder activations and error vectors, saving them and the
        token embeddings.

        Args:
            inputs (str): the inputs to attribute - hard coded to be a single string (no
                batching) for now
            sparse (bool): whether to return activations as a sparse tensor or not
            zero_bos (bool): whether to zero out the activations and error vectors at the
                bos position
        """

        if isinstance(inputs, torch.Tensor):
            tokens = inputs.squeeze(0)
            assert tokens.ndim == 1, "Tokens must be a 1D tensor"
        else:
            assert isinstance(inputs, str), "Inputs must be a string"
            # tokenized = self.tokenizer(inputs, return_tensors="pt").input_ids.to(self.cfg.device)
            tokenized = self.tokenizer(inputs, return_tensors="pt").input_ids.to(self.main_device)

            tokens = tokenized.squeeze(0)

        special_tokens = []
        
        for special_token in self.tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                special_tokens.extend(special_token)
            else:
                special_tokens.append(special_token)

        special_token_ids = self.tokenizer.convert_tokens_to_ids(special_tokens)
        zero_bos = (
            zero_bos and tokens[0].cpu().item() in special_token_ids
        )  # == self.tokenizer.bos_token_id

        # cache activations and MLP in
        activation_matrix, activation_hooks = self._get_activation_caching_hooks(
            sparse=sparse, zero_bos=zero_bos
        )
        mlp_in_cache, mlp_in_caching_hooks, _ = self.get_caching_hooks(
            lambda name: self.feature_input_hook in name
        )

        error_vectors = torch.zeros(
            [self.cfg.n_layers, len(tokens), self.cfg.d_model],
            # device=self.cfg.device,
            device = "cpu",
            dtype=self.cfg.dtype,
        )

        fvu_values = torch.zeros(
            [self.cfg.n_layers, len(tokens)],
            # device=self.cfg.device,
            device = "cpu",
            dtype=torch.float32,
        )

        # hook into MLP out to compute errors
        def compute_error_hook(acts, hook, layer):
            in_hook = f"blocks.{layer}.{self.feature_input_hook}"
            reconstruction = self.transcoders[layer](mlp_in_cache[in_hook])[0]
            error = acts - reconstruction
            # error_vectors[layer] = error
            # total_variance = (acts - acts.mean(dim=-2, keepdim=True)).pow(2).sum(dim=-1)
            # fvu_values[layer] = error.pow(2).sum(dim=-1) / total_variance
            
            err_cpu = error.to('cpu')
            error_vectors[layer] = err_cpu
            total_variance = (acts - acts.mean(dim=-2, keepdim=True)).pow(2).sum(dim=-1)
            fvu_values[layer] = (error.pow(2).sum(dim=-1) / total_variance).to('cpu')


        error_hooks = [
            (f"blocks.{layer}.{self.feature_output_hook}", partial(compute_error_hook, layer=layer))
            for layer in range(self.cfg.n_layers)
        ]

        # note: activation_hooks must come before error_hooks
        logits = self.run_with_hooks(
            tokens, fwd_hooks=activation_hooks + mlp_in_caching_hooks + error_hooks
        )

        if zero_bos:
            error_vectors[:, 0] = 0

        activation_matrix = torch.stack(activation_matrix)
        if sparse:
            activation_matrix = activation_matrix.coalesce()

        token_vectors = self.W_E[tokens].detach()  # (n_pos, d_model)
        return logits, activation_matrix, error_vectors, token_vectors

    def setup_intervention_with_freeze(
        self, inputs: Union[str, torch.Tensor], direct_effects: bool = False
    ) -> List[Tuple[str, Callable]]:
        """Sets up an intervention with either frozen attention (default) or frozen
        attention, LayerNorm, and MLPs, for direct effects

        Args:
            inputs (Union[str, torch.Tensor]): The inputs to intervene on
            direct_effects (bool, optional): Whether to freeze not just attention, but also
                LayerNorm and MLPs. Defaults to False.

        Returns:
            List[Tuple[str, Callable]]: The freeze hooks needed to run the desired intervention.
        """

        if direct_effects:
            hookpoints_to_freeze = ["hook_pattern", "hook_scale", self.feature_output_hook]
            if self.skip_transcoder:
                hookpoints_to_freeze.append(self.feature_input_hook)
        else:
            hookpoints_to_freeze = ["hook_pattern"]

        freeze_cache, cache_hooks, _ = self.get_caching_hooks(
            names_filter=lambda name: any(hookpoint in name for hookpoint in hookpoints_to_freeze)
        )
        self.run_with_hooks(inputs, fwd_hooks=cache_hooks)

        def freeze_hook(activations, hook):
            cached_values = freeze_cache[hook.name]

            # if we're doing open-ended generation, the position dimensions won't match
            # so we'll just freeze the previous positions, and leave the new ones unfrozen
            if "hook_pattern" in hook.name and activations.shape[2:] != cached_values.shape[2:]:
                new_activations = activations.clone()
                new_activations[:, :, : cached_values.shape[2], : cached_values.shape[3]] = (
                    cached_values
                )
                return new_activations

            elif (
                "hook_scale" in hook.name or self.feature_output_hook in hook.name
            ) and activations.shape[1] != cached_values.shape[1]:
                new_activations = activations.clone()
                new_activations[:, : cached_values.shape[1]] = cached_values
                return new_activations

            # if other positions don't match, that's no good
            assert activations.shape == cached_values.shape, (
                f"Activations shape {activations.shape} does not match cached values"
                f" shape {cached_values.shape} at hook {hook.name}"
            )
            return cached_values

        fwd_hooks = [
            (hookpoint, freeze_hook)
            for hookpoint in freeze_cache.keys()
            if self.feature_input_hook not in hookpoint
        ]

        if not direct_effects:
            return fwd_hooks

        if self.skip_transcoder:
            skip_diffs = {}

            def diff_hook(activations, hook, layer: int):
                # The MLP hook out freeze hook sets the value of the MLP to the value it
                # had when run on the inputs normally. We subtract out the skip that
                # corresponds to such a run, and add in the skip with direct effects.
                frozen_skip = self.transcoders[layer].compute_skip(freeze_cache[hook.name])
                normal_skip = self.transcoders[layer].compute_skip(activations)

                skip_diffs[layer] = normal_skip - frozen_skip

            def add_diff_hook(activations, hook, layer: int):
                return activations + skip_diffs[layer]

            fwd_hooks += [
                (f"blocks.{layer}.{self.feature_input_hook}", partial(diff_hook, layer=layer))
                for layer in range(self.cfg.n_layers)
            ]
            fwd_hooks += [
                (f"blocks.{layer}.{self.feature_output_hook}", partial(add_diff_hook, layer=layer))
                for layer in range(self.cfg.n_layers)
            ]
        return fwd_hooks

    def _get_feature_intervention_hooks(
        self,
        inputs: Union[str, torch.Tensor],
        interventions: List[
            Tuple[int, Union[int, slice, torch.Tensor], int, Union[int, torch.Tensor]]
        ],
        direct_effects: bool = False,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
    ):
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, allowing all effects to propagate (optionally allowing its effects to
        propagate through transcoders)

        Args:
            input (_type_): the input prompt to intervene on
            intervention_dict (List[Tuple[int, Union[int, slice, torch.Tensor]], int,
                Union[int, torch.Tensor]]): A list of interventions to perform, formatted as
                a list of (layer, position, feature_idx, value)
            direct_effects (bool): whether to freeze all MLPs/transcoders / attn patterns /
                layernorm denominators
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
        """

        interventions_by_layer = defaultdict(list)
        for layer, pos, feature_idx, value in interventions:
            interventions_by_layer[layer].append((pos, feature_idx, value))

        # This activation cache will fill up during our forward intervention pass
        activation_cache, activation_hooks = self._get_activation_caching_hooks(
            apply_activation_function=apply_activation_function
        )

        def intervention_hook(activations, hook, layer, layer_interventions):
            transcoder_activations = activation_cache[layer]
            if not apply_activation_function:
                transcoder_activations = (
                    self.transcoders[layer]
                    .activation_function(transcoder_activations.unsqueeze(0))
                    .squeeze(0)
                )
            transcoder_output = self.transcoders[layer].decode(transcoder_activations)
            for pos, feature_idx, value in layer_interventions:
                transcoder_activations[pos, feature_idx] = value
            new_transcoder_output = self.transcoders[layer].decode(transcoder_activations)
            steering_vector = new_transcoder_output - transcoder_output
            return activations + steering_vector

        intervention_hooks = [
            (
                f"blocks.{layer}.{self.feature_output_hook}",
                partial(intervention_hook, layer=layer, layer_interventions=layer_interventions),
            )
            for layer, layer_interventions in interventions_by_layer.items()
        ]

        all_hooks = (
            self.setup_intervention_with_freeze(inputs, direct_effects=direct_effects)
            if freeze_attention or direct_effects
            else []
        )
        all_hooks += activation_hooks + intervention_hooks

        return all_hooks, activation_cache

    @torch.no_grad
    def feature_intervention(
        self,
        inputs: Union[str, torch.Tensor],
        interventions: List[
            Tuple[int, Union[int, slice, torch.Tensor], int, Union[int, torch.Tensor]]
        ],
        direct_effects: bool = False,
        freeze_attention: bool = True,
        apply_activation_function: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given the input, and a dictionary of features to intervene on, performs the
        intervention, and returns the logits and feature activations. If direct_effects is
        True, attention patterns will be frozen, along with MLPs and LayerNorms. If it is
        False, the effects of the intervention will propagate through transcoders /
        LayerNorms

        Args:
            input (_type_): the input prompt to intervene on
            interventions (List[Tuple[int, Union[int, slice, torch.Tensor]], int,
                Union[int, torch.Tensor]]): A list of interventions to perform, formatted as
                a list of (layer, position, feature_idx, value)
            direct_effects (bool): whether to freeze all MLPs/transcoders / attn patterns /
                layernorm denominators
            apply_activation_function (bool): whether to apply the activation function when
                recording the activations to be returned. This is useful to set to False for
                testing purposes, as attribution predicts the change in pre-activation
                feature values.
        """

        feature_intervention_hook_output = self._get_feature_intervention_hooks(
            inputs,
            interventions,
            direct_effects=direct_effects,
            freeze_attention=freeze_attention,
            apply_activation_function=apply_activation_function,
        )

        hooks, activation_cache = feature_intervention_hook_output

        with self.hooks(hooks):
            logits = self(inputs)

        activation_cache = torch.stack(activation_cache)

        return logits, activation_cache
