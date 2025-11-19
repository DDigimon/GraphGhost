"""
Build an **attribution graph** that captures the *direct*, *linear* effects
between features and next-token logits for a *prompt-specific*
**local replacement model**.

High-level algorithm (matches the 2025 ``Attribution Graphs`` paper):
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

1. **Local replacement model** - we configure gradients to flow only through
   linear components of the network, effectively bypassing attention mechanisms,
   MLP non-linearities, and layer normalization scales.
2. **Forward pass** - record residual-stream activations and mark every active
   feature.
3. **Backward passes** - for each source node (feature or logit), inject a
   *custom* gradient that selects its encoder/decoder direction.  Because the
   model is linear in the residual stream under our freezes, this contraction
   equals the *direct effect* A_{s->t}.
4. **Assemble graph** - store edge weights in a dense matrix and package a
   ``Graph`` object.  Downstream utilities can *prune* the graph to the subset
   needed for interpretation.
"""

import contextlib
import logging
import time
import weakref
from functools import partial
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from einops import einsum
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from circuit_tracer.graph import Graph
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.disk_offload import offload_modules


class AttributionContext:
    """Manage hooks for computing attribution rows.

    This helper caches residual-stream activations **(forward pass)** and then
    registers backward hooks that populate a write-only buffer with
    *direct-effect rows* **(backward pass)**.

    The buffer layout concatenates rows for **feature nodes**, **error nodes**,
    **token-embedding nodes**

    Args:
        activation_matrix (torch.sparse.Tensor):
            Sparse `(n_layers, n_pos, n_features)` tensor indicating **which**
            features fired at each layer/position.
        error_vectors (torch.Tensor):
            `(n_layers, n_pos, d_model)` - *residual* the CLT / PLT failed to
            reconstruct ("error nodes").
        token_vectors (torch.Tensor):
            `(n_pos, d_model)` - embeddings of the prompt tokens.
        decoder_vectors (torch.Tensor):
            `(total_active_features, d_model)` - decoder rows **only for active
            features**, already multiplied by feature activations so they
            represent a_s * W^dec.
    """

    def __init__(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        feature_output_hook: str,
    ) -> None:
        n_layers, n_pos, _ = activation_matrix.shape

        # Forward-pass cache
        self._resid_activations: List[torch.Tensor | None] = [None] * (n_layers + 1)
        self._batch_buffer: torch.Tensor | None = None
        self.n_layers: int = n_layers

        # Assemble all backward hooks up-front
        self._attribution_hooks = self._make_attribution_hooks(
            activation_matrix, error_vectors, token_vectors, decoder_vecs, feature_output_hook
        )

        total_active_feats = activation_matrix._nnz()
        self._row_size: int = total_active_feats + (n_layers + 1) * n_pos  # + logits later

    def _caching_hooks(self, feature_input_hook: str) -> List[Tuple[str, Callable]]:
        """Return hooks that store residual activations layer-by-layer."""

        proxy = weakref.proxy(self)

        def _cache(acts: torch.Tensor, hook: HookPoint, *, layer: int) -> torch.Tensor:
            proxy._resid_activations[layer] = acts
            return acts

        hooks = [
            (f"blocks.{layer}.{feature_input_hook}", partial(_cache, layer=layer))
            for layer in range(self.n_layers)
        ]
        hooks.append(("unembed.hook_pre", partial(_cache, layer=self.n_layers)))
        return hooks

    def _compute_score_hook(
        self,
        hook_name: str,
        output_vecs: torch.Tensor,
        write_index: slice,
        read_index: slice | np.ndarray = np.s_[:],
    ) -> Tuple[str, Callable]:
        """
        Factory that contracts *gradients* with an **output vector set**.
        The hook computes A_{s->t} and writes the result into an in-place buffer row.
        """

        proxy = weakref.proxy(self)

        def _hook_fn(grads: torch.Tensor, hook: HookPoint) -> None:
            # proxy._batch_buffer[write_index] = einsum(
            #     grads.to(output_vecs.dtype)[read_index],
            #     output_vecs,
            #     "batch position d_model, position d_model -> position batch",
            # )
            dev = grads.device  # 以梯度所在设备为准（例如 cuda:7）

            # 1) 让 output_vecs 上同一设备；并对齐 dtype（沿用你“以 output_vecs 的 dtype 为准”的策略）
            ov = output_vecs
            if ov.device != dev or ov.dtype != output_vecs.dtype:
                ov = ov.to(device=dev, dtype=output_vecs.dtype, non_blocking=True)

            g = grads[read_index]
            if g.dtype != ov.dtype:
                g = g.to(dtype=ov.dtype)

            # 2) 计算
            out = einsum(
                g, ov,
                "batch position d_model, position d_model -> position batch",
            )

            # 3) 确保 buffer 在同一设备再写入（如果它是预分配在 CPU 的，这里搬一次）
            if proxy._batch_buffer.device != dev:
                proxy._batch_buffer = proxy._batch_buffer.to(dev, non_blocking=True)

            proxy._batch_buffer[write_index] = out

        return hook_name, _hook_fn

    def _make_attribution_hooks(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        feature_output_hook: str,
    ) -> List[Tuple[str, Callable]]:
        """Create the complete backward-hook for computing attribution scores."""

        n_layers, n_pos, _ = activation_matrix.shape
        nnz_layers, nnz_positions, _ = activation_matrix.indices()

        # Map each layer → slice in flattened active-feature list
        _, counts = torch.unique_consecutive(nnz_layers, return_counts=True)
        edges = [0] + counts.cumsum(0).tolist()
        layer_spans = list(zip(edges[:-1], edges[1:]))

        # Feature nodes
        feature_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                decoder_vecs[start:end],
                write_index=np.s_[start:end],
                read_index=np.s_[:, nnz_positions[start:end]],
            )
            for layer, (start, end) in enumerate(layer_spans)
            if start != end
        ]

        # Error nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            return activation_matrix._nnz() + layer * n_pos

        error_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            )
            for layer in range(n_layers)
        ]

        # Token-embedding nodes
        tok_start = error_offset(n_layers)
        token_hook = [
            self._compute_score_hook(
                "hook_embed",
                token_vectors,
                write_index=np.s_[tok_start : tok_start + n_pos],
            )
        ]

        return feature_hooks + error_hooks + token_hook

    @contextlib.contextmanager
    def install_hooks(self, model: "ReplacementModel"):
        """Context manager instruments the hooks for the forward and backward passes."""
        with model.hooks(
            fwd_hooks=self._caching_hooks(model.feature_input_hook),
            bwd_hooks=self._attribution_hooks,
        ):
            yield

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Return attribution rows for a batch of (layer, pos) nodes.

        The routine overrides gradients at **exact** residual-stream locations
        triggers one backward pass, and copies the rows from the internal buffer.

        Args:
            layers: 1-D tensor of layer indices *l* for the source nodes.
            positions: 1-D tensor of token positions *c* for the source nodes.
            inject_values: `(batch, d_model)` tensor with outer product
                a_s * W^(enc/dec) to inject as custom gradient.

        Returns:
            torch.Tensor: ``(batch, row_size)`` matrix - one row per node.
        """

        batch_size = self._resid_activations[0].shape[0]
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        # Custom gradient injection (per-layer registration)
        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject(grads, *, batch_indices, pos_indices, values):
            # grads_out = grads.clone().to(values.dtype)
            # grads_out.index_put_((batch_indices, pos_indices), values)
            # return grads_out.to(grads.dtype)


            dev = grads.device

            if batch_indices.dtype != torch.long:
                batch_indices = batch_indices.to(torch.long)
            if pos_indices.dtype != torch.long:
                pos_indices = pos_indices.to(torch.long)
            if batch_indices.device != dev:
                batch_indices = batch_indices.to(dev, non_blocking=True)
            if pos_indices.device != dev:
                pos_indices = pos_indices.to(dev, non_blocking=True)

            # print("[dbg] grads", grads.device, grads.dtype,
            #     "| batch_idx", batch_indices.device, batch_indices.dtype,
            #     "| pos_idx", pos_indices.device, pos_indices.dtype,
            #     "| values", values.device, values.dtype)

            g = grads.clone()
            v = values.to(device=dev, dtype=values.dtype, non_blocking=True)
            g = g.to(v.dtype)
            g[batch_indices, pos_indices] = v
            return g.to(grads.dtype)
        
        handles = []
        layers_in_batch = layers.unique().tolist()
        for layer in layers_in_batch:
            mask = layers == layer
            if not mask.any():
                continue
            fn = partial(
                _inject,
                batch_indices=batch_idx[mask],
                pos_indices=positions[mask],
                values=inject_values[mask],
            )
            resid_activations = self._resid_activations[int(layer)]
            assert resid_activations is not None, "Residual activations are not cached"
            handles.append(resid_activations.register_hook(fn))

        try:
            last_layer = max(layers_in_batch)
            self._resid_activations[last_layer].backward(
                gradient=torch.zeros_like(self._resid_activations[last_layer]),
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[: len(layers)]


@torch.no_grad()
def compute_salient_logits(
    logits: torch.Tensor,
    unembed_proj: torch.Tensor,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the smallest logit set whose cumulative prob >= *desired_logit_prob*.

    Args:
        logits: ``(d_vocab,)`` vector (single position).
        unembed_proj: ``(d_model, d_vocab)`` unembedding matrix.
        max_n_logits: Hard cap *k*.
        desired_logit_prob: Cumulative probability threshold *p*.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            * logit_indices - ``(k,)`` vocabulary ids.
            * logit_probs   - ``(k,)`` softmax probabilities.
            * demeaned_vecs - ``(k, d_model)`` unembedding columns, demeaned.
    """

    probs = torch.softmax(logits, dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    cols = unembed_proj[:, top_idx]
    demeaned = cols - unembed_proj.mean(dim=-1, keepdim=True)
    return top_idx, top_p, demeaned.T


@torch.no_grad()
def select_scaled_decoder_vecs(
    activations: torch.sparse.Tensor, transcoders: Sequence
) -> torch.Tensor:
    """Return decoder rows for **active** features only.

    The return value is already scaled by the feature activation, making it
    suitable as ``inject_values`` during gradient overrides.
    """
    gather_device = torch.device("cpu")
    rows: List[torch.Tensor] = []
    # for layer, row in enumerate(activations):
    #     _, feat_idx = row.coalesce().indices()
    #     rows.append(transcoders[layer].W_dec[feat_idx])
    # return torch.cat(rows) * activations.values()[:, None]
    
    # 确保稀疏张量是 coalesced（indices/values 一一对应）
    activations = activations.coalesce()

    # 逐层取特征索引，收集 W_dec 的行
    for layer, row in enumerate(activations):     # 假设 activations 第一维是 layer
        row = row.coalesce()
        # indices() 形状取决于你的稀疏布局，这里按你原来的写法保留
        _, feat_idx = row.indices()
        if feat_idx.numel() == 0:
            continue
        # 取该层 W_dec 的对应行，并搬到统一设备
        rows.append(transcoders[layer].W_dec[feat_idx].to(gather_device))

    if not rows:
        # 没有激活特征时返回空张量（按需设定列数/dtype）
        d_out = transcoders[0].W_dec.shape[1]
        return torch.empty(0, d_out, device=gather_device, dtype=transcoders[0].W_dec.dtype)

    # 拼接后按激活值缩放（激活值也搬到同设备）
    cat_rows = torch.cat(rows, dim=0)  # 现在都在 gather_device
    scales  = activations.values().to(gather_device)[:, None]
    return cat_rows * scales
    

@torch.no_grad()
def select_encoder_rows(
    activation_matrix: torch.sparse.Tensor, transcoders: Sequence
) -> torch.Tensor:
    """Return encoder rows for **active** features only."""

    # rows: List[torch.Tensor] = []
    # for layer, row in enumerate(activation_matrix):
    #     _, feat_idx = row.coalesce().indices()
    #     rows.append(transcoders[layer].W_enc.T[feat_idx])
    # return torch.cat(rows)
    gather_device = torch.device("cpu")  # 也可以换成 torch.device("cuda:0") 等

    rows: List[torch.Tensor] = []
    for layer, row in enumerate(activation_matrix):
        row = row.coalesce()
        _, feat_idx = row.indices()
        if feat_idx.numel() == 0:
            continue
        # 把每层取到的行移动到同一设备再收集
        rows.append(transcoders[layer].W_enc.T[feat_idx].to(gather_device))

    if not rows:
        # 没有激活特征时返回空张量（按形状/dtype 构造）
        # W_enc 形状通常是 [d_features, d_model] 或 [d_model, d_features]
        # 这里我们取 W_enc.T 的列数作为第二维
        d_cols = transcoders[0].W_enc.T.shape[1]
        return torch.empty(0, d_cols, device=gather_device, dtype=transcoders[0].W_enc.dtype)

    return torch.cat(rows, dim=0)  # 此时都在 gather_device 上



def compute_partial_influences(edge_matrix, logit_p, row_to_node_index, max_iter=128, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
    normalized_matrix = normalized_matrix.abs_()
    normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

    influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod[-len(logit_p) :] = logit_p

    for _ in range(max_iter):
        prod = prod[row_to_node_index] @ normalized_matrix
        if not prod.any():
            break
        influences += prod
    else:
        raise RuntimeError("Failed to converge")

    return influences


def ensure_tokenized(prompt: Union[str, torch.Tensor, List[int]], tokenizer) -> torch.Tensor:
    """Convert *prompt* → 1-D tensor of token ids (no batch dim)."""

    if isinstance(prompt, str):
        return tokenizer(prompt, return_tensors="pt").input_ids[0]
    if isinstance(prompt, torch.Tensor):
        return prompt.squeeze(0) if prompt.ndim == 2 else prompt
    if isinstance(prompt, list):
        return torch.tensor(prompt, dtype=torch.long)
    raise TypeError(f"Unsupported prompt type: {type(prompt)}")


def attribute(
    prompt: Union[str, torch.Tensor, List[int]],
    model: ReplacementModel,
    input_ids: Optional[torch.Tensor] = None,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: Optional[int] = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
    print_log: bool = True,
) -> Graph:
    """Compute an attribution graph for *prompt*.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``ReplacementModel``
        max_n_logits: Max number of logit nodes.
        desired_logit_prob: Keep logits until cumulative prob >= this value.
        batch_size: How many source nodes to process per backward pass.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
                 Options are "cpu" (move to CPU), "disk" (save to disk),
                 or None (no offloading).
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """

    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if verbose and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    offload_handles = []
    try:
        return _run_attribution(
            model=model,
            prompt=prompt,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            offload_handles=offload_handles,
            update_interval=update_interval,
            logger=logger,
            input_ids=input_ids,
            print_log=print_log,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()

        logger.removeHandler(handler)
        
def _mem(tag: str):
    try:
        n = torch.cuda.device_count()
        stats = []
        for i in range(n):
            alloc = torch.cuda.memory_allocated(i)
            resv  = torch.cuda.memory_reserved(i)
            stats.append(f"dev{i}: alloc={alloc/1e6:.1f}MB resv={resv/1e6:.1f}MB")
        cur = torch.cuda.current_device()
        print(f"[MEM] {tag} | current={cur} | " + " | ".join(stats))
    except Exception as e:
        print(f"[MEM] {tag} | cuda not ready? {e}")
    
    
def _run_attribution(
    model,
    prompt,
    max_n_logits,
    desired_logit_prob,
    batch_size,
    max_feature_nodes,
    offload,
    verbose,
    offload_handles,
    update_interval=4,
    logger=None,
    input_ids=None,
    print_log = True,
):
    start_time = time.time()
    # Phase 0: precompute
    if print_log:
        logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()
    if input_ids is None:
        input_ids = ensure_tokenized(prompt, model.tokenizer)
    
    try:
        dev0 = model.device_map[0]           # torch.device("cuda", 4)
        torch.cuda.set_device(dev0.index)    # 从此“默认设备”= cuda:4
    except Exception as e:
        print("[DBG] set_device skipped:", e)
    

    logits, activation_matrix, error_vecs, token_vecs = model.setup_attribution(
        input_ids, sparse=True
    )

    
    decoder_vecs = select_scaled_decoder_vecs(activation_matrix, model.transcoders)
    encoder_rows = select_encoder_rows(activation_matrix, model.transcoders)
    ctx = AttributionContext(
        activation_matrix, error_vecs, token_vecs, decoder_vecs, model.feature_output_hook
    )
    if print_log:
        logger.info(f"Precomputation completed in {time.time() - phase_start:.2f}s")
        logger.info(f"Found {activation_matrix._nnz()} active features")

    if offload:
        offload_handles += offload_modules(model.transcoders, offload)

    # Phase 1: forward pass
    if print_log:
        logger.info("Phase 1: Running forward pass")
    phase_start = time.time()
    
    with ctx.install_hooks(model):
        residual = model.forward(input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
        # print(residual.require_grad)
        # exit()
        # out = model.ln_final(residual)
        ctx._resid_activations[-1] = model.ln_final(residual)

        # ctx._resid_activations[model.cfg.n_layers] = out 
        
    if print_log:
        logger.info(f"Forward pass completed in {time.time() - phase_start:.2f}s")
        logger.info("Phase 2: Building input vectors")

    if offload:
        offload_handles += offload_modules([block.mlp for block in model.blocks], offload)

    # Phase 2: build input vector list
    
    phase_start = time.time()
    feat_layers, feat_pos, _ = activation_matrix.indices()
    n_layers, n_pos, _ = activation_matrix.shape

    total_active_feats = activation_matrix._nnz()

    
    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        logits[0, -1],
        model.unembed.W_U,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )
    if print_log:
        logger.info(
            f"Selected {len(logit_idx)} logits with cumulative probability {logit_p.sum().item():.4f}"
        )

    if offload:
        offload_handles += offload_modules([model.unembed, model.embed], offload)

    logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
    n_logits = len(logit_idx)
    total_nodes = logit_offset + n_logits

    max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
    

    edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    # Maps row indices in edge_matrix to original feature/node indices
    # First populated with logit node IDs, then feature IDs in attribution order
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)
    if print_log:
        logger.info(f"Will include {max_feature_nodes} of {total_active_feats} feature nodes")
        logger.info(f"Input vectors built in {time.time() - phase_start:.2f}s")

        # Phase 3: logit attribution
        logger.info("Phase 3: Computing logit attributions")
    phase_start = time.time()
    for i in range(0, len(logit_idx), batch_size):
        batch = logit_vecs[i : i + batch_size]
        rows = ctx.compute_batch(
            layers=torch.full((batch.shape[0],), n_layers),
            positions=torch.full((batch.shape[0],), n_pos - 1),
            inject_values=batch,
        )
        edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()
        row_to_node_index[i : i + batch.shape[0]] = (
            torch.arange(i, i + batch.shape[0]) + logit_offset
        )
    if print_log:
        logger.info(f"Logit attributions completed in {time.time() - phase_start:.2f}s")

        # Phase 4: feature attribution
        logger.info("Phase 4: Computing feature attributions")
    phase_start = time.time()
    st = n_logits
    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    n_visited = 0

    if print_log:
        pbar = tqdm(total=max_feature_nodes, desc="Feature influence computation", disable=not verbose)

    while n_visited < max_feature_nodes:
        if max_feature_nodes == total_active_feats:
            pending = torch.arange(total_active_feats)
        else:
            influences = compute_partial_influences(
                edge_matrix[:st], logit_p, row_to_node_index[:st]
            )
            feature_rank = torch.argsort(influences[:total_active_feats], descending=True).cpu()
            queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
            pending = feature_rank[~visited[feature_rank]][:queue_size]

        queue = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

        for idx_batch in queue:
            n_visited += len(idx_batch)

            rows = ctx.compute_batch(
                layers=feat_layers[idx_batch],
                positions=feat_pos[idx_batch],
                inject_values=encoder_rows[idx_batch],
                retain_graph=n_visited < max_feature_nodes,
            )

            end = min(st + batch_size, st + rows.shape[0])
            edge_matrix[st:end, :logit_offset] = rows.cpu()
            row_to_node_index[st:end] = idx_batch
            visited[idx_batch] = True
            st = end
            if print_log:
                pbar.update(len(idx_batch))
    if print_log:
        pbar.close()
    
    if print_log:
        logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 5: packaging graph
    selected_features = torch.where(visited)[0]
    if max_feature_nodes < total_active_feats:
        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        col_read = torch.cat([selected_features, non_feature_nodes])
        edge_matrix = edge_matrix[:, col_read]
        
    # print(selected_features.shape)
    # print(selected_features)

    # sort rows such that features are in order
    edge_matrix = edge_matrix[row_to_node_index.argsort()]
    final_node_count = edge_matrix.shape[1]
    full_edge_matrix = torch.zeros(final_node_count, final_node_count)
    full_edge_matrix[:max_feature_nodes] = edge_matrix[:max_feature_nodes]
    full_edge_matrix[-n_logits:] = edge_matrix[max_feature_nodes:]
    # print('input_ids',input_ids.shape)
    input_ids = input_ids.squeeze()
    graph = Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_tokens=logit_idx,
        logit_probabilities=logit_p,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix,
        cfg=model.cfg,
        scan=model.scan,
    )
    
    # print(f"Graph has {graph.nodes} nodes and {graph.edges} edges")
    # print(graph.nodes[:10])  # Display first 10 nodes for debugging
    # print(model.tokenizer.decode(input_ids))
    # print(graph.adjacency_matrix.shape)
    # print(graph.adjacency_matrix)
    # print(graph.active_features)  # Display first 10x10 submatrix for debugging
    
    

    total_time = time.time() - start_time
    if print_log:
        logger.info(f"Attribution completed in {total_time:.2f}s")

    return graph
