import os
import gzip

from collections import namedtuple
from importlib import resources
from typing import Callable, Optional
from jaxtyping import Float
import pickle
from tqdm import tqdm
import einops
import numpy as np
import torch
import torch.nn.functional as F
from wandb import config
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
import circuit_tracer
from circuit_tracer.transcoder.activation_functions import JumpReLU,Relu
from circuit_tracer.utils.hf_utils import download_hf_uris, parse_hf_uri
from circuit_tracer.transcoder.utils import compute_geometric_median


class SingleLayerTranscoder(nn.Module):
    d_model: int
    d_transcoder: int
    layer_idx: int
    W_enc: nn.Parameter
    W_dec: nn.Parameter
    b_enc: nn.Parameter
    b_dec: nn.Parameter
    W_skip: Optional[nn.Parameter]
    activation_function: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        configs,
        activation_function,
        skip_connection = False,
    ):
        """Single layer transcoder implementation, adapted from the JumpReLUSAE implementation here:
        https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp

        Args:
            d_model (int): The dimension of the model.
            d_transcoder (int): The dimension of the transcoder.
            activation_function (nn.Module): The activation function.
            layer_idx (int): The layer index.
            skip_connection (bool): Whether there is a skip connection,
                as in https://arxiv.org/abs/2501.18823
        """
        super().__init__()
        # print('inside configs')
        # print(configs)
        self.cfg = configs
        self.d_model = configs.d_in# d_model
        self.d_transcoder = configs.d_transcoder# d_transcoder
        self.layer_idx = configs.target_layer
        self.dtype = torch.float32 # configs.dtype
        self.device = configs.device
        self.l1_coefficient = configs.l1_coefficient

        # self.W_enc = nn.Parameter(torch.zeros(self.d_model, self.d_transcoder))
        # self.W_dec = nn.Parameter(torch.zeros(self.d_transcoder, self.d_model))
        # self.b_enc = nn.Parameter(torch.zeros(self.d_transcoder))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))
        
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_model, self.d_transcoder, dtype=self.dtype, device=self.device)
            )   
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_transcoder, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_transcoder, self.d_model, dtype=self.dtype, device=self.device)
            )
        )
        
        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_model, dtype=self.dtype, device=self.device)
        )
        
        # self.b_dec_out = nn.Parameter(
        #         torch.zeros(self.d_model, dtype=self.dtype, device=self.device)
        #     )
        self.b_dec_out = None


        if skip_connection:
            self.W_skip = nn.Parameter(torch.zeros(self.d_model, self.d_model))
        else:
            self.W_skip = None

        self.activation_function = activation_function

    def encode(self, input_acts, return_pres = False, apply_activation_function: bool = True):
        # print('input_acts', input_acts.shape, 'W_enc', self.W_enc.shape, 'b_enc', self.b_enc.shape)
        pre_acts = input_acts.to(self.W_enc.dtype) @ self.W_enc + self.b_enc
        if not apply_activation_function:
            return pre_acts
        acts = self.activation_function(pre_acts)
        
        if return_pres:
            return acts, pre_acts
        else:
            return acts

    def decode(self, acts):
        if acts.is_sparse:
            return (
                torch.bmm(acts, self.W_dec.unsqueeze(0).expand(acts.size(0), *self.W_dec.size()))
                + self.b_dec
            )
        else:
            # print('acts', acts.shape, 'W_dec', self.W_dec.shape, 'b_dec', self.b_dec.shape)
            return acts @ self.W_dec + self.b_dec

    def compute_skip(self, input_acts):
        if self.W_skip is not None:
            return input_acts @ self.W_skip.T
        else:
            raise ValueError("Transcoder has no skip connection")

    def forward(self, input_acts, dead_neuron_mask = None, mse_target = None):
        # print('input_acts', input_acts)
        transcoder_acts, hidden_pre = self.encode(input_acts, return_pres = True)
        # print('transcoder_acts', transcoder_acts)
        # print('hidden_pre', hidden_pre)
        decoded = self.decode(transcoder_acts)
        # print('decoded',decoded)
        
        if self.training == False:
            decoded = decoded.detach()
            decoded.requires_grad = True

        if self.W_skip is not None:
            skip = self.compute_skip(input_acts)
            decoded = decoded + skip
            
        if mse_target is None:
            mse_loss = (torch.pow((decoded-input_acts.float()), 2) / (input_acts**2).sum(dim=-1, keepdim=True).sqrt())
        else:
            mse_loss = (torch.pow((decoded-mse_target.float()), 2) / (mse_target**2).sum(dim=-1, keepdim=True).sqrt())

        mse_loss_ghost_resid = torch.tensor(0.0, dtype=self.dtype, device=self.device)

            
        if self.training and dead_neuron_mask.sum() > 0:
            assert dead_neuron_mask is not None 
            
            # ghost protocol
            
            # 1.
            residual = input_acts - decoded
            
            l2_norm_residual = torch.norm(residual, dim=-1)
            feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
            ghost_out =  feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask,:]
            l2_norm_ghost_out = torch.norm(ghost_out, dim = -1)
            norm_scaling_factor = l2_norm_residual / (1e-6+ l2_norm_ghost_out* 2)
            ghost_out = ghost_out*norm_scaling_factor[:, None].detach()
            
            # 3. 
            mse_loss_ghost_resid = (
                torch.pow((ghost_out - residual.detach().float()), 2) / (residual.detach()**2).sum(dim=-1, keepdim=True).sqrt()
            )
            mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
            mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid
        
        mse_loss_ghost_resid = mse_loss_ghost_resid.mean()
        mse_loss = mse_loss.mean()
        # print(feature_acts.shape)
        sparsity = torch.abs(transcoder_acts).sum(dim=-1).mean(dim=(0,))
        l1_loss = self.l1_coefficient * sparsity
        loss = mse_loss + l1_loss + mse_loss_ghost_resid
        

        return decoded, transcoder_acts, loss, mse_loss, l1_loss, mse_loss_ghost_resid
    
    def get_sparse_connection_loss(self, sparse_connection_l1_coeff):
        dots = self.spacon_sae_W_dec @ self.W_dec.T
        # each row is an sae feature, each column is a transcoder feature
        loss = torch.sum(dots.abs(), dim=1).mean() # mean over each sae feature of L1 of transcoder features activated
        return sparse_connection_l1_coeff * loss
    
    @torch.no_grad()
    def initialize_b_dec(self, activation_store):
        
        if self.cfg.b_dec_init_method == "geometric_median":
            self.initialize_b_dec_with_geometric_median(activation_store)
        elif self.cfg.b_dec_init_method == "mean":
            self.initialize_b_dec_with_mean(activation_store)
        elif self.cfg.b_dec_init_method == "zeros":
            pass
        else:
            raise ValueError(f"Unexpected b_dec_init_method: {self.cfg.b_dec_init_method}")

    @torch.no_grad()
    def initialize_b_dec_with_geometric_median(self, activation_store):
        # assert(self.cfg.is_transcoder == activation_store.cfg.is_transcoder)

        previous_b_dec = self.b_dec.clone().cpu()
        all_activations = activation_store.storage_buffer.detach().cpu()
        out = compute_geometric_median(
            all_activations,
            skip_typechecks=True, 
            maxiter=100, per_component=False
        ).median
        
        
        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)
        
        print("Reinitializing b_dec with geometric median of activations")
        print(f"Previous distances: {previous_distances.median(0).values.mean().item()}")
        print(f"New distances: {distances.median(0).values.mean().item()}")
        
        out = torch.tensor(out, dtype=self.dtype, device=self.device)
        self.b_dec.data = out

        if self.b_dec is not None:
            # stupid code duplication
            previous_b_dec_out = self.b_dec.clone().cpu()
            all_activations_out = activation_store.storage_buffer_out.detach().cpu()
            out_out = compute_geometric_median(
                all_activations_out,
                skip_typechecks=True, 
                maxiter=100, per_component=False
            ).median
            
            previous_distances_out = torch.norm(all_activations_out - previous_b_dec_out, dim=-1)
            distances_out = torch.norm(all_activations_out - out_out, dim=-1)
            
            print("Reinitializing b_dec with geometric median of activations")
            print(f"Previous distances: {previous_distances_out.median(0).values.mean().item()}")
            print(f"New distances: {distances_out.median(0).values.mean().item()}")
            
            out_out = torch.tensor(out_out, dtype=self.dtype, device=self.device)
            self.b_dec.data = out_out
        
    @torch.no_grad()
    def initialize_b_dec_with_mean(self, activation_store):
        # assert(self.cfg.is_transcoder == activation_store.cfg.is_transcoder)
        
        previous_b_dec = self.b_dec.clone().cpu()
        if isinstance(activation_store.storage_buffer,list) :
            all_activations = torch.cat(activation_store.storage_buffer)
        else:
            all_activations = activation_store.storage_buffer
        all_activations = all_activations.detach().cpu()
        out = all_activations.mean(dim=0)
        
        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)
        
        print("Reinitializing b_dec with mean of activations")
        print(f"Previous distances: {previous_distances.median(0).values.mean().item()}")
        print(f"New distances: {distances.median(0).values.mean().item()}")
        
        self.b_dec.data = out.to(self.dtype).to(self.device)

        if self.b_dec is not None:
            # stupid code duplication        
            previous_b_dec_out = self.b_dec.clone().cpu()
            if isinstance(activation_store.storage_buffer_out, list):
                expanded_list = [t.unsqueeze(0) for t in activation_store.storage_buffer_out]
                activation_store.storage_buffer_out = torch.cat(expanded_list,axis=0)
            all_activations_out = activation_store.storage_buffer_out.detach().cpu()
            print('all_activations_out', all_activations_out.shape)
            print('previous_b_dec_out', previous_b_dec_out.shape)
            # exit()
            # if self.cfg.is_crosscoder:
            #     out_out = all_activations_out.mean(dim=(1))
            
            out_out = all_activations_out.mean(dim=0)
            print('out_out', out_out.shape)
            
            # previous_distances_out = torch.norm(all_activations_out - previous_b_dec_out, dim=-1)
            # distances_out = torch.norm(all_activations_out - out_out, dim=-1)
            
            # print("Reinitializing b_dec with mean of activations")
            # print(f"Previous distances: {previous_distances_out.median(0).values.mean().item()}")
            # print(f"New distances: {distances_out.median(0).values.mean().item()}")
            
            self.b_dec.data = out_out.to(self.dtype).to(self.device)
        

    @torch.no_grad()
    def resample_neurons_l2(
        self,
        x: Float[Tensor, "batch_size n_hidden"],
        feature_sparsity: Float[Tensor, "n_hidden_ae"],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        '''
        Resamples neurons that have been dead for `dead_neuron_window` steps, according to `frac_active`.
        
        I'll probably break this now and fix it later!
        '''
        
        feature_reinit_scale = self.cfg.feature_reinit_scale
        
        sae_out, _, _, _, _ = self.forward(x)
        per_token_l2_loss = (sae_out - x).pow(2).sum(dim=-1).squeeze()

        # Find the dead neurons in this instance. If all neurons are alive, continue
        is_dead = (feature_sparsity < self.cfg.dead_feature_threshold)
        dead_neurons = torch.nonzero(is_dead).squeeze(-1)
        alive_neurons = torch.nonzero(~is_dead).squeeze(-1)
        n_dead = dead_neurons.numel()
        
        if n_dead == 0:
            return 0 # If there are no dead neurons, we don't need to resample neurons
        
        # Compute L2 loss for each element in the batch
        # TODO: Check whether we need to go through more batches as features get sparse to find high l2 loss examples. 
        if per_token_l2_loss.max() < 1e-6:
            return 0 # If we have zero reconstruction loss, we don't need to resample neurons
        
        # Draw `n_hidden_ae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss squared
        per_token_l2_loss = per_token_l2_loss.to(torch.float32) # wont' work with bfloat16
        distn = Categorical(probs = per_token_l2_loss.pow(2) / (per_token_l2_loss.pow(2).sum()))
        replacement_indices = distn.sample((n_dead,)) # shape [n_dead]

        # Index into the batch of hidden activations to get our replacement values
        replacement_values = (x - self.b_dec)[replacement_indices] # shape [n_dead n_input_ae]

        # unit norm
        replacement_values = (replacement_values / (replacement_values.norm(dim=1, keepdim=True) + 1e-8))

        # St new decoder weights
        self.W_dec.data[is_dead, :] = replacement_values

        # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
        W_enc_norm_alive_mean = 1.0 if len(alive_neurons) == 0 else self.W_enc[:, alive_neurons].norm(dim=0).mean().item()
        
        # Lastly, set the new weights & biases
        self.W_enc.data[:, is_dead] = (replacement_values * W_enc_norm_alive_mean * feature_reinit_scale).T
        self.b_enc.data[is_dead] = 0.0
        
        
        # reset the Adam Optimiser for every modified weight and bias term
        # Reset all the Adam parameters
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    assert k.data.shape == (self.d_in, self.d_sae)
                    v[v_key][:, is_dead] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.d_sae,)
                    v[v_key][is_dead] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.d_sae, self.d_out)
                    v[v_key][is_dead, :] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.d_out,)
                else:
                    if not self.cfg.is_transcoder:
                        raise ValueError(f"Unexpected dict_idx {dict_idx}")
                        # if we're a transcoder, then this is fine, because we also have b_dec_out
                
        # Check that the opt is really updated
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    if k.data.shape != (self.d_in, self.d_sae):
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked, there are shapes mismatches"
                        )
                    if v[v_key][:, is_dead].abs().max().item() > 1e-6:
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked"
                        )
        
        return n_dead

    @torch.no_grad()
    def resample_neurons_anthropic(
        self, 
        dead_neuron_indices, 
        model,
        optimizer, 
        activation_store):
        """
        Arthur's version of Anthropic's feature resampling
        procedure.
        """
        # collect global loss increases, and input activations
        global_loss_increases, global_input_activations = self.collect_anthropic_resampling_losses(
            model, activation_store
        )

        # sample according to losses
        probs = global_loss_increases / global_loss_increases.sum()
        sample_indices = torch.multinomial(
            probs,
            min(len(dead_neuron_indices), probs.shape[0]),
            replacement=False,
        )
        # if we don't have enough samples for for all the dead neurons, take the first n
        if sample_indices.shape[0] < len(dead_neuron_indices):
            dead_neuron_indices = dead_neuron_indices[:sample_indices.shape[0]]

        # Replace W_dec with normalized differences in activations
        self.W_dec.data[dead_neuron_indices, :] = (
            (
                global_input_activations[sample_indices]
                / torch.norm(global_input_activations[sample_indices], dim=1, keepdim=True)
            )
            .to(self.dtype)
            .to(self.device)
        )
        
        # Lastly, set the new weights & biases
        self.W_enc.data[:, dead_neuron_indices] = self.W_dec.data[dead_neuron_indices, :].T
        self.b_enc.data[dead_neuron_indices] = 0.0
        
        # Reset the Encoder Weights
        if dead_neuron_indices.shape[0] < self.d_sae:
            sum_of_all_norms = torch.norm(self.W_enc.data, dim=0).sum()
            sum_of_all_norms -= len(dead_neuron_indices)
            average_norm = sum_of_all_norms / (self.d_sae - len(dead_neuron_indices))
            self.W_enc.data[:, dead_neuron_indices] *= self.cfg.feature_reinit_scale * average_norm

            # Set biases to resampled value
            relevant_biases = self.b_enc.data[dead_neuron_indices].mean()
            self.b_enc.data[dead_neuron_indices] = relevant_biases * 0 # bias resample factor (put in config?)

        else:
            self.W_enc.data[:, dead_neuron_indices] *= self.cfg.feature_reinit_scale
            self.b_enc.data[dead_neuron_indices] = -5.0
        
        # TODO: Refactor this resetting to be outside of resampling.
        # reset the Adam Optimiser for every modified weight and bias term
        # Reset all the Adam parameters
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    assert k.data.shape == (self.d_in, self.d_sae)
                    v[v_key][:, dead_neuron_indices] = 0.0
                elif dict_idx == 1:
                    assert k.data.shape == (self.d_sae,)
                    v[v_key][dead_neuron_indices] = 0.0
                elif dict_idx == 2:
                    assert k.data.shape == (self.d_sae, self.d_out)
                    v[v_key][dead_neuron_indices, :] = 0.0
                elif dict_idx == 3:
                    assert k.data.shape == (self.d_out,)
                else:
                    if not self.cfg.is_transcoder:
                        raise ValueError(f"Unexpected dict_idx {dict_idx}")
                        # if we're a transcoder, then this is fine, because we also have b_dec_out
                
        # Check that the opt is really updated
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0:
                    if k.data.shape != (self.d_in, self.d_sae):
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked, there are shapes mismatches"
                        )
                    if v[v_key][:, dead_neuron_indices].abs().max().item() > 1e-6:
                        print(
                            "Warning: it does not seem as if resetting the Adam parameters worked"
                        )
        
        return 

    @torch.no_grad()
    def collect_anthropic_resampling_losses(self, model, activation_store):
        """
        Collects the losses for resampling neurons (anthropic)
        """
        
        batch_size = self.cfg.store_batch_size
        
        # we're going to collect this many forward passes
        number_final_activations = self.cfg.resample_batches * batch_size
        # but have seq len number of tokens in each
        number_activations_total = number_final_activations * self.cfg.context_size
        anthropic_iterator = range(0, number_final_activations, batch_size)
        anthropic_iterator = tqdm(anthropic_iterator, desc="Collecting losses for resampling...")
        
        global_loss_increases = torch.zeros((number_final_activations,), dtype=self.dtype, device=self.device)
        global_input_activations = torch.zeros((number_final_activations, self.d_model), dtype=self.dtype, device=self.device)

        for refill_idx in anthropic_iterator:
            
            # get a batch, calculate loss with/without using SAE reconstruction.
            batch_tokens = activation_store.get_batch_tokens()
            ce_loss_with_recons = self.get_test_loss(batch_tokens, model)
            ce_loss_without_recons, normal_activations_cache = model.run_with_cache(
                batch_tokens,
                names_filter=self.cfg.hook_point,
                return_type = "loss",
                loss_per_token = True,
            )
            # ce_loss_without_recons = model.loss_fn(normal_logits, batch_tokens, True)
            # del normal_logits
            
            normal_activations = normal_activations_cache[self.cfg.hook_point]
            if self.cfg.hook_point_head_index is not None:
                normal_activations = normal_activations[:,:,self.cfg.hook_point_head_index]

            # calculate the difference in loss
            changes_in_loss = ce_loss_with_recons - ce_loss_without_recons
            changes_in_loss = changes_in_loss.cpu()
            
            # sample from the loss differences
            probs = F.relu(changes_in_loss) / F.relu(changes_in_loss).sum(dim=1, keepdim=True)
            changes_in_loss_dist = Categorical(probs)
            samples = changes_in_loss_dist.sample()
            
            assert samples.shape == (batch_size,), f"{samples.shape=}; {self.cfg.store_batch_size=}"
            
            end_idx = refill_idx + batch_size
            global_loss_increases[refill_idx:end_idx] = changes_in_loss[torch.arange(batch_size), samples]
            global_input_activations[refill_idx:end_idx] = normal_activations[torch.arange(batch_size), samples]
        
        return global_loss_increases, global_input_activations
    
    @torch.no_grad()
    def get_test_loss(self, batch_tokens, model):
        """
        A method for running the model with the SAE activations in order to return the loss.
        returns per token loss when activations are substituted in.
        """

        # if not self.cfg.is_transcoder:
        if self.cfg.is_sae:
            head_index = self.cfg.hook_point_head_index
            
            def standard_replacement_hook(activations, hook):
                activations = self.forward(activations)[0].to(activations.dtype)
                return activations
            
            def head_replacement_hook(activations, hook):
                new_actions = self.forward(activations[:,:,head_index])[0].to(activations.dtype)
                activations[:,:,head_index] = new_actions
                return activations
    
            replacement_hook = standard_replacement_hook if head_index is None else head_replacement_hook
            
            ce_loss_with_recons = model.run_with_hooks(
                batch_tokens,
                return_type="loss",
                fwd_hooks=[(self.cfg.hook_point, replacement_hook)],
            )
        elif self.cfg.is_transcoder:
            # TODO: currently, this only works with MLP transcoders
            assert("mlp" in self.cfg.out_hook_point)
            
            old_mlp = model.blocks[self.cfg.target_layer]
            class TranscoderWrapper(torch.nn.Module):
                def __init__(self, transcoder):
                    super().__init__()
                    self.transcoder = transcoder
                def forward(self, x):
                    return self.transcoder(x)[0]
            model.blocks[self.cfg.target_layer].mlp = TranscoderWrapper(self)
            ce_loss_with_recons = model.run_with_hooks(
                batch_tokens,
                return_type="loss"
            )
            model.blocks[self.cfg.target_layer] = old_mlp
        # elif self.cfg.is_crosscoder:
        #     old_mlp = model.blocks[self.cfg.hook_point_layer]
        #     class CrosscoderWrapper(torch.nn.Module):
        #         def __init__(self, crosscoder):
        #             super().__init__()
        #             self.crosscoder = crosscoder
        #         def forward(self, x):
        #             return self.crosscoder(x)[0]
        #     model.blocks[self.cfg.hook_point_layer].mlp = CrosscoderWrapper(self)
        #     ce_loss_with_recons = model.run_with_hooks(
        #         batch_tokens,
        #         return_type="loss"
        #     )
        #     model.blocks[self.cfg.hook_point_layer] = old_mlp
        
        return ce_loss_with_recons
        

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        
    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        '''
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        '''
        # print(self.W_dec.grad.shape, self.W_dec.data.shape)
        # if self.cfg.is_crosscoder:
        #     parallel_component = einops.einsum(
        #         self.W_dec.grad,
        #         self.W_dec.data,
        #         "layers d_sae d_out, layers d_sae d_out -> layers d_sae",
        #     )
            
        #     self.W_dec.grad -= einops.einsum(
        #         parallel_component,
        #         self.W_dec.data,
        #         "layers d_sae, layers d_sae d_out -> layers d_sae d_out",
        #     )
        # else:
        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_out, d_sae d_out -> d_sae",
        )
        
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_out -> d_sae d_out",
        )
    
    def save_model(self, path: str):
        '''
        Basic save function for the model. Saves the model's state_dict and the config used to train it.
        '''
        
        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        
        state_dict = {
            "cfg": self.cfg,
            "state_dict": self.state_dict()
        }
        
        if path.endswith(".pt"):
            torch.save(state_dict, path)
        elif path.endswith("pkl.gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            raise ValueError(f"Unexpected file extension: {path}, supported extensions are .pt and .pkl.gz")
        
        
        print(f"Saved model to {path}")
    
    @classmethod
    def load_from_pretrained(cls,configs, path: str):
        '''
        Load function for the model. Loads the model's state_dict and the config used to train it.
        This method can be called directly on the class, without needing an instance.
        '''

        # Ensure the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at specified path: {path}")

        # Load the state dictionary
        if path.endswith(".pt"):
            try:
                print(torch.backends.mps.is_available())
                if torch.backends.mps.is_available():
                    state_dict = torch.load(path, map_location="mps", weights_only=False)
                    state_dict["cfg"].device = "mps"
                else:
                    state_dict = torch.load(path, weights_only=False)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pt file: {e}")
            
        elif path.endswith(".pkl.gz"):
            try:
                with gzip.open(path, 'rb') as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl.gz file: {e}")
        elif path.endswith(".pkl"):
            try:
                with open(path, 'rb') as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl file: {e}")
        else:
            raise ValueError(f"Unexpected file extension: {path}, supported extensions are .pt, .pkl, and .pkl.gz")

        # Ensure the loaded state contains both 'cfg' and 'state_dict'
        if 'cfg' not in state_dict or 'state_dict' not in state_dict:
            raise ValueError("The loaded state dictionary must contain 'cfg' and 'state_dict' keys")

        # Create an instance of the class using the loaded configuration
        instance = cls(cfg=state_dict["cfg"])
        instance.load_state_dict(state_dict["state_dict"])

        return instance

    def get_name(self):
        sae_name = f"sparse_autoencoder_{self.cfg.model_name}_{self.cfg.hook_point}_{self.cfg.d_sae}"
        return sae_name


def load_gemma_scope_transcoder(
    path: str,
    layer: int,
    device: Optional[torch.device] = torch.device("cuda"),
    dtype: Optional[torch.dtype] = torch.float32,
    revision: Optional[str] = None,
) -> SingleLayerTranscoder:
    if os.path.isfile(path):
        path_to_params = path
    else:
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-transcoders",
            filename=path,
            revision=revision,
            force_download=False,
        )

    # load the parameters, have to rename the threshold key,
    # as ours is nested inside the activation_function module
    param_dict = np.load(path_to_params)
    param_dict = {k: torch.tensor(v, device=device, dtype=dtype) for k, v in param_dict.items()}
    param_dict["activation_function.threshold"] = param_dict["threshold"]
    del param_dict["threshold"]

    # create the transcoders
    d_model = param_dict["W_enc"].shape[0]
    d_transcoder = param_dict["W_enc"].shape[1]

    # dummy JumpReLU; will get loaded via load_state_dict
    activation_function = JumpReLU(0.0, 0.1)
    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(d_model, d_transcoder, activation_function, layer)
    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder

from contextlib import contextmanager
import copy

@contextmanager
def patch_attr(obj, name, value):
    old = getattr(obj, name, None)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)

def load_my_transcoder(param_dict, configs, layer: int):
    # with torch.device("meta"):
    # print(configs)
    dev = configs.device_map[layer]
    # 如果是字符串，转成 torch.device
    if isinstance(dev, str):
        dev = torch.device(dev)

    # 临时覆写 configs.device，仅在构造这个层时生效
    with patch_attr(configs, "device", dev):
        # print(dev, configs.device)
        transcoder = SingleLayerTranscoder(configs, Relu(), layer)

    # transcoder = SingleLayerTranscoder(configs, Relu(), layer)
    transcoder.load_state_dict(param_dict, assign=True, strict=False)
    # transcoder.load_from_pretrained(model_path)
    return transcoder

def load_relu_transcoder(
    path: str,
    layer: int,
    device: torch.device = torch.device("cuda"),
    dtype: Optional[torch.dtype] = torch.float32,
):
    param_dict = load_file(path, device=device.type)
    W_enc = param_dict["W_enc"]
    d_sae, d_model = W_enc.shape

    param_dict["W_enc"] = param_dict["W_enc"].T.contiguous()
    param_dict["W_dec"] = param_dict["W_dec"].T.contiguous()

    assert param_dict.get("log_thresholds") is None
    activation_function = F.relu
    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(
            d_model,
            d_sae,
            activation_function,
            layer,
            skip_connection=param_dict["W_skip"] is not None,
        )
    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder.to(dtype)


TranscoderSettings = namedtuple(
    "TranscoderSettings", ["transcoders", "feature_input_hook", "feature_output_hook", "scan"]
)


def load_trained_transcoder(
    configs,
    transcoder_config_file: str):
    # print(configs)
    max_layers = configs.n_layers
    transcoders = {}
    for l in range(max_layers):
        layer_path = os.path.join(transcoder_config_file, f"{l}.pt")

        state_dict = torch.load(layer_path, weights_only=False, map_location='cpu')['state_dict']
        configs.target_layer = l
        # configs.d_in = 192
        # configs.d_out = 192
        # configs.l1_coefficient = 0.0001
        transcoders[l] = load_my_transcoder(state_dict, configs,l)
        
        # state_dict = None
        # transcoders[l] = load_my_transcoder(state_dict, configs,l)
        
    return transcoders

def load_transcoder_set(
    transcoder_config_file: str,
    device: Optional[torch.device] = torch.device("cuda"),
    dtype: Optional[torch.dtype] = torch.float32,
    local_dir: Optional[str] = None
) -> TranscoderSettings:
    """Loads either a preset set of transformers, or a set specified by a file.

    Args:
        transcoder_config_file (str): _description_
        device (Optional[torch.device], optional): _description_. Defaults to torch.device('cuda').

    Returns:
        TranscoderSettings: A namedtuple consisting of the transcoder dict,
        and their feature input hook, feature output hook and associated scan.
    """

    scan = None
    # try to match a preset, and grab its config
    if transcoder_config_file == "gemma":
        package_path = resources.files(circuit_tracer)
        transcoder_config_file = package_path / "configs/gemmascope-l0-0.yaml"
        scan = "gemma-2-2b"
    elif transcoder_config_file == "llama":
        package_path = resources.files(circuit_tracer)
        transcoder_config_file = package_path / "configs/llama-relu.yaml"
        scan = "llama-3-131k-relu"

    with open(transcoder_config_file, "r") as file:
        config = yaml.safe_load(file)

    sorted_transcoder_configs = sorted(config["transcoders"], key=lambda x: x["layer"])
    if scan is None:
        # the scan defaults to a list of transcoder ids, preceded by the model's name
        model_name_no_slash = config["model_name"].split("/")[-1]
        scan = [
            f"{model_name_no_slash}/{transcoder_config['id']}"
            for transcoder_config in sorted_transcoder_configs
        ]

    hf_paths = [
        t["filepath"] for t in sorted_transcoder_configs if t["filepath"].startswith("hf://")
    ]
    local_map = download_hf_uris(hf_paths, local_dir=local_dir)

    transcoders = {}
    for transcoder_config in sorted_transcoder_configs:
        path = transcoder_config["filepath"]
        if path.startswith("hf://"):
            local_path = local_map[path]
            repo_id = parse_hf_uri(path).repo_id
            if "gemma-scope" in repo_id:
                transcoder = load_gemma_scope_transcoder(
                    local_path, transcoder_config["layer"], device=device, dtype=dtype
                )
            else:
                transcoder = load_relu_transcoder(
                    local_path, transcoder_config["layer"], device=device, dtype=dtype
                )
        else:
            transcoder = load_relu_transcoder(
                path, transcoder_config["layer"], device=device, dtype=dtype
            )
        assert transcoder.layer_idx not in transcoders, (
            f"Got multiple transcoders for layer {transcoder.layer_idx}"
        )
        transcoders[transcoder.layer_idx] = transcoder

    # we don't know how many layers the model has, but we need all layers from 0 to max covered
    assert set(transcoders.keys()) == set(range(max(transcoders.keys()) + 1)), (
        f"Each layer should have a transcoder, but got transcoders for layers "
        f"{set(transcoders.keys())}"
    )
    feature_input_hook = config["feature_input_hook"]
    feature_output_hook = config["feature_output_hook"]
    return TranscoderSettings(transcoders, feature_input_hook, feature_output_hook, scan)
