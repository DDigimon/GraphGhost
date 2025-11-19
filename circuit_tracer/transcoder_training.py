import torch
import os
from abc import ABC
from tqdm import tqdm
from torch.optim import Adam
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from functools import partial
import sys


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .transcoder.single_layer_transcoder import SingleLayerTranscoder
from .utils.activationstores_utils import ActivationsStore
from .utils.train_utils import get_scheduler
from .evaluation import eval_transcoder_l0_ce



def train_sae_on_language_model(
    configs,
    model: HookedTransformer,
    sparse_autoencoder: SingleLayerTranscoder,
    activation_store: ActivationsStore,
    n_checkpoints: int = 0,
    feature_sampling_method: str = "None",  # None, l2, or anthropic
    feature_sampling_window: int = 1000,  # how many training steps between resampling the features / considiring neurons dead
    feature_reinit_scale: float = 0.2,  # how much to scale the resampled features by
    use_wandb: bool = False,
    eval_fre: int = 50,
    use_eval: bool = True
):
    

    total_training_tokens = configs.total_training_tokens
    dead_feature_window = configs.dead_feature_window  # unless this window is larger feature sampling,
    dead_feature_estimation_method = configs.dead_feature_estimation_method
    dead_feature_threshold = configs.dead_feature_threshold
    resample_batches = configs.resample_batches
    is_sparse_connection = configs.is_sparse_connection
    checkpoint_path = configs.checkpoint_path
    lr_scheduler_name = configs.lr_scheduler_name
    lr_warm_up_steps = configs.lr_warm_up_steps
    lr = configs.lr
    batch_size = configs.batch_size
    
    save_transcoder_path = os.path.join(checkpoint_path)
    if os.path.exists(save_transcoder_path) is False:
        os.makedirs(save_transcoder_path, exist_ok=True)
    # print(lr_scheduler_name, lr_warm_up_steps, checkpoint_path)
    if feature_sampling_method is not None:
        feature_sampling_method = feature_sampling_method.lower()

    total_training_steps = total_training_tokens - 1 // batch_size 
    print(total_training_tokens)
    n_training_steps = 0
    n_resampled_neurons = 0
    steps_before_reset = 0
    if n_checkpoints > 0:
        checkpoint_thresholds = list(range(0, total_training_tokens, total_training_tokens // (n_checkpoints+1)))[1:]
    
    # n_forward_passes_since_fired = torch.zeros(configs.d_in, device=configs.device)
    # act_freq_scores = torch.zeros(configs.d_in, device=configs.device)
    
    n_forward_passes_since_fired = torch.zeros(configs.d_transcoder, device=configs.device)
    act_freq_scores = torch.zeros(configs.d_transcoder, device=configs.device)
    
    n_frac_active_tokens = 0
    
    optimizer = Adam(sparse_autoencoder.parameters(),
                     lr = lr)
    # print(lr_scheduler_name)
    scheduler = get_scheduler(
        scheduler_name = lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end = lr / 10, # heuristic for now. 
    )
    # sparse_autoencoder.initialize_b_dec(activation_store)
    sparse_autoencoder.to(configs.device)
    sparse_autoencoder.train()
    

    # if sparse_autoencoder.cfg.use_tqdm:
    for e in range(configs.epoch):
        print(f"Epoch {e+1}/{configs.epoch}")
        pbar = tqdm(total=total_training_tokens*configs.epoch, desc="Training SAE")
        max_ratio = float('inf')
        # Reset the activation store for each epoch
        # activation_store.reset()
        n_training_tokens = 0
        while n_training_tokens < total_training_tokens:
            # Do a training step.
            sparse_autoencoder.train()
            # Make sure the W_dec is still zero-norm
            sparse_autoencoder.set_decoder_norm_to_unit_norm()


            if (feature_sampling_method=="anthropic") and ((n_training_steps + 1) % dead_feature_window == 0):
                
                feature_sparsity = act_freq_scores / n_frac_active_tokens
                
                # if reset criterion is frequency in window, then then use that to generate indices.
                if dead_feature_estimation_method == "no_fire":
                    dead_neuron_indices = (act_freq_scores == 0).nonzero(as_tuple=False)[:, 0]
                elif dead_feature_estimation_method == "frequency":
                    dead_neuron_indices = (feature_sparsity < dead_feature_threshold).nonzero(as_tuple=False)[:, 0]
                
                if len(dead_neuron_indices) > 0:
                    
                    if len(dead_neuron_indices) > resample_batches * configs.store_batch_size:
                        print("Warning: more dead neurons than number of tokens. Consider sampling more tokens when resampling.")
                    
                    sparse_autoencoder.resample_neurons_anthropic(
                        dead_neuron_indices, 
                        model,
                        optimizer, 
                        activation_store
                    )

                    if use_wandb:
                        n_resampled_neurons = min(len(dead_neuron_indices), configs.store_batch_size * resample_batches)
                        # wandb.log(
                        #     {
                        #         "metrics/n_resampled_neurons": n_resampled_neurons,
                        #     },
                        #     step=n_training_steps,
                        # )
                    
                    # for now, we'll hardcode this.
                    current_lr = scheduler.get_last_lr()[0]
                    reduced_lr = current_lr / 10_000
                    increment = (current_lr - reduced_lr) / 10_000
                    optimizer.param_groups[0]['lr'] = reduced_lr
                    steps_before_reset = 10_000
                else:
                    print("No dead neurons, skipping resampling")
                
            # Resample dead neurons
            if (feature_sampling_method == "l2") and ((n_training_steps + 1) % dead_feature_window == 0):
                print("no l2 resampling currently. Please use anthropic resampling")
                
            # after resampling, reset the sparsity:
            if (n_training_steps + 1) % feature_sampling_window == 0:
                feature_sparsity = act_freq_scores / n_frac_active_tokens
                log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()

                # if use_wandb:
                #     wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
                #     wandb.log(
                #         {   
                #             "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
                #             "plots/feature_density_line_chart": wandb_histogram,
                #         },
                #         step=n_training_steps,
                #     )
                # if sparse_autoencoder.cfg.is_crosscoder:
                #     act_freq_scores = torch.zeros(sparse_autoencoder.cfg.max_layer, sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device)
                # else:
                act_freq_scores = torch.zeros(configs.d_transcoder, device=configs.device)
                n_frac_active_tokens = 0



            if (steps_before_reset > 0) and n_training_steps > 0:
                steps_before_reset -= 1
                optimizer.param_groups[0]['lr'] += increment
                if steps_before_reset == 0:
                    optimizer.param_groups[0]['lr'] = current_lr
            else:
                scheduler.step()
        
            optimizer.zero_grad(set_to_none=True)
            
            
            ghost_grad_neuron_mask = (n_forward_passes_since_fired > dead_feature_window).bool()
            # try:
            next_batch = activation_store.next_batch()
            # except StopIteration:
            #     break
            # assert(sparse_autoencoder.cfg.is_transcoder == activation_store.cfg.is_transcoder)
            # if not sparse_autoencoder.cfg.is_transcoder:
            if configs.is_sae:
                sae_in = next_batch
                # Forward and Backward Passes
                sae_out, feature_acts, loss, mse_loss, l1_loss, ghost_grad_loss = sparse_autoencoder(
                    sae_in,
                    ghost_grad_neuron_mask,
                    mse_target=sae_in
                )
            elif configs.is_transcoder:
                sae_in = next_batch[:, :configs.d_in]
                mlp_out = next_batch[:, configs.d_in:]
                sae_out, feature_acts, loss, mse_loss, l1_loss, ghost_grad_loss = sparse_autoencoder(
                    sae_in,
                    ghost_grad_neuron_mask,
                    mse_target=mlp_out
                )
            # elif sparse_autoencoder.cfg.is_crosscoder:
            #     sae_in = next_batch[:, : , :sparse_autoencoder.cfg.d_in]
            #     mlp_out = next_batch[:, : ,sparse_autoencoder.cfg.d_in:]
            #     sae_out, feature_acts, loss, mse_loss, l1_loss, ghost_grad_loss = sparse_autoencoder(
            #         sae_in,
            #         ghost_grad_neuron_mask,
            #         mse_target=mlp_out
            #     )


            spacon_loss = 0
            if is_sparse_connection:
                spacon_loss = sparse_autoencoder.get_sparse_connection_loss(l1_coeff)
                loss = loss + spacon_loss

            
            # if sparse_autoencoder.cfg.is_crosscoder:
                
            #     for i in range(sparse_autoencoder.cfg.max_layer):
            #         did_fire = ((feature_acts[i] > 0).float().sum(0) > 0)
            #         # print(feature_acts[i].shape, n_forward_passes_since_fired[i].shape, did_fire.shape)
            #         n_forward_passes_since_fired[i] += 1
            #         n_forward_passes_since_fired[i][did_fire] = 0
            # else:
            did_fire = ((feature_acts > 0).float().sum(0) > 0)
            n_forward_passes_since_fired += 1
            # print(f"n_forward_passes_since_fired: {n_forward_passes_since_fired.shape}, did_fire: {did_fire.shape}")
            n_forward_passes_since_fired[did_fire] = 0
            
            n_training_tokens += batch_size
            
            

            with torch.no_grad():
                # Calculate the sparsities, and add it to a list, calculate sparsity metrics
                n_frac_active_tokens += batch_size
                # if sparse_autoencoder.cfg.is_crosscoder:
                #     feature_sparsity = []
                #     for i in range(sparse_autoencoder.cfg.max_layer):
                #         # print(f'feature_acts[{i}].shape: {feature_acts[i].shape}')
                #         # print(f'act_freq_scores[{i}].shape: {act_freq_scores.shape}')
                #         act_freq_scores[i] += (feature_acts[i].abs() > 0).float().sum(0)
                #         feature_sparsity.append(act_freq_scores[i] / n_frac_active_tokens)
                #     # feature_sparsity = feature_sparsity.mean(0)  # average across layers
                #     feature_sparsity = torch.stack(feature_sparsity, dim=0).mean(0)  # average across layers
                # else:
                # print(f"feature_acts.shape: {feature_acts.shape}, act_freq_scores.shape: {act_freq_scores.shape}")
                act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
                feature_sparsity = act_freq_scores / n_frac_active_tokens

                if use_wandb and ((n_training_steps + 1) % eval_fre == 0):
                    # metrics for currents acts
                    l0 = (feature_acts > 0).float().sum(-1).mean()
                    current_learning_rate = optimizer.param_groups[0]["lr"]
                    
                    per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
                    total_variance = sae_in.pow(2).sum(-1)
                    explained_variance = 1 - per_token_l2_loss/total_variance
                    
                    # wandb.log(
                    #     {
                    #         # losses
                    #         "losses/mse_loss": mse_loss.item(),
                    #         "losses/l1_loss": l1_loss.item() / sparse_autoencoder.l1_coefficient, # normalize by l1 coefficient
                    #         "losses/ghost_grad_loss": ghost_grad_loss.item(),
                    #         "losses/overall_loss": loss.item(),
                    #         # variance explained
                    #         "metrics/explained_variance": explained_variance.mean().item(),
                    #         "metrics/explained_variance_std": explained_variance.std().item(),
                    #         "metrics/l0": l0.item(),
                    #         # sparsity
                    #         "sparsity/mean_passes_since_fired": n_forward_passes_since_fired.mean().item(),
                    #         "sparsity/n_passes_since_fired_over_threshold": ghost_grad_neuron_mask.sum().item(),
                    #         "sparsity/below_1e-5": (feature_sparsity < 1e-5)
                    #         .float()
                    #         .mean()
                    #         .item(),
                    #         "sparsity/below_1e-6": (feature_sparsity < 1e-6)
                    #         .float()
                    #         .mean()
                    #         .item(),
                    #         "sparsity/dead_features": (
                    #             feature_sparsity < dead_feature_threshold
                    #         )
                    #         .float()
                    #         .mean()
                    #         .item(),
                    #         "details/n_training_tokens": n_training_tokens,
                    #         "details/current_learning_rate": current_learning_rate,
                    #     },
                    #     step=n_training_steps,
                    # )

                # record loss frequently, but not all the time.
                # if use_wandb and ((n_training_steps + 1) % (wandb_log_frequency * 10) == 0):
                #     sparse_autoencoder.eval()
                #     run_evals(configs, sparse_autoencoder, activation_store, model, n_training_steps)
                #     sparse_autoencoder.train()
                # min_l0s = float('inf')# 1e6
                
                record_scores = {}
                
                ## evaluation
                # if (n_training_steps + 1) % (eval_fre * 10) == 0:
                #     sparse_autoencoder.eval()
                #     eval_results = run_evals(configs, sparse_autoencoder, activation_store, model, n_training_steps)
                #     ratios = eval_results['l0s']/eval_results['ce_loss']
                #     print(f"Ratio: {ratios}, CE {eval_results['ce_loss']}, L0 {eval_results['l0s']}")
                #     if use_eval:
                #         if ratios < max_ratio:
                #             max_ratio = ratios
                #             record_scores = eval_results
                #             path = os.path.join(save_transcoder_path,f"{configs.target_layer}.pt")
                #             # log_feature_sparsity_path = f"{checkpoint_path}/final_{sparse_autoencoder.get_name()}_log_feature_sparsity.pt"
                #             print(path)
                #             sparse_autoencoder.save_model(path)
                #             print(f"Ratio: {ratios}, CE loss: {eval_results['ce_loss']}, model saved to {path}")
                #     else:
                #         path = os.path.join(save_transcoder_path,f"{configs.target_layer}.pt")
                #         # log_feature_sparsity_path = f"{checkpoint_path}/final_{sparse_autoencoder.get_name()}_log_feature_sparsity.pt"
                #         print(path)
                #         sparse_autoencoder.save_model(path)
                #         print(f"Ratio: {ratios}, CE loss: {eval_results['ce_loss']}, model saved to {path}")

                #     sparse_autoencoder.train()
                
                
                # if sparse_autoencoder.cfg.use_tqdm:
                if is_sparse_connection:
                    pbar.set_description(
                        f"{n_training_steps}| MSE Loss {mse_loss.item():.3f} | L1 {l1_loss.item():.3f} | SCST {spacon_loss.item():.3f}"
                    )
                else:
                    pbar.set_description(
                        f"{n_training_steps}| MSE Loss {mse_loss.item():.3f} | L1 {l1_loss.item():.3f}"
                    )
                pbar.update(batch_size)
            
            loss.backward()
            sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
            optimizer.step()
            
            del sae_in, mlp_out
            del ghost_grad_neuron_mask
            del next_batch
            torch.cuda.empty_cache()

        

            # checkpoint if at checkpoint frequency
            if n_checkpoints > 0 and n_training_tokens > checkpoint_thresholds[0]:
                # cfg = sparse_autoencoder.cfg
                path = f"{checkpoint_path}/{n_training_tokens}_{sparse_autoencoder.get_name()}.pt"
                log_feature_sparsity_path = f"{checkpoint_path}/{n_training_tokens}_{sparse_autoencoder.get_name()}_log_feature_sparsity.pt"
                # sparse_autoencoder.save_model(path)
                try: log_feature_sparsity
                except NameError:
                    feature_sparsity = act_freq_scores / n_frac_active_tokens
                    log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()
                torch.save(log_feature_sparsity, log_feature_sparsity_path)
                checkpoint_thresholds.pop(0)
                if len(checkpoint_thresholds) == 0:
                    n_checkpoints = 0
                # if cfg.log_to_wandb:
                #     model_artifact = wandb.Artifact(
                #         f"{sparse_autoencoder.get_name()}", type="model", metadata=dict(cfg.__dict__)
                #     )
                #     model_artifact.add_file(path)
                #     wandb.log_artifact(model_artifact)
                    
                #     sparsity_artifact = wandb.Artifact(
                #         f"{sparse_autoencoder.get_name()}_log_feature_sparsity", type="log_feature_sparsity", metadata=dict(cfg.__dict__)
                #     )
                #     sparsity_artifact.add_file(log_feature_sparsity_path)
                #     wandb.log_artifact(sparsity_artifact)
                    
                
            n_training_steps += 1
            
        if use_eval == False:
            path = os.path.join(save_transcoder_path,f"{configs.target_layer}.pt")
            sparse_autoencoder.save_model(path)
            print(f"Model saved to {path}")

    
    # torch.save(log_feature_sparsity, log_feature_sparsity_path)
    # if sparse_autoencoder.cfg.log_to_wandb:
    #     sparsity_artifact = wandb.Artifact(
    #             f"{sparse_autoencoder.get_name()}_log_feature_sparsity", type="log_feature_sparsity", metadata=dict(cfg.__dict__)
    #         )
    #     sparsity_artifact.add_file(log_feature_sparsity_path)
    #     wandb.log_artifact(sparsity_artifact)
        

    return sparse_autoencoder, record_scores


@torch.no_grad()
def run_evals(configs, sparse_autoencoder: SingleLayerTranscoder, activation_store: ActivationsStore, model: HookedTransformer, n_training_steps: int):
    
    hook_point = configs.hook_point
    hook_point_layer = configs.target_layer
    hook_point_head_index = configs.hook_point_head_index
    
    ### Evals
    # if configs.from_pretrained:
    eval_tokens = activation_store.get_batch_tokens()
    # else:
    #     eval_tokens = activation_store.get_my_batch()
    
    # Get Reconstruction Score
    recons_score, ntp_loss, recons_loss, zero_abl_loss = get_recons_loss(configs, sparse_autoencoder, model, activation_store, eval_tokens)
    
    # get cache
    _, cache = model.run_with_cache(eval_tokens, prepend_bos=False, names_filter=[get_act_name("pattern", hook_point_layer), hook_point])
    
    # get act
    if configs.hook_point_head_index is not None:
        original_act = cache[configs.hook_point][:,:,configs.hook_point_head_index]
    else:
        original_act = cache[configs.hook_point]
        
    sae_out, feature_acts, _, _, _, _ = sparse_autoencoder(
        original_act
    )
    patterns_original = cache[get_act_name("pattern", hook_point_layer)][:,hook_point_head_index].detach().cpu()
    del cache
    
    if "cuda" in str(configs.device):
        torch.cuda.empty_cache()
    
    l2_norm_in = torch.norm(original_act, dim=-1)
    l2_norm_out = torch.norm(sae_out, dim=-1)
    l2_norm_ratio = l2_norm_out / l2_norm_in
    
    # wandb.log(
    #     {

    #         # l2 norms
    #         "metrics/l2_norm": l2_norm_out.mean().item(),
    #         "metrics/l2_ratio": l2_norm_ratio.mean().item(),
            
    #         # CE Loss
    #         "metrics/CE_loss_score": recons_score,
    #         "metrics/ce_loss_without_sae": ntp_loss,
    #         "metrics/ce_loss_with_sae": recons_loss,
    #         "metrics/ce_loss_with_ablation": zero_abl_loss,
            
    #     },
    #     step=n_training_steps,
    # )
    
    print(f"metrics/l2_norm {l2_norm_out.mean().item()}",
            f"metrics/l2_ratio {l2_norm_ratio.mean().item()}",
            f"metrics/CE_loss_score (reconstruction score) {recons_score}",
            f"metrics/ce_loss_without_sae (next token prediction loss) {ntp_loss}",
            f"metrics/ce_loss_with_sae (reconstruction loss) {recons_loss}",
            f"metrics/ce_loss_with_ablation {zero_abl_loss}")
    eval_results = eval_transcoder_l0_ce(configs, model, sparse_autoencoder)
    
    
    head_index = configs.hook_point_head_index

    def standard_replacement_hook(activations, hook):
        activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
        return activations

    def head_replacement_hook(activations, hook):
        new_actions = sparse_autoencoder.forward(activations[:,:,head_index])[0].to(activations.dtype)
        activations[:,:,head_index] = new_actions
        return activations

    head_index = configs.hook_point_head_index
    replacement_hook = standard_replacement_hook if head_index is None else head_replacement_hook
    
    # get attn when using reconstructed activations
    with model.hooks(fwd_hooks=[(hook_point, partial(replacement_hook))]):
        _, new_cache = model.run_with_cache(eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)])
        patterns_reconstructed = new_cache[get_act_name("pattern", hook_point_layer)][:,hook_point_head_index].detach().cpu()
        del new_cache
        
    # get attn when using reconstructed activations
    with model.hooks(fwd_hooks=[(hook_point, partial(zero_ablate_hook))]):
        _, zero_ablation_cache = model.run_with_cache(eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)])
        patterns_ablation = zero_ablation_cache[get_act_name("pattern", hook_point_layer)][:,hook_point_head_index].detach().cpu()
        del zero_ablation_cache

    # if dealing with a head SAE, do the head metrics.
    if configs.hook_point_head_index:
        
        # show patterns before/after
        # fig_patterns_original = px.imshow(patterns_original[0].numpy(), title="original attn scores",
        #     color_continuous_midpoint=0, color_continuous_scale="RdBu")
        # fig_patterns_original.update_layout(coloraxis_showscale=False)         # hide colorbar 
        # wandb.log({"attention/patterns_original": wandb.Plotly(fig_patterns_original)}, step = n_training_steps)
        # fig_patterns_reconstructed = px.imshow(patterns_reconstructed[0].numpy(), title="reconstructed attn scores",
        #         color_continuous_midpoint=0, color_continuous_scale="RdBu")
        # fig_patterns_reconstructed.update_layout(coloraxis_showscale=False)         # hide colorbar
        # wandb.log({"attention/patterns_reconstructed": wandb.Plotly(fig_patterns_reconstructed)}, step = n_training_steps)
        
        kl_result_reconstructed = kl_divergence_attention(patterns_original, patterns_reconstructed)
        kl_result_reconstructed = kl_result_reconstructed.sum(dim=-1).numpy()
        # print(kl_result.mean().item())
        # px.imshow(kl_result, title="KL Divergence", width=800, height=800,
        #       color_continuous_midpoint=0, color_continuous_scale="RdBu").show()
        # px.histogram(kl_result.flatten()).show()
        # px.line(kl_result.mean(0), title="KL Divergence by Position").show()
        
        kl_result_ablation = kl_divergence_attention(patterns_original, patterns_ablation)
        kl_result_ablation = kl_result_ablation.sum(dim=-1).numpy()
        # print(kl_result.mean().item())
        # # px.imshow(kl_result, title="KL Divergence", width=800, height=800,
        # #       color_continuous_midpoint=0, color_continuous_scale="RdBu").show()
        # px.histogram(kl_result.flatten()).show()
        # px.line(kl_result.mean(0), title="KL Divergence by Position").show()
    
        # wandb.log(
        #     {

        #       "metrics/kldiv_reconstructed": kl_result_reconstructed.mean().item(),
        #       "metrics/kldiv_ablation": kl_result_ablation.mean().item(),
                
        #     },
        #     step=n_training_steps,
        # )
    return eval_results

@torch.no_grad()
def get_recons_loss(configs,sparse_autoencoder, model, activation_store, batch_tokens):
    hook_point = configs.hook_point
    loss = model(batch_tokens, return_type="loss")

    head_index = configs.hook_point_head_index

    def standard_replacement_hook(activations, hook):
        activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
        return activations

    def head_replacement_hook(activations, hook):
        new_actions = sparse_autoencoder.forward(activations[:,:,head_index])[0].to(activations.dtype)
        activations[:,:,head_index] = new_actions
        return activations

    replacement_hook = standard_replacement_hook if head_index is None else head_replacement_hook
    recons_loss = model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(hook_point, partial(replacement_hook))],
    )

    zero_abl_loss = model.run_with_hooks(
        batch_tokens, return_type="loss", fwd_hooks=[(hook_point, zero_ablate_hook)]
    )

    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    return score, loss, recons_loss, zero_abl_loss


def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1]).to(mlp_post.dtype)
    return mlp_post


def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.0
    return mlp_post


def kl_divergence_attention(y_true, y_pred):

    # Compute log probabilities for KL divergence
    log_y_true = torch.log2(y_true + 1e-10)
    log_y_pred = torch.log2(y_pred + 1e-10)

    return y_true * (log_y_true - log_y_pred)

