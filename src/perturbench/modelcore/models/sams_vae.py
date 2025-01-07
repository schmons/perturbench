import torch
import torch.nn.functional as F
import lightning as L
from torch.distributions.kl import kl_divergence
import torch.distributions as dist
from ..nn.vae import BaseEncoder
from ..nn.mlp import MLP
from .base import PerturbationModel
from perturbench.data.types import Batch
from typing import Tuple

from ..nn.mlp import gumbel_softmax_bernoulli

class SparseAdditiveVAE(PerturbationModel):

    """
    Sparse Additive Variational Autoencoder (VAE) model, following the model proposed in the paper:

    Bereket, Michael, and Theofanis Karaletsos. 
    "Modelling Cellular Perturbations with the Sparse Additive Mechanism Shift Variational Autoencoder." 
    Advances in Neural Information Processing Systems 36 (2024).

    Attributes:
        n_genes (int): Number of genes.
        n_perts (int): Number of perturbations.
        lr (int): Learning rate.
        wd (int): Weight decay.
        lr_scheduler_freq (int): Frequency of the learning rate scheduler.
        lr_scheduler_patience (int): Patience of the learning rate scheduler.
        lr_scheduler_factor (int): Factor of the learning rate scheduler.
        latent_dim (int): Latent dimension.
        sparse_additive_mechanism (bool): Whether to use sparse additive mechanism.
        mean_field_encoding (bool): Whether to use mean field encoding.
        inject_covariates_encoder (bool): Whether to inject covariates in the encoder.
        inject_covariates_decoder (bool): Whether to inject covariates in the decoder.
        mask_prior_probability (float): The target probability for the masks.
        datamodule (L.LightningDataModule | None): LightningDataModule for data loading.

    Methods:
        reparameterize(mu, log_var): Reparametrizes the Gaussian distribution.
        training_step(batch, batch_idx): Performs a training step.
        validation_step(batch, batch_idx): Performs a validation step.
        configure_optimizers(): Configures the optimizers.

    """

    def __init__(
        self, 
        n_genes: int,
        n_perts: int,
        n_layers_encoder_x: int = 2,
        n_layers_encoder_e: int = 2,
        n_layers_decoder: int = 3,
        hidden_dim_x: int = 850,
        hidden_dim_cond: int = 128,
        latent_dim: int = 40,
        dropout: float = 0.2,
        inject_covariates_encoder: bool = False,
        inject_covariates_decoder: bool = False,
        mask_prior_probability: float = 0.01,
        lr: int | None = None,
        wd: int | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: int | None = None,
        softplus_output: bool = True,
        datamodule: L.LightningDataModule | None=None,
    ) -> None:
        """
        Initializes the SparseAdditiveVAE model.

        Args:
            n_genes (int): Number of genes.
            n_perts (int): Number of perturbations.
            n_layers_encoder_x (int): Number of layers in the encoder for x.
            n_layers_encoder_e (int): Number of layers in the encoder for e.
            n_layers_decoder (int): Number of layers in the decoder.
            hidden_dim_x (int): Hidden dimension for x.
            hidden_dim_cond (int): Hidden dimension for the conditional input.
            latent_dim (int): Latent dimension.
            lr (int): Learning rate.
            wd (int): Weight decay.
            lr_scheduler_freq (int): Frequency of the learning rate scheduler.
            lr_scheduler_patience (int): Patience of the learning rate scheduler.
            lr_scheduler_factor (int): Factor of the learning rate scheduler.
            inject_covariates_encoder (bool): Whether to inject covariates in the encoder.
            inject_covariates_decoder (bool): Whether to inject covariates in the decoder.
            mask_prior_probability (float): The target probability for the masks.
            softplus_output (bool): Whether to apply a softplus activation to the
                output of the decoder to enforce non-negativity
            datamodule (L.LightningDataModule | None): LightningDataModule for data loading.

        Returns:
            None
        """
        
        super(SparseAdditiveVAE, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor
        )
        self.save_hyperparameters(ignore=["datamodule"])

        if n_genes is not None:
            self.n_genes = n_genes
        if n_perts is not None:
            self.n_perts = n_perts

        self.latent_dim = latent_dim
        self.latent_dim_pert = latent_dim * self.n_perts
        self.inject_covariates_encoder = inject_covariates_encoder
        self.inject_covariates_decoder = inject_covariates_decoder
        self.mask_prior_probability = mask_prior_probability
        self.softplus_output = softplus_output
        
        print(datamodule.train_dataset.transform.keys())
        
        perturbations_all = datamodule.train_dataset.transform['perturbations'](list(datamodule.train_dataset.perturbations))
        self.perturbations_all_sum = perturbations_all.sum(axis=0)

        if inject_covariates_encoder or inject_covariates_decoder:
            if datamodule is None or datamodule.context is None:
                raise ValueError("If inject_covariates is True, datamodule must be provided")

        encoder_input_dim = self.n_genes + self.n_total_covariates if self.inject_covariates_encoder else self.n_genes
        decoder_input_dim = latent_dim + self.n_total_covariates if self.inject_covariates_decoder else latent_dim


        self.encoder_x = BaseEncoder(
            input_dim=encoder_input_dim + self.latent_dim, 
            hidden_dim=hidden_dim_x, 
            latent_dim=latent_dim,
            n_layers=n_layers_encoder_x
        )

        self.encoder_e = BaseEncoder(
                input_dim=latent_dim + self.n_perts, 
                hidden_dim=hidden_dim_x, 
                latent_dim=latent_dim,
                n_layers=n_layers_encoder_e
            )

        self.m_logits = torch.nn.Parameter(-torch.ones((self.n_perts, self.latent_dim)))
      
        self.decoder = MLP(
            decoder_input_dim, 
            hidden_dim_x, 
            self.n_genes, 
            n_layers_decoder,
            dropout=dropout
            )
        
    def forward(
        self, 
        observed_perturbed_expression: torch.Tensor,   
        perturbation: torch.Tensor,                   
        covariates: dict
        ) -> Tuple:
        batch_size = observed_perturbed_expression.shape[0]

        perturbations_per_cell = perturbation.sum(axis=1)

        if self.inject_covariates_encoder or self.inject_covariates_decoder:
            merged_covariates = torch.cat([cov.squeeze() for cov in covariates.values()], dim=1)
        
        if self.inject_covariates_encoder:
            observed_expression_with_covariates = torch.cat([observed_perturbed_expression, merged_covariates.to(self.device)], dim=1)
        else:
            observed_expression_with_covariates = observed_perturbed_expression

        m_probs = torch.sigmoid(self.m_logits)     
        m = gumbel_softmax_bernoulli(m_probs)      
        
        # Get indices where perturbations are active (1s)
        z_p_index_batch, z_p_index_pert = torch.where(perturbation.bool())
        
        # Initialize z_p with zeros early
        z_p = torch.zeros((batch_size, self.latent_dim), device=self.device)
        # Initialize e_mu and e_log_var as None
        e_mu, e_log_var = None, None
        
        # Only process perturbations if there are any in the batch
        if z_p_index_batch.nelement() > 0:
            m_t = torch.cat([
                m[perturbation[i].bool()] for i in range(batch_size) if perturbation[i].bool().any()
            ])
            perturbation_expanded = perturbation.repeat_interleave(perturbations_per_cell.int(), dim=0)
            mask_and_perturbation = torch.cat([m_t, perturbation_expanded], dim=-1)
            e_mu, e_log_var = self.encoder_e(mask_and_perturbation)
            e_t = self.reparameterize(e_mu, e_log_var)
            
            # Calculate element-wise product
            combined_effect = m_t * e_t  
            
            # Use scatter_add_ to sum the effects for each batch sample
            z_p.index_add_(0, z_p_index_batch, combined_effect)

        observed_expression_with_covariates_and_z_p = torch.cat(
            [observed_expression_with_covariates, z_p], dim=-1)                         
        z_mu_x, z_log_var_x = self.encoder_x(observed_expression_with_covariates_and_z_p)  
            
        z_basal = self.reparameterize(z_mu_x, z_log_var_x)                                 
        z = z_basal + z_p                                                                   

        if self.inject_covariates_decoder:
            z = torch.cat([z, merged_covariates], dim=1)

        if self.softplus_output:
            x_sample = F.softplus(self.decoder(z))
        else:
            x_sample = self.decoder(z)

        # Define distributions for kl_divergence     
        q_z = dist.Normal(loc=z_mu_x, scale=torch.exp(0.5 * z_log_var_x))                   # Shape: (batch_size, latent_dim)
        p_z = dist.Normal(loc=torch.zeros_like(z_mu_x),                                     # Shape: (batch_size, latent_dim)
                         scale=torch.ones_like(z_mu_x))
        
        # Initialize kl_qe_pe with zeros
        kl_qe_pe = torch.zeros(batch_size, device=self.device)
        
        # Calculate KL divergence only if there are active perturbations
        if e_mu is not None:
            # Calculate KL divergence for active perturbations
            q_e = dist.Normal(loc=e_mu, scale=torch.exp(0.5*e_log_var))
            p_e = dist.Normal(loc=torch.zeros_like(e_mu), scale=torch.ones_like(e_mu))
            kl_per_pert = kl_divergence(q_e, p_e).sum(axis=-1)  # Sum over latent dimensions
            
            # Add KL terms to the correct batch samples
            kl_qe_pe.index_add_(0, z_p_index_batch, kl_per_pert)

        # Apply adjustment factor
        adjustment_factor = 1 / (perturbation @ self.perturbations_all_sum.to(self.device))

        # Set adjustment_factor to 0 if it is 0 to avoid division by 0 for control values
        adjustment_factor[adjustment_factor.isinf()] = 0
        kl_qe_pe = kl_qe_pe * adjustment_factor

        mse = torch.nn.MSELoss(reduction='none')(observed_perturbed_expression, x_sample).sum(axis=-1).mean()  
        kl_qz_pz = kl_divergence(q_z, p_z).sum(axis=-1)                                      
        kl_sum = (kl_qz_pz + kl_qe_pe).mean()                                                       
            
        q_m = dist.Bernoulli(probs=torch.sigmoid(self.m_logits))                                            
        p_m = dist.Bernoulli(
            probs=self.mask_prior_probability *                                        
                torch.ones_like(self.m_logits)
            )
        kl_qm_pm = kl_divergence(q_m, p_m).sum(axis=-1) * perturbation.sum(axis=0) / self.perturbations_all_sum.to(self.device)
        kl_sum += kl_qm_pm.mean()
            
        return x_sample, mse, kl_sum

    def training_step(
        self, 
        batch: Batch, 
        batch_idx: int
        ) -> torch.Tensor:

        observed_perturbed_expression = batch.gene_expression.squeeze()
        perturbation = batch.perturbations.squeeze()
        covariates = batch.covariates

        _, mse, kl_sum  = self(observed_perturbed_expression, perturbation, covariates)
        loss = mse + kl_sum
        self.log("recon_loss", mse, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch))
        self.log("kl_div", kl_sum, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch))

        return loss

    def validation_step(
        self, 
        batch: Batch, 
        batch_idx: int
        ) -> torch.Tensor:

        observed_perturbed_expression = batch.gene_expression.squeeze()
        perturbation = batch.perturbations.squeeze()
        covariates = batch.covariates

        _, mse, kl_sum  = self(observed_perturbed_expression, perturbation, covariates)
        val_loss = mse + kl_sum
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch))
        return val_loss
    
    def predict(
        self, 
        batch: Batch
        ) -> torch.Tensor:

        observed_perturbed_expression = batch.gene_expression.squeeze().to(self.device)
        perturbation = batch.perturbations.squeeze().to(self.device)
        covariates = batch.covariates

        x_sample, mse, kl_sum  = self(observed_perturbed_expression, perturbation, covariates)
        return x_sample

    def reparameterize(
        self, 
        mu: torch.Tensor, 
        log_var: torch.Tensor,
        ) -> torch.Tensor:
        """
        Reparametrizes the Gaussian distribution so (stochastic) backpropagation can be applied.
        """
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std) 
        
        return mu + eps*std 
