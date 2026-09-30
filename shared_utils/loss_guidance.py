from functools import partial

import numpy as np
import torch
from torchdiffeq import odeint
from torchsde import sdeint

from sc_exp_design.constants import PredictionFields, DataFields
from sc_exp_design.utils import match_shapes

from lambda_schedulers import LambdaScheduler
from sde import SDE


class LossGuidedFlow:
    def __init__(
        self,
        prior_flow,
        forward_model,
        loss_fn,
        prior_flow_map=None,
        fix_noise=True,
        num_forward_pass_per_sample=1_000,
        regularization=None,
        reg_strength=1e-3,
        n_time_steps_forward_model=10,
        solver_kwargs_forward_model={"method":"euler"},
        target_id="Region",
    ):
        self.prior_flow = prior_flow
        self.forward_model = forward_model
        self.loss_fn = loss_fn
        self.prior_flow_map = prior_flow_map
        self.fix_noise = fix_noise
        self.num_forward_pass_per_sample=num_forward_pass_per_sample
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.n_time_steps_forward_model = n_time_steps_forward_model
        self.solver_kwargs_forward_model = solver_kwargs_forward_model
        self.target_id = target_id
        self.prior_vf = prior_flow.velocity_field

    def compute_target_loss(self, target_pred_dict, optimal_condition):
        """"""
        if self.fix_noise:
            return torch.sum(
                torch.stack(
                    [(loss_fn(target_pred_dict[covariate], optimal_condition[covariate].to(target_pred_dict[covariate].device)).mean(1)) 
                    for covariate, loss_fn in self.loss_fn.items()],
                    dim = 0)
                , dim=0)
        else:
            return torch.sum(
                torch.stack(
                    [loss_fn(target_pred_dict[covariate], optimal_condition[covariate].to(target_pred_dict[covariate].device)) 
                    for covariate, loss_fn in self.loss_fn.items()],
                    dim = 0)
                , dim=0)

    def compute_one_step_prediction(
        self,
        t,
        xt,
        cond=None,
        source=None,
        cfg_guidance_strength=1.0
    ):
        # computing velocity field
        if self.prior_flow_map is not None:
            t_input = t[..., 0]
            return self.prior_flow_map.flow_map(
                t_input,
                torch.ones_like(t_input),
                xt
            )
        t = match_shapes(t, xt)
        vf_fn = self.prior_vf.get_vf_fn(
            cond=cond,
            source=source,
            cfg_guidance_strength=cfg_guidance_strength,
        )
        vt = vf_fn(t[:, 0], xt)
        return xt + (1 - t)*vt

    def compute_loss_gradients(self, t, xt, optimal_condition, non_linearity, noise):
        # handling shape of time tensor
        t = match_shapes(t, xt)
        if self.fix_noise:
            optimal_condition = {
                cov: cond.unsqueeze(1).repeat(xt.shape[0], self.num_forward_pass_per_sample, 1).to(xt.device) 
                for cov, cond in optimal_condition.items()
            }
        else:
            optimal_condition = {
                cov: cond.repeat(xt.shape[0], 1).to(xt.device) 
                for cov, cond in optimal_condition.items()
            }

        # function for computing loss from x_t
        def _compute_loss(xt):
            x1 = self.compute_one_step_prediction(t, xt)
            if non_linearity is not None:
                x1 = non_linearity(x1)
            batch_dict = {}
            x1_fwd = x1
            if noise is not None:
                batch_dict[DataFields.SOURCE_STATE] = noise
                if self.fix_noise:
                    x1_fwd = x1.unsqueeze(1).repeat(1, self.num_forward_pass_per_sample, 1)
            
            condition_repr = next(iter(self.forward_model.forward_model.train_data.data.perturbation_covariates))
            batch_dict[DataFields.PERTURBATION_DATA] = {
                condition_repr: x1_fwd
            }
            forward_out = self.forward_model.predict(
                batch_dict,
                no_grad=False,
                fix_noise=self.fix_noise,
                num_time_steps=self.n_time_steps_forward_model,
                solver_kwargs=self.solver_kwargs_forward_model
            )
            pred = forward_out[PredictionFields.TARGET_PREDICTION_DATA]
            phen_loss = self.compute_target_loss(pred, optimal_condition)
            if self.regularization is None:   
                return phen_loss
            elif self.regularization == "l1":
                reg = torch.abs(x1).sum(-1)
                return phen_loss + self.reg_strength* reg
            elif self.regularization == "l2":
                reg = (x1 **2).sum(-1)
                return phen_loss + self.reg_strength * reg
            else:
                raise NotImplementedError
        return torch.autograd.functional.vjp(_compute_loss, xt, torch.ones((xt.shape[0],), device=xt.device))

    def guided_vf_fn(
        self,
        t,
        xt,
        optimal_condition=None,
        lambda_scheduler: LambdaScheduler=None,
        non_linearity=None,
        noise=None,
    ):
        # computing unguided vf
        t = match_shapes(t, xt)
        vf_fn = self.prior_flow.velocity_field.get_vf_fn()
        vt = vf_fn(t[:, 0], xt)

        # computing guidance term
        loss, gt = self.compute_loss_gradients(t, xt, optimal_condition, non_linearity, noise=noise)
        self._loss_history.append(loss)

        # optional decay
        guidance_strength = lambda_scheduler.compute_lambda_t(t) if lambda_scheduler is not None else 1.0
        self._lambda_history.append(guidance_strength)
        return vt - guidance_strength*gt

    def recompute_losses_and_lambda_scheduler(
        self,
        time,
        traj,
        optimal_condition,
        noise,
        lambda_scheduler,
        non_linearity,
    ):
        losses = []
        for idx, t in enumerate(time):
            loss, gt = self.compute_loss_gradients(
                t,
                traj[idx, :, :],
                optimal_condition,
                non_linearity,
                noise,
            )
            losses.append(loss)
        losses = torch.stack(losses, dim=1)
        lambdas = lambda_scheduler.compute_lambda_t(time)
        return losses, lambdas

    def sample_posterior(
        self,
        N,
        optimal_condition,
        num_time_steps=50,
        solver_kwargs=None,
        lambda_scheduler:LambdaScheduler=None,
        non_linearity=None,
        sde_sampling=False,
    ):
        # initializing history for current sampling session
        self._loss_history = []
        self._lambda_history = []

        # default values for the solver arguments
        if solver_kwargs is None:
            solver_kwargs = {}
        solver_kwargs.setdefault("method", "euler")
        solver_kwargs.setdefault("atol", 1e-5)
        solver_kwargs.setdefault("rtol", 1e-5)

        # define discretization
        time = torch.linspace(0.0, 1.0, num_time_steps)

        # sampling source states
        source = torch.randn((N, self.prior_flow.cvf_config.flow_dim), device=self.prior_flow.device_id).float()

        # sampling fixed noise for the forward model
        if self.fix_noise:
            noise = self.forward_model.forward_model.noise_distribution(
                (N, self.num_forward_pass_per_sample, self.forward_model.forward_model.velocity_field.config.flow_dim)
            ).squeeze(dim=0).to(self.forward_model.forward_model.device)
        else:
            noise = None

        # set evaluation mode (determinism)
        self.forward_model.target_prediction_model.eval()
        self.forward_model.forward_model.velocity_field.eval()

        # handling velocity field function
        vf_fn = partial(
            self.guided_vf_fn,
            optimal_condition=optimal_condition,
            lambda_scheduler=lambda_scheduler,
            non_linearity=non_linearity,
            noise=noise,
        )
        Lstar = -torch.sum(optimal_condition[self.target_id ]*torch.log(optimal_condition[self.target_id ] + 1e-10), dim=-1, keepdim=True).item()
        if sde_sampling:
            sde = SDE(vf_fn)
            traj = sdeint(sde, source, time, **solver_kwargs)#.detach().cpu().numpy()
            loss_history, lambda_history = self.recompute_losses_and_lambda_scheduler(
                time,
                traj,
                optimal_condition,
                noise,
                lambda_scheduler,
                non_linearity,
            )
            traj = traj.detach().cpu().numpy()
            loss_history = loss_history.detach().cpu().numpy().T - Lstar
            lambda_history = lambda_history.detach().cpu().numpy()
        else:
            traj =  odeint(vf_fn, source, time, **solver_kwargs).detach().cpu().numpy()
            loss_history = torch.stack(self._loss_history, dim=0).detach().cpu().numpy() - Lstar
            lambda_history = torch.stack(self._lambda_history, dim=0).squeeze().detach().cpu().numpy()
        traj = np.permute_dims(traj, (1, 0, 2))
        return traj, loss_history.T, lambda_history.T, noise
