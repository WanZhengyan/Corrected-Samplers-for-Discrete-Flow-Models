# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from math import ceil
from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch.nn import functional as F
from tqdm import tqdm

from flow_matching.path import MixtureDiscreteProbPath, MixtureDiscreteSoftmaxProbPath

from flow_matching.solver.solver import Solver
from flow_matching.utils import categorical, ModelWrapper
from .utils import get_nearest_times


class MixtureDiscreteEulerSolver(Solver):
    r"""Solver that simulates the CTMC process :math:`(X_t)_{t_{\text{init}}\leq t\leq t_{\text{final}}}` defined by :math:`p_t` the marginal probability path of ``path``.
    Given :math:`X_t \sim p_t`, the algorithm of solver step from :math:`t` to :math:`t+h` for the i-th coordinate is:

    .. math::

        \begin{align*}
            & X_1^i \sim p_{1|t}^i(\cdot|X_t)\\
            & \lambda^i \gets \sum_{x^i\ne X_t^i} u_t^i(x^i, X_t^i|X_1^i)\\
            & Z^i_{\text{change}} \sim U[0,1]\\
            & X_{t+h}^i \sim \begin{cases}
                \frac{u_t^i(\cdot, X_t^i|X_1^i)}{\lambda^i}(1-\delta_{X_t^i}(\cdot)) \text{ if $Z^i_{\text{change}}\le 1-e^{-h\lambda^i}$}\\
                \delta_{X_t^i}(\cdot) \text{ else }
            \end{cases}
        \end{align*}

    Where :math:`p_{1|t}(\cdot|X_t)` is the output of ``model``, and the conditional probability velocity is of the mixture probability path is:

    .. math::

        u_t^i(x^i, y^i|x_1^i) = \hat{u}_t^i(x^i, y^i|x_1^i) + c_{\text{div\_free}}\left[\hat{u}_t^i(x^i, y^i|x_1^i) - \check{u}_t^i(x^i, y^i|x_1^i) \right],

    where

    .. math::
        \hat{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{1-\kappa_t} \left[ \delta_{x_1^i}(x^i) - \delta_{y^i}(x^i) \right],

    and

    .. math::

        \check{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{\kappa_t}\left[ \delta_{y^i}(x^i) - p(x^i) \right].

    The source distribution :math:`p(x^i)` is given by ``p``.

    Args:
        model (ModelWrapper): trained with x-prediction, outputting posterior probabilities (in the range :math:`[0,1]`), output must be [..., vocabulary_size].
        path (MixtureDiscreteProbPath): Probability path used for x-prediction training.
        vocabulary_size (int): size of the discrete vocabulary.
        source_distribution_p (Optional[Tensor], optional): Source distribution, must be of shape [vocabulary_size]. Required only when divergence-free term for the probability velocity is non-zero. Defaults to None.
    """

    def __init__(
        self,
        model: ModelWrapper,
        path: MixtureDiscreteProbPath,
        vocabulary_size: int,
        source_distribution_p: Optional[Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.path = path
        self.vocabulary_size = vocabulary_size

        if source_distribution_p is not None:
            assert source_distribution_p.shape == torch.Size(
                [vocabulary_size]
            ), f"Source distribution p dimension must match the vocabulary size {vocabulary_size}. Got {source_distribution_p.shape}."

        self.source_distribution_p = source_distribution_p

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        div_free: Union[float, Callable[[float], float]] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        """
        Sample a sequence of discrete values from the given model.

        .. code-block:: python

            import torch
            from flow_matching.utils import ModelWrapper
            from flow_matching.solver import MixtureDiscreteEulerSolver

            class DummyModel(ModelWrapper):
                def __init__(self):
                    super().__init__(None)
                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    return ...

            model = DummyModel()
            solver = MixtureDiscreteEulerSolver(model=model)

            x_init = torch.LongTensor([122, 725])
            step_size = 0.001
            time_grid = torch.tensor([0.0, 1.0])

            result = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid)

        Args:
            x_init (Tensor): The initial state.
            step_size (Optional[float]): If float then time discretization is uniform with the given step size. If None then time discretization is set to be time_grid.
            div_free (Union[float, Callable[[float], float]]): The coefficient of the divergence-free term in the probability velocity. Can be either a float or a time dependent function. Defaults to 0.0.
            dtype_categorical (torch.dtype): Precision to use for categorical sampler. Defaults to torch.float32.
            time_grid (Tensor): The CTMC process is solved in the interval [time_grid[0], time_grid[-1]] and if step_size is None then time discretization is set by the time grid. Defaults to torch.tensor([0.0,1.0]).
            return_intermediates (bool): If True then return intermediate time steps according to time_grid. Defaults to False.
            verbose (bool): Whether to print progress bars. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Tensor: The sampled sequence of discrete values.
        """
        if not div_free == 0.0:
            assert (
                self.source_distribution_p is not None
            ), "Source distribution p must be specified in order to add a divergence-free term to the probability velocity."

        # Initialize the current state `x_t` with the initial state `X_0`.
        time_grid = time_grid.to(device=x_init.device)

        if step_size is None:
            # If step_size is None then set the t discretization to time_grid.
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
        else:
            # If step_size is float then t discretization is uniform with step size set by step_size.
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            assert (
                t_final - t_init
            ) > step_size, f"Time interval [time_grid[0], time_grid[-1]] must be larger than step_size. Got a time interval [{t_init}, {t_final}] and step_size {step_size}."

            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [t_init + step_size * i for i in range(n_steps)] + [t_final],
                device=x_init.device,
            )

            if return_intermediates:
                # get order of intermediate steps:
                order = torch.argsort(time_grid)
                # Compute intermediate steps to return via nearest points in t_discretization to time_grid.
                time_grid = get_nearest_times(
                    time_grid=time_grid, t_discretization=t_discretization
                )

        x_t = x_init.clone()
        steps_counter = 0
        res = []

        if return_intermediates:
            res = [x_init.clone()]

        if verbose:
            ctx = tqdm(total=t_final, desc=f"Step: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx:
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                # Sample x_1 ~ p_1|t( \cdot |x_t)
                p_1t = self.model(x=x_t, t=t.repeat(x_t.shape[0]), **model_extras)
                x_1 = categorical(p_1t.to(dtype=dtype_categorical))

                # Checks if final step
                if i == n_steps - 1:
                    x_t = x_1
                else:
                    # Compute u_t(x|x_t,x_1)
                    scheduler_output = self.path.scheduler(t=t)

                    k_t = scheduler_output.alpha_t
                    d_k_t = scheduler_output.d_alpha_t

                    delta_1 = F.one_hot(x_1, num_classes=self.vocabulary_size).to(
                        k_t.dtype
                    )
                    u = d_k_t / (1 - k_t) * delta_1

                    # Add divergence-free part
                    div_free_t = div_free(t) if callable(div_free) else div_free

                    if div_free_t > 0:
                        p_0 = self.source_distribution_p[(None,) * x_t.dim()]
                        u = u + div_free_t * d_k_t / (k_t * (1 - k_t)) * (
                            (1 - k_t) * p_0 + k_t * delta_1
                        )

                    # Set u_t(x_t|x_t,x_1) = 0
                    delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size)
                    u = torch.where(
                        delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                    )

                    # Sample x_t ~ u_t( \cdot |x_t,x_1)
                    intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                    mask_jump = torch.rand(
                        size=x_t.shape, device=x_t.device
                    ) < 1 - torch.exp(-h * intensity)

                    if mask_jump.sum() > 0:
                        x_t[mask_jump] = categorical(
                            u[mask_jump].to(dtype=dtype_categorical)
                        )

                steps_counter += 1
                t = t + h

                if return_intermediates and (t in time_grid):
                    res.append(x_t.clone())

                if verbose:
                    ctx.n = t.item()
                    ctx.refresh()
                    ctx.set_description(f"Step: {steps_counter}")

        if return_intermediates:
            if step_size is None:
                return torch.stack(res, dim=0)
            else:
                return torch.stack(res, dim=0)[order]
        else:
            return x_t


# New class
class MixtureDiscreteSoftmaxEulerSolver(Solver):
    r"""Solver that simulates the CTMC process :math:`(X_t)_{t_{\text{init}}\leq t\leq t_{\text{final}}}` defined by :math:`p_t` the marginal probability path of ``path``.
    Given :math:`X_t \sim p_t`, the algorithm of solver step from :math:`t` to :math:`t+h` for the i-th coordinate is:

    .. math::

        \begin{align*}
            & X_1^i \sim p_{1|t}^i(\cdot|X_t)\\
            & \lambda^i \gets \sum_{x^i\ne X_t^i} u_t^i(x^i, X_t^i|X_1^i)\\
            & Z^i_{\text{change}} \sim U[0,1]\\
            & X_{t+h}^i \sim \begin{cases}
                \frac{u_t^i(\cdot, X_t^i|X_1^i)}{\lambda^i}(1-\delta_{X_t^i}(\cdot)) \text{ if $Z^i_{\text{change}}\le 1-e^{-h\lambda^i}$}\\
                \delta_{X_t^i}(\cdot) \text{ else }
            \end{cases}
        \end{align*}

    Where :math:`p_{1|t}(\cdot|X_t)` is the output of ``model``, and the conditional probability velocity is of the mixture probability path is:

    .. math::

        u_t^i(x^i, y^i|x_1^i) = \hat{u}_t^i(x^i, y^i|x_1^i) + c_{\text{div\_free}}\left[\hat{u}_t^i(x^i, y^i|x_1^i) - \check{u}_t^i(x^i, y^i|x_1^i) \right],

    where

    .. math::
        \hat{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{1-\kappa_t} \left[ \delta_{x_1^i}(x^i) - \delta_{y^i}(x^i) \right],

    and

    .. math::

        \check{u}_t^i(x^i, y^i|x_1^i) = \frac{\dot{\kappa}_t}{\kappa_t}\left[ \delta_{y^i}(x^i) - p(x^i) \right].

    The source distribution :math:`p(x^i)` is given by ``p``.

    Args:
        model (ModelWrapper): trained with x-prediction, outputting posterior probabilities (in the range :math:`[0,1]`), output must be [..., vocabulary_size].
        path (MixtureDiscreteProbPath): Probability path used for x-prediction training.
        vocabulary_size (int): size of the discrete vocabulary.
        source_distribution_p (Optional[Tensor], optional): Source distribution, must be of shape [vocabulary_size]. Required only when divergence-free term for the probability velocity is non-zero. Defaults to None.
    """

    def __init__(
        self,
        model: ModelWrapper,
        path_txt: MixtureDiscreteSoftmaxProbPath,
        path_img: MixtureDiscreteSoftmaxProbPath,
        vocabulary_size_txt: int,
        vocabulary_size_img: int,
    ):
        super().__init__()
        self.model = model
        self.path_txt = path_txt
        self.path_img = path_img
        self.vocabulary_size_txt = vocabulary_size_txt
        self.vocabulary_size_img = vocabulary_size_img

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: Optional[float],
        div_free: Union[float, Callable[[float], float]] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        # callback: bool = False,
        **model_extras,
    ) -> Tensor:
        """
        Sample a sequence of discrete values from the given model.

        .. code-block:: python

            import torch
            from flow_matching.utils import ModelWrapper
            from flow_matching.solver import MixtureDiscreteEulerSolver

            class DummyModel(ModelWrapper):
                def __init__(self):
                    super().__init__(None)
                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    return ...

            model = DummyModel()
            solver = MixtureDiscreteEulerSolver(model=model)

            x_init = torch.LongTensor([122, 725])
            step_size = 0.001
            time_grid = torch.tensor([0.0, 1.0])

            result = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid)

        Args:
            x_init (Tensor): The initial state.
            step_size (Optional[float]): If float then time discretization is uniform with the given step size. If None then time discretization is set to be time_grid.
            div_free (Union[float, Callable[[float], float]]): The coefficient of the divergence-free term in the probability velocity. Can be either a float or a time dependent function. Defaults to 0.0.
            dtype_categorical (torch.dtype): Precision to use for categorical sampler. Defaults to torch.float32.
            time_grid (Tensor): The CTMC process is solved in the interval [time_grid[0], time_grid[-1]] and if step_size is None then time discretization is set by the time grid. Defaults to torch.tensor([0.0,1.0]).
            return_intermediates (bool): If True then return intermediate time steps according to time_grid. Defaults to False.
            verbose (bool): Whether to print progress bars. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Tensor: The sampled sequence of discrete values.
        """
        if not div_free == 0.0:
            assert (
                self.source_distribution_p is not None
            ), "Source distribution p must be specified in order to add a divergence-free term to the probability velocity."

        # Initialize the current state `x_t` with the initial state `X_0`.
        time_grid = time_grid.to(device=x_init.device)

        if step_size is None:
            # If step_size is None then set the t discretization to time_grid.
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
        else:
            # If step_size is float then t discretization is uniform with step size set by step_size.
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            assert (
                t_final - t_init
            ) > step_size, f"Time interval [time_grid[0], time_grid[-1]] must be larger than step_size. Got a time interval [{t_init}, {t_final}] and step_size {step_size}."

            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [t_init + step_size * i for i in range(n_steps)] + [t_final],
                device=x_init.device,
            )

            if return_intermediates:
                # get order of intermediate steps:
                order = torch.argsort(time_grid)
                # Compute intermediate steps to return via nearest points in t_discretization to time_grid.
                time_grid = get_nearest_times(
                    time_grid=time_grid, t_discretization=t_discretization
                )

        x_t = x_init.clone()
        steps_counter = 0
        res = []
        # Here are different
        if return_intermediates:
            if self.model.g_or_u == 'generation':
                res = [x_init.clone()[model_extras['datainfo']['image_token_mask']==1].reshape(x_init.shape[0], -1)]
            elif self.model.g_or_u =='understanding':
                res = [x_init.clone()[model_extras['datainfo']['text_token_mask']==1].reshape(x_init.shape[0], -1)]
            else:
                res = [x_init.clone()]


        if verbose:
            ctx = tqdm(total=time_grid[-1].item(), desc=f"Step: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx:
            original_x_t = x_t.clone()
            batch_size = original_x_t.shape[0]
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                # Sample x_1 ~ p_1|t( \cdot |x_t)
                # Here are different
                # length of p_1t_txt and p_1t_img are the same maybe
                # Here, the input of model is x_t, which is a concatenation of text and image tokens;
                # the image information is in data_info['understanding_img'] if data_info['has_understanding_img'] = True
                # p_1t_txt, p_1t_img, data_info = self.model(x=x_t, **model_extras)

                p_1t_txt, p_1t_img, data_info = self.model(x=x_t, **model_extras)
                if p_1t_txt is None: # only generate image
                    x_1 = categorical(p_1t_img.to(dtype=dtype_categorical))
                    x_1 = x_1[data_info['image_token_mask']==1].reshape(batch_size, -1)
                    x_t = x_t[data_info['image_token_mask']==1].reshape(batch_size, -1)
                    # x_1 = x_1 * data_info['image_token_mask'] + x_t * (1 - data_info['image_token_mask']) 
                elif p_1t_img is None: # only generate text
                    x_1 = categorical(p_1t_txt.to(dtype=dtype_categorical))
                    x_1 = x_1[data_info['text_token_mask']==1].reshape(batch_size, -1)
                    x_t = x_t[data_info['text_token_mask']==1].reshape(batch_size, -1)
                    # x_1 = x_1 * data_info['text_token_mask'] + x_t * (1 - data_info['text_token_mask']) 
                else: # both text and image
                    x_1_img = categorical(p_1t_img.to(dtype=dtype_categorical))
                    x_1_txt = categorical(p_1t_txt.to(dtype=dtype_categorical))
                    x_1_img = x_1_img[data_info['image_token_mask']==1].reshape(batch_size, -1)
                    x_1_txt = x_1_txt[data_info['text_token_mask']==1].reshape(batch_size, -1)
                    x_t_img = x_t[data_info['image_token_mask']==1].reshape(batch_size, -1)
                    x_t_txt = x_t[data_info['text_token_mask']==1].reshape(batch_size, -1)
                    # x_1_txt = x_1_txt * data_info['text_token_mask'] + x_t * (1 - data_info['text_token_mask']) 
                    # x_1_img = x_1_img * data_info['image_token_mask'] + x_t * (1 - data_info['image_token_mask']) 
                    # x_1 = x_1_txt * (1 - data_info['image_token_mask']) + x_1_img * data_info['image_token_mask'] 


                # Checks if final step
                # Here are different
                if i == n_steps - 1:
                    if p_1t_txt is None:
                        x_t = x_1
                    elif p_1t_img is None:
                        x_t = x_1
                    else:
                        x_t = original_x_t.clone()
                        x_t[data_info['image_token_mask']==1] = x_1_img.flatten()
                        x_t[data_info['text_token_mask']==1] = x_1_txt.flatten()

                    if return_intermediates:
                        res.append(x_t.clone())
                else:
                    if p_1t_txt is None:
                        # Compute p_t(x|x_1)
                        emb_x_1 = self.path_img.embedding(x_1)
                        prob_x_t = self.path_img.get_prob_distribution(emb_x_1, t) 
                        prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
                        # Comptute the metric
                        emb_x_t = self.path_img.embedding(x_t)
                        emb_x_t_flattened = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
                        emb_x_1_flattened = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)
                        distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) + torch.sum(emb_x_1_flattened**2, dim=1, keepdim=True) - 2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
                        distance_x_1_2_x = self.path_img.metric(emb_x_1)
                        distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
                        if t ==0 :
                            d_beta_t = 0
                        else:
                            d_beta_t = self.path_img.c * self.path_img.a * ((t / (1 - t)) ** (self.path_img.a - 1)) *  1 / ((1 - t) ** 2)
                        # get u (KO conditional velocity)
                        u = prob_x_t * d_beta_t * distance
                        # print(f"prob_x_t:{prob_x_t}")
                        # print(f"d_beta_t:{d_beta_t}")
                        # print(f"distance:{distance}")
                        # print(f"t:{t}, {t.dtype}")
                        u = u.reshape(x_1.shape[0], x_1.shape[1], -1)

                        # Set u_t(x_t|x_t,x_1) = 0
                        delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size_img)
                        u = torch.where(
                            delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                        )

                        # Sample x_t ~ u_t( \cdot |x_t,x_1)
                        intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                        # print(f"intensity:{intensity.sum()}")
                        mask_jump = torch.rand(
                            size=x_t.shape, device=x_t.device
                        ) < 1 - torch.exp(-h * intensity)
                        # torch.save(u, f'u_{u.device}.pt')
                        if mask_jump.sum() > 0:
                            x_t[mask_jump] = categorical(
                                u[mask_jump].to(dtype=dtype_categorical)
                            )
                        if return_intermediates:
                            res.append(x_t.clone())
                            # if callback:
                            #     yield x_t
                            # res.append(x_1.clone())
                        original_x_t[data_info['image_token_mask']==1] = x_t.flatten()
                        # original_x_t[data_info['image_token_mask']==1] = x_1.flatten()
                        x_t = original_x_t.clone()
                    elif p_1t_img is None:
                        # Compute p_t(x|x_1)
                        emb_x_1 = self.path_txt.embedding(x_1)
                        prob_x_t = self.path_txt.get_prob_distribution(emb_x_1, t) 
                        prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
                        # Comptute the metric
                        emb_x_t = self.path_txt.embedding(x_t)
                        emb_x_t_flattened = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
                        emb_x_1_flattened = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)
                        distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) + torch.sum(emb_x_1_flattened**2, dim=1, keepdim=True) - 2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
                        distance_x_1_2_x = self.path_txt.metric(emb_x_1)
                        distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
                        if t ==0 :
                            d_beta_t = 0
                        else:
                            d_beta_t = self.path_txt.c * self.path_txt.a * ((t / (1 - t)) ** (self.path_txt.a - 1)) *  1 / ((1 - t) ** 2)
                        # get u
                        u = prob_x_t * d_beta_t * distance
                        # print(f"prob_x_t:{prob_x_t}")
                        # print(f"d_beta_t:{d_beta_t}")
                        # print(f"distance:{distance}")
                        # print(f"t:{t}, {t.dtype}")
                        u = u.reshape(x_1.shape[0], x_1.shape[1], -1)

                        # Set u_t(x_t|x_t,x_1) = 0
                        delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size_txt)
                        u = torch.where(
                            delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                        )

                        # Sample x_t ~ u_t( \cdot |x_t,x_1)
                        intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                        mask_jump = torch.rand(
                            size=x_t.shape, device=x_t.device
                        ) < 1 - torch.exp(-h * intensity)
                        # torch.save(u, f'u_{u.device}.pt')
                        if mask_jump.sum() > 0:
                            x_t[mask_jump] = categorical(
                                u[mask_jump].to(dtype=dtype_categorical)
                            )
                        if return_intermediates:
                            res.append(x_t.clone())
                            # if callback:
                            #     yield x_t
                        original_x_t[data_info['text_token_mask']==1] = x_t.flatten()
                        x_t = original_x_t.clone()
                    else:
                        # The text part
                        x_t = x_t_txt.clone()
                        x_1 = x_1_txt.clone()
                        # Compute p_t(x|x_1)
                        emb_x_1 = self.path_txt.embedding(x_1)
                        prob_x_t = self.path_txt.get_prob_distribution(emb_x_1, t) 
                        prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
                        # Comptute the metric
                        emb_x_t = self.path_txt.embedding(x_t)
                        emb_x_t_flattened = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
                        emb_x_1_flattened = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)
                        distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) + torch.sum(emb_x_1_flattened**2, dim=1, keepdim=True) - 2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
                        distance_x_1_2_x = self.path_txt.metric(emb_x_1)
                        distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
                        if t ==0 :
                            d_beta_t = 0
                        else:
                            d_beta_t = self.path_txt.c * self.path_txt.a * ((t / (1 - t)) ** (self.path_txt.a - 1)) *  1 / ((1 - t) ** 2)
                        # get u
                        u = prob_x_t * d_beta_t * distance
                        # print(f"prob_x_t:{prob_x_t}")
                        # print(f"d_beta_t:{d_beta_t}")
                        # print(f"distance:{distance}")
                        # print(f"t:{t}, {t.dtype}")
                        u = u.reshape(x_1.shape[0], x_1.shape[1], -1)

                        # Set u_t(x_t|x_t,x_1) = 0
                        delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size_txt)
                        u = torch.where(
                            delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                        )

                        # Sample x_t ~ u_t( \cdot |x_t,x_1)
                        intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                        mask_jump = torch.rand(
                            size=x_t.shape, device=x_t.device
                        ) < 1 - torch.exp(-h * intensity)
                        # torch.save(u, f'u_{u.device}.pt')
                        if mask_jump.sum() > 0:
                            x_t[mask_jump] = categorical(
                                u[mask_jump].to(dtype=dtype_categorical)
                            )
                        original_x_t[data_info['text_token_mask']==1] = x_t.flatten()

                        # The image part
                        x_t = x_t_img.clone()
                        x_1 = x_1_img.clone()
                        emb_x_1 = self.path_img.embedding(x_1)
                        prob_x_t = self.path_img.get_prob_distribution(emb_x_1, t) 
                        prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
                        # Comptute the metric
                        emb_x_t = self.path_img.embedding(x_t)
                        emb_x_t_flattened = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
                        emb_x_1_flattened = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)
                        distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) + torch.sum(emb_x_1_flattened**2, dim=1, keepdim=True) - 2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
                        distance_x_1_2_x = self.path_img.metric(emb_x_1)
                        distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
                        if t ==0 :
                            d_beta_t = 0
                        else:
                            d_beta_t = self.path_img.c * self.path_img.a * ((t / (1 - t)) ** (self.path_img.a - 1)) *  1 / ((1 - t) ** 2)
                        # get u
                        u = prob_x_t * d_beta_t * distance
                        # print(f"prob_x_t:{prob_x_t}")
                        # print(f"d_beta_t:{d_beta_t}")
                        # print(f"distance:{distance}")
                        # print(f"t:{t}, {t.dtype}")
                        u = u.reshape(x_1.shape[0], x_1.shape[1], -1)

                        # Set u_t(x_t|x_t,x_1) = 0
                        delta_t = F.one_hot(x_t, num_classes=self.vocabulary_size_img)
                        u = torch.where(
                            delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                        )

                        # Sample x_t ~ u_t( \cdot |x_t,x_1)
                        intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                        mask_jump = torch.rand(
                            size=x_t.shape, device=x_t.device
                        ) < 1 - torch.exp(-h * intensity)
                        # torch.save(u, f'u_{u.device}.pt')
                        if mask_jump.sum() > 0:
                            x_t[mask_jump] = categorical(
                                u[mask_jump].to(dtype=dtype_categorical)
                            )
                        original_x_t[data_info['image_token_mask']==1] = x_t.flatten()
                        
                        x_t = original_x_t.clone()
                        if return_intermediates:
                            res.append(x_t.clone())

                steps_counter += 1
                t = t + h

                if verbose:
                    ctx.n = t.item()
                    ctx.refresh()
                    ctx.set_description(f"Step: {steps_counter}")

        # if return_intermediates and not callback:
        if return_intermediates:
            return torch.stack(res, dim=0)[:, 0, :].reshape(n_steps+1, -1)
        # elif callback:
        #     yield x_t
        else:
            return x_t



class MixtureDiscreteTimeCorrectedSolver(Solver):
    r"""Time-corrected sampler for discrete flow matching (Image Generation Only).
    
    Implements the time-corrected sampling algorithm from the paper.
    For each step k:
      1. Sample x_1 ~ p_{1|t}(·|Y_{k-1})
      2. Calculate λ_{k,i}^d at m grid points s_i = t_{k-1} + (i-1)/m * (t_k - t_{k-1})
      3. Sample (T_k^d, l_k^d) based on discretized process
      4. If T_k^d ≠ t_k, sample jump target z^d from Q_{T_k^d}
    
    Args:
        model (ModelWrapper): Model outputting posterior probabilities.
        path_img (MixtureDiscreteSoftmaxProbPath): Probability path for image generation.
        vocabulary_size_img (int): Size of the image vocabulary.
    """

    def __init__(
        self,
        model: ModelWrapper,
        path_img: MixtureDiscreteSoftmaxProbPath,
        vocabulary_size_img: int,
    ):
        super().__init__()
        self.model = model
        self.path_img = path_img
        self.vocabulary_size_img = vocabulary_size_img

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: float,
        m: int = 1,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        """
        Time-corrected sampler with numerical integration.

        Args:
            x_init (Tensor): The initial state.
            step_size (float): Time discretization step size.
            m (int): Number of grid points for numerical integration. Defaults to 1.
            dtype_categorical (torch.dtype): Precision for categorical sampler. Defaults to torch.float32.
            time_grid (Tensor): Process solved in [time_grid[0], time_grid[-1]]. Defaults to [0.0, 1.0].
            return_intermediates (bool): If True, return intermediate steps. Defaults to False.
            verbose (bool): Whether to print progress bars. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Tensor: The sampled sequence of discrete values.
        """
        # Initialize
        time_grid = time_grid.to(device=x_init.device)
        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()
        
        assert (t_final - t_init) > step_size, \
            f"Time interval must be larger than step_size. Got [{t_init}, {t_final}] and step_size {step_size}."

        n_steps = ceil((t_final - t_init) / step_size)
        t_discretization = torch.tensor(
            [t_init + step_size * i for i in range(n_steps)] + [t_final],
            device=x_init.device,
        )

        x_t = x_init.clone()
        steps_counter = 0
        res = []
        
        if return_intermediates:
            res = [x_init.clone()[model_extras['datainfo']['image_token_mask']==1].reshape(x_init.shape[0], -1)]

        if verbose:
            ctx = tqdm(total=time_grid[-1].item(), desc=f"Step: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx:
            original_x_t = x_t.clone()
            batch_size = original_x_t.shape[0]
            
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                # Step 1: Sample x_1 ~ p_{1|t}(·|Y_{k-1})
                _, p_1t_img, data_info = self.model(x=x_t, **model_extras)
                
                # Only process image generation
                x_1 = categorical(p_1t_img.to(dtype=dtype_categorical))
                x_1 = x_1[data_info['image_token_mask']==1].reshape(batch_size, -1)
                x_t = x_t[data_info['image_token_mask']==1].reshape(batch_size, -1)

                # Check if final step
                if i == n_steps - 1:
                    x_t = x_1
                    if return_intermediates:
                        res.append(x_t.clone())
                else:
                    # Time-corrected sampling
                    path = self.path_img
                    vocabulary_size = self.vocabulary_size_img
                    D = x_t.shape[1]
                    emb_x_1 = path.embedding(x_1)
                    
                    # Step 2: Calculate λ_{k,i}^d for i=1 to m at grid points s_i
                    lambda_ki_list = []  # Store intensity at each grid point
                    Q_list = []          # Store rate matrices at each grid point
                    
                    for j in range(1, m + 1):
                        s_j = t + (j - 1) / m * h
                        
                        # Compute Q_{s_j}(x, y | x_1) - the conditional rate matrix
                        prob_x_t = path.get_prob_distribution(emb_x_1, s_j)
                        prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
                        
                        emb_x_t = path.embedding(x_t)
                        emb_x_t_flattened = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
                        emb_x_1_flattened = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)
                        distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) + 
                                               torch.sum(emb_x_1_flattened**2, dim=1, keepdim=True) - 
                                               2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
                        distance_x_1_2_x = path.metric(emb_x_1)
                        distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
                        
                        if s_j == 0:
                            d_beta_t = 0
                        else:
                            d_beta_t = path.c * path.a * ((s_j / (1 - s_j)) ** (path.a - 1)) * 1 / ((1 - s_j) ** 2)
                        
                        Q_j = prob_x_t * d_beta_t * distance
                        Q_j = Q_j.reshape(x_t.shape[0], x_t.shape[1], -1)  # [batch, seq_len, vocab]
                        
                        # Set Q(x_t^d | x_t^d, x_1^d) = 0 (no self-transitions)
                        mask = F.one_hot(x_t, num_classes=vocabulary_size).bool()
                        Q_j = torch.where(mask, torch.zeros_like(Q_j), Q_j)
                        
                        # Calculate λ_{k,i}^d = sum_{z^d ≠ Y_{k-1}^d} Q_{s_i}^d(Y_{k-1}^d, z^d | x_1^d)
                        lambda_j = Q_j.sum(dim=-1)  # [batch, seq_len]
                        lambda_ki_list.append(lambda_j)
                        Q_list.append(Q_j)
                    
                    # Stack lambdas: [m, batch, seq_len]
                    lambda_ki = torch.stack(lambda_ki_list, dim=0)
                    lambda_ki = lambda_ki.permute(1, 2, 0)  # [batch, seq_len, m]
                    
                    # Step 3: Sample (T_k^d, l_k^d) for each coordinate d using discretized process
                    delta_t = h / m
                    
                    # Compute jump probabilities for each grid point l
                    # P(T_k^d = s_l, l_k^d = l) = exp(-δt * Σ_{i=1}^{l-1} λ_{k,i}^d) - exp(-δt * Σ_{i=1}^l λ_{k,i}^d)
                    exp_cumsum = torch.exp(-delta_t * lambda_ki.cumsum(dim=2))  # [batch, seq_len, m]
                    exp_cumsum_pad = torch.cat([torch.ones(batch_size, D, 1, device=x_t.device), exp_cumsum[:, :, :-1]], dim=2)  # [batch, seq_len, m]
                    
                    # P(no jump at coord d) = exp(-δt * Σ_{i=1}^m λ_{k,i}^d)
                    prob_no_jump = exp_cumsum[:, :, -1]  # [batch, seq_len]
                    
                    # Compute p_grid for jump probabilities
                    p_grid = exp_cumsum_pad - exp_cumsum  # [batch, seq_len, m]
                    
                    # Combine all probabilities: [batch, seq_len, m+1]
                    all_probs = torch.cat([p_grid, prob_no_jump.unsqueeze(2)], dim=2)
                    
                    # Sample jump location for each coordinate
                    jump_indices = categorical(all_probs.to(dtype=dtype_categorical))  # [batch, seq_len]

                    # Step 4: For coordinates where T_k^d ≠ t_k (jump_indices < m), sample target state
                    for b in range(batch_size):
                        mask_jump_b = jump_indices[b] < m  # [seq_len]
                        
                        if mask_jump_b.sum() > 0:
                            # Stack Q matrices for this sample: [m, seq_len, vocab]
                            Q_stacked_b = torch.stack([Q_list[j][b] for j in range(m)], dim=0)
                            Q_stacked_b = Q_stacked_b.permute(1, 0, 2)  # [seq_len, m, vocab]
                            
                            # Get jump indices for coordinates that jump
                            jump_idx_b = jump_indices[b][mask_jump_b]  # [num_jumps], where num_jumps = mask_jump_b.sum()
                            Q_stacked_b_masked = Q_stacked_b[mask_jump_b]  # [num_jumps, m, vocab]
                            
                            # Gather Q at jump times
                            jump_idx_expanded = jump_idx_b.unsqueeze(1).unsqueeze(2).expand(-1, -1, vocabulary_size)  # [num_jumps, 1, vocab]
                            Q_at_jump_b = torch.gather(Q_stacked_b_masked, 1, jump_idx_expanded).squeeze(1)  # [num_jumps, vocab]
                            
                            # Sample target state for jumping coordinates
                            x_t[b, mask_jump_b] = categorical(Q_at_jump_b.to(dtype=dtype_categorical))
                    
                    if return_intermediates:
                        res.append(x_t.clone())
                    
                    original_x_t[data_info['image_token_mask']==1] = x_t.flatten()
                    x_t = original_x_t.clone()

                steps_counter += 1
                t = t + h

                if verbose:
                    ctx.n = t.item()
                    ctx.refresh()
                    ctx.set_description(f"Step: {steps_counter}")

        if return_intermediates:
            return torch.stack(res, dim=0)[:, 0, :].reshape(n_steps+1, -1)
        else:
            return x_t


# New class
class MixtureDiscreteLocationCorrectedSolver(Solver):
    r"""Location-corrected sampler for discrete flow matching (Image Generation Only).
    
    Implements the location-corrected sampling algorithm from the paper.
    For each step k:
      1. Sample x_1^d ~ p_{1|t}(·|Y_{k-1}) for all coordinates d (in parallel)
      2. Calculate λ_{k,i}^d at m grid points for all coordinates d
      3. Sample (T_k^d, l_k^d) for all coordinates d based on discretized process
      4. Find T_k^{(j)} = j-th order statistic of {T_k^d}
      5. If T_k^{(j)} = t_k: jump coordinates with T_k^d ≠ t_k, set Y_k
      6. If T_k^{(j)} < t_k: 
         - Jump coordinates with T_k^d ≤ T_k^{(j)}
         - Resample x_1 and continue from T_k^{(j)}
    
    Args:
        model (ModelWrapper): Model outputting posterior probabilities.
        path_img (MixtureDiscreteSoftmaxProbPath): Probability path for image generation.
        vocabulary_size_img (int): Size of the image vocabulary.
    """

    def __init__(
        self,
        model: ModelWrapper,
        path_img: MixtureDiscreteSoftmaxProbPath,
        vocabulary_size_img: int,
    ):
        super().__init__()
        self.model = model
        self.path_img = path_img
        self.vocabulary_size_img = vocabulary_size_img

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: float,
        m: int = 1,
        j: int = 1,
        t_theta: float = 0.5,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        """
        Location-corrected sampler with numerical integration.

        Args:
            x_init (Tensor): The initial state.
            step_size (float): Time discretization step size.
            m (int): Number of grid points for numerical integration. Defaults to 1.
            j (int): Order statistic to use (1 for minimum, 2 for second minimum, etc.). Defaults to 1.
            dtype_categorical (torch.dtype): Precision for categorical sampler. Defaults to torch.float32.
            time_grid (Tensor): Process solved in [time_grid[0], time_grid[-1]]. Defaults to [0.0, 1.0].
            return_intermediates (bool): If True, return intermediate steps. Defaults to False.
            verbose (bool): Whether to print progress bars. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Tensor: The sampled sequence of discrete values.
        """
        # Initialize
        j = int(j)  # ensure j is an integer for indexing
        time_grid = time_grid.to(device=x_init.device)
        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()
        
        assert (t_final - t_init) > step_size, \
            f"Time interval must be larger than step_size. Got [{t_init}, {t_final}] and step_size {step_size}."

        n_steps = ceil((t_final - t_init) / step_size)
        t_discretization = torch.tensor(
            [t_init + step_size * i for i in range(n_steps)] + [t_final],
            device=x_init.device,
        )

        x_t = x_init.clone()
        steps_counter = 0
        res = []
        
        if return_intermediates:
            res = [x_init.clone()[model_extras['datainfo']['image_token_mask']==1].reshape(x_init.shape[0], -1)]

        if verbose:
            ctx = tqdm(total=time_grid[-1].item(), desc=f"Step: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx:
            original_x_t = x_t.clone()
            batch_size = original_x_t.shape[0]
            
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]
                if t < t_theta:

                    # Step 1: Sample x_1 ~ p_{1|t}(·|Y_{k-1})
                    _, p_1t_img, data_info = self.model(x=x_t, **model_extras)
                    
                    # Only process image generation
                    x_1 = categorical(p_1t_img.to(dtype=dtype_categorical))
                    x_1 = x_1[data_info['image_token_mask']==1].reshape(batch_size, -1)
                    x_t = x_t[data_info['image_token_mask']==1].reshape(batch_size, -1)

                    # Check if final step
                    if i == n_steps - 1:
                        x_t = x_1
                        if return_intermediates:
                            res.append(x_t.clone())
                    else:
                        # Time-corrected sampling
                        path = self.path_img
                        vocabulary_size = self.vocabulary_size_img
                        D = x_t.shape[1]
                        emb_x_1 = path.embedding(x_1)
                        
                        # Step 2: Calculate λ_{k,i}^d for i=1 to m at grid points s_i
                        lambda_ki_list = []  # Store intensity at each grid point
                        Q_list = []          # Store rate matrices at each grid point
                        
                        for grid_idx in range(1, m + 1):
                            s_j = t + (grid_idx - 1) / m * h
                            
                            # Compute Q_{s_j}(x, y | x_1) - the conditional rate matrix
                            prob_x_t = path.get_prob_distribution(emb_x_1, s_j)
                            prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
                            
                            emb_x_t = path.embedding(x_t)
                            emb_x_t_flattened = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
                            emb_x_1_flattened = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)
                            distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) + 
                                                torch.sum(emb_x_1_flattened**2, dim=1, keepdim=True) - 
                                                2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
                            distance_x_1_2_x = path.metric(emb_x_1)
                            distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
                            
                            if s_j == 0:
                                d_beta_t = 0
                            else:
                                d_beta_t = path.c * path.a * ((s_j / (1 - s_j)) ** (path.a - 1)) * 1 / ((1 - s_j) ** 2)
                            
                            Q_j = prob_x_t * d_beta_t * distance
                            Q_j = Q_j.reshape(x_t.shape[0], x_t.shape[1], -1)  # [batch, seq_len, vocab]
                            
                            # Set Q(x_t^d | x_t^d, x_1^d) = 0 (no self-transitions)
                            mask = F.one_hot(x_t, num_classes=vocabulary_size).bool()
                            Q_j = torch.where(mask, torch.zeros_like(Q_j), Q_j)
                            
                            # Calculate λ_{k,i}^d = sum_{z^d ≠ Y_{k-1}^d} Q_{s_i}^d(Y_{k-1}^d, z^d | x_1^d)
                            lambda_j = Q_j.sum(dim=-1)  # [batch, seq_len]
                            lambda_ki_list.append(lambda_j)
                            Q_list.append(Q_j)
                        
                        # Stack lambdas: [m, batch, seq_len]
                        lambda_ki = torch.stack(lambda_ki_list, dim=0)
                        lambda_ki = lambda_ki.permute(1, 2, 0)  # [batch, seq_len, m]
                        
                        # Step 3: Sample (T_k^d, l_k^d) for each coordinate d using discretized process
                        delta_t = h / m
                        
                        # Compute jump probabilities for each grid point l
                        # P(T_k^d = s_l, l_k^d = l) = exp(-δt * Σ_{i=1}^{l-1} λ_{k,i}^d) - exp(-δt * Σ_{i=1}^l λ_{k,i}^d)
                        exp_cumsum = torch.exp(-delta_t * lambda_ki.cumsum(dim=2))  # [batch, seq_len, m]
                        exp_cumsum_pad = torch.cat([torch.ones(batch_size, D, 1, device=x_t.device), exp_cumsum[:, :, :-1]], dim=2)  # [batch, seq_len, m]
                        
                        # P(no jump at coord d) = exp(-δt * Σ_{i=1}^m λ_{k,i}^d)
                        prob_no_jump = exp_cumsum[:, :, -1]  # [batch, seq_len]
                        
                        # Compute p_grid for jump probabilities
                        p_grid = exp_cumsum_pad - exp_cumsum  # [batch, seq_len, m]
                        
                        # Combine all probabilities: [batch, seq_len, m+1]
                        all_probs = torch.cat([p_grid, prob_no_jump.unsqueeze(2)], dim=2)
                        
                        # Sample jump location for each coordinate
                        jump_indices = categorical(all_probs.to(dtype=dtype_categorical))  # [batch, seq_len]

                        # Step 4: For coordinates where T_k^d ≠ t_k (jump_indices < m), sample target state
                        for b in range(batch_size):
                            mask_jump_b = jump_indices[b] < m  # [seq_len]
                            
                            if mask_jump_b.sum() > 0:
                                # Stack Q matrices for this sample: [m, seq_len, vocab]
                                Q_stacked_b = torch.stack([Q_list[qi][b] for qi in range(m)], dim=0)
                                Q_stacked_b = Q_stacked_b.permute(1, 0, 2)  # [seq_len, m, vocab]
                                
                                # Get jump indices for coordinates that jump
                                jump_idx_b = jump_indices[b][mask_jump_b]  # [num_jumps], where num_jumps = mask_jump_b.sum()
                                Q_stacked_b_masked = Q_stacked_b[mask_jump_b]  # [num_jumps, m, vocab]
                                
                                # Gather Q at jump times
                                jump_idx_expanded = jump_idx_b.unsqueeze(1).unsqueeze(2).expand(-1, -1, vocabulary_size)  # [num_jumps, 1, vocab]
                                Q_at_jump_b = torch.gather(Q_stacked_b_masked, 1, jump_idx_expanded).squeeze(1)  # [num_jumps, vocab]
                                
                                # Sample target state for jumping coordinates
                                x_t[b, mask_jump_b] = categorical(Q_at_jump_b.to(dtype=dtype_categorical))
                        
                        if return_intermediates:
                            res.append(x_t.clone())
                        
                        original_x_t[data_info['image_token_mask']==1] = x_t.flatten()
                        x_t = original_x_t.clone()

                    steps_counter += 1
                    t = t + h

                    if verbose:
                        ctx.n = t.item()
                        ctx.refresh()
                        ctx.set_description(f"Step: {steps_counter}")
                else:

                    # Step 1: Sample x_1^d ~ p_{1|t}(·|Y_{k-1}) for all d (in parallel)
                    _, p_1t_img, data_info = self.model(x=x_t, **model_extras)
                    
                    # Only process image generation
                    x_1 = categorical(p_1t_img.to(dtype=dtype_categorical))
                    x_1 = x_1[data_info['image_token_mask']==1].reshape(batch_size, -1)
                    x_t = x_t[data_info['image_token_mask']==1].reshape(batch_size, -1)

                    # Check if final step
                    if i == n_steps - 1:
                        x_t = x_1
                        if return_intermediates:
                            res.append(x_t.clone())
                    else:
                        # Location-corrected sampling
                        path = self.path_img
                        vocabulary_size = self.vocabulary_size_img
                        D = x_t.shape[1]
                        emb_x_1 = path.embedding(x_1)
                        
                        # Step 2: Calculate λ_{k,i}^d for all coordinates d at m grid points
                        lambda_ki_list = []  # Store intensity at each grid point
                        Q_list = []          # Store rate matrices at each grid point
                        
                        for grid_idx in range(1, m + 1):
                            s_j = t + (grid_idx - 1) / m * h
                            
                            # Compute Q_{s_j}(x, y | x_1) - the conditional rate matrix
                            prob_x_t = path.get_prob_distribution(emb_x_1, s_j)
                            prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
                            
                            emb_x_t = path.embedding(x_t)
                            emb_x_t_flattened = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
                            emb_x_1_flattened = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)
                            distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) + 
                                                torch.sum(emb_x_1_flattened**2, dim=1, keepdim=True) - 
                                                2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
                            distance_x_1_2_x = path.metric(emb_x_1)
                            distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
                            
                            if s_j == 0:
                                d_beta_t = 0
                            else:
                                d_beta_t = path.c * path.a * ((s_j / (1 - s_j)) ** (path.a - 1)) * 1 / ((1 - s_j) ** 2)
                            
                            Q_j = prob_x_t * d_beta_t * distance
                            Q_j = Q_j.reshape(x_t.shape[0], x_t.shape[1], -1)  # [batch, seq_len, vocab]
                            
                            # Set Q(x_t^d | x_t^d, x_1^d) = 0 (no self-transitions)
                            mask = F.one_hot(x_t, num_classes=vocabulary_size).bool()
                            Q_j = torch.where(mask, torch.zeros_like(Q_j), Q_j)
                            
                            # Calculate λ_{k,i}^d = sum_{z^d ≠ Y_{k-1}^d} Q_{s_i}^d(Y_{k-1}^d, z^d | x_1^d)
                            lambda_j = Q_j.sum(dim=-1)  # [batch, seq_len]
                            lambda_ki_list.append(lambda_j)
                            Q_list.append(Q_j)
                        
                        # Stack lambdas: [m, batch, seq_len] -> [batch, seq_len, m]
                        lambda_ki = torch.stack(lambda_ki_list, dim=0)
                        lambda_ki = lambda_ki.permute(1, 2, 0)  # [batch, seq_len, m]
                        
                        # Step 3: Sample (T_k^d, l_k^d) for each coordinate d using discretized process
                        delta_t = h / m
                        
                        # Compute jump probabilities for each grid point l
                        # P(T_k^d = s_l, l_k^d = l) = exp(-δt * Σ_{i=1}^{l-1} λ_{k,i}^d) - exp(-δt * Σ_{i=1}^l λ_{k,i}^d)
                        exp_cumsum = torch.exp(-delta_t * lambda_ki.cumsum(dim=2))  # [batch, seq_len, m]
                        exp_cumsum_pad = torch.cat([torch.ones(batch_size, D, 1, device=x_t.device), exp_cumsum[:, :, :-1]], dim=2)
                        
                        # P(no jump at coord d) = exp(-δt * Σ_{i=1}^m λ_{k,i}^d)
                        prob_no_jump = exp_cumsum[:, :, -1]  # [batch, seq_len]
                        
                        # Compute p_grid for jump probabilities
                        p_grid = exp_cumsum_pad - exp_cumsum  # [batch, seq_len, m]
                        
                        # Combine all probabilities: [batch, seq_len, m+1]
                        all_probs = torch.cat([p_grid, prob_no_jump.unsqueeze(2)], dim=2)
                        
                        # Sample jump location for each coordinate
                        jump_indices = categorical(all_probs.to(dtype=dtype_categorical))  # [batch, seq_len]
                        
                        # Step 4: Find T_k^{(j)} - the j-th order statistic of jump times across all coordinates
                        # Create actual jump times from indices: T_k^d = t + l_k^d / m * h
                        # If jump_indices < m, then T_k^d = t + jump_indices / m * h
                        # If jump_indices == m, then T_k^d = t + h (no jump within this interval)
                        jump_times = t + (jump_indices.float() / m) * h  # [batch, seq_len]
                        jump_times = torch.where(jump_indices < m, jump_times, t + h)  # [batch, seq_len]
                        
                        # Find j-th smallest jump time for each batch sample
                        # Sort jump times to get order statistics
                        sorted_jump_times, _ = torch.sort(jump_times, dim=1)  # [batch, seq_len], ascending order
                        T_k_j = sorted_jump_times[:, min(j-1, D-1)]  # [batch], j-th order statistic (0-indexed, so j-1)
                        
                        # Step 5: Check if T_k^{(j)} == t_k (i.e., no coordinate jumps before or at T_k^{(j)})
                        mask_no_jump_at_all = T_k_j >= (t + h - 1e-8)  # [batch], allowing small numerical error
                        
                        if (~mask_no_jump_at_all).sum() > 0:
                            # Some samples have at least one jump
                            # Step 6a: First, let ALL samples jump their early-jumping coordinates
                            for b in range(batch_size):
                                if mask_no_jump_at_all[b]:
                                    # No jump for this sample, x_t[b] stays the same
                                    continue
                                
                                T_k_order = T_k_j[b]
                                mask_jump_early = jump_times[b] <= T_k_order
                                
                                if mask_jump_early.sum() > 0:
                                    jump_idx = jump_indices[b][mask_jump_early]
                                    Q_stacked_b = torch.stack([Q_list[jj][b] for jj in range(m)], dim=0)
                                    Q_stacked_b = Q_stacked_b.permute(1, 0, 2)  # [seq_len, m, vocab]
                                    
                                    jump_idx_expanded = jump_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, vocabulary_size)
                                    Q_at_jump_b = torch.gather(Q_stacked_b[mask_jump_early], 1, jump_idx_expanded).squeeze(1)
                                    x_t[b, mask_jump_early] = categorical(Q_at_jump_b.to(dtype=dtype_categorical))
                            
                            # Step 6b: Identify samples that need resampling (T_k^{(j)} < t_k)
                            mask_need_resample = T_k_j < (t + h - 1e-8)  # [batch]
                            
                            
                            # Batch all samples that need resampling together
                            original_x_t[data_info['image_token_mask'] == 1] = x_t.flatten()
                            samples_to_resample = original_x_t[mask_need_resample]
                            
                            # Single function evaluation for ALL samples needing resampling
                            _, p_1t_batch, _ = self.model(x=samples_to_resample, **model_extras)
                            x_1_batch = categorical(p_1t_batch.to(dtype=dtype_categorical))
                            num_resample = mask_need_resample.sum().item()
                            x_1_batch = x_1_batch[data_info['image_token_mask'][mask_need_resample] == 1].reshape(num_resample, -1)
                            
                            # Step 6c: Now apply time-corrected sampling to each resampled sample
                            resample_idx = 0
                            for b in range(batch_size):
                                if not mask_need_resample[b]:
                                    continue
                                
                                T_k_order = T_k_j[b]
                                h_remaining = t + h - T_k_order
                                x_1_resample = x_1_batch[resample_idx:resample_idx+1]  # [1, seq_len]
                                resample_idx += 1
                                
                                # Recalculate λ and Q for the remaining interval [T_k^{(j)}, t_k]
                                lambda_ki_resample = []
                                Q_list_resample = []
                                emb_x_1_resample = path.embedding(x_1_resample)
                                
                                for jj in range(1, m + 1):
                                    s_j = T_k_order + (jj - 1) / m * h_remaining
                                    
                                    prob_x_t = path.get_prob_distribution(emb_x_1_resample, s_j)
                                    prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
                                    
                                    emb_x_t_single = path.embedding(x_t[b:b+1])
                                    emb_x_t_flattened = F.normalize(emb_x_t_single.view(-1, emb_x_t_single.shape[-1]), p=2, dim=-1)
                                    emb_x_1_flattened = F.normalize(emb_x_1_resample.view(-1, emb_x_1_resample.shape[-1]), p=2, dim=-1)
                                    distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) + 
                                                            torch.sum(emb_x_1_flattened**2, dim=1, keepdim=True) - 
                                                            2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
                                    distance_x_1_2_x = path.metric(emb_x_1_resample)
                                    distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
                                    
                                    if s_j == 0:
                                        d_beta_t = 0
                                    else:
                                        d_beta_t = path.c * path.a * ((s_j / (1 - s_j)) ** (path.a - 1)) * 1 / ((1 - s_j) ** 2)
                                    
                                    Q_j = prob_x_t * d_beta_t * distance
                                    Q_j = Q_j.reshape(1, D, -1)
                                    
                                    mask = F.one_hot(x_t[b:b+1], num_classes=vocabulary_size).bool()
                                    Q_j = torch.where(mask, torch.zeros_like(Q_j), Q_j)
                                    
                                    lambda_j = Q_j.sum(dim=-1)
                                    lambda_ki_resample.append(lambda_j)
                                    Q_list_resample.append(Q_j)
                                
                                # Apply time-corrected sampling for the remaining interval
                                lambda_ki_resample = torch.stack(lambda_ki_resample, dim=0).permute(1, 2, 0)  # [1, seq_len, m]
                                delta_t_remaining = h_remaining / m
                                
                                exp_cumsum_resample = torch.exp(-delta_t_remaining * lambda_ki_resample.cumsum(dim=2))
                                exp_cumsum_pad_resample = torch.cat([torch.ones(1, D, 1, device=x_t.device), exp_cumsum_resample[:, :, :-1]], dim=2)
                                prob_no_jump_resample = exp_cumsum_resample[:, :, -1]
                                p_grid_resample = exp_cumsum_pad_resample - exp_cumsum_resample
                                all_probs_resample = torch.cat([p_grid_resample, prob_no_jump_resample.unsqueeze(2)], dim=2)
                                
                                jump_indices_resample = categorical(all_probs_resample.to(dtype=dtype_categorical))  # [1, seq_len]
                                mask_jump_resample = jump_indices_resample[0] < m
                                
                                if mask_jump_resample.sum() > 0:
                                    Q_stacked_resample = torch.stack(Q_list_resample, dim=0)[: , 0, :, :]  # [m, seq_len, vocab]
                                    Q_stacked_resample = Q_stacked_resample.permute(1, 0, 2)  # [seq_len, m, vocab]
                                    
                                    jump_idx_resample = jump_indices_resample[0][mask_jump_resample]
                                    jump_idx_expanded = jump_idx_resample.unsqueeze(1).unsqueeze(2).expand(-1, -1, vocabulary_size)
                                    Q_at_jump_resample = torch.gather(Q_stacked_resample[mask_jump_resample], 1, jump_idx_expanded).squeeze(1)
                                    
                                    x_t[b, mask_jump_resample] = categorical(Q_at_jump_resample.to(dtype=dtype_categorical))
                        
                        if return_intermediates:
                            res.append(x_t.clone())
                        
                        original_x_t[data_info['image_token_mask']==1] = x_t.flatten()
                        x_t = original_x_t.clone()

                    steps_counter += 1
                    t = t + h

                    if verbose:
                        ctx.n = t.item()
                        ctx.refresh()
                        ctx.set_description(f"Step: {steps_counter}")

        if return_intermediates:
            return torch.stack(res, dim=0)[:, 0, :].reshape(n_steps+1, -1)
        else:
            return x_t


class MixtureDiscreteSoftmaxRK2Solver(Solver):
    r"""Second-order Runge-Kutta solver for discrete flow matching with softmax probability paths (Image Generation Only).
    
    Implements the RK2 sampling algorithm adapted for the softmax/embedding-based probability path.
    For each step k from t to t+h:
      1. Compute u_left at (x_t, t) using the softmax probability path
      2. Take an Euler predictor step with step size theta*h to get x_mid
      3. Compute u_mid at (x_mid, t + theta*h)  
      4. Combine: final_u = (1 - 1/(2*theta)) * u_left + 1/(2*theta) * u_mid
      5. Jump from x_t using the combined rate
    
    theta = 1/2: midpoint method
    theta = 1: Heun's method
    theta = 2/3: Ralston's method
    
    Steps = n_steps (each non-final step uses 2 model evaluations, final step uses 1)
    
    Args:
        model (ModelWrapper): Model outputting posterior probabilities.
        path_img (MixtureDiscreteSoftmaxProbPath): Probability path for image generation.
        vocabulary_size_img (int): Size of the image vocabulary.
    """

    def __init__(
        self,
        model: ModelWrapper,
        path_img: MixtureDiscreteSoftmaxProbPath,
        vocabulary_size_img: int,
    ):
        super().__init__()
        self.model = model
        self.path_img = path_img
        self.vocabulary_size_img = vocabulary_size_img

    def _compute_u(self, path, vocabulary_size, x_t, x_1, t, dtype_categorical):
        """Compute the conditional probability velocity u at (x_t, t) given x_1.
        
        Args:
            path: MixtureDiscreteSoftmaxProbPath instance
            vocabulary_size: vocabulary size
            x_t: current state [batch, seq_len]
            x_1: predicted clean state [batch, seq_len]
            t: current time (tensor)
            dtype_categorical: dtype for categorical sampling
            
        Returns:
            u: rate matrix [batch, seq_len, vocab] with u(x_t|x_t,x_1) = 0
        """
        emb_x_1 = path.embedding(x_1)
        prob_x_t = path.get_prob_distribution(emb_x_1, t)
        prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
        
        # Compute the metric / distance
        emb_x_t = path.embedding(x_t)
        emb_x_t_flattened = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
        emb_x_1_flattened = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)
        distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) +
                               torch.sum(emb_x_1_flattened ** 2, dim=1, keepdim=True) -
                               2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
        distance_x_1_2_x = path.metric(emb_x_1)
        distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
        
        if t == 0:
            d_beta_t = 0
        else:
            d_beta_t = path.c * path.a * ((t / (1 - t)) ** (path.a - 1)) * 1 / ((1 - t) ** 2)
        
        u = prob_x_t * d_beta_t * distance
        u = u.reshape(x_1.shape[0], x_1.shape[1], -1)
        
        # Set u(x_t | x_t, x_1) = 0
        delta_t = F.one_hot(x_t, num_classes=vocabulary_size)
        u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)
        
        return u

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: float,
        theta: float = 0.5,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        """
        RK2 sampler for discrete flow matching with softmax probability paths.

        Args:
            x_init (Tensor): The initial state.
            step_size (float): Time discretization step size.
            theta (float): RK2 parameter in [0.5, 1.0]. 0.5=midpoint, 1=Heun, 2/3=Ralston.
            dtype_categorical (torch.dtype): Precision for categorical sampler.
            time_grid (Tensor): Process solved in [time_grid[0], time_grid[-1]].
            return_intermediates (bool): If True, return intermediate steps.
            verbose (bool): Whether to print progress bars.
            **model_extras: Additional input for the model.

        Returns:
            Tensor: The sampled sequence of discrete values.
        """
        assert 0.5 <= theta <= 1.0, f"theta must be in [0.5, 1.0], got {theta}"
        
        # Initialize
        time_grid = time_grid.to(device=x_init.device)
        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()
        
        assert (t_final - t_init) > step_size, \
            f"Time interval must be larger than step_size. Got [{t_init}, {t_final}] and step_size {step_size}."

        n_steps = ceil((t_final - t_init) / step_size)
        t_discretization = torch.tensor(
            [t_init + step_size * i for i in range(n_steps)] + [t_final],
            device=x_init.device,
        )

        x_t = x_init.clone()
        steps_counter = 0
        res = []
        
        if return_intermediates:
            res = [x_init.clone()[model_extras['datainfo']['image_token_mask']==1].reshape(x_init.shape[0], -1)]

        if verbose:
            ctx = tqdm(total=time_grid[-1].item(), desc=f"Step: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx:
            original_x_t = x_t.clone()
            batch_size = original_x_t.shape[0]
            
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                # ============================================================
                # Stage 1: Model evaluation at t to get x_1, compute u_left
                # ============================================================
                _, p_1t_img, data_info = self.model(x=x_t, **model_extras)
                
                x_1 = categorical(p_1t_img.to(dtype=dtype_categorical))
                x_1 = x_1[data_info['image_token_mask']==1].reshape(batch_size, -1)
                x_t = x_t[data_info['image_token_mask']==1].reshape(batch_size, -1)

                # Check if final step
                if i == n_steps - 1:
                    x_t = x_1
                    if return_intermediates:
                        res.append(x_t.clone())
                    steps_counter += 1  # final step (1 model evaluation)
                else:
                    path = self.path_img
                    vocabulary_size = self.vocabulary_size_img
                    
                    # Compute u_left at (x_t, t)
                    u_left = self._compute_u(path, vocabulary_size, x_t, x_1, t, dtype_categorical)
                    
                    # Euler predictor step with step size theta*h to get x_mid
                    intensity_pred = u_left.sum(dim=-1)
                    mask_jump_pred = torch.rand(
                        size=x_t.shape, device=x_t.device
                    ) < 1 - torch.exp(-theta * h * intensity_pred)
                    
                    x_mid = x_t.clone()
                    if mask_jump_pred.sum() > 0:
                        x_mid[mask_jump_pred] = categorical(
                            u_left[mask_jump_pred].to(dtype=dtype_categorical)
                        )
                    
                    # ============================================================
                    # Stage 2: Model evaluation at t_mid, compute u_mid
                    # ============================================================
                    t_mid = t + theta * h
                    
                    # Put x_mid back into original_x_t for model call
                    original_x_t_mid = original_x_t.clone()
                    original_x_t_mid[data_info['image_token_mask']==1] = x_mid.flatten()
                    
                    _, p_1t_img_mid, _ = self.model(x=original_x_t_mid, **model_extras)
                    
                    x_1_mid = categorical(p_1t_img_mid.to(dtype=dtype_categorical))
                    x_1_mid = x_1_mid[data_info['image_token_mask']==1].reshape(batch_size, -1)
                    
                    # Compute u_mid at (x_mid, t_mid)
                    u_mid = self._compute_u(path, vocabulary_size, x_mid, x_1_mid, t_mid, dtype_categorical)
                    
                    # ============================================================
                    # Combine: final_u = (1 - 1/(2*theta)) * u_left + 1/(2*theta) * u_mid
                    # Re-zero u_mid at x_t positions (we jump from x_t)
                    # ============================================================
                    w1 = 1 - 1 / (2 * theta)
                    w2 = 1 / (2 * theta)
                    
                    delta_t_for_combined = F.one_hot(x_t, num_classes=vocabulary_size)
                    u_mid_rezeroed = torch.where(delta_t_for_combined.to(dtype=torch.bool), torch.zeros_like(u_mid), u_mid)
                    
                    final_u = w1 * u_left + w2 * u_mid_rezeroed
                    final_u = torch.clamp(final_u, min=0)
                    
                    # Jump from x_t using the combined rate
                    intensity = final_u.sum(dim=-1)
                    mask_jump = torch.rand(
                        size=x_t.shape, device=x_t.device
                    ) < 1 - torch.exp(-h * intensity)
                    
                    if mask_jump.sum() > 0:
                        x_t[mask_jump] = categorical(
                            final_u[mask_jump].to(dtype=dtype_categorical)
                        )
                    
                    if return_intermediates:
                        res.append(x_t.clone())
                    
                    original_x_t[data_info['image_token_mask']==1] = x_t.flatten()
                    x_t = original_x_t.clone()
                    steps_counter += 1  # 1 step (2 model evaluations)

                t = t + h

                if verbose:
                    ctx.n = t.item()
                    ctx.refresh()
                    ctx.set_description(f"Step: {steps_counter}")

        if return_intermediates:
            return torch.stack(res, dim=0)[:, 0, :].reshape(n_steps+1, -1)
        else:
            return x_t


class MixtureDiscreteSoftmaxRK2TrapezoidSolver(Solver):
    r"""Theta-trapezoidal tau-leaping solver for discrete flow matching with softmax probability paths (Image Generation Only).
    
    Implements the theta-trapezoidal tau-leaping algorithm adapted for the softmax/embedding-based probability path.
    For each step k from t to t+h:
      1. Compute u_left at (x_t, t) using the softmax probability path
      2. Take an Euler predictor step with step size theta*h to get x_mid
      3. Compute u_mid at (x_mid, t + theta*h)
      4. Combine: final_u = alpha1(theta) * u_mid - alpha2(theta) * u_left (clamped to >= 0)
         where alpha1(theta) = 1/(2*theta), alpha2(theta) = ((1-theta)^2 + theta^2)/(2*theta)
      5. Jump from x_mid using the combined rate (NOT from x_t)
    
    Steps = n_steps (each non-final step uses 2 model evaluations, final step uses 1)
    
    Args:
        model (ModelWrapper): Model outputting posterior probabilities.
        path_img (MixtureDiscreteSoftmaxProbPath): Probability path for image generation.
        vocabulary_size_img (int): Size of the image vocabulary.
    """

    def __init__(
        self,
        model: ModelWrapper,
        path_img: MixtureDiscreteSoftmaxProbPath,
        vocabulary_size_img: int,
    ):
        super().__init__()
        self.model = model
        self.path_img = path_img
        self.vocabulary_size_img = vocabulary_size_img

    def _compute_u(self, path, vocabulary_size, x_t, x_1, t, dtype_categorical):
        """Compute the conditional probability velocity u at (x_t, t) given x_1.
        
        Args:
            path: MixtureDiscreteSoftmaxProbPath instance
            vocabulary_size: vocabulary size
            x_t: current state [batch, seq_len]
            x_1: predicted clean state [batch, seq_len]
            t: current time (tensor)
            dtype_categorical: dtype for categorical sampling
            
        Returns:
            u: rate matrix [batch, seq_len, vocab] with u(x_t|x_t,x_1) = 0
        """
        emb_x_1 = path.embedding(x_1)
        prob_x_t = path.get_prob_distribution(emb_x_1, t)
        prob_x_t = prob_x_t.reshape(-1, prob_x_t.shape[-1])
        
        # Compute the metric / distance
        emb_x_t = path.embedding(x_t)
        emb_x_t_flattened = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
        emb_x_1_flattened = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)
        distance_x_t_2_x_1 = (torch.sum(emb_x_t_flattened ** 2, dim=1, keepdim=True) +
                               torch.sum(emb_x_1_flattened ** 2, dim=1, keepdim=True) -
                               2 * torch.einsum('bd,bd->b', emb_x_t_flattened, emb_x_1_flattened).unsqueeze(1)) ** 2
        distance_x_1_2_x = path.metric(emb_x_1)
        distance = F.relu(distance_x_t_2_x_1 - distance_x_1_2_x)
        
        if t == 0:
            d_beta_t = 0
        else:
            d_beta_t = path.c * path.a * ((t / (1 - t)) ** (path.a - 1)) * 1 / ((1 - t) ** 2)
        
        u = prob_x_t * d_beta_t * distance
        u = u.reshape(x_1.shape[0], x_1.shape[1], -1)
        
        # Set u(x_t | x_t, x_1) = 0
        delta_t = F.one_hot(x_t, num_classes=vocabulary_size)
        u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)
        
        return u

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: float,
        theta: float = 0.5,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        """
        Theta-trapezoidal tau-leaping sampler for discrete flow matching with softmax probability paths.

        Args:
            x_init (Tensor): The initial state.
            step_size (float): Time discretization step size.
            theta (float): Trapezoidal parameter in (0, 1.0].
            dtype_categorical (torch.dtype): Precision for categorical sampler.
            time_grid (Tensor): Process solved in [time_grid[0], time_grid[-1]].
            return_intermediates (bool): If True, return intermediate steps.
            verbose (bool): Whether to print progress bars.
            **model_extras: Additional input for the model.

        Returns:
            Tensor: The sampled sequence of discrete values.
        """
        assert 0 < theta <= 1.0, f"theta must be in (0, 1.0], got {theta}"
        
        # Trapezoidal coefficients
        alpha1 = 1 / (2 * theta)
        alpha2 = ((1 - theta)**2 + theta**2) / (2 * theta)
        
        # Initialize
        time_grid = time_grid.to(device=x_init.device)
        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()
        
        assert (t_final - t_init) > step_size, \
            f"Time interval must be larger than step_size. Got [{t_init}, {t_final}] and step_size {step_size}."

        n_steps = ceil((t_final - t_init) / step_size)
        t_discretization = torch.tensor(
            [t_init + step_size * i for i in range(n_steps)] + [t_final],
            device=x_init.device,
        )

        x_t = x_init.clone()
        steps_counter = 0
        res = []
        
        if return_intermediates:
            res = [x_init.clone()[model_extras['datainfo']['image_token_mask']==1].reshape(x_init.shape[0], -1)]

        if verbose:
            ctx = tqdm(total=time_grid[-1].item(), desc=f"Step: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx:
            original_x_t = x_t.clone()
            batch_size = original_x_t.shape[0]
            
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                # ============================================================
                # Stage 1: Model evaluation at t to get x_1, compute u_left
                # ============================================================
                _, p_1t_img, data_info = self.model(x=x_t, **model_extras)
                
                x_1 = categorical(p_1t_img.to(dtype=dtype_categorical))
                x_1 = x_1[data_info['image_token_mask']==1].reshape(batch_size, -1)
                x_t = x_t[data_info['image_token_mask']==1].reshape(batch_size, -1)

                # Check if final step
                if i == n_steps - 1:
                    x_t = x_1
                    if return_intermediates:
                        res.append(x_t.clone())
                    steps_counter += 1  # final step (1 model evaluation)
                else:
                    path = self.path_img
                    vocabulary_size = self.vocabulary_size_img
                    
                    # Compute u_left at (x_t, t)
                    u_left = self._compute_u(path, vocabulary_size, x_t, x_1, t, dtype_categorical)
                    
                    # Euler predictor step with step size theta*h to get x_mid
                    intensity_pred = u_left.sum(dim=-1)
                    mask_jump_pred = torch.rand(
                        size=x_t.shape, device=x_t.device
                    ) < 1 - torch.exp(-theta * h * intensity_pred)
                    
                    x_mid = x_t.clone()
                    if mask_jump_pred.sum() > 0:
                        x_mid[mask_jump_pred] = categorical(
                            u_left[mask_jump_pred].to(dtype=dtype_categorical)
                        )
                    
                    # ============================================================
                    # Stage 2: Model evaluation at t_mid, compute u_mid
                    # ============================================================
                    t_mid = t + theta * h
                    
                    # Put x_mid back into original_x_t for model call
                    original_x_t_mid = original_x_t.clone()
                    original_x_t_mid[data_info['image_token_mask']==1] = x_mid.flatten()
                    
                    _, p_1t_img_mid, _ = self.model(x=original_x_t_mid, **model_extras)
                    
                    x_1_mid = categorical(p_1t_img_mid.to(dtype=dtype_categorical))
                    x_1_mid = x_1_mid[data_info['image_token_mask']==1].reshape(batch_size, -1)
                    
                    # Compute u_mid at (x_mid, t_mid)
                    u_mid = self._compute_u(path, vocabulary_size, x_mid, x_1_mid, t_mid, dtype_categorical)
                    
                    # ============================================================
                    # Combine: final_u = alpha1 * u_mid - alpha2 * u_left
                    # Re-zero u_left at x_mid positions (we jump from x_mid)
                    # ============================================================
                    delta_t_for_combined = F.one_hot(x_mid, num_classes=vocabulary_size)
                    u_left_rezeroed = torch.where(delta_t_for_combined.to(dtype=torch.bool), torch.zeros_like(u_left), u_left)
                    
                    final_u = alpha1 * u_mid - alpha2 * u_left_rezeroed
                    final_u = torch.clamp(final_u, min=0)  # clamp negative rates to 0
                    
                    # Jump from x_mid using the combined rate (NOT from x_t)
                    intensity = final_u.sum(dim=-1)
                    mask_jump = torch.rand(
                        size=x_mid.shape, device=x_mid.device
                    ) < 1 - torch.exp(-h * intensity)
                    
                    if mask_jump.sum() > 0:
                        x_mid[mask_jump] = categorical(
                            final_u[mask_jump].to(dtype=dtype_categorical)
                        )
                    
                    # Update x_t from x_mid (trapezoidal jumps from x_mid)
                    x_t = x_mid
                    
                    if return_intermediates:
                        res.append(x_t.clone())
                    
                    original_x_t[data_info['image_token_mask']==1] = x_t.flatten()
                    x_t = original_x_t.clone()
                    steps_counter += 1  # 1 step (2 model evaluations)

                t = t + h

                if verbose:
                    ctx.n = t.item()
                    ctx.refresh()
                    ctx.set_description(f"Step: {steps_counter}")

        if return_intermediates:
            return torch.stack(res, dim=0)[:, 0, :].reshape(n_steps+1, -1)
        else:
            return x_t
