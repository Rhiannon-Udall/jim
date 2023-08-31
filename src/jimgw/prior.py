import jax
import jax.numpy as jnp
from flowMC.nfmodel.base import Distribution
from jaxtyping import Array, Float
from typing import Callable, Union
from dataclasses import field

class Prior(Distribution):
    """
    A thin wrapper build on top of flowMC distributions to do book keeping.

    Should not be used directly since it does not implement any of the real method.
    """

    naming: list[str]
    transforms: dict[tuple[str,Callable]] = field(default_factory=dict)

    @property
    def n_dim(self):
        return len(self.naming)
    
    def __init__(self, naming: list[str], transforms: dict[tuple[str,Callable]] = {}):
        """
        Parameters
        ----------
        naming : list[str]
            A list of names for the parameters of the prior.
        transforms : dict[tuple[str,Callable]]
            A dictionary of transforms to apply to the parameters. The keys are
            the names of the parameters and the values are a tuple of the name
            of the transform and the transform itself.
        """
        self.naming = naming
        self.transforms = []
        for name in naming:
            if name in transforms:
                self.transforms.append(transforms[name])
            else:
                self.transforms.append((name,lambda x: x))

    def transform(self, x: Array) -> Array:
        """
        Apply the transforms to the parameters.

        Parameters
        ----------
        x : Array
            The parameters to transform.

        Returns
        -------
        x : Array
            The transformed parameters.
        """
        for i,transform in enumerate(self.transforms):
            x = x.at[i].set(transform[1](x[i]))
        return x

    def add_name(self, x: Array, with_transform: bool = False) -> dict:
        """
        Turn an array into a dictionary
        """
        if with_transform:
            naming = []
            for i,transform in enumerate(self.transforms):
                naming.append(transform[0])
        else:
            naming = self.naming
        return dict(zip(naming, x))


class Uniform(Prior):

    xmin: Array
    xmax: Array

    def __init__(self, xmin: Union[float,Array], xmax: Union[float,Array], **kwargs):
        super().__init__(kwargs.get("naming"), kwargs.get("transforms"))
        self.xmax = jnp.array(xmax)
        self.xmin = jnp.array(xmin)
    
    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        """
        Sample from a uniform distribution.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : Array
            An array of shape (n_samples, n_dim) containing the samples.
        
        """
        samples = jax.random.uniform(rng_key, (n_samples,self.n_dim), minval=self.xmin, maxval=self.xmax)
        return samples # TODO: remember to cast this to a named array

    def log_prob(self, x: Array) -> Float:
        output = jnp.sum(jnp.where((x>=self.xmax) | (x<=self.xmin), jnp.zeros_like(x)-jnp.inf, jnp.zeros_like(x)))
        return output + jnp.sum(jnp.log(1./(self.xmax-self.xmin))) 


class PowerLaw(Prior):
    """A power law distribution over a number of parameters, governed by a set of spectral indices alpha
    p(x) = \\Pi_{i=1}^N c_i x_i ^ {\\alpha_i}
    """
    xmin: Array
    """Array of lower bounds"""
    xmax: Array
    """Array of upper bounds"""
    alpha: Array
    """Array of power law exponential parameters"""
    normalization: Array
    """Array of the normalization constants"""

    def __init__(self, xmin: Union[float, Array], xmax: Union[float, Array], alpha: Union[float, Array], **kwargs):
        """Setup a prior on one or multiple parameters following (possibly different) power laws

        Parameters
        ==========
        xmin : float | Array
            The minimum values for the parameters
        xmax : float | Array
            The maximum values for the parameters
        alpha : float | Array
            The spectral indices for each parameter - 0 corresponds to a uniform distribution
        naming : list
            A list of the names for each parameter
        transforms : dict
            A dictionary of name:transform for parameters needing transforms
        """
        super().__init__(kwargs.get("naming"), kwargs.get("transforms"))
        self.xmax = jnp.array(xmax)
        self.xmin = jnp.array(xmin)
        self.alpha = jnp.array(alpha)
        self.normalization = jnp.where(
            self.alpha == -1,
            1 / (jnp.log(self.xmax) - jnp.log(self.xmin)),
            (self.alpha + 1) / (self.xmax ** (self.alpha + 1) - self.xmin**(self.alpha + 1))
        )

    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        """
        Sample from a power law distribution.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : Array
            An array of shape (n_samples, n_dim) containing the samples.
        
        """
        samples = jax.random.uniform(rng_key, (n_samples,self.n_dim))
        return jnp.apply_along_axis(self.rescale, axis=1, arr=samples) # TODO: remember to cast this to a named array
    
    def rescale(self, x : Array) -> Array:
        """Rescale values drawn from the unit hypercube to be distributed according to this prior
        
        Parameters
        ==========
        x : Array
            Values drawn from the unit cube to be rescaled by way of an inverse CDF
        
        Returns
        =======
        Array
            The values rescaled to the new prior
        """
        return jnp.where(
            self.alpha == -1,
            jnp.exp(x / self.normalization) * self.xmin,
            ((self.alpha + 1) * x / self.normalization + self.xmin ** (self.alpha + 1)) ** (1 / (self.alpha + 1))
        )

    def log_prob(self, x: Array) -> Float:
        """The log of the probability for this point in parameter space

        Parameters
        ==========
        x : Array
            The point in parameter space for which to compute the probability
        
        Returns
        =======
        float
            The log probability, being -inf if e.g. the point is out of bounds
        """
        output = jnp.nan_to_num(self.alpha * jnp.log(x))
        output = jnp.where((x<=self.xmax) & (x>=self.xmin), output, -jnp.inf).sum()
        return output + jnp.sum(jnp.log(self.normalization))
