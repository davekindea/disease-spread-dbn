"""Disease spread modeling using Dynamic Bayesian Networks.

Models temporal epidemic spread through a population with:
  - Latent variables: individual SEIR infection states
  - Observations: reported symptoms / test results
  - PGM pillars: representation, inference (belief propagation), learning (EM)
"""

__version__ = "1.0.0"
