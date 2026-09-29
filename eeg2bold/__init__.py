"""eeg2bold: predict fMRI BOLD from simultaneous EEG."""
from .config import Config
from .frontend import MSSE, MorletCWT, build_frontend
from .encoder import Encoder, TransformerBlock
from .readout import FIRKernel, LeadfieldFIRReadout, RegressionHeads, leadfield_prior
from .model import EEG2BOLD
from .losses import CompositeLoss
from .metrics import evaluate, evaluate_grouped

__all__ = [
    "Config", "MSSE", "MorletCWT", "build_frontend", "Encoder", "TransformerBlock",
    "FIRKernel", "LeadfieldFIRReadout", "RegressionHeads", "leadfield_prior",
    "EEG2BOLD", "CompositeLoss", "evaluate", "evaluate_grouped",
]
__version__ = "0.1.0"
