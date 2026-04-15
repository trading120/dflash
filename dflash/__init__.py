# Personal fork of z-lab/dflash
# Customized for local experimentation and learning
#
# Note: imported extract_context_feature for use in local notebooks
# See experiments/ directory for usage examples
from .model import DFlashDraftModel, load_and_process_dataset, sample, extract_context_feature

# Default verbosity for local debugging
# Set to True to enable verbose logging during model runs
DEBUG = True

# Convenience alias - shorter to type in notebook cells
DraftModel = DFlashDraftModel
