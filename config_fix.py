"""
替换你的 config.py 开头部分（大约第 10-26 行）
"""

from pydantic.typing import Literal
from utils import BaseSettings
from models.modified_cgcnn import CGCNNConfig
from models.icgcnn import ICGCNNConfig
from models.gcn import SimpleGCNConfig
from models.densegcn import DenseGCNConfig
from models.alignn import ALIGNNConfig
from models.dense_alignn import DenseALIGNNConfig

# Optional imports - may fail if files don't exist or dependencies missing
try:
    from models.alignn_cgcnn import ACGCNNConfig
    HAS_ACGCNN = True
except (ImportError, ModuleNotFoundError):
    ACGCNNConfig = None
    HAS_ACGCNN = False

try:
    from models.alignn_layernorm import ALIGNNConfig as ALIGNN_LN_Config
    HAS_ALIGNN_LN = True
except (ImportError, ModuleNotFoundError):
    ALIGNN_LN_Config = None
    HAS_ALIGNN_LN = False
