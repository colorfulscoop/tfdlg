from pydantic import BaseModel


class Config(BaseModel):
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    vocab_size: int
    max_position_encoding: int
    attention_dropout_rate: float
    residual_dropout_rate: float
    epsilon: float


class TransformerConfig(BaseModel):
    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    vocab_size: int = 32000
    max_position_encoding: int = 512
    attention_dropout_rate: float = 0.1
    residual_dropout_rate: float = 0.1
    epsilon: float = 1e-6


# ===== Config for GPT2 =====


class GPT2SmallConfig(BaseModel):
    """Config for GPT2 small model with 117M parameters"""
    num_layers: int = 12
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 3072  # == 4 * d_model
    vocab_size: int = 50257
    max_position_encoding: int = 1024
    attention_dropout_rate: float = 0.1
    residual_dropout_rate: float = 0.1
    epsilon: float = 1e-6


class GPT2MediumConfig(GPT2SmallConfig):
    """Config for GPT2 medium model with 345 parameters"""
    num_layers: int = 24
    d_model: int = 1024
    num_heads: int = 16
    d_ff: int = 4096


class GPT2LargeConfig(GPT2SmallConfig):
    """Config for GPT2 large model with 774 parameters"""
    num_layers: int = 36
    d_model: int = 1280
    num_heads: int = 20
    d_ff: int = 5120


class GPT2XLConfig(GPT2SmallConfig):
    """Config for GPT2 XL model with 1558 parameters"""
    num_layers: int = 48
    d_model: int = 1600
    num_heads: int = 25
    d_ff: int = 6400
