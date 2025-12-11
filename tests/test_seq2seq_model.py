"""Unit tests for Seq2Seq model components.

These tests verify the basic functionality of the Seq2Seq model
and its components (encoders, policy, etc.).

Note: These tests require PyTorch and torchvision to be installed.
Run with: pytest tests/test_seq2seq_model.py
"""

import pytest
import torch
import numpy as np
from omegaconf import OmegaConf


# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Skip all tests if PyTorch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed. Install with: pip install torch torchvision"
)


class TestInstructionEncoder:
    """Test instruction encoder functionality."""
    
    def test_import(self):
        """Test that instruction encoder can be imported."""
        from satnav.models.encoders.instruction_encoder import InstructionEncoder
        assert InstructionEncoder is not None
    
    def test_forward_random_init(self):
        """Test forward pass with random initialization (no GloVe)."""
        from satnav.models.encoders.instruction_encoder import InstructionEncoder
        from omegaconf import OmegaConf
        
        # Create config for random initialization
        config = OmegaConf.create({
            "sensor_uuid": "instruction",
            "use_pretrained_embeddings": False,
            "vocab_size": 100,
            "embedding_size": 50,
            "hidden_size": 128,
            "rnn_type": "LSTM",
            "bidirectional": False,
            "final_state_only": True,
        })
        
        # Create encoder
        encoder = InstructionEncoder(config)
        
        # Test forward pass
        batch_size = 4
        seq_len = 10
        observations = {
            "instruction": torch.randint(0, 100, (batch_size, seq_len))
        }
        
        output = encoder(observations)
        
        # Check output shape
        assert output.shape == (batch_size, 128)
        assert not torch.isnan(output).any()


class TestVisualEncoder:
    """Test visual encoder functionality."""
    
    def test_import(self):
        """Test that visual encoder can be imported."""
        from satnav.models.encoders.visual_encoder import TorchVisionResNet50
        assert TorchVisionResNet50 is not None
    
    def test_forward(self):
        """Test forward pass of visual encoder."""
        from satnav.models.encoders.visual_encoder import TorchVisionResNet50
        
        # Create encoder
        encoder = TorchVisionResNet50(
            output_size=256,
            normalize_visual_inputs=False,
            trainable=False,
            spatial_output=False,
        )
        
        # Test forward pass
        batch_size = 2
        H, W = 224, 224
        observations = {
            "rgb": torch.randint(0, 256, (batch_size, H, W, 3)).float()
        }
        
        output = encoder(observations)
        
        # Check output shape
        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()


class TestRNNStateEncoder:
    """Test RNN state encoder functionality."""
    
    def test_import(self):
        """Test that state encoder can be imported."""
        from satnav.models.encoders.rnn_state_encoder import build_rnn_state_encoder
        assert build_rnn_state_encoder is not None
    
    def test_build_gru(self):
        """Test building GRU state encoder."""
        from satnav.models.encoders.rnn_state_encoder import build_rnn_state_encoder
        
        encoder = build_rnn_state_encoder(
            input_size=384,
            hidden_size=512,
            rnn_type="GRU",
            num_layers=1,
        )
        
        assert encoder is not None
        # RNN encoder has 1 layer (internal property, not exposed in Net interface)
    
    def test_forward(self):
        """Test forward pass of state encoder."""
        from satnav.models.encoders.rnn_state_encoder import build_rnn_state_encoder
        
        encoder = build_rnn_state_encoder(
            input_size=384,
            hidden_size=512,
            rnn_type="GRU",
            num_layers=1,
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 384)
        hidden_states = torch.zeros(1, batch_size, 512)
        masks = torch.ones(batch_size, 1)
        
        output, hidden_out = encoder(x, hidden_states, masks)
        
        assert output.shape == (batch_size, 512)
        assert hidden_out.shape == (1, batch_size, 512)
        assert not torch.isnan(output).any()


class TestModelRegistry:
    """Test model registry functionality."""
    
    def test_registry_import(self):
        """Test that model registry can be imported."""
        from satnav.models import ModelRegistry
        assert ModelRegistry is not None
    
    def test_seq2seq_registered(self):
        """Test that Seq2Seq model is registered."""
        from satnav.models import ModelRegistry
        
        # Check that seq2seq is in registry
        models = ModelRegistry.list_models()
        assert "seq2seq" in models["baseline"]
    
    def test_get_seq2seq(self):
        """Test getting Seq2Seq model from registry."""
        from satnav.models import ModelRegistry
        
        model_class = ModelRegistry.get_model("seq2seq")
        assert model_class is not None
        assert model_class.__name__ == "Seq2SeqPolicy"


class TestSeq2SeqPolicy:
    """Test Seq2Seq policy end-to-end."""
    
    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        return OmegaConf.create({
            "MODEL": {
                "INSTRUCTION_ENCODER": {
                    "sensor_uuid": "instruction",
                    "use_pretrained_embeddings": False,
                    "vocab_size": 100,
                    "embedding_size": 50,
                    "hidden_size": 128,
                    "rnn_type": "LSTM",
                    "bidirectional": False,
                    "final_state_only": True,
                },
                "RGB_ENCODER": {
                    "cnn_type": "TorchVisionResNet50",
                    "output_size": 256,
                    "trainable": False,
                },
                "SEQ2SEQ": {
                    "hidden_size": 512,
                    "rnn_type": "GRU",
                    "use_prev_action": True,
                },
                "normalize_rgb": False,
            }
        })
    
    @pytest.fixture
    def mock_spaces(self):
        """Create mock observation and action spaces."""
        class MockSpace:
            def __init__(self, n=None):
                self.n = n
        
        return MockSpace(), MockSpace(n=4)
    
    def test_import(self):
        """Test that Seq2Seq policy can be imported."""
        from satnav.models.baselines.seq2seq_policy import Seq2SeqPolicy
        assert Seq2SeqPolicy is not None
    
    def test_create_policy(self, config, mock_spaces):
        """Test creating Seq2Seq policy instance."""
        from satnav.models.baselines.seq2seq_policy import Seq2SeqPolicy
        
        obs_space, act_space = mock_spaces
        policy = Seq2SeqPolicy(obs_space, act_space, config.MODEL)
        
        assert policy is not None
        assert policy.net is not None
    
    def test_forward_pass(self, config, mock_spaces):
        """Test forward pass through Seq2Seq network."""
        from satnav.models.baselines.seq2seq_policy import Seq2SeqPolicy
        
        obs_space, act_space = mock_spaces
        policy = Seq2SeqPolicy(obs_space, act_space, config.MODEL)
        
        # Create dummy inputs
        batch_size = 2
        observations = {
            "instruction": torch.randint(0, 100, (batch_size, 10)),
            "rgb": torch.randint(0, 256, (batch_size, 224, 224, 3)).float(),
        }
        rnn_states = torch.zeros(1, batch_size, 512)
        prev_actions = torch.zeros(batch_size, 1).long()
        masks = torch.ones(batch_size, 1)
        
        # Forward pass
        action, new_rnn_states = policy.act(
            observations, rnn_states, prev_actions, masks
        )
        
        # Check outputs
        assert action.shape == (batch_size, 1)
        assert new_rnn_states.shape == (1, batch_size, 512)
        assert not torch.isnan(action.float()).any()


class TestVocabBuilder:
    """Test vocabulary building utilities."""
    
    def test_import(self):
        """Test that vocab builder can be imported."""
        from satnav.utils.build_vocab import tokenize, build_vocab_from_episodes
        assert tokenize is not None
        assert build_vocab_from_episodes is not None
    
    def test_tokenize(self):
        """Test tokenization function."""
        from satnav.utils.build_vocab import tokenize
        
        sentence = "Go forward, then turn left."
        tokens = tokenize(sentence)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
        # Check lowercase
        assert all(t.islower() or not t.isalpha() for t in tokens)
    
    def test_build_vocab_from_episodes(self):
        """Test building vocabulary from episodes."""
        from satnav.utils.build_vocab import build_vocab_from_episodes
        
        episodes = [
            {"instruction": {"instruction_text": "go forward"}},
            {"instruction": {"instruction_text": "turn left"}},
            {"instruction": {"instruction_text": "go left"}},
        ]
        
        word2idx = build_vocab_from_episodes(episodes, min_count=1)
        
        # Check structure
        assert "<pad>" in word2idx
        assert "<unk>" in word2idx
        assert word2idx["<pad>"] == 0
        assert word2idx["<unk>"] == 1
        
        # Check words
        assert "go" in word2idx
        assert "left" in word2idx
        assert "forward" in word2idx
        assert "turn" in word2idx


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

