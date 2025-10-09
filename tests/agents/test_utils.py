import sys
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dynaconf.base import LazySettings
from SRAgent.agents.utils import load_settings, set_model


class TestLoadSettings:
    """Tests for load_settings function"""

    def test_load_settings_returns_lazy_settings(self):
        """Test that load_settings returns a LazySettings object"""
        settings = load_settings()
        assert isinstance(settings, LazySettings)

    @patch("SRAgent.agents.utils.Dynaconf")
    def test_load_settings_has_expected_keys(self, mock_dynaconf):
        """Test that settings are loaded and can be accessed"""
        # Setup mock settings that behave more like a Dynaconf object
        # Need to mock the __getitem__ behaviour for nested access like settings['models']['default']
        mock_dynaconf_instance = MagicMock()

        # For simplicity here, just make it return a MagicMock instance
        mock_dynaconf.return_value = mock_dynaconf_instance

        # Call load_settings
        settings = load_settings()

        # Verify that Dynaconf was called correctly
        mock_dynaconf.assert_called_once()
        # Optionally assert that settings is the mock instance
        assert settings is mock_dynaconf_instance


class TestSetModel:
    """Tests for set_model function"""

    @patch("SRAgent.agents.utils.load_settings")
    def test_set_model_default_settings(self, mock_load_settings):
        """Test set_model with default settings"""

        # Create mock settings that handles KeyError for service_tier and flex_timeout properly
        def settings_getitem(key):
            available_settings = {
                "models": {"default": "gpt-5-mini"},
                "temperature": {"default": 0.1},
                "reasoning_effort": {"default": "low"},
            }
            if key in available_settings:
                return available_settings[key]
            else:
                raise KeyError(key)

        mock_settings = MagicMock()
        mock_settings.__getitem__.side_effect = settings_getitem
        mock_load_settings.return_value = mock_settings

        # Test with o1/o3/o4 model
        with patch("SRAgent.agents.utils.FlexTierChatOpenAI") as mock_chat:
            model = set_model()
            mock_chat.assert_called_once_with(
                model_name="gpt-5-mini",
                temperature=None,
                reasoning_effort="low",
                max_tokens=None,
                service_tier="default",
                timeout=None,
            )

    @patch("SRAgent.agents.utils.load_settings")
    def test_set_model_with_gpt4o(self, mock_load_settings):
        """Test set_model with gpt-4o model"""

        # Create mock settings that handles KeyError for service_tier and flex_timeout properly
        def settings_getitem(key):
            available_settings = {
                "models": {"default": "gpt-4o-mini"},
                "temperature": {"default": 0.1},
                "reasoning_effort": {"default": "low"},
            }
            if key in available_settings:
                return available_settings[key]
            else:
                raise KeyError(key)

        mock_settings = MagicMock()
        mock_settings.__getitem__.side_effect = settings_getitem
        mock_load_settings.return_value = mock_settings

        # Test with GPT-4o-mini model
        with patch("SRAgent.agents.utils.FlexTierChatOpenAI") as mock_chat:
            model = set_model()
            mock_chat.assert_called_once_with(
                model_name="gpt-4o-mini",
                temperature=0.1,
                reasoning_effort=None,
                max_tokens=None,
                service_tier="default",
                timeout=None,
            )

    @patch("SRAgent.agents.utils.load_settings")
    def test_set_model_with_overrides(self, mock_load_settings):
        """Test set_model with parameter overrides"""

        # Create mock settings that handles KeyError for service_tier and flex_timeout properly
        def settings_getitem(key):
            available_settings = {
                "models": {"default": "gpt-4o-mini"},
                "temperature": {"default": 0.1},
                "reasoning_effort": {"default": "low"},
            }
            if key in available_settings:
                return available_settings[key]
            else:
                raise KeyError(key)

        mock_settings = MagicMock()
        mock_settings.__getitem__.side_effect = settings_getitem
        mock_load_settings.return_value = mock_settings

        # Test with override parameters
        with patch("SRAgent.agents.utils.FlexTierChatOpenAI") as mock_chat:
            model = set_model(
                model_name="gpt-4o-mini", temperature=0.5, reasoning_effort="high"
            )
            mock_chat.assert_called_once_with(
                model_name="gpt-4o-mini",
                temperature=0.5,
                reasoning_effort=None,
                max_tokens=None,
                service_tier="default",
                timeout=None,
            )

    @patch("SRAgent.agents.utils.load_settings")
    def test_set_model_specific_agent(self, mock_load_settings):
        """Test set_model with specific agent settings"""

        # Create mock settings that handles KeyError for service_tier and flex_timeout properly
        def settings_getitem(key):
            available_settings = {
                "models": {"default": "gpt-5-mini", "entrez": "gpt-5-mini"},
                "temperature": {"default": 0.1, "entrez": 0.2},
                "reasoning_effort": {"default": "low", "entrez": "medium"},
            }
            if key in available_settings:
                return available_settings[key]
            else:
                raise KeyError(key)

        mock_settings = MagicMock()
        mock_settings.__getitem__.side_effect = settings_getitem
        mock_load_settings.return_value = mock_settings

        # Test with agent_name parameter
        with patch("SRAgent.agents.utils.FlexTierChatOpenAI") as mock_chat:
            model = set_model(agent_name="entrez")
            mock_chat.assert_called_once_with(
                model_name="gpt-5-mini",
                temperature=None,
                reasoning_effort="medium",
                max_tokens=None,
                service_tier="default",
                timeout=None,
            )

    @patch("SRAgent.agents.utils.load_settings")
    def test_set_model_unsupported_model(self, mock_load_settings):
        """Test set_model with unsupported model"""
        # Mock settings
        mock_settings = {
            "models": {"default": "unsupported-model"},
            "temperature": {"default": 0.1},
            "reasoning_effort": {"default": "low"},
        }
        mock_load_settings.return_value = mock_settings

        # Test with unsupported model
        with pytest.raises(ValueError, match="Model unsupported-model not supported"):
            set_model()

    @patch("SRAgent.agents.utils.load_settings")
    def test_set_model_with_claude(self, mock_load_settings):
        """Test set_model with claude model"""
        # Mock settings
        mock_settings = {
            "models": {"default": "claude-sonnet-4-5"},
            "temperature": {"default": 0.1},
            "reasoning_effort": {"default": "low"},
        }
        mock_load_settings.return_value = mock_settings

        # Test with claude model and low reasoning effort
        with patch("SRAgent.agents.utils.ChatAnthropic") as mock_chat:
            model = set_model()
            mock_chat.assert_called_once_with(
                model="claude-sonnet-4-5",
                temperature=None,
                thinking={"type": "enabled", "budget_tokens": 1024},
                max_tokens=2048,
            )

        # Test with claude model and medium reasoning effort
        mock_settings["reasoning_effort"]["default"] = "medium"
        with patch("SRAgent.agents.utils.ChatAnthropic") as mock_chat:
            model = set_model()
            mock_chat.assert_called_once_with(
                model="claude-sonnet-4-5",
                temperature=None,
                thinking={"type": "enabled", "budget_tokens": 2048},
                max_tokens=3072,
            )

        # Test with claude model and high reasoning effort
        mock_settings["reasoning_effort"]["default"] = "high"
        with patch("SRAgent.agents.utils.ChatAnthropic") as mock_chat:
            model = set_model()
            mock_chat.assert_called_once_with(
                model="claude-sonnet-4-5",
                temperature=None,
                thinking={"type": "enabled", "budget_tokens": 4096},
                max_tokens=5120,
            )

        # Test with claude model and disabled reasoning effort
        mock_settings["reasoning_effort"]["default"] = "none"
        with patch("SRAgent.agents.utils.ChatAnthropic") as mock_chat:
            model = set_model()
            mock_chat.assert_called_once_with(
                model="claude-sonnet-4-5",
                temperature=0.1,
                thinking={"type": "disabled"},
                max_tokens=1024,
            )
