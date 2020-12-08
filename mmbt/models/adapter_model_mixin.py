from abc import ABC, abstractmethod


class ModelAdaptersMixin(ABC):
    """Mixin for transformer models adding support for loading/ saving adapters."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model_name = None
        self._active_adapter_names = None

    @abstractmethod
    def add_adapter(self, adapter_name: str):
        """Adds a new adapter module of the specified type to the model.
        Args:
            adapter_name (str): The name of the adapter module to be added.
            adapter_type (AdapterType): The adapter type.
            config (str or dict or AdapterConfig, optional): The adapter configuration, can be either:
                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
        """
        pass

    @abstractmethod
    def train_adapter(self, adapter_names: list):
        """Sets the model into mode for training the given adapters."""
        pass

    def freeze_model(self, freeze=True):
        """Freezes all weights of the model."""
        # first freeze/ unfreeze all model weights
        for param in self.parameters():
            param.requires_grad = not freeze
        self.model_freezed = freeze
