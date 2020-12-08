import torch
from torch import nn
from mmbt.models.adapter_model_mixin import ModelAdaptersMixin
from mmbt.models.mmadapter_modeling import BertMultimodalAdapter


class BertSelfOutputAdaptersMixin:
    """Adds adapters to the BertSelfOutput module."""
    
    def _init_adapter_modules(self):
        #self.attention_text_task_adapters = nn.ModuleDict(dict())
        self.attention_multimodal_task_adapters = nn.ModuleDict(dict())
        #self.adapter_fusion_layer = nn.ModuleDict(dict())

    def add_adapter(self, adapter_name: str, config):
        adapter = BertMultimodalAdapter(
            hidden_size=config.hidden_size,
            m_hidden_size=config.modality_size,
            adapter_size=config.adapter_size,
            adapter_activation=config.adapter_activation
        )
        if adapter_name in ["image", "video", "audio"]:
            self.attention_multimodal_task_adapters[adapter_name] = adapter
        else:
            raise ValueError("Invalid adapter type '{}'.".format(adapter_name))

    '''
    def add_fusion_layer(self, adapter_names):
        """See BertModel.add_attention_layer"""
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(",")
        self.adapter_fusion_layer[",".join(adapter_names)] = BertFusion(self.config)
    '''

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool):
        """Unfreezes a given list of adapters, the adapter fusion layer, or both
        :param adapter_names: names of adapters to unfreeze (or names of adapters part of the fusion layer to unfreeze)
        :param unfreeze_adapters: whether the adapters themselves should be unfreezed
        :param unfreeze_fusion: whether the adapter attention layer for the given adapters should be unfreezed
        """
        if unfreeze_adapters:
            if isinstance(adapter_names, str):
                adapter_names = [adapter_names]
            for adapter_name in adapter_names:
                layer = self.get_adapter_layer(adapter_name)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True

    def get_adapter_layer(self, adapter_name):
        """
        Depending on the adapter type we retrieve the correct layer. If no adapter for that name was set at that layer
        we return None
        Args:
            adapter_name: string name of the adapter
        Returns: layer | None
        """
        if adapter_name in self.attention_multimodal_task_adapters:
            return self.attention_multimodal_task_adapters[adapter_name]
        #if adapter_name in self.attention_text_task_adapters:
        #    return self.attention_text_task_adapters[adapter_name]
        return None

    def adapter_stack_layer(self, hidden_states, mod, adapter_stack):
        """
        One layer of stacked adapters. This either passes through a single adapter and prepares the data to be passed
        into a subsequent adapter, or the next transformer layer
        OR
        IFF more than one adapter names is set for one stack layer, we assume that fusion is activated. Thus, the
        adapters are fused together.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            input_tensor: residual connection of transformer
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters
        Returns: hidden_states
        """
        # We assume that all adapters have the same residual connection and layer norm setting as the first adapter in
        # the stack
        
        adapter_layer = self.get_adapter_layer(adapter_stack)
        if adapter_layer is not None:
            hidden_states, _, _ = adapter_layer(hidden_states, mod)

        return hidden_states

    def adapters_forward(self, hidden_states, mod, adapter_names=None):
        for adapter_stack in adapter_names:
            hidden_states = self.adapter_stack_layer(
                hidden_states=hidden_states,
                mod=mod,
                adapter_stack=adapter_stack,
            )
        
        return hidden_states


class BertOutputAdaptersMixin:
    """Adds adapters to the BertOutput module."""
    
    def _init_adapter_modules(self):
        #self.attention_text_task_adapters = nn.ModuleDict(dict())
        self.attention_multimodal_task_adapters = nn.ModuleDict(dict())
        #self.adapter_fusion_layer = nn.ModuleDict(dict())

    def add_adapter(self, adapter_name: str, config):
        adapter = BertMultimodalAdapter(
            hidden_size=config.hidden_size,
            m_hidden_size=config.modality_size,
            adapter_size=config.adapter_size,
            adapter_activation=config.adapter_activation
        )
        if adapter_name in ["image", "video", "audio"]:
            self.attention_multimodal_task_adapters[adapter_name] = adapter
        else:
            raise ValueError("Invalid adapter type '{}'.".format(adapter_name))

    '''
    def add_fusion_layer(self, adapter_names):
        """See BertModel.add_attention_layer"""
        adapter_names = adapter_names if isinstance(adapter_names, list) else adapter_names.split(",")
        self.adapter_fusion_layer[",".join(adapter_names)] = BertFusion(self.config)
    '''

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool):
        """Unfreezes a given list of adapters, the adapter fusion layer, or both
        :param adapter_names: names of adapters to unfreeze (or names of adapters part of the fusion layer to unfreeze)
        :param unfreeze_adapters: whether the adapters themselves should be unfreezed
        :param unfreeze_fusion: whether the adapter attention layer for the given adapters should be unfreezed
        """
        if unfreeze_adapters:
            if isinstance(adapter_names, str):
                adapter_names = [adapter_names]
            for adapter_name in adapter_names:
                layer = self.get_adapter_layer(adapter_name)
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True

    def get_adapter_layer(self, adapter_name):
        """
        Depending on the adapter type we retrieve the correct layer. If no adapter for that name was set at that layer
        we return None
        Args:
            adapter_name: string name of the adapter
        Returns: layer | None
        """
        if adapter_name in self.attention_multimodal_task_adapters:
            return self.attention_multimodal_task_adapters[adapter_name]
        #if adapter_name in self.attention_text_task_adapters:
        #    return self.attention_text_task_adapters[adapter_name]
        return None

    def adapter_stack_layer(self, hidden_states, mod, adapter_stack):
        """
        One layer of stacked adapters. This either passes through a single adapter and prepares the data to be passed
        into a subsequent adapter, or the next transformer layer
        OR
        IFF more than one adapter names is set for one stack layer, we assume that fusion is activated. Thus, the
        adapters are fused together.
        Args:
            hidden_states: output of the previous transformer layer or adapter
            input_tensor: residual connection of transformer
            adapter_stack: names of adapters for the current stack. Iff len(adapter_stack) == 1, we pass through a
                            single adapter. iff len(adapter_stack) > 1 we fuse the adapters
        Returns: hidden_states
        """
        # We assume that all adapters have the same residual connection and layer norm setting as the first adapter in
        # the stack
        
        adapter_layer = self.get_adapter_layer(adapter_stack)
        if adapter_layer is not None:
            hidden_states, _, _ = adapter_layer(hidden_states, mod)

        return hidden_states

    def adapters_forward(self, hidden_states, mod, adapter_names=None):
        for adapter_stack in adapter_names:
            hidden_states = self.adapter_stack_layer(
                hidden_states=hidden_states,
                mod=mod,
                adapter_stack=adapter_stack,
            )
        
        return hidden_states


class BertLayerAdaptersMixin:
    """Adds adapters to the BertLayer module."""
    
    '''
    def add_fusion_layer(self, adapter_names):
        self.attention.output.add_fusion_layer(adapter_names)
        self.output.add_fusion_layer(adapter_names)
    '''
    
    def add_adapter(self, adapter_name: str, config):
        self.attention.output.add_adapter(adapter_name, config)
        self.output.add_adapter(adapter_name, config)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool):
        self.attention.output.enable_adapters(adapter_names, unfreeze_adapters)
        self.output.enable_adapters(adapter_names, unfreeze_adapters)


class BertEncoderAdaptersMixin:
    """Adds adapters to the BertEncoder module."""

    '''
    def add_fusion_layer(self, adapter_names):
        for layer in self.layer:
            layer.add_fusion_layer(adapter_names)
    '''
    
    def add_adapter(self, adapter_name: str, config):
        for i, layer in enumerate(self.layer):
            layer.add_adapter(adapter_name, config)

    def enable_adapters(self, adapter_names: list, unfreeze_adapters: bool):
        for layer in self.layer:
            layer.enable_adapters(adapter_names, unfreeze_adapters)


class BertModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the BertModel module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_adapter_modules(self):
        super()._init_adapter_modules()

        # language adapters
        '''
        for language in self.config.adapters.adapter_list(AdapterType.text_lang):
            self.encoder.add_adapter(language, AdapterType.text_lang)
            self.add_invertible_lang_adapter(language)
        '''
        # multimodal adapters
        #for task in self.config.adapters.adapter_list(AdapterType.text_task):
        #    self.encoder.add_adapter(task, self.config)
        # fusion
        '''
        if hasattr(self.config, "fusion_models"):
            for fusion_adapter_names in self.config.fusion_models:
                self.add_fusion_layer(fusion_adapter_names)
        '''

    def train_adapter(self, adapter_names: list):
        """Sets the model into mode for training the given adapters."""
        self.train()
        self.freeze_model(True)
        #adapter_names_flat = flatten_adapter_names(adapter_names)
        self.encoder.enable_adapters(adapter_names, True)
        #self.enable_invertible_adapters(adapter_names_flat)
        # use the adapters to be trained by default in every forward pass
        #self.set_active_adapters(adapter_names)

    '''
    def train_fusion(self, adapter_names: list):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        self.train()
        self.freeze_model(True)
        adapter_names_flat = flatten_adapter_names(adapter_names)
        self.encoder.enable_adapters(adapter_names_flat, False, True)
        # use the adapters to be trained by default in every forward pass
        self.set_active_adapters(adapter_names)
        # TODO implement fusion for invertible adapters
    '''

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
        if adapter_name not in ["image", "video", "audio"]:
            raise ValueError("Invalid adapter type {}".format(adapter_type))
        self.encoder.add_adapter(adapter_name, self.config)
    
    '''
    def _add_fusion_layer(self, adapter_names):
        self.encoder.add_fusion_layer(adapter_names)
    '''