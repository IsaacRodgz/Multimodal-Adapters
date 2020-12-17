#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmbt.models.adapter import BertAdapterClf
from mmbt.models.mmadapter import MultimodalAdapterClf
from mmbt.models.mmadapter_fusion import MultimodalAdapterFusionClf
#from mmbt.models.mmadapter import MultimodalBertAdapterMClf, MultimodalBertAdapterMTropesClf
#from mmbt.models.mmadapterfull import MultimodalAdapterFullClf


MODELS = {
    "adapter": BertAdapterClf,
    "mmadapter": MultimodalAdapterClf,
    "mmadapterfus": MultimodalAdapterFusionClf,
    #"mmadapterfull": MultimodalAdapterFullClf,
    #"mmadapter": MultimodalBertAdapterMClf,
}


def get_model(args):
    return MODELS[args.model](args)