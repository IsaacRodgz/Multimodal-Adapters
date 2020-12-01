#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmbt.models.adapter import MultimodalBertAdapterClf
from mmbt.models.mmadapter import MultimodalBertAdapterMClf, MultimodalBertAdapterMTropesClf
from mmbt.models.mmadapterfull import MultimodalAdapterFullClf


MODELS = {
    "adapter": MultimodalBertAdapterClf,
    "mmadapter": MultimodalBertAdapterMClf,
    "mmadapterfull": MultimodalAdapterFullClf,
}


def get_model(args):
    return MODELS[args.model](args)