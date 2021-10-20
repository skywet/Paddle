# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
from __future__ import print_function

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.nn as nn
from paddle import ParamAttr
from paddle.utils.download import get_weights_path_from_url



__all__ = []

model_urls = {
    'densenet121':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet121_pretrained.pdparams',
     'db1b239ed80a905290fd8b01d3af08e4'),
    'densenet161':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet161_pretrained.pdparams',
     '62158869cb315098bd25ddbfd308a853'),
    'densenet169':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet169_pretrained.pdparams',
     '82cc7c635c3f19098c748850efb2d796'),
    'densenet201':
    ('https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet161_pretrained.pdparams',
     '16ca29565a7712329cf9e36e02caaf58')
}

class _BNConvLayer(nn.Layer):
    def __init__(
        self,
        num_init_features,
        kernel_size,
        num_filters,
        stride=1,
        pad=0,
        groups=1,
        name=None):
        super(_BNConvLayer, self).__init__()

        self._batch_norm = nn.BatchNorm(
            num_init_features,
            param_attr=ParamAttr(name + '_bn_scale'),
            bias_attr=ParamAttr(name + '_bn_offset'),
            moving_mean_name=name + '_bn_mean',
            moving_variance_name=name + '_bn_variance'
        )

        self.relu = nn.ReLU()

        self._conv = nn.Conv2D(
            in_channels = num_init_features,
            out_channels = num_filters,
            kernel_size = kernel_size,
            stride = stride,
            padding=pad,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False
        )
    
    def forward(self, x):
        output = self._batch_norm(x)
        output = self.relu(output)
        return self._conv(output)

class _ConvBNLayer(nn.Layer):
    def __init__(self, num_channels, num_filters, filter_size, stride=1, name=None):
        super(_ConvBNLayer, self).__init__()

        self._conv = nn.Conv2D(
            in_channels = num_channels, 
            out_channels = num_filters,
            kernel_size = filter_size,
            stride = 1,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False
        )
        self._batch_norm = nn.BatchNorm(
            num_filters,
            param_attr=ParamAttr(name=name + '_bn_scale'),
            bias_attr=ParamAttr(name + '_bn_offset'),
            moving_mean_name=name + '_bn_mean',
            moving_variance_name=name + '_bn_variance'
        )
        self.max2d = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        output = self._conv(x)
        output = self._batch_norm(output)
        return output

class _DenseLayer(nn.Layer):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, name=None):
        super(_DenseLayer, self).__init__()
        
        self.bn_layer1 = _BNConvLayer(num_input_features, 1, bn_size * growth_rate, 1, 0,name=name+"_x1")
        self.bn_layer2 = _BNConvLayer(bn_size * growth_rate, 3, growth_rate, 1, 1, name=name+"_x2")

        self.drop_rate = drop_rate
        if self.drop_rate > 0:
            self.dropout = nn.Dropout(p=drop_rate, mode="downscale_in_infer")
    
    def forward(self, x):
        output = x
        output = self.bn_layer1(output)
        output = self.bn_layer2(output)
        if self.drop_rate > 0:
            output = self.dropout(output)
        output = paddle.concat([x, output], axis=1)
        return output

class _DenseBlock(nn.Layer):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, name=None):
        super(_DenseBlock, self).__init__()
        self.layer_arr = []
        for i in range(num_layers):
            self.layer_arr.append(
                self.add_sublayer(
                    "{}_{}".format(name, i+1),
                    _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, name=name+"_"+str(i+1)),
                )
            )
    
    def forward(self, x):
        output = x
        for layer in self.layer_arr:
            output = layer(output)
        return output

class _Transition(nn.Layer):
    def __init__(self, num_input_features, num_output_features, name=None):
        super(_Transition, self).__init__()
        self.BNConv = _BNConvLayer(num_input_features, 1, num_output_features, 1, name=name)
        self.avgpool2d = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        output = self.BNConv(x)
        output = self.avgpool2d(output)
        return output

class DenseNet(nn.Layer):
    """Densenet-BC model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        with_pool (bool) - use pool before the last fc layer or not

    Examples:
    .. code-block:: python

        from paddle.vision.models import DenseNet

        config = (6,12,32,32)

        densenet = DenseNet(block_config=config, num_classes=10)
    """
    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000,
                 with_pool=True):
        super(DenseNet, self).__init__()

        self.block_config = block_config

        self.convbnlayer = _ConvBNLayer(3, num_init_features, 7, 2, name="conv1")
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        
        num_features = num_init_features
        self.dense_blocks = []
        self.transition_layers = []

        for i, num_layers in enumerate(block_config):
            self.dense_blocks.append(
                self.add_sublayer(
                    "db_conv_{}".format(i+2),
                    _DenseBlock(
                        num_layers=num_layers,
                        num_input_features=num_features,
                        bn_size=bn_size,
                        growth_rate=growth_rate,
                        drop_rate=drop_rate,
                        name='conv' + str(i + 2))
                    )
            ) 
               
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                self.transition_layers.append(
                    self.add_sublayer(
                        "tr_conv{}_blk".format(i+2),
                        _Transition(
                            num_input_features=num_features,
                            num_output_features=num_features // 2,
                            name='conv' + str(i + 2) + "_blk"
                        )
                    )
                )
                num_features = num_features // 2
                
        self.batch_norm = nn.BatchNorm(
            num_features,
            act="relu",
            param_attr=ParamAttr(name='conv5_blk_bn_scale'),
            bias_attr=ParamAttr(name='conv5_blk_bn_offset'),
            moving_mean_name='conv5_blk_bn_mean',
            moving_variance_name='conv5_blk_bn_variance')
        # self.features.add_sublayer('norm5', nn.BatchNorm2D(num_features))
        self.with_pool = with_pool
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.num_classes = num_classes
        if num_classes > 0:
            stdv = 1.0 / np.sqrt(num_features * 1.0)
            self.classifier = nn.Linear(
                num_features,
                num_classes,
                weight_attr=ParamAttr(
                    initializer=nn.initializer.Uniform(-stdv, stdv), name="fc_weights"),
                bias_attr=ParamAttr(name="fc_offset"))

    def forward(self, x):
        out = self.convbnlayer(x)
        out = self.relu1(out)
        out = self.maxpool(out)
        
        for i, num_layers in enumerate(self.block_config):
            out = self.dense_blocks[i](out)
            if i != len(self.block_config) - 1:
                out = self.transition_layers[i](out)

        out = self.batch_norm(out)
        if self.with_pool:
            out = self.avgpool(out)
        if self.num_classes > 0:
            out = paddle.flatten(out, 1)
            out = self.classifier(out)
        return out

        
def _densenet(arch, block_cfg, pretrained, **kwargs):
    model = DenseNet(block_config=block_cfg, **kwargs)

    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0],
                                                model_urls[arch][1])
        param = paddle.load(weight_path)
        model.set_dict(param)
    return model


def densenet121(pretrained=False, **kwargs):
    """Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Examples:
    .. code-block:: python

        from paddle.vision.models import densenet121

        # build model
        model = densenet121()
    """
    model_name = 'densenet121'
    return _densenet(model_name, (6, 12, 24, 16), pretrained, **kwargs)


def densenet161(pretrained=False, **kwargs):
    """Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
    .. code-block:: python

        from paddle.vision.models import densenet161

        # build model
        model = densenet161()
    """
    model_name = 'densenet161'
    return _densenet(model_name, (6, 12, 32, 32), pretrained, **kwargs)


def densenet169(pretrained=False, **kwargs):
    """Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
    .. code-block:: python

        from paddle.vision.models import densenet169

        # build model
        model = densenet169()
    """
    model_name = 'densenet169'
    return _densenet(model_name, (6, 12, 48, 32), pretrained, **kwargs)


def densenet201(pretrained=False, **kwargs):
    """Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    
    Examples:
    .. code-block:: python

        from paddle.vision.models import densenet201

        # build model
        model = densenet201()
    """
    model_name = 'densenet201'
    return _densenet(model_name, (6, 12, 64, 48), pretrained, **kwargs)

