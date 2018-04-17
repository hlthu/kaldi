# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2018    Lu Huang(THU)
# Apache 2.0.


""" This module has the implementations of different LSTM layers.
"""
from __future__ import print_function
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase


# This class is for lines like
#   'vfsmn-layer name=fsmn1 input=[-1] dim=1 lookhead-num=-1 lookback-num=-1 look-stride=-1'
# It generates a scale FSMN
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   dim=-1                   [Dimension of the output]
#   lookhead-num=-1          [number of lookhead frames, future]
#   lookback-num=-1          [number of lookback frames, history]
#   self_repair_scale_nonlinearity=1e-5 [It is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent]
#                                           i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent ]
#   ng-per-element-scale-options=''     [Additional options used for the diagonal matrices]
#   ng-affine-options=''                [Additional options used for the full matrices]
#   l2-regularize=0.0                   [Constant controlling l2 regularization for this layer]
class XconfigvFSMNLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "vfsmn-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                        'dim' : -1,
                        'lookhead-num' : -1,
                        'lookback-num' : -1,
                        'look-stride' : 1,
                        'ng-per-element-scale-options' : ' max-change=0.75',
                        'ng-affine-options' : ' max-change=0.75 ',
                        'self-repair-scale-nonlinearity' : 0.00001,
                        'l2-regularize': 0.0,
                        }

    def set_derived_configs(self):
        if self.config['dim'] <= 0:
            self.config['dim'] = self.descriptors['input']['dim']

    def check_configs(self):
        key = 'dim'
        if self.config['dim'] <= 0:
            raise RuntimeError("dim has invalid value {0}.".format(self.config[key]))

        for key in ['self-repair-scale-nonlinearity']:
            if self.config[key] < 0.0 or self.config[key] > 1.0:
                raise RuntimeError("{0} has invalid value {1}.".format(key, self.config[key]))

    def auxiliary_outputs(self):
        return ['m_t']

    def output_name(self, auxiliary_output = None):
        node_name = 'y_t'
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                node_name = auxiliary_output
            else:
                raise RuntimeError("Unknown auxiliary output name {0}".format(auxiliary_output))

        return '{0}.{1}'.format(self.name, node_name)

    def output_dim(self, auxiliary_output = None):
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                if node_name == 'm_t':
                    return self.config['dim']
            else:
                raise RuntimeError("Unknown auxiliary output name {0}".format(auxiliary_output))

        return self.config['dim']

    def get_full_config(self):
        ans = []
        config_lines = self.generate_fsmn_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                ans.append((config_name, line))
        return ans

    # convenience function to generate the FSMN config
    def generate_fsmn_config(self):

        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        output_dim = self.config['dim']
        lookhead_num = self.config['lookhead-num']
        lookback_num = self.config['lookback-num']
        look_stride = self.config['look-stride']


        repair_nonlin = self.config['self-repair-scale-nonlinearity']
        repair_nonlin_str = "self-repair-scale={0:.10f}".format(repair_nonlin) if repair_nonlin is not None else ''
        affine_str = self.config['ng-affine-options']
        # Natural gradient per element scale parameters
        ng_per_element_scale_options = self.config['ng-per-element-scale-options']
        if re.search('param-mean', ng_per_element_scale_options) is None and \
           re.search('param-stddev', ng_per_element_scale_options) is None:
           ng_per_element_scale_options += " param-mean=0.0 param-stddev=0.05 "
        pes_str = ng_per_element_scale_options
        l2_regularize = self.config['l2-regularize']
        l2_regularize_option = ('l2-regularize={0} '.format(l2_regularize)
                                if l2_regularize != 0.0 else '')


        configs = []

        configs.append("### Begin vFSMN layer '{0}'".format(name))
        configs.append("# the transform matrix:")
        configs.append("component name={0}.W_all type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, 2 * input_dim, output_dim,
                                                       affine_str, l2_regularize_option))
        configs.append("# the diagal matrices:")
        for his_cnt in range(0, lookback_num + 1):
            configs.append("component name={0}.U_h_{1} type=NaturalGradientPerElementScaleComponent "
                       " dim={2} {3} {4}".format(name, his_cnt * look_stride, input_dim, pes_str,
                                                 l2_regularize_option))
        for fut_cnt in range(1, lookhead_num + 1):
            configs.append("component name={0}.U_f_{1} type=NaturalGradientPerElementScaleComponent "
                       " dim={2} {3} {4}".format(name, fut_cnt * look_stride, input_dim, pes_str,
                                                 l2_regularize_option))

        configs.append('component name={0}.noop type=NoOpComponent dim={1}'.format(name, input_dim))

        configs.append("# Weighting the history and future :")
        for his_cnt in range(0, lookback_num + 1):
            configs.append("component-node name={0}.h_t_{1} component={0}.U_h_{1}  input=Offset({2}, -{1})".format(name, his_cnt * look_stride, input_descriptor))
        for fut_cnt in range(1, lookhead_num + 1):
            configs.append("component-node name={0}.f_t_{1} component={0}.U_f_{1}  input=Offset({2}, {1})".format(name, fut_cnt * look_stride, input_descriptor))

        configs.append("# Sumimg up the history and future :")
        sum_str = "component-node name={0}.m_t component={0}.noop input=Sum(".format(name)
        for his_cnt in range(0, lookback_num + 1):
            sum_str += "{0}.h_t_{1}, ".format(name, his_cnt * look_stride)
        for fut_cnt in range(1, lookhead_num):
            sum_str += "{0}.f_t_{1}, ".format(name, fut_cnt * look_stride)
        sum_str += "{0}.f_t_{1})".format(name, lookhead_num * look_stride)
        configs.append(sum_str)

        configs.append("component-node name={0}.h_t component={0}.W_all input=Append({1}, {0}.m_t)".format(name, input_descriptor))
        

        configs.append("component name={0}.relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, repair_nonlin_str))
        configs.append("component-node name={0}.relu_t component={0}.relu input={0}.h_t".format(name))

        configs.append("component name={0}.batchnorm type=BatchNormComponent dim={1} target-rms=1.0".format(name, output_dim))
        configs.append("component-node name={0}.y_t component={0}.batchnorm input={0}.relu_t".format(name))

        configs.append("### End vFSMN layer '{0}'".format(name))
        return configs

# This class is for lines like
#   'cfsmn-layer name=fsmn2 input=[-1] projection-dim=1 output-dim=1 lookhead-num=-1 lookback-num=-1 look-stride=-1 resnet-opts=fsmn1'
# It generates a scale FSMN
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   projection-dim=-1        [project the input to the this dim]
#   output-dim=-1            [Dimension of the output]
#   lookhead-num=-1          [number of lookhead frames, future]
#   lookback-num=-1          [number of lookback frames, history]
#   resnet-opts=''           [the residual node to connect, layer's name]
#   self_repair_scale_nonlinearity=1e-5 [It is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent]
#                                           i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent ]
#   ng-per-element-scale-options=''     [Additional options used for the diagonal matrices]
#   ng-affine-options=''                [Additional options used for the full matrices]
#   l2-regularize=0.0                   [Constant controlling l2 regularization for this layer]
#   dropout-proportion=-1               [if -1, no dropout]
#   dropout-per-frame=False             [if dropout ]
class XconfigcFSMNLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "cfsmn-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                        'projection-dim' : -1,
                        'output-dim' : -1,
                        'lookhead-num' : -1,
                        'lookback-num' : -1,
                        'look-stride' : 1,
                        'resnet-opts' : '',
                        'ng-per-element-scale-options' : ' max-change=0.75',
                        'ng-affine-options' : ' max-change=0.75 ',
                        'self-repair-scale-nonlinearity' : 0.00001,
                        'l2-regularize': 0.0,
                        'dropout-proportion': -1.0,
                        'dropout-per-frame': False,
                        }

    def set_derived_configs(self):
        if self.config['output-dim'] <= 0:
            self.config['output-dim'] = self.descriptors['input']['dim']
        if self.config['projection-dim'] <= 0:
            self.config['projection-dim'] = self.descriptors['input']['dim']/4

    def check_configs(self):
        key = 'output-dim'
        if self.config['output-dim'] <= 0:
            raise RuntimeError("output-dim has invalid value {0}.".format(self.config[key]))
        key = 'projection-dim'
        if self.config['projection-dim'] <= 0:
            raise RuntimeError("projection-dim has invalid value {0}.".format(self.config[key]))
        for key in ['self-repair-scale-nonlinearity']:
            if self.config[key] < 0.0 or self.config[key] > 1.0:
                raise RuntimeError("{0} has invalid value {1}.".format(key, self.config[key]))

    def auxiliary_outputs(self):
        return ['m_t']

    def output_name(self, auxiliary_output = None):
        node_name = 'y_t'
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                node_name = auxiliary_output
            else:
                raise RuntimeError("Unknown auxiliary output name {0}".format(auxiliary_output))

        return '{0}.{1}'.format(self.name, node_name)

    def output_dim(self, auxiliary_output = None):
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                if node_name == 'm_t':
                    return self.config['output-dim']
            else:
                raise RuntimeError("Unknown auxiliary output name {0}".format(auxiliary_output))

        return self.config['output-dim']

    def get_full_config(self):
        ans = []
        config_lines = self.generate_fsmn_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                ans.append((config_name, line))
        return ans

    # convenience function to generate the FSMN config
    def generate_fsmn_config(self):

        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        prj_dim = self.config['projection-dim']
        output_dim = self.config['output-dim']
        lookhead_num = self.config['lookhead-num']
        lookback_num = self.config['lookback-num']
        look_stride = self.config['look-stride']
        res_opts = self.config['resnet-opts']


        repair_nonlin = self.config['self-repair-scale-nonlinearity']
        repair_nonlin_str = "self-repair-scale={0:.10f}".format(repair_nonlin) if repair_nonlin is not None else ''
        affine_str = self.config['ng-affine-options']
        # Natural gradient per element scale parameters
        ng_per_element_scale_options = self.config['ng-per-element-scale-options']
        if re.search('param-mean', ng_per_element_scale_options) is None and \
           re.search('param-stddev', ng_per_element_scale_options) is None:
           ng_per_element_scale_options += " param-mean=0.0 param-stddev=0.05 "
        pes_str = ng_per_element_scale_options
        l2_regularize = self.config['l2-regularize']
        l2_regularize_option = ('l2-regularize={0} '.format(l2_regularize)
                                if l2_regularize != 0.0 else '')

        dropout_proportion = self.config['dropout-proportion']
        dropout_per_frame = 'true' if self.config['dropout-per-frame'] else 'false'

        configs = []

        configs.append("### Begin cFSMN layer '{0}'".format(name))
        configs.append("# the transform matrices:")
        configs.append("component name={0}.W_in type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, input_dim, prj_dim,
                                                       affine_str, l2_regularize_option))
        configs.append("component name={0}.W_out type=NaturalGradientAffineComponent input-dim={1} "
                       "output-dim={2} {3} {4}".format(name, prj_dim, output_dim,
                                                       affine_str, l2_regularize_option))

        configs.append("# the diagal matrices:")
        for his_cnt in range(0, lookback_num + 1):
            configs.append("component name={0}.U_h_{1} type=NaturalGradientPerElementScaleComponent "
                       " dim={2} {3} {4}".format(name, his_cnt * look_stride, prj_dim, pes_str,
                                                 l2_regularize_option))
        for fut_cnt in range(1, lookhead_num + 1):
            configs.append("component name={0}.U_f_{1} type=NaturalGradientPerElementScaleComponent "
                       " dim={2} {3} {4}".format(name, fut_cnt * look_stride, prj_dim, pes_str,
                                                 l2_regularize_option))

        configs.append('component name={0}.noop type=NoOpComponent dim={1}'.format(name, prj_dim))

        if dropout_proportion != -1.0:
            configs.append("component name={0}.dropout type=DropoutComponent dim={1} "
                            "dropout-proportion={2} dropout-per-frame={3}"
                            .format(name, prj_dim, dropout_proportion, dropout_per_frame))

        configs.append("component-node name={0}.p_t component={0}.W_in  input={1}".format(name, input_descriptor))
        
        configs.append("# Weighting the history and future :")
        if dropout_proportion != -1.0:
            for his_cnt in range(0, lookback_num + 1):
                configs.append("component-node name={0}.h_t_{1}_pre component={0}.U_h_{1}  input=Offset({0}.p_t, -{1})".format(name, his_cnt * look_stride))
                configs.append("component-node name={0}.h_t_{1} component={0}.dropout input={0}.h_t_{1}_pre".format(name, his_cnt * look_stride))
            for fut_cnt in range(1, lookhead_num + 1):
                configs.append("component-node name={0}.f_t_{1}_pre component={0}.U_f_{1}  input=Offset({0}.p_t, {1})".format(name, fut_cnt * look_stride))
                configs.append("component-node name={0}.f_t_{1} component={0}.dropout input={0}.f_t_{1}_pre".format(name, fut_cnt * look_stride))
        else:
            for his_cnt in range(0, lookback_num + 1):
                configs.append("component-node name={0}.h_t_{1} component={0}.U_h_{1}  input=Offset({0}.p_t, -{1})".format(name, his_cnt * look_stride))
            for fut_cnt in range(1, lookhead_num + 1):
                configs.append("component-node name={0}.f_t_{1} component={0}.U_f_{1}  input=Offset({0}.p_t, {1})".format(name, fut_cnt * look_stride))

        configs.append("# Sumimg up the history and future :")
        sum_str = "component-node name={0}.m_t component={0}.noop input=Sum({0}.p_t, ".format(name)
        if res_opts is not '':
            sum_str += "{0}.m_t, ".format(res_opts)
        for his_cnt in range(0, lookback_num + 1):
            sum_str += "{0}.h_t_{1}, ".format(name, his_cnt * look_stride)
        for fut_cnt in range(1, lookhead_num):
            sum_str += "{0}.f_t_{1}, ".format(name, fut_cnt * look_stride)
        sum_str += "{0}.f_t_{1})".format(name, lookhead_num * look_stride)
        configs.append(sum_str)

        configs.append("component-node name={0}.h_t component={0}.W_out input={0}.m_t".format(name))
        

        configs.append("component name={0}.relu type=RectifiedLinearComponent dim={1} {2}".format(name, output_dim, repair_nonlin_str))
        configs.append("component-node name={0}.relu_t component={0}.relu input={0}.h_t".format(name))

        configs.append("component name={0}.batchnorm type=BatchNormComponent dim={1} target-rms=1.0".format(name, output_dim))
        configs.append("component-node name={0}.y_t component={0}.batchnorm input={0}.relu_t".format(name))

        configs.append("### End cFSMN layer '{0}'".format(name))
        return configs