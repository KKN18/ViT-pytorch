# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

from .modeling_resnet import ResNetV2

import pdb

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

HIDDEN_SIZE = 768
HEAD_SIZE = 12

#global PATCH_X
#global PATCH_Y

#PATCH_X = 0
#PATCH_Y = 0

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        # num_heads = 12 (default)
        self.num_attention_heads = config.transformer["num_heads"]
        # attention_head_size = 768 / 12 = 64 (default)
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        # all_head_size = 12 x 64 = 768 (default)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear(768, 768) (default)
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

        #self.key_conv2d = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=3, padding='same', padding_mode='zeros')
        #self.value_conv2d = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=3, padding='same', padding_mode='zeros')
        
        global PATCH_X
        global PATCH_Y
        assert(PATCH_X == PATCH_Y and PATCH_X != 0)
        n_in = PATCH_X

        self.kernel_size=3
        self.stride=2
        self.padding=1
        
        self.n_out = (n_in+2*self.padding-self.kernel_size)//self.stride + 1

        self.key_conv2d = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, padding_mode='zeros') 
        self.value_conv2d = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, padding_mode='zeros')

        nn.init.xavier_uniform(self.key_conv2d.weight)
        nn.init.xavier_uniform(self.value_conv2d.weight)
       
        #self.is_shortcut_possible = False
        #if (n_in % self.n_out == 0):
        #    self.is_shortcut_possible = True
        #    ratio = n_in // self.n_out
        #    self.extend_hidden_size = config.hidden_size * (ratio ** 2)
        #    self.fc = nn.Linear(self.extend_hidden_size, config.hidden_size)
        #    nn.init.xavier_uniform(self.fc.weight)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # hidden_states shape [51, 197, 768]
        h_shape = hidden_states.shape
        cls, img_tokens = torch.split(hidden_states, [1, h_shape[1]-1], dim=1)
        query_layer = self.query(img_tokens)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        k_shape = key_layer.shape
        v_shape = value_layer.shape
        pdb.set_trace()
       
        global PATCH_X # 14
        global PATCH_Y # 14 
  
        # (Removed) remove cls token
        # key_cls, key_layer = torch.split(mixed_key_layer, [1, k_shape[1]-1], dim=1)
        # value_cls, value_layer = torch.split(mixed_value_layer, [1, v_shape[1]-1], dim=1)

        # (Removed) shape without cls [51, 196, 768] = [B, HxW, C]
        # only_key_shape = key_layer.shape
        # only_value_shape = value_layer.shape

        # reshape from [B, HxW, C] to [B, H, W, C]
        reshaped_key_layer = key_layer.reshape(k_shape[0], PATCH_X, PATCH_Y, k_shape[2])
        reshaped_value_layer = value_layer.reshape(v_shape[0], PATCH_X, PATCH_Y, v_shape[2])


        # change to CNN input dimension [B, C, H, W]
        mixed_key_layer_4D = reshaped_key_layer.permute(0, 3, 1, 2).contiguous()
        mixed_value_layer_4D = reshaped_value_layer.permute(0, 3, 1, 2).contiguous()
    
        # CNN forward
        conv_key_layer = self.key_conv2d(mixed_key_layer_4D)
        conv_value_layer = self.value_conv2d(mixed_value_layer_4D)

        # restore original dimension [B, H, W, C]
        conv_key_layer = conv_key_layer.permute(0, 2, 3, 1).contiguous()
        conv_value_layer = conv_value_layer.permute(0, 2, 3, 1).contiguous()

        # restore initial shape without cls [51, 49, 768]
        after_conv_shape = (only_key_shape[0], self.n_out ** 2, only_key_shape[2])
        mixed_key_layer = conv_key_layer.reshape(after_conv_shape)
        mixed_value_layer = conv_value_layer.reshape(after_conv_shape)

        # add shortcut if possible
        # if (self.is_shortcut_possible):
        #    shuffled_key_layer = reshaped_key_layer.reshape(k_shape[0], self.n_out, self.n_out, self.extend_hidden_size)
        #    shuffled_value_layer = reshaped_value_layer.reshape(k_shape[0], self.n_out, self.n_out, self.extend_hidden_size)
            
        #    shortcut_key_layer = self.fc(shuffled_key_layer).reshape(k_shape[0], self.n_out ** 2, k_shape[2])
        #    shortcut_value_layer = self.fc(shuffled_value_layer).reshape(v_shape[0], self.n_out ** 2, v_shape[2])

        #    mixed_key_layer = mixed_key_layer + shortcut_key_layer
        #    mixed_value_layer = mixed_value_layer + shortcut_value_layer
        
        # cat cls token
        cat_key_layer = torch.cat((key_cls, mixed_key_layer), dim=1)
        cat_value_layer = torch.cat((value_cls, mixed_value_layer), dim=1)

        # final shape [51, 50, 768], Not used
        final_key_layer = cat_key_layer 
        final_value_layer = cat_value_layer

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(cat_key_layer)
        value_layer = self.transpose_for_scores(cat_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        pdb.set_trace()
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        HIDDEN_SIZE = config.hidden_size


        global PATCH_X
        global PATCH_Y

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            PATCH_X = img_size[0] // 16
            PATCH_Y = img_size[1] // 16
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            PATCH_X = img_size[0] // patch_size[0]
            PATCH_Y = img_size[1] // patch_size[1]
            #print('\nn_patches : '+str(n_patches))
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        #print('B : '+str(B))
        #print('X shape before embedding')
        #print(x.shape)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        #print('Embedding shape')
        #print(x.shape)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():

            if (HEAD_SIZE > 12):
                print("Head Size Error")
            quotient = self.hidden_size // HEAD_SIZE
            if (quotient > 64):
                print("Quotient Error")
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")][:self.hidden_size,:HEAD_SIZE,:quotient]).reshape(-1).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")][:self.hidden_size,:HEAD_SIZE, :quotient]).reshape(-1).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")][:self.hidden_size,:HEAD_SIZE, :quotient]).reshape(-1).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")][:HEAD_SIZE, :quotient,:self.hidden_size]).reshape(-1).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")][:HEAD_SIZE, :quotient]).reshape(-1).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")][:HEAD_SIZE, :quotient]).reshape(-1).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")][:HEAD_SIZE, :quotient]).reshape(-1).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")][:self.hidden_size]).reshape(-1).view(-1)
            

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")][:self.hidden_size,:]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")][:,:self.hidden_size]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")][:self.hidden_size]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")][:self.hidden_size]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")][:self.hidden_size]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")][:self.hidden_size]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")][:self.hidden_size]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            #print("##Weight Shape and type")
            #print(weights["embedding/kernel"][:700].shape)
            #print(weights["embedding/kernel"][:,:,:,:700].shape)
            #print(weights["cls"].shape)
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"][:,:,:,:HIDDEN_SIZE], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"][:HIDDEN_SIZE]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"][:,:,:HIDDEN_SIZE]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"][:HIDDEN_SIZE]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"][:HIDDEN_SIZE]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb[:,:,:HIDDEN_SIZE]))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
