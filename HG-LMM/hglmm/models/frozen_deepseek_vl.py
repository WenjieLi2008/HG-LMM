import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from xtuner.model.utils import LoadWoInit
from mmengine.logging import print_log
from hglmm.utils import compute_mask_IoU
import time
from mmengine.config import Config
import numpy as np
from sklearn.cluster import KMeans
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np
from torch.nn.functional import interpolate
from featup.util import norm, unnorm, pca, remove_axes
from featup.plotting import plot_feats, plot_lang_heatmaps
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from typing import List
from segment_anything import SamPredictor, sam_model_registry
import cv2
import torchvision.models as models
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

from scipy.ndimage import center_of_mass
import math
# import wandb
# wandb.init(project='F-LMM', name='llava_1.5_vicuna_7b')



# ========================= Blob loss ================================



class FrozenDeepseekVL(BaseModel):

    def __init__(self,
                 model,
                 tokenizer,
                 merge='mean',
                 loss_mask=None,
                 loss_dice=None,
                 **kwargs):
        super().__init__()
        with LoadWoInit():
            self.deepseek_vl = BUILDER.build(model)
        self.deepseek_vl.requires_grad_(False)
        self.tokenizer = BUILDER.build(tokenizer)
        self.image_token_idx = self.tokenizer.encode('<image_placeholder>', add_special_tokens=False)[-1]
        print_log(f"Image token: {self.tokenizer.decode(self.image_token_idx)}")
        self.lifg = LIFG()
        self.ldsi = LDSI()
        self.lcb = LCB()
        self.fuse_maskhead = Fuse_Maskhead()
        self.merge = merge
        assert merge in ['mean', 'max']
        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)
        self.patch_size = 16   # hard-code use siglip_large_patch16_384
        self.clip_shape = 24
        self._generation_ready = False

    def apply_merge(self, x, merge, dim=1):
        if merge == 'mean':
            return x.mean(dim=dim)
        elif merge =='last':
            return x[:, -1]
        elif merge == 'max':
            return x.max(dim=dim).values
        else:
            raise NotImplementedError

    def init_weights(self):
        pass

    def train(self, mode=True):
        super().train(mode=mode)
        self.deepseek_vl.train(mode=False)
        self.training = mode
        return self

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data)
        elif mode == 'predict':
            return self.predict(data)
        elif mode == 'tensor':
            return self._forward(data)
        else:
            raise NotImplementedError

    def _compute(self, pred_masks, gt_masks):
        mask_cnt = pred_masks.shape[0]
        loss_dice = self.loss_dice(
            pred_masks.view(mask_cnt, -1), gt_masks.view(mask_cnt, -1),
            avg_factor=mask_cnt)
        loss_mask = self.loss_mask(
            pred_masks.view(-1),
            gt_masks.view(-1),
            avg_factor=pred_masks.numel())
        accuracy = torch.eq((pred_masks.detach().sigmoid() > 0.5).to(gt_masks),
                            gt_masks).to(gt_masks).mean()
        aiou = compute_mask_IoU((pred_masks.detach().sigmoid() > 0.5).to(gt_masks).view(mask_cnt, -1),
                                gt_masks.view(mask_cnt, -1)).mean()

        return loss_dice, loss_mask, accuracy, aiou

class ResidualConvUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.conv(x)  # 残差连接

class DynamicConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 256
        self.out_channels = 48
        self.text_dim = 256
        self.controller = nn.Sequential(
            nn.Linear(self.text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.in_channels * self.out_channels + self.out_channels * self.out_channels + self.out_channels + self.out_channels)  # 权重 + 偏置
        )
        self.ln48 = nn.LayerNorm(normalized_shape=48, eps=1e-5, elementwise_affine=True)


    def forward(self, image_embeds, llm_text_embeds):
        B, input_channels, H, W = image_embeds.shape
        n = llm_text_embeds.shape[0]

        # 1. 动态生成卷积核参数
        params = self.controller(llm_text_embeds).to(image_embeds.device)  # (n, in_channels*out_channels + out_channels)
        num_layers = 2
        output_channels = self.out_channels
        input_channels = self.in_channels

        # 拆分权重和偏置
        weights = [
            params[:, :input_channels * output_channels].reshape(n * output_channels, input_channels, 1, 1),
            params[:,
            input_channels * output_channels: input_channels * output_channels + output_channels * output_channels].reshape(
                n * output_channels, output_channels, 1, 1)
        ]
        biases = [
            params[:,
            input_channels * output_channels + output_channels * output_channels: input_channels * output_channels + output_channels * output_channels + output_channels].reshape(
                n * output_channels),
            params[:, input_channels * output_channels + output_channels * output_channels + output_channels:].reshape(
                n * output_channels)
        ]

        image_embeds_n = image_embeds.expand(n, -1, -1, -1, -1).permute(1, 0, 2, 3, 4).reshape(B, n * input_channels, H, W)
        output = image_embeds_n
        # 分组卷积：每个文本独立处理
        for i in range(num_layers):
            output = F.conv2d(
                output,
                weights[i],
                bias=biases[i],
                stride=1,
                padding=0,
                dilation=1,
                groups=n
            )
            if i < num_layers - 1:
                output = F.relu(output)
        dynamic_image_embeds = output.reshape(B, n, output_channels, H, W)

        dynamic_image_embeds = self.ln48(dynamic_image_embeds.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        dynamic_image_embeds_list = []
        for i in range(4):
            dynamic_image_embeds_list.append(dynamic_image_embeds[i * 2:(i + 1) * 2])
        dynamic_image_embeds = torch.stack(dynamic_image_embeds_list).permute(0, 2, 1, 3, 4, 5).reshape(4,-1,96,24,24)
        return dynamic_image_embeds

class LIFG(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, out_channels=128):
        super().__init__()
        self.dynamic_conv = DynamicConv()
        self.conv1 = nn.Conv2d(2048, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 256, kernel_size=3, padding=1)
        self.ln2048 = nn.LayerNorm(normalized_shape=2048, eps=1e-5, elementwise_affine=True)
    def process_clip_features(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x

    def forward(self, clip_multilayer, llm_text_embeds):
        clip_multilayer = self.ln2048(clip_multilayer)
        clip_multilayer = clip_multilayer.permute(0, 2, 1).reshape(8, 2048, 24, 24)
        processed_clipfeatures = self.process_clip_features(clip_multilayer)
        dynamic_fused_image_embeds = self.dynamic_conv(processed_clipfeatures, llm_text_embeds)

        return dynamic_fused_image_embeds

class LCB(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, out_channels=128):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 第一阶段：1/4下采样
        x = self.conv1(x)  # 1/2
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)  # 1/4 -> F1

        # 第二阶段：多尺度特征提取
        shallow_features = self.proj(x)  # 1/4
        return shallow_features

class LDSI(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv96to128 = nn.Conv2d(96, 128, kernel_size=3, padding=1)
        self.conv384to96 = nn.Conv2d(384, 96, kernel_size=3, padding=1)
        self.conv192to128 = nn.Conv2d(192, 128, kernel_size=3, padding=1)

        self.target_sizes = {
            "stage1": 96,  # 1/4
            "stage2": 48,  # 1/8
            "stage3": 24,  # 1/16
            "stage4": 12  # 1/32
        }
        self.oupt_channels = 64
        # 1. Reassemble模块: Resample到不同分辨率
        self.reassemble_projs = nn.ModuleDict({
            f"stage{i + 1}": nn.Sequential(
                nn.Conv2d(128, self.oupt_channels, kernel_size=1),
                nn.Upsample(size=size, mode='bilinear')
                # nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
                if i < 3 else
                nn.Conv2d(self.oupt_channels, self.oupt_channels, kernel_size=3, stride=2, padding=1)
            )
            for i, (stage, size) in enumerate(self.target_sizes.items())
        })
        self.spm_adapters = nn.Sequential(
                nn.Conv2d(128, self.oupt_channels, kernel_size=1),
                nn.BatchNorm2d(self.oupt_channels),
                nn.ReLU(inplace=True)
        )


    def forward(self, dynamic_fused_image_embeds,mask_attentions,shallow_features):
        mask_attentions = self.conv384to96(mask_attentions).expand(3, -1, 96, 24, 24)
        dynamic_fused_image_embeds_deep = dynamic_fused_image_embeds[1:4]
        mask_embeds = torch.cat([mask_attentions, dynamic_fused_image_embeds_deep], dim=2)
        mask_embeds_list = []
        mask_embeds_list.append(self.conv96to128(dynamic_fused_image_embeds[0]))
        for i in range(3):
            mask_embeds_list.append(self.conv192to128(mask_embeds[i]))
        mask_embeds = torch.stack(mask_embeds_list)

        reassembled = []
        for i, stage in enumerate(self.target_sizes.keys()):
            proj_module = self.reassemble_projs[stage]
            # 应用Resample
            x = mask_embeds[i]
            resampled = proj_module(x)
            if i == 0:
                spm_feat = self.spm_adapters(shallow_features)
                resampled = resampled + spm_feat
            reassembled.append(resampled)
        return reassembled

class Fuse_Maskhead(nn.Module):
    def __init__(self):
        super().__init__()
        self.oupt_channels = 64
        # 2. 上采样路径
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)

        # 3. RefineNet融合模块 (按照DPT图1右结构)
        self.resconvunit =ResidualConvUnit(self.oupt_channels)

        self.refine_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualConvUnit(self.oupt_channels),
                self.upsample,
                nn.Conv2d(self.oupt_channels, self.oupt_channels, kernel_size=3, padding=1)
                # nn.ConvTranspose2d(oupt_channels, oupt_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            ) for _ in range(4)
        ])

        # 4. Mask预测头
        self.mask_head_fpn = nn.Sequential(
            nn.Conv2d(self.oupt_channels, self.oupt_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.oupt_channels),
            nn.Dropout(p=0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.oupt_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, multi_features):
        #  自底向上融合 (从深层到浅层)
        #最后一层单独处理
        hidden_states = []
        stages_num = 3
        x = self.refine_blocks[stages_num](multi_features[stages_num])
        hidden_states.append(x)
        # 融合过程 (参考DPT图1右)，前三层
        for i in range(stages_num, 0, -1):
            # 添加跳跃连接 (skip connection)
            skip_feat = multi_features[i - 1]  # 获取同级特征
            skip_feat = self.resconvunit(skip_feat)
            x = x + skip_feat  # 特征融合
            # RefineNet处理
            x = self.refine_blocks[stages_num - i](x)
            hidden_states.append(x)
        # Step 3: 分割头上采样到原图分辨率
        mask_pred = self.mask_head_fpn(x)  # (3,1,384,384)
        return mask_pred

class HGFrozenDeepseekVL(FrozenDeepseekVL):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_proj = nn.Linear(self.deepseek_vl.config.language_config.hidden_size,
                                   256)
        self.text_layer_weights = nn.Parameter(
            torch.ones(self.deepseek_vl.config.language_config.num_hidden_layers))

    def get_text_layer_weights(self):
        return torch.softmax(self.text_layer_weights, dim=0)


    def _forward(self, data_sample):
        # with torch.no_grad():
        text_layer_weights = self.get_text_layer_weights()
        pixel_values = data_sample['pixel_values'][None, None].to(
            device=self.deepseek_vl.device,
            dtype=self.deepseek_vl.dtype)
        input_ids = data_sample['input_ids'][None].to(self.deepseek_vl.device)
        images_seq_mask = input_ids == self.image_token_idx
        images_emb_mask = torch.ones((1, 1, images_seq_mask.sum()), dtype=torch.bool,
                                     device=self.deepseek_vl.device)


        inputs_embeds,image_embeds,clip_attns, clip_hidden_states = self.deepseek_vl.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask)
        with torch.no_grad():
            outputs = self.deepseek_vl.language_model(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
                use_cache=False)


        mask_ids = data_sample['mask_ids'].to(self.deepseek_vl.device)
        meta_data = data_sample['meta_data']
        attentions = [attn[0, ..., images_seq_mask[0]] for attn in outputs.attentions]
        attentions = [attn.view(*attn.shape[:-1], self.clip_shape, self.clip_shape) for attn in attentions]
        hidden_states = outputs.hidden_states[-self.deepseek_vl.config.language_config.num_hidden_layers:]
        hidden_states = torch.stack([hs[0] for hs in hidden_states])  # num_layers, seq_len, dim
        hidden_states_tea = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # seq_len, dim
        del outputs


        ###LITA
        masks = data_sample['masks']
        mask_attentions = []
        llm_text_embeds = []
        for mask_id in range(len(masks)):
            matched = mask_ids == mask_id
            assert matched.sum() > 0
            merge = 'mean'
            mask_attentions.append(
                torch.cat(
                    [self.apply_merge(attn[:, matched], merge ,dim=1) for attn in attentions]
                )
            )
            llm_text_embeds.append(self.text_proj(hidden_states_tea[matched]).mean(0))

            del  matched

        llm_text_embeds = torch.stack(llm_text_embeds)
        mask_attentions = torch.stack(mask_attentions).to(self.ldsi.dtype)
        del attentions


        ###LIFG
        clip_hidden_states = torch.stack(clip_hidden_states)  # (24, 576, 2048)
        selected_layers = [3,6,9,12,15,18,20,23]
        clip_multilayer = torch.stack([clip_hidden_states[i] for i in selected_layers]).squeeze(1)
        dynamic_fused_image_embeds = self.lifg(clip_multilayer,llm_text_embeds)

        ###LDSI
        shallow_features = self.lcb(pixel_values[0])
        multi_features = self.ldsi(dynamic_fused_image_embeds,mask_attentions,shallow_features)

        ###Maskhead
        pred_masks = self.fuse_maskhead(multi_features)
        del dynamic_fused_image_embeds


        pred_masks = pred_masks[:, 0]
        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        before_height = int(meta_data['padding']['before_height'] )
        before_width = int(meta_data['padding']['before_width'] )
        mask_h = int(meta_data['image_shape']['height'] + 0.5)
        mask_w = int(meta_data['image_shape']['width'] + 0.5)

        pred_masks = F.interpolate(pred_masks[None].float(),
                                   size=(padded_h, padded_w), mode='bilinear')[0]
        pred_masks \
            = pred_masks[:, before_height:before_height + mask_h, before_width:before_width + mask_w].contiguous()



        output = dict(pred_masks=pred_masks,mask_ids=mask_ids, hidden_states=hidden_states,
                      mask_attentions=mask_attentions)

        return output

    @torch.no_grad()
    def predict(self, data_sample):
        return self._forward(data_sample)['pred_masks']

    def compute_loss(self, data):
        mask_cnts = 0

        loss_dice = 0
        loss_mask = 0
        accuracy = 0
        aiou = 0

        for data_sample in data:
            forward_output = self._forward(data_sample)
            pred_masks = forward_output['pred_masks']
            masks = data_sample['masks'].to(self.deepseek_vl.device)
            gt_masks = F.interpolate(masks[None].float(),
                                     size=pred_masks.shape[-2:])[0].to(pred_masks)
            mask_cnt = pred_masks.shape[0]
            assert pred_masks.shape == gt_masks.shape
            mask_cnts += mask_cnt

            loss_dice_, loss_mask_, accuracy_, aiou_ = self._compute(pred_masks, gt_masks)
            loss_dice += loss_dice_ * mask_cnt
            loss_mask += loss_mask_ * mask_cnt
            accuracy += accuracy_ * mask_cnt
            aiou += aiou_ * mask_cnt


        assert mask_cnts > 0

        loss_dict = {'loss_mask': loss_mask / mask_cnts,
                     'loss_dice': loss_dice / mask_cnts,
                     'accuracy': accuracy / mask_cnts,
                     'aiou': aiou / mask_cnts,
                     }

        return loss_dict
    def _prepare_for_generation(self,
                                image_processor,
                                prompt_template,
                                max_thought_tokens=16,
                                max_new_tokens=512,
                                lmm_name='',
                                additional_prompt=' Please briefly answer the question.',
                                with_memory=True,
                                box_scale=1.0,
                                use_sam=True,
                                kmeans=False,
                                **kwargs):
        from deepseek_vl.models import VLChatProcessor
        from transformers import StoppingCriteriaList
        from xtuner.utils import StopWordStoppingCriteria
        if isinstance(image_processor, dict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        self.vl_chat_processor = VLChatProcessor.from_pretrained(lmm_name)
        self.prompt_template = prompt_template
        self.max_thought_tokens = max_thought_tokens
        self.max_new_tokens = max_new_tokens

        stop_words = self.prompt_template.get('STOP_WORDS', []) + ['.']   # only need the first sentence
        self.stop_criteria = StoppingCriteriaList()
        self.stop_word_ids = [self.tokenizer.encode(word, add_special_tokens=False)[-1]
                              for word in stop_words]
        for word in stop_words:
            self.stop_criteria.append(
                StopWordStoppingCriteria(self.tokenizer, word))
        self._generation_ready = True
        self.additional_prompt = additional_prompt
        self.with_memory = with_memory
        assert self.with_memory, "For now we only support with_memory"
        self.box_scale = box_scale
        self.use_sam = use_sam
        self.kmeans = kmeans
        self.config = self.deepseek_vl.config
        print_log(f"USE SAM? {use_sam}")
        print_log(f"KMeans? {kmeans}")


    @torch.no_grad()
    def _conversation(self, conversation, images):
        # prepare for inputs
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=images, force_batchify=True
        )[0].to(self.deepseek_vl.device)

        # run image encoder to get the image embeddings
        inputs_embeds, images_embeds, attns, clip_hidden_states = self.deepseek_vl.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.deepseek_vl.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    @torch.no_grad()
    def answer(self, image, question, *args, **kwargs):
        assert self._generation_ready
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{question}",
                "images": ["image"],
            },
            {"role": "Assistant", "content": ""},
        ]
        # prepare for inputs
        prepare_inputs, meta_datas = self.vl_chat_processor(
            conversations=conversation, images=[image], force_batchify=True
        )
        prepare_inputs = prepare_inputs.to(self.deepseek_vl.device)
        # run image encoder to get the image embeddings
        inputs_embeds,image_embeds,clip_attns, clip_hidden_states = self.deepseek_vl.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.deepseek_vl.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        ## collect attentions and embeddings
        num_layers = self.deepseek_vl.config.language_config.num_hidden_layers
        num_heads = self.deepseek_vl.config.language_config.num_attention_heads
        # collect attentions
        images_seq_mask = prepare_inputs.images_seq_mask[0]
        attention_maps = torch.cat([torch.cat([attns[layer_id][0, ..., torch.where(images_seq_mask)[0]]
                                               for attns in outputs.attentions[1:]], dim=-2)
                                    for layer_id in range(num_layers)], dim=0).view(num_layers*num_heads, -1,
                                                                                    self.clip_shape, self.clip_shape)
        # collect embeddings
        text_layer_weights = self.get_text_layer_weights()
        hidden_states = torch.stack([
            torch.cat([hs[layer_id+1][0] for hs in outputs.hidden_states[1:]], dim=0)
            for layer_id in range(num_layers)], dim=0)  # num_layers, seq_len, dim
        text_hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # seq_len, dim

        output_ids = outputs.sequences[0, :-1]   # discard the last one
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)

        return dict(output_ids=output_ids, output_text=output_text, text_hidden_states=text_hidden_states,
                    attention_maps=attention_maps, meta_data=meta_datas[0], clip_hidden_states=clip_hidden_states,
                    pixel_values= prepare_inputs['pixel_values'], hidden_states=hidden_states,)

    def ground(self, positive_ids,  attention_maps, meta_data,clip_hidden_states,pixel_values,text_hidden_states, **kwargs):
        mask_attentions = []
        llm_text_embeds = []

        for start_id, end_id in positive_ids:
            assert end_id > start_id
            mask_attentions.append(
                self.apply_merge(attention_maps[:, start_id:end_id], merge='mean', dim=1)
            )
            llm_text_embeds.append(self.text_proj(text_hidden_states[start_id:end_id]).mean(0))

        mask_attentions = torch.stack(mask_attentions).to(self.ldsi.dtype)
        llm_text_embeds = torch.stack(llm_text_embeds)

        ###LIFG
        clip_hidden_states = torch.stack(clip_hidden_states)  # (24, 576, 2048)
        selected_layers = [3, 6, 9, 12, 15, 18, 20, 23]
        clip_multilayer = torch.stack([clip_hidden_states[i] for i in selected_layers]).squeeze(1)
        dynamic_fused_image_embeds = self.lifg(clip_multilayer, llm_text_embeds)

        ###LDSI
        shallow_features = self.lcb(pixel_values[0])
        multi_features = self.ldsi(dynamic_fused_image_embeds, mask_attentions, shallow_features)

        ###Maskhead
        pred_masks = self.fuse_maskhead(multi_features)
        del dynamic_fused_image_embeds

        pred_masks = pred_masks[:, 0]
        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        before_height = int(meta_data['padding']['before_height'])
        before_width = int(meta_data['padding']['before_width'])
        mask_h = int(meta_data['image_shape']['height'] + 0.5)
        mask_w = int(meta_data['image_shape']['width'] + 0.5)

        pred_masks = F.interpolate(pred_masks[None].float(),
                                   size=(padded_h, padded_w), mode='bilinear')[0]
        pred_masks \
            = pred_masks[:, before_height:before_height + mask_h, before_width:before_width + mask_w].contiguous()

        return pred_masks,



if __name__ == '__main__':
    from PIL import Image
    from xtuner.model.utils import guess_load_checkpoint
    from mmengine.config import Config
    image = Image.open('images/dog_a.png')
    question = "<image_placeholder>What category does the dog belong to?"
    cfg = Config.fromfile('configs/deepseek_vl/frozen_deepseek_vl_1_3b_chat_unet_sam_l_refcoco_png.py')
    model = BUILDER.build(cfg.model)
    _ = model.load_state_dict(guess_load_checkpoint('checkpoints/frozen_deepseek_vl_1_3b_unet_sam_l_iter_95080.pth'),
                              strict=False)
    model._prepare_for_generation(image_processor=cfg.image_processor,
                                  prompt_template=cfg.prompt_template,
                                  max_thought_tokens=16,
                                  max_new_tokens=512)
    model = model.cuda().eval()
    output = model.visual_cot_v1(image, question)




