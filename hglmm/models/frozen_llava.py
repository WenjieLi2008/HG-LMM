import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.registry import BUILDER
from mmengine.model import BaseModel
from xtuner.model.utils import guess_load_checkpoint
from hglmm.utils import compute_mask_IoU




class FrozenLlava(BaseModel):

    def __init__(self,
                 model,
                 tokenizer,
                 merge='mean',
                 loss_mask=None,
                 loss_dice=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()
        self.llava = BUILDER.build(model)
        self.llava.requires_grad_(False)
        self.patch_size = self.llava.config.vision_config.patch_size
        self.merge = merge
        assert merge in ['mean', 'max']

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)
        self.tokenizer = BUILDER.build(tokenizer)

        self.lifg = LIFG()
        self.ldsi = LDSI()
        self.lcb = LCB()
        self.fuse_maskhead = Fuse_Maskhead()

        self.text_layer_weights = nn.Parameter(
            torch.ones(self.llava.config.text_config.num_hidden_layers))

        if pretrained is not None:
            _ = self.load_state_dict(guess_load_checkpoint(pretrained), strict=False)
        
    def get_text_layer_weights(self):
        return torch.softmax(self.text_layer_weights, dim=0)

    def apply_merge(self, x, dim=1):
        if self.merge == 'mean':
            return x.mean(dim=dim)
        elif self.merge == 'max':
            return x.max(dim=dim).values
        else:
            raise NotImplementedError

    def init_weights(self):
        pass

    def train(self, mode=True):
        super().train(mode=mode)
        self.llava.train(mode=False)
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
        self.conv1 = nn.Conv2d(4096, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(192, 256, kernel_size=3, padding=1)
        self.ln4096 = nn.LayerNorm(normalized_shape=4096, eps=1e-5, elementwise_affine=True)
    def process_clip_features(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x

    def forward(self, clip_multilayer, llm_text_embeds):
        clip_multilayer = self.ln4096(clip_multilayer)
        clip_multilayer = clip_multilayer.permute(0, 2, 1).reshape(8, 4096, 24, 24)
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
        self.conv1024to384 = nn.Conv2d(1024, 384, kernel_size=3, padding=1)
        self.conv384to96 = nn.Conv2d(384, 96, kernel_size=3, padding=1)
        self.conv96to128 = nn.Conv2d(96, 128, kernel_size=3, padding=1)
        self.conv192to128 = nn.Conv2d(192, 128, kernel_size=3, padding=1)

        self.target_sizes = {
            "stage1": 84,
            "stage2": 42,
            "stage3": 21,
            "stage4": 10
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
        mask_attentions = self.conv1024to384(mask_attentions)
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

class HGFrozenLlava(FrozenLlava):
    def __init__(self, *args, **kwargs):
        pretrained = kwargs.pop('pretrained', None)
        super().__init__(*args, **kwargs)
        self.text_proj = nn.Linear(4096,256)
        if pretrained is not None:
            _ = self.load_state_dict(guess_load_checkpoint(pretrained), strict=False)






    def _forward(self, data_sample):
        text_layer_weights = self.get_text_layer_weights()
        inputs = dict(input_ids=data_sample['input_ids'][None].to(self.llava.device),
                      mask_ids=data_sample['mask_ids'][None].to(self.llava.device),
                      pixel_values=data_sample['pixel_values'][None].to(device=self.llava.device,
                                                                        dtype=self.llava.dtype),
                      labels=data_sample['labels'][None].to(self.llava.device)
                      )
        attention_mask = torch.ones(inputs['input_ids'].shape, device=self.llava.device,
                                    dtype=torch.bool)
        meta_data = data_sample['meta_data']
        pixel_values = data_sample['pixel_values'][None].to(device=self.llava.device,
                                                            dtype=self.llava.dtype),
        with torch.no_grad():
            outputs,clip_hidden_states,inputs_embeds = self.llava(**inputs,
                                                     attention_mask=attention_mask,
                                                     output_hidden_states=True,
                                                     output_attentions=True)

        mask_ids = outputs['mask_ids'][0]
        attentions = [attn[0, ..., outputs['image_to_overwrite'][0]]
                      for attn in outputs.attentions]
        hidden_states = outputs.hidden_states[-self.llava.config.text_config.num_hidden_layers:]
        clip_hidden_states =clip_hidden_states[-24:]
        labels = outputs.labels[0]

        hidden_states = torch.stack([hs[0] for hs in hidden_states])  # num_layers, seq_len, dim
        hidden_states = (hidden_states * text_layer_weights.view(-1, 1, 1)).sum(0)  # seq_len, dim

        del outputs

        padded_h, padded_w = meta_data['padded_shape']['height'], meta_data['padded_shape']['width']
        llava_h, llava_w = padded_h // self.patch_size, padded_w // self.patch_size

        attentions = [attn.view(*attn.shape[:-1], llava_h, llava_w) for attn in attentions]
        masks = data_sample['masks']
        mask_attentions = []
        llm_text_embeds = []
        for mask_id in range(len(masks)):
            matched = mask_ids == mask_id
            assert matched.sum() > 0
            mask_attentions.append(torch.cat(
                [self.apply_merge(attn[:, matched], dim=1) for attn in attentions]))
            llm_text_embeds.append(self.text_proj(hidden_states[matched]).mean(0))

        mask_attentions = torch.stack(mask_attentions)
        llm_text_embeds = torch.stack(llm_text_embeds)
        del attentions
        if self.training:
            mask_attentions.requires_grad = True



        ###LIFG
        clip_hidden_states = torch.stack(clip_hidden_states)  # (24, 576, 2048)
        selected_layers = [3,6,9,12,15,18,20,23]
        clip_multilayer = torch.stack([clip_hidden_states[i] for i in selected_layers]).squeeze(1)
        dynamic_fused_image_embeds = self.lifg(clip_multilayer,llm_text_embeds)

        ###LDSI
        shallow_features = self.lcb(pixel_values[0])
        multi_features = self.ldsi(dynamic_fused_image_embeds, mask_attentions, shallow_features)

        ###Fuse_Maskhead
        pred_masks = self.fuse_maskhead(multi_features)
        del dynamic_fused_image_embeds

        # todo: unpad pred_masks
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


        output = dict(pred_masks=pred_masks,labels=labels, mask_ids=mask_ids, hidden_states=llm_text_embeds)

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

        sam_loss_dice = 0
        sam_loss_mask = 0
        sam_accuracy = 0
        sam_aiou = 0

        for data_sample in data:
            forward_output = self._forward(data_sample)
            pred_masks, sam_pred_masks = forward_output['pred_masks'], forward_output['sam_pred_masks']
            masks = data_sample['masks'].to(self.llava.device)
            gt_masks = F.interpolate(masks[None].float(),
                                     size=pred_masks.shape[-2:])[0].to(pred_masks)
            sam_gt_masks = F.interpolate(masks[None].float(),
                                         size=sam_pred_masks.shape[-2:])[0].to(sam_pred_masks)

            mask_cnt = pred_masks.shape[0]
            assert pred_masks.shape == gt_masks.shape
            mask_cnts += mask_cnt

            loss_dice_, loss_mask_, accuracy_, aiou_ = self._compute(pred_masks, gt_masks)
            loss_dice += loss_dice_ * mask_cnt
            loss_mask += loss_mask_ * mask_cnt
            accuracy += accuracy_ * mask_cnt
            aiou += aiou_ * mask_cnt

            sam_loss_dice_, sam_loss_mask_, sam_accuracy_, sam_aiou_ = self._compute(sam_pred_masks, sam_gt_masks)
            sam_loss_dice += sam_loss_dice_ * mask_cnt
            sam_loss_mask += sam_loss_mask_ * mask_cnt
            sam_accuracy += sam_accuracy_ * mask_cnt
            sam_aiou += sam_aiou_ * mask_cnt

        assert mask_cnts > 0

        loss_dict = {'loss_mask': loss_mask / mask_cnts,
                     'loss_dice': loss_dice / mask_cnts,
                     'accuracy': accuracy / mask_cnts,
                     'aiou': aiou / mask_cnts,
                     'sam_loss_mask': sam_loss_mask / mask_cnts,
                     'sam_loss_dice': sam_loss_dice / mask_cnts,
                     'sam_accuracy': sam_accuracy / mask_cnts,
                     'sam_aiou': sam_aiou / mask_cnts,
                     }

        return loss_dict





