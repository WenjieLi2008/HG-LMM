from LISA.utils.dataset import HybridDataset, ValDataset, collate_fn
from LISA.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

import torch.nn.functional as F
import argparse
from hglmm.datasets.png import PNGDataset
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
from functools import partial
import tqdm
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import torch
from mmengine.config import Config
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint
from scripts.demo.utils import colors

import spacy
nlp = spacy.load("en_core_web_sm")
import random
random.shuffle(colors)


def process_noun_chunks(noun_chunks):
    new_noun_chunks = []
    for i in range(len(noun_chunks)):
        noun_chunk = noun_chunks[i]
        if 'image' in noun_chunk.lower():
            continue
        if noun_chunk.lower() in ['it', 'this', 'that', 'those', 'these', 'them',
                                  'he', 'she', 'you', 'i', 'they', 'me', 'her',
                                  'him', 'a', 'what', 'which', 'whose', 'who']:
            continue
        keep = True
        for j in range(len(noun_chunks)):  # de-duplicate
            if i != j and noun_chunk in noun_chunks[j]:
                if len(noun_chunk) < len(noun_chunks[j]) or i > j:
                    keep = False
                    break
        if keep:
            new_noun_chunks.append(noun_chunk)

    return new_noun_chunks
def extract_noun_phrases(output_text):
    doc = nlp(output_text)
    noun_chunks = list(set(chunk.text for chunk in doc.noun_chunks))
    if len(noun_chunks) == 0:
        noun_chunks = [output_text]
    last_end = 0
    noun_chunks = process_noun_chunks(noun_chunks)
    noun_chunks = sorted(noun_chunks, key=lambda x: output_text.find(x))

    # noun_chunks = [noun_chunk for noun_chunk in noun_chunks
    #                if int(input(f'Ground {noun_chunk}?')) == 1]

    positive_ids = []
    phrases = []
    for noun_chunk in noun_chunks:
        obj_start = output_text.find(noun_chunk)
        if obj_start < last_end:
            continue
        obj_end = obj_start + len(noun_chunk)
        last_end = obj_end
        positive_ids.append((obj_start, obj_end))
        phrases.append(noun_chunk)

    return positive_ids, phrases
def find_interval(intervals, idx):
    for interval_id, (start_id, end_id) in enumerate(intervals):
        if (idx >= start_id) and (idx < end_id):
            return interval_id
    return len(intervals)

def validate(val_loader, model, epoch, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            image = input_dict["original_images"][0]
            question = input_dict["questions_list"][0][0]+('?Based on the image, answer the name of the object that is most relevant to the answer to the question in just one or two words, and do not answer yes or no')
            output = model.answer(image, question)
            output_ids = output.pop('output_ids').cpu()
            output_text = output.pop('output_text')
            positive_ids = [(0,len(output_ids))]

            with torch.no_grad():
                pred_masks = model.ground(positive_ids=positive_ids, **output)

        pred_masks = F.interpolate(pred_masks[None].float().sigmoid(),
                                  size=(image.height, image.width), mode='bilinear')[0]
        masks_list = input_dict["masks_list"][0].int()
        output_list = pred_masks
        mismatch_num = 0
        print(output_text)
        assert len(pred_masks) == 1
        if output_list.shape != masks_list.shape:
            print("masks shape mismatch")
            print(output_list.shape, masks_list.shape)
            print(mismatch_num)
            continue
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])


    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]


    return giou, ciou



def main():
    # Create model
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="/Storage/data", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    args = parser.parse_args()

    log_base_dir = os.path.join('', 'logs')
    exp_name = os.path.basename(args.config).split('.')[0]
    log_dir = os.path.join(log_base_dir, exp_name)

    # loading model
    accelerator = Accelerator()
    # each GPU creates a string
    message = [f"Hello this is GPU {accelerator.process_index}"]
    # collect the messages from all GPUs
    messages = gather_object(message)
    # output the messages only on the main process with accelerator.print()
    accelerator.print(messages)

    cfg = Config.fromfile(args.config)
    prompt_template = cfg.prompt_template
    image_processor = cfg.image_processor

    print(f'Device: {accelerator.device}', flush=True)
    model = BUILDER.build(cfg.model)
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Unexpected parameters: {unexpected}")

    model._prepare_for_generation(image_processor=image_processor,
                                  prompt_template=prompt_template,
                                  max_thought_tokens=16,
                                  max_new_tokens=1024,
                                  lmm_name=cfg.lmm_name,
                                  additional_prompt='')
    model = model.to(device=accelerator.device)
    # model.eval()
    tokenizer = model.tokenizer
    val_dataset = ValDataset(
                args.dataset_dir,
                tokenizer,
                args.vision_tower,
                args.val_dataset,
                args.image_size,
            )

     # validation dataset
    if val_dataset is not None:
        val_batch_size = 1
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            pin_memory=False,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                use_mm_start_end=args.use_mm_start_end,
            ),
        )

    giou, ciou = validate(val_loader, model, 0, args)
    print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))


if __name__ == "__main__":
    main()