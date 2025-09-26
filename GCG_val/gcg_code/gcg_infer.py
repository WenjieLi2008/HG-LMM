
import json

from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, CLIPImageProcessor
from accelerate import Accelerator
from accelerate.utils import gather_object
from GCG_val.utils import *
from GCG_val.ddp import *

import argparse
import torch
import numpy as np
from mmengine.config import Config
from xtuner.registry import BUILDER
from PIL import Image
from xtuner.model.utils import guess_load_checkpoint
from scripts.demo.utils import colors

import spacy

nlp = spacy.load("en_core_web_sm")
import random

random.shuffle(colors)


def custom_collate_fn(batch):
    image_id = [item[0] for item in batch]
    image_path = [item[1] for item in batch]

    return image_id, image_path

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


def inference(model, instruction, image_path):
    image = Image.open(image_path)
    output = model.answer(image, instruction)
    output_ids = output.pop('output_ids').cpu()
    output_text = output.pop('output_text')

    encoded = model.tokenizer(output_text, add_special_tokens=False, return_tensors='pt')
    offsets = encoded.encodings[0].offsets
    str_places, phrases = extract_noun_phrases(output_text)
    positive_ids = []
    for start_id, end_id in str_places:
        start_token_place = find_interval(offsets, start_id)
        end_token_place = max(start_token_place + 1, find_interval(offsets, end_id))
        positive_ids.append((start_token_place, end_token_place))
    with torch.no_grad():
        pred_masks = model.ground(positive_ids=positive_ids, **output)

    masks = F.interpolate(pred_masks[None].float().sigmoid(),
                          size=(image.height, image.width), mode='bilinear')[0].cpu()
    masks = (masks > 0.5).int()
    return output_text, masks, phrases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument("--img_dir", default="data/GCG/GranDf_HA_images/val_test", type=str)
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--output_dir", default="CGC_val/gcg_result/test", type=str)
    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    init_distributed_mode(args)

    instruction = "Please give me a detailed description of the image."
    gt_cap_path = f"data/GCG/GranDf/annotations/val_test/{args.split}_gcg_coco_caption_gt.json"
    PROGRESS_FILE = os.path.join(args.output_dir, "processed_image_ids.json")

    # loading model
    accelerator = Accelerator()
    # each GPU creates a string
    message = [f"Hello this is GPU {accelerator.process_index}"]
    # collect the messages from all GPUs
    messages = gather_object(message)
    # output the messages only on the main process with accelerator.print()
    accelerator.print(messages)

    # Load the processed image_id
    processed_ids = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            processed_ids = set(json.load(f))
        accelerator.print(f"Progress file loaded, {len(processed_ids)} images processed.")
    else:
        accelerator.print("Progress file not found, processing will start from the beginning.")

    # Get the image names of the split
    all_images_ids = []
    with open(gt_cap_path, 'r') as f:
        contents = json.load(f)
        for image in contents['images']:
            all_images_ids.append(image['id'])

    cfg = Config.fromfile(args.config)
    prompt_template = cfg.prompt_template
    tokenizer = cfg.tokenizer
    image_processor = cfg.image_processor
    prompt = cfg.get('prompt', None)

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
    model.eval()

    dataset = GCGEvalDDP(args.img_dir)
    # distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2,
                            collate_fn=custom_collate_fn)

    # Iterate over all the images, run inference and save results
    for (image_id, image_path) in tqdm(dataloader):
        image_id, image_path = image_id[0], image_path[0]
        if image_id[:-4] not in all_images_ids:
            # accelerator.print(f"not val data")
            continue
        # If the image has already been processed, skip
        if image_id[:-4] in processed_ids:
            accelerator.print(f"Processed images {image_id}, skip.")
            continue

        output_path = f"{args.output_dir}/{image_id[:-4]}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_caption, pred_masks, phrases = inference(model, instruction, image_path)  # GLaMM Inference

        # Convert the predicted masks into RLE format
        pred_masks_tensor = pred_masks.cpu()
        binary_pred_masks = pred_masks_tensor > 0
        uncompressed_mask_rles = mask_to_rle_pytorch(binary_pred_masks)
        rle_masks = []
        for m in uncompressed_mask_rles:
            rle_masks.append(coco_encode_rle(m))

        # Create results dictionary
        result_dict = {
            "image_id": image_id[:-4],
            "caption": result_caption,
            "phrases": phrases,
            "pred_masks": rle_masks
        }

        # Save the inference results
        with open(output_path, 'w') as f:
            json.dump(result_dict, f)

        # Save progress: record the current image_id processed
        processed_ids.add(image_id[:-4])
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(list(processed_ids), f)

        accelerator.print(f"processed {image_id}，Current total processed：{len(processed_ids)}")