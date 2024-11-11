import json

split = 'train'
dataset = json.load(open(f"segsub_data/segsub_data_{split}_v4.json", "r"))
segsub_dir = "segsub_data/segsub_images"
coco_image_dir = f"/home/pcarragh/dev/webqa/images/images/{split}2014"
webqa_image_dir = "/home/pcarragh/dev/webqa/images/webqa"

def get_image_paths(sample):
    paths = []
    for image in sample['image']:
        if isinstance(image, int):
            assert(sample['dataset'] == 'webqa')
            paths.append(f"{webqa_image_dir}/{image}.jpeg")
        elif sample['type'] == 'original':
            assert(sample['dataset'] in ['vqa', 'okvqa'])
            paths.append(f"{coco_image_dir}/{image}")
        else:
            paths.append(f"{segsub_dir}/{image}")
    return paths

def convert_format(dataset):
    output = []
    for sample in dataset:   
        image_paths = get_image_paths(sample)   
        output.append({
            "query": sample['conversations'][0]['value'],
            "response": sample['conversations'][1]['value'],
            "images": image_paths,
        })
    return output


dataset = convert_format(dataset)
with open(f"/home/pcarragh/dev/webqa/ms-swift/data/{split}.jsonl", 'w') as f:
    for entry in dataset:
        json.dump(entry, f)
        
# dataset_val_qwen = json.load(open("/home/pcarragh/dev/webqa/lmms-finetune/webqa/data/webqa_val_gen_formatted_v2.json", "r"))
# dataset_val_qwen = convert_format(dataset_val_qwen)
# with open("/home/pcarragh/dev/webqa/ms-swift/data/qwen_val.jsonl", 'w') as f:
#     for entry in dataset_val_qwen:
#         json.dump(entry, f)