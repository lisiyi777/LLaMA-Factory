import os 

datasets = [
    "actions",
    "aerial-airport",
    "all-elements",
    "aquarium-combined",
    "defect-detection",
    "dentalai",
    "flir-camera-objects",
    "gwhd2021",
    "lacrosse-object-detection",
    "new-defects-in-wood",
    "orionproducts",
    "paper-parts",
    "recode-waste",
    "soda-bottles",
    "the-dreidel-project",
    "trail-camera",
    "water-meter",
    "wb-prova",
    "wildfire-smoke",
    "x-ray-id"
]

versions = [
    # "single_class", 
    "single_instruction", 
    # "multi_class", 
    # "multi_instruction"
    ]

for dataset in datasets:
    for version in versions:
        os.system(f"llamafactory-cli train configs/qwen3vl_lora_sft.yaml dataset={dataset}_{version} output_dir=saves/qwen3vl-8b/{dataset}/{version}")
        
