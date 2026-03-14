import json 

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

versions = [("single_class", "by_class_label_only"),
           ("single_instruction", "by_class_with_description"),
           ("multi_class", "by_image_label_only"),
           ("multi_instruction", "by_image_with_description")]

configs = {}
for dataset in datasets:
    for type, file in versions:
        configs[f"{dataset}_{type}"] = {
                "file_name": f"/data3/nperi/sharegpt4v_datasets/{dataset}/train/{file}.json",
                "formatting": "sharegpt",
                "columns": {
                "messages": "messages",
                "images": "images"
                },
                "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
                }
            }
        
json.dump(configs, open("config.json", "w"), indent=4)
