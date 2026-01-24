# WordArt-V1.5 Dataset

**ICDAR 2024 Competition on Artistic Text Recognition**

This repository contains the **WordArt-V1.5** dataset, created for the **ICDAR 2024 Competition on Artistic Text Recognition**.


## Overview

<div align="left">
    <picture>
        <kbd>
            <img
                src="https://github.com/lucas-azdias/Artistic-Text-Recognition/raw/main/thumbnails/dataset-examples.png"
                height="300px"
            />
        </kbd>
    </picture>
</div>

<br/>

The competition focused on the **recognition of text in artistic and stylized images** — a challenging problem that combines aspects of OCR, computer vision, and typography understanding.

**WordArt-V1.5** includes a wide variety of:
- Artistic and decorative text styles,
- Diverse fonts and visual distortions,
- Real-world and synthetic samples from creative media such as posters, advertisements, and digital artwork.


## Related Research

This dataset was used in the paper:  
> **Evaluating State-of-the-Art Approaches to Artistic Text Recognition**  
> Available [here](https://github.com/lucas-azdias/Artistic-Text-Recognition)


## Purpose

The dataset was originally released as part of the **official ICDAR 2024 competition** and is provided here **for research and reproducibility purposes**.


## Dataset Composition

| Split   | Number of Images |
|:--|:--|
| **Train** | 6,000 |
| **Test A** | 3,000 |
| **Test B** | 3,000 |

Total: 12,000 images.


## Annotation Format
Each split of the dataset includes a `labels.txt` file containing image paths and corresponding text labels in each line.
```
images/320.png Ford
images/355.png PARA
images/356.png JUNIPER
...
```

There is also a structured MMOCR JSON-based annotation format (`.json` / `.jsonl`) used by MMOCR for text recognition datasets.
```
// JSON
{
    "metainfo": {
        "dataset_type": "TextRecogDataset",
        "task_name": "textrecog"
    },
    "data_list": [
        {
            "instances": [
                {
                    "text": "Ford"
                }
            ],
            "img_path": "full_images/train/images/320.png"
        },
        {
            "instances": [
                {
                    "text": "PARA"
                }
            ],
            "img_path": "full_images/train/images/355.png"
        },
        ...
    ]
}
```
```
// JSONL
{"filename": "full_images/train/images/320.png", "text": "Ford"}
{"filename": "full_images/train/images/355.png", "text": "PARA"}
{"filename": "full_images/train/images/356.png", "text": "JUNIPER"}
...
```


## Competition links
| Resource | Link |
|:--|:--|
| **Competition Page** | [View here](https://sites.google.com/view/icdar-2024-competition-wordart/) |
| **Competition Results** | [View on CodaLab](https://codalab.lisn.upsaclay.fr/competitions/17182#participate) |
| **Train Dataset** | [Download](https://drive.google.com/file/d/1Lq6xKNbD7Kvs-i1myJPmwBLLG5YnoM9N/view) |
| **Test A Dataset** | [Download](https://drive.google.com/file/d/15tkLbdXYzIILVWIg4kqjPMJ51p-vD2Ej/view) |
| **Test B Dataset** | [Download](https://drive.google.com/file/d/1Q7kAqFITGntZAn-HuCh8vQpHTpDkPSAH/view) |
