""" Modified version of https://github.com/chibohe/CdistNet-pytorch/blob/main/tool/create_lmdb_dataset.py"""

import fire
import os
import lmdb
import cv2
import numpy as np
import io

from PIL import Image


def checkImageIsValid(imageBin):
    """Check if an image is valid by decoding it with OpenCV."""
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH == 0 or imgW == 0:
        return False
    return True


def writeCache(env, cache):
    """Write a cache dictionary to LMDB."""
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def load_clean_image(path):
    """
    Open image using Pillow and remove ICC profile to avoid libpng warnings.
    Return binary image data.
    """
    img = Image.open(path)
    buf = io.BytesIO()
    img.save(buf, format="PNG", icc_profile=None)
    return buf.getvalue()


def estimate_map_size(inputPath, gtFile):
    """Estimate the required map_size for LMDB based on image sizes."""
    total_size = 0
    with open(gtFile, "r", encoding="utf-8") as f:
        datalist = f.readlines()

    for line in datalist:
        splitted = line.strip("\n").split(" ")
        imagePath = os.path.join(inputPath, splitted[0])
        if os.path.exists(imagePath):
            total_size += os.path.getsize(imagePath)
        # Rough estimate of label/key sizes
        total_size += 100

    # Add 20% buffer
    return int(total_size * 1.4)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.

    Args:
        inputPath (str): Folder path where image files are located.
        gtFile (str): Text file with image paths and labels.
        outputPath (str): Output folder for LMDB.
        checkValid (bool): Whether to check image validity.
    """
    os.makedirs(outputPath, exist_ok=True)

    # --- Estimate required map_size ---
    map_size = estimate_map_size(inputPath, gtFile)
    env = lmdb.open(outputPath, map_size=map_size)

    cache = {}
    cnt = 1

    with open(gtFile, "r", encoding="utf-8") as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        splitted = datalist[i].strip("\n").split(" ")
        imagePath, label = splitted[0], " ".join(splitted[1:])
        fullImagePath = os.path.join(inputPath, imagePath)

        if not os.path.exists(fullImagePath):
            print(f"{fullImagePath} does not exist")
            continue

        try:
            imageBin = load_clean_image(fullImagePath)
        except Exception as e:
            print(f"Error loading image {fullImagePath}: {e}")
            continue

        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print(f"{fullImagePath} is not a valid image")
                    continue
            except Exception as e:
                print(f"Error validating image {fullImagePath}: {e}")
                with open(os.path.join(outputPath, "error_image_log.txt"), "a") as log:
                    log.write(f"{i}-th image data caused error: {fullImagePath}\n")
                continue

        imageKey = f"image-{cnt:09d}".encode()
        labelKey = f"label-{cnt:09d}".encode()
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print(f"Written {cnt} / {nSamples}")

        cnt += 1

    # Write remaining cache and sample count
    nSamples = cnt - 1
    cache["num-samples".encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print(f"Created dataset with {nSamples} samples")


if __name__ == "__main__":
    fire.Fire(createDataset)
