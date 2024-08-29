# COCO

Interface for accessing the Microsoft COCO dataset.

Microsoft COCO is a large image dataset designed for object detection,
segmentation, and caption generation. pycocotools is a Python API that
assists in loading, parsing and visualizing the annotations in COCO.
Please visit https://cocodataset.org/ for more information on COCO, including
for the data, paper, and tutorials. The exact format of the annotations
is also described on the COCO website. For example usage of the pycocotools
please see pycocotools_demo.ipynb. In addition to this API, please download both
the COCO images and annotations in order to run the demo.

An alternative to using the API is to load the annotations directly
into Python dictionary
Using the API provides additional utility functions. Note that this API
supports both *instance* and *caption* annotations. In the case of
captions not all functions are defined (e.g. categories are undefined).

The following API functions are defined:
- COCO       - COCO api class that loads COCO annotation file and prepare data structures.
- decodeMask - Decode binary mask M encoded via run-length encoding.
- encodeMask - Encode binary mask M using run-length encoding.
- getAnnIds  - Get ann ids that satisfy given filter conditions.
- getCatIds  - Get cat ids that satisfy given filter conditions.
- getImgIds  - Get img ids that satisfy given filter conditions.
- loadAnns   - Load anns with the specified ids.
- loadCats   - Load cats with the specified ids.
- loadImgs   - Load imgs with the specified ids.
- annToMask  - Convert segmentation in an annotation to binary mask.
- showAnns   - Display the specified annotations.
- loadRes    - Load algorithm results and create API for accessing them.
- download   - Download COCO images from mscoco.org server.

Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
Help on each functions can be accessed by: "help COCO>function".
