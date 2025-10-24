# Mask API - Batched Operations

This directory contains the internal mask API with optimized batched versions for performance.

## Overview

The mask API provides methods for working with Run-Length Encoded (RLE) masks and bounding boxes. Each method now has a corresponding batched version (with `Batch` suffix) that can process multiple items more efficiently.

## Batched Functions

### RLE Operations

- **rleAreaBatch**: Compute area of encoded masks in a batched manner
  - Returns a tensor of areas instead of a list
  - More efficient for downstream tensor operations

- **rleEncodeBatch**: Encode binary masks using RLE (already batched in original)
  - Alias to `rleEncode` which already processes batches efficiently

- **rleDecodeBatch**: Decode binary masks encoded via RLE (already batched in original)
  - Alias to `rleDecode` which already processes batches efficiently

- **rleToBboxBatch**: Get bounding boxes surrounding encoded masks (already batched in original)
  - Alias to `rleToBbox` which already handles variable-length RLEs

- **rleIouBatch**: Compute intersection over union between masks (already batched in original)
  - Alias to `rleIou` which already processes pairs efficiently

- **rleMergeBatch**: Compute union or intersection of encoded masks for multiple sets
  - Processes multiple RLE sets in a batch

- **rleFrBboxBatch**: Convert bounding boxes to encoded masks (already batched in original)
  - Alias to `rleFrBbox` which already processes batches

- **rleFrPolyBatch**: Convert polygon to encoded masks (single polygon operation)
  - Alias to `rleFrPoly`

- **rleFrStringBatch**: Convert from compressed string representation (batch version)
  - Processes multiple strings at once

- **rleToStringBatch**: Convert to compressed string representation (already exists)
  - Already implemented in the codebase

- **rleNmsBatch**: Non-maximum suppression for masks (optimized batch)
  - Returns a boolean tensor instead of a list

### Bounding Box Operations

- **bbIouBatch**: Compute IoU between bounding boxes (already batched in original)
  - Alias to `bbIou` which already processes matrices efficiently

- **bbNmsBatch**: Non-maximum suppression for bounding boxes (optimized batch)
  - Uses vectorized IoU computation
  - Returns a boolean tensor for filtering

## Usage

```python
from pytorchcocotools.internal.mask_api import (
    rleAreaBatch,
    bbIouBatch,
    bbNmsBatch,
)

# Compute areas for multiple RLE masks
areas = rleAreaBatch(rles)  # Returns tensor instead of list

# Compute IoU between bounding boxes
iou_matrix = bbIouBatch(detections, ground_truth, iscrowd)

# Apply NMS to bounding boxes
keep_mask = bbNmsBatch(bounding_boxes, threshold=0.5)
filtered_boxes = bounding_boxes[keep_mask]
```

## Performance Considerations

Many of the original implementations already process batches efficiently:
- `rleEncode` and `rleDecode` use vectorized operations
- `rleToBbox` handles variable-length RLEs
- `bbIou` computes full IoU matrices

The batched versions either:
1. Alias to the original (when already optimal)
2. Add tensor output instead of lists for better integration
3. Provide truly batched operations where beneficial

## Testing

Tests for batched versions are in `tests/mask/test_batch_versions.py`.

Run tests with:
```bash
pytest tests/mask/test_batch_versions.py -v
```
