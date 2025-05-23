# Mask
Interface for manipulating masks stored in RLE format.

RLE is a simple yet efficient format for storing binary masks. RLE
first divides a vector (or vectorized image) into a series of piecewise
constant regions and then for each piece simply stores the length of
that piece. For example, given M=[0 0 1 1 1 0 1] the RLE counts would
be [2 3 1 1], or for M=[1 1 1 1 1 1 0] the counts would be [0 6 1]
(note that the odd counts are always the numbers of zeros). Instead of
storing the counts directly, additional compression is achieved with a
variable bitrate representation based on a common scheme called LEB128.

Compression is greatest given large piecewise constant regions.
Specifically, the size of the RLE is proportional to the number of
*boundaries* in M (or for an image the number of boundaries in the y
direction). Assuming fairly simple shapes, the RLE representation is
O(sqrt(n)) where n is number of pixels in the object. Hence space usage
is substantially lower, especially for large simple objects (large n).

Many common operations on masks can be computed directly using the RLE
(without need for decoding). This includes computations such as area,
union, intersection, etc. All of these operations are linear in the
size of the RLE, in other words they are O(sqrt(n)) where n is the area
of the object. Computing these operations on the original mask is O(n).
Thus, using the RLE can result in substantial computational savings.

Usage:
 Rs     = encode( masks )
 masks  = decode( Rs )
 R      = merge( Rs, intersect=false )
 o      = iou( dt, gt, iscrowd )
 a      = area( Rs )
 bbs    = toBbox( Rs )
 Rs     = frPyObjects( [pyObjects], h, w )

In the API the following formats are used:
 Rs      - [dict] Run-length encoding of binary masks
 R       - dict Run-length encoding of binary mask
 masks   - [hxwxn] Binary mask(s) (must have type torch.ndarray(dtype=uint8) in column-major order)
 iscrowd - [nx1] list of torch.ndarray. 1 indicates corresponding gt image has crowd region to ignore
 bbs     - [nx4] Bounding box(es) stored as [x y w h]
 poly    - Polygon stored as [[x1 y1 x2 y2...],[x1 y1 ...],...] (2D list)
 dt,gt   - May be either bounding boxes or encoded masks
Both poly and bbs are 0-indexed (bbox=[0 0 1 1] encloses first pixel).
