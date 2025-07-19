from typing import List, Optional, Union

import torch
import torchvision

from PIL import Image
import numpy as np

import torchvision.transforms.functional
from transformers import DetrFeatureExtractor
from transformers.image_transforms import center_to_corners_format
from transformers.utils import TensorType
from transformers.feature_extraction_utils import BatchFeature

import model.transform as T


class RtDetrFeatureExtractor(DetrFeatureExtractor):
    # only different is the keep part
    def prepare_coco_detection(self, image, target, return_segmentation_masks=False):
        """
        Convert the target in COCO format into the format expected by DETR.
        """
        w, h = image.size

        image_id = target["image_id"]
        image_id = np.asarray([image_id], dtype=np.int64)

        # get all COCO annotations for the given image
        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = np.asarray(classes, dtype=np.int64)

        if return_segmentation_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = self.convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = np.asarray(keypoints, dtype=np.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.reshape((-1, 3))

        keep = (boxes[:, 3] >= boxes[:, 1]) & (boxes[:, 2] >= boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if return_segmentation_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["class_labels"] = classes
        if return_segmentation_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = np.asarray([obj["area"] for obj in anno], dtype=np.float32)
        iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno], dtype=np.int64)
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = np.asarray([int(h), int(w)], dtype=np.int64)
        target["size"] = np.asarray([int(h), int(w)], dtype=np.int64)

        return image, target
    
    # only different post_process and pad_and_create_pixel_mask
    def post_process(self, outputs, target_sizes):
        """
        Converts the output of [`DeformableDetrForObjectDetection`] into the format expected by the COCO api. Only
        supports PyTorch.
        Args:
            outputs ([`DeformableDetrObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (height, width) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation). For visualization, this should be the image size
                after data augment, but before padding.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if len(out_logits) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )
        if target_sizes.shape[1] != 2:
            raise ValueError(
                "Each element of target_sizes must contain the size (h, w) of each image of the batch"
            )

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), 100, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="trunc")
        labels = topk_indexes % out_logits.shape[2]
        boxes = center_to_corners_format(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]

        return results

    def pad_and_create_pixel_mask(
        self, pixel_values_list: List["torch.Tensor"], return_tensors: Optional[Union[str, TensorType]] = None
    ):
        """
        Dummy function, resize all images to 640x640.

        Args:
            pixel_values_list (`List[torch.Tensor]`):
                List of images (pixel values) to be padded. Each image should be a tensor of shape (C, H, W).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when `pad_and_return_pixel_mask=True` or if
              *"pixel_mask"* is in `self.model_input_names`).

        """

        h, w = 640, 640
        resized_images = [torchvision.transforms.functional.resize(image, (h, w)) for image in pixel_values_list]
        pixel_mask = [torch.ones((h, w), dtype=torch.int64) for _ in pixel_values_list]

        resized_images = torch.stack(resized_images)
        pixel_mask = torch.stack(pixel_mask)

        # return as BatchFeature
        data = {"pixel_values": resized_images, "pixel_mask": pixel_mask}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
    


class RtDetrFeatureExtractorWithAugmentorNoCrop(RtDetrFeatureExtractor):
    def _resize(self, image, size, target=None, max_size=None):
        """
        Resize the image to the given size. Size can be min_size (scalar) or (w, h) tuple. If size is an int, smaller
        edge of the image will be matched to this number.
        If given, also resize the target accordingly.
        """
        if not isinstance(image, Image.Image):
            image = self.to_pil_image(image)
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            # T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
            ]
        )
        rescaled_image, target = transform(image, target)
        return rescaled_image, target