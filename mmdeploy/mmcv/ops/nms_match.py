# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
from torch.onnx import symbolic_helper as sym_help

import mmdeploy
from mmdeploy.core import mark


class ONNXNMSMatchOp(torch.autograd.Function):
    """Create onnx::NonMaxSuppressionMatch op.

    NMS_Match in mmcv only supports one class with no batch info. This class
    assists in exporting NMS_Match of ONNX's definition.
    """

    @staticmethod
    def forward(ctx, boxes: Tensor, scores: Tensor, iou_threshold: float,
                score_threshold: float) -> Tensor:
        """Get NMS_Match_Fake output indices.

        Args:
            ctx (Context): The context with meta information.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.

        Returns:
            Tensor: Selected indices of boxes. 2-D tensor of shape
            (num_selected_indices, 4) with each row of
            [batch_index, class_index, box_index, suppresion_index].
        """
        from mmcv.ops import nms_match
        batch_size, num_class, _ = scores.shape

        indices = []
        score_threshold = float(score_threshold)
        iou_threshold = float(iou_threshold)
        for batch_id in range(batch_size):
            for cls_id in range(num_class):
                _boxes = boxes[batch_id, ...]
                _scores = scores[batch_id, cls_id, ...].contiguous()
                _dets = torch.cat((_boxes, _scores.unsqueeze(1)), dim=1)
                box_inds = nms_match(_dets, iou_threshold)
                batch_inds = torch.zeros(1) + batch_id
                cls_inds = torch.zeros(1) + cls_id
                both_inds = torch.cat([batch_inds, cls_inds])
                for box in box_inds:
                    if box.size() == 1:
                        continue
                    keep = box[0]
                    box = box[1:]
                    if _dets[keep][-1] < score_threshold:
                        continue
                    for supp in box:
                        indices.append(
                            torch.cat((both_inds, keep.unsqueeze(0),
                                       supp.unsqueeze(0))))
        return torch.stack(indices).to(torch.int64)

    @staticmethod
    def symbolic(g, boxes: Tensor, scores: Tensor, iou_threshold: float,
                 score_threshold: float):
        """Symbolic function for mmdeploy::NMSMatch.

        Args:
            g (Graph): The traced onnx graph.
            boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
            scores (Tensor): The detection scores of shape
                [N, num_boxes, num_classes].
            iou_threshold (float): IOU threshold of nms.
            score_threshold (float): score threshold of nms.

        Returns:
            NonMaxSuppressionMatch op for onnx.
        """
        if not sym_help._is_value(iou_threshold):
            iou_threshold = g.op(
                'Constant',
                value_t=torch.tensor([iou_threshold], dtype=torch.float))

        if not sym_help._is_value(score_threshold):
            score_threshold = g.op(
                'Constant',
                value_t=torch.tensor([score_threshold], dtype=torch.float))
        return g.op('mmdeploy::NMSMatch', boxes, scores, iou_threshold,
                    score_threshold)


def _select_nms_index(scores: torch.Tensor,
                      boxes: torch.Tensor,
                      nms_index,
                      batch_size: int,
                      keep_top_k: int = -1):
    """Transform NMS_Match output."""
    batch_inds, cls_inds = nms_index[:, 0], nms_index[:, 1]
    box_inds = nms_index[:, 2]
    
    # Index by nms output
    scores = scores[batch_inds, cls_inds, box_inds].unsqueeze(1)
    boxes = boxes[batch_inds, box_inds, ...]
    dets = torch.cat([boxes, scores], dim=1)

    # 1. Fix Dets Masking (Ensure float32 matches float32)
    batched_dets = dets.unsqueeze(0).repeat(batch_size, 1, 1)
    batch_template = torch.arange(
        0, batch_size, dtype=batch_inds.dtype, device=batch_inds.device)
    
    det_cond = (batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1)
    # Use new_zeros(1) with explicit dtype to be safe
    batched_dets = batched_dets.where(det_cond, batched_dets.new_zeros(1, dtype=torch.float32))

    # 2. Fix Labels Masking (This is the /Where_2 node causing the error)
    # First, create the batched labels
    batched_labels = cls_inds.unsqueeze(0).repeat(batch_size, 1)
    
    # Now apply the Type-Lock for Int64
    label_cond = (batch_inds == batch_template.unsqueeze(1))
    # We create an explicit Int64 tensor for the -1 value
    # This stops the ONNX tracer from promoting it to float
    else_val = torch.full_like(batched_labels, -1, dtype=torch.int64)
    
    # Use functional torch.where to be extra strict during export
    batched_labels = torch.where(label_cond, batched_labels.to(torch.int64), else_val)

    N = batched_dets.shape[0]

    # expand tensor to eliminate [0, ...] tensor
    batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 5))), 1)
    
    # Ensure the padding here is also int64
    label_pad = torch.zeros((N, 1), dtype=torch.int64, device=batched_labels.device)
    batched_labels = torch.cat((batched_labels, label_pad), 1)
    
    # sort
    is_use_topk = keep_top_k > 0 and \
        (torch.onnx.is_in_onnx_export() or keep_top_k < batched_dets.shape[1])
    if is_use_topk:
        _, topk_inds = batched_dets[:, :, -1].topk(keep_top_k, dim=1)
    else:
        _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)
        
    topk_batch_inds = torch.arange(
        batch_size, dtype=topk_inds.dtype,
        device=topk_inds.device).view(-1, 1)
        
    batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
    batched_labels = batched_labels[topk_batch_inds, topk_inds, ...]
    
    return batched_dets, batched_labels


def _multiclass_nms_match(boxes: Tensor,
                          scores: Tensor,
                          max_output_boxes_per_class: int = 1000,
                          iou_threshold: float = 0.5,
                          score_threshold: float = 0.05,
                          pre_top_k: int = -1,
                          keep_top_k: int = -1,
                          output_index: bool = False):
    """Create a dummy onnx::NonMaxSuppressionMatch op while exporting to ONNX.

    This function helps exporting to onnx with batch and multiclass NMSMatch
    op. It only supports class-agnostic detection results. That is, the scores
    is of shape (N, num_bboxes, num_classes) and the boxes is of shape (N,
    num_boxes, 4).
    """
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    batch_size = scores.shape[0]
    topk_inds = None
    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(pre_top_k)
        batch_inds = torch.arange(
            batch_size, device=scores.device).view(-1, 1).long()
        boxes = boxes[batch_inds, topk_inds, :]
        scores = scores[batch_inds, topk_inds, :]

    scores = scores.permute(0, 2, 1)
    selected_indices = ONNXNMSMatchOp.apply(boxes, scores, iou_threshold,
                                            score_threshold)
    return _select_nms_index(
        scores, boxes, selected_indices, batch_size, keep_top_k=keep_top_k)


@mark(
    'multiclass_nms_match',
    inputs=['boxes', 'scores'],
    outputs=['dets', 'labels'])
def multiclass_nms_match(boxes: Tensor,
                         scores: Tensor,
                         max_output_boxes_per_class: int = 1000,
                         iou_threshold: float = 0.1,
                         score_threshold: float = 0.05,
                         pre_top_k: int = -1,
                         keep_top_k: int = -1):
    """Wrapper function for `_multiclass_nms_match`."""
    return mmdeploy.mmcv.ops.nms_match._multiclass_nms_match(
        boxes,
        scores,
        max_output_boxes_per_class=max_output_boxes_per_class,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_top_k,
        keep_top_k=keep_top_k)
