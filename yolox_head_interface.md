# YOLOX Head + Neck Interface (baseline) — Stage 2 deliverable
**Source:** `external/ssms_event_cameras/RVT/models/detection/...` (read 2026-06-06)
This documents the *actual* reused interfaces so our backbone produces the correct format. Supersedes the generic Stage 2 doc.

---

## Detector wiring (`yolox_extension/models/detector.py` — `YoloXDetector`)
```
backbone = build_recurrent_backbone(cfg.backbone)        # returns (features: dict[int->BCHW], states: LstmStates)
in_channels = backbone.get_stage_dims(cfg.fpn.in_stages) # e.g. stages (2,3,4)
fpn  = build_yolox_fpn(cfg.fpn, in_channels)             # YOLOPAFPN
head = build_yolox_head(cfg.head, in_channels, strides=backbone.get_strides(in_stages))

forward(x, previous_states, retrieve_detections, targets) -> (outputs, losses, states)
   backbone_features, states = backbone(x, previous_states)   # STATE carried here, at BACKBONE level
   fpn_features              = fpn(backbone_features)
   outputs, losses          = head(fpn_features, targets?)
```
**Key:** temporal state is a **backbone-level** contract (`previous_states` in, `states` out). FPN + head are stateless per window.

## Backbone contract (`recurrent_backbone/base.py`, `maxvit_rnn.py`)
- `forward(x:(L,B,C,H,W), prev_states:Optional[list], token_mask, train_step) -> (features:Dict[int,FeatureMap], states:LstmStates)`
- `features` = dict keyed by **stage number 1..4**, each `(L·B, C_stage, H, W)` after internal reshape (NCHW per window).
- `states` = **list of length num_stages (4)**; each entry one stage's recurrent state `(B, C_stage, H, W)` for S5. `None` allowed (first window).
- `get_stage_dims(stages)`, `get_strides(stages)` — must be implemented by any backbone.
- **Temporal axis (S5/our Mamba):** internally `rearrange "(L B) C H W -> (B H W) L C"` → scan over **L = time**, per spatial location → state `(B,C,H,W)`. **This is Formulation A.**

## Neck (`yolox_extension/models/yolo_pafpn.py` — `YOLOPAFPN`)
- `__init__(depth, in_stages=(2,3,4), in_channels=(256,512,1024), depthwise, act)`
- `forward(input: Dict[int, Tensor]) -> tuple(pan_out2, pan_out1, pan_out0)`
- Picks `input[f] for f in in_stages` → `x2,x1,x0` (high-res→low-res).
- **Path-aggregation** (FPN top-down + PAN bottom-up). Output channels follow the input pyramid `(in_channels[0], in_channels[1], in_channels[2])` — **NOT unified to 256.**
- Upsample = `interpolate(scale_factor=2, mode="nearest-exact")` → requires **even** spatial dims (guaranteed by input padding to multiple of 32).
- Output order: `(stride-8, stride-16, stride-32)`.

## Head (`yolox/models/yolo_head.py` — `YOLOXHead`)
- `__init__(num_classes, strides=(8,16,32), in_channels=(256,512,1024), act, depthwise, compile_cfg)`
- **`forward(xin, labels=None)`**
  - `xin` = **list/tuple of 3 feature maps**, ordered **(stride8, stride16, stride32)** — matches PAFPN output order.
  - `in_channels` per scale **may differ**; head hidden width `= int(256 * in_channels[-1] / 1024)`. ⚠ If you feed uniform 256, hidden width collapses to 64. Keep the pyramid widths.
  - `labels` shape `(B, max_obj, 5)` = `(class_id, cx, cy, w, h)` in **pixel** coords (cxcywh).
- **Returns `(outputs, losses)`**:
  - `outputs`: decoded predictions `(B, n_anchors_all, 5+num_classes)` = `[x, y, w, h, obj, cls...]` (decoded in both train and eval — modified from stock YOLOX).
  - `losses`: dict `{loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg}` in training; `None` in eval.
- **Losses actually used (⚠ corrects Stage 0 "Focal+GIoU"):**
  - cls: `BCEWithLogitsLoss`  | obj/conf: `BCEWithLogitsLoss`
  - reg: `IOUloss(loss_type="iou")` = `1 − iou²` (giou available but **not** default — verify config)
  - assignment: **SimOTA** (dynamic-k, center-radius 1.5)
  - total `= 5.0·iou_loss + obj_loss + cls_loss (+ l1 if enabled)`; `use_l1=False` by default.
  - "Focal Loss" appears only as bias init (`prior_prob=0.01`), not as a loss term.

## Standalone shape test (run to confirm on your env)
```python
import torch
from models.detection.yolox.models.yolo_head import YOLOXHead
head = YOLOXHead(num_classes=2, strides=(8,16,32), in_channels=(128,256,512)).cuda().eval()
# ResNet-18 stages (2,3,4) padded-input feature sizes: 32x40 / 16x20 / 8x10
xin = [torch.randn(2,128,32,40).cuda(), torch.randn(2,256,16,20).cuda(), torch.randn(2,512,8,10).cuda()]
out, losses = head(xin)
print(out.shape, losses)   # expect (2, 1680, 7), None    [1680 = 32*40+16*20+8*10]
```

## Implications for our backbone
1. Output a **dict keyed by stage number**, channels = ResNet stage dims (stages 2/3/4 → 128/256/512).
2. Implement `get_stage_dims` / `get_strides` (strides 8/16/32 for stages 2/3/4).
3. Return **per-stage states** via the `LstmStates` list contract — our Mamba state (SSM + conv caches) packed per stage.
4. Then PAFPN + head + losses + detector are **reused unmodified**.
