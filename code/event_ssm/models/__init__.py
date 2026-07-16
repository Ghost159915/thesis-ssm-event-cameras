"""Thesis-B detector models: EventSSM (ResNet spatial) and PureSSM (BiMamba spatial).

Only the spatial mixer differs between the two; the recurrent skeleton
(event_ssm.backbone.resnet_mamba) and Mamba temporal path are shared.
"""
