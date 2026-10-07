"""Stage 22 — synaptic-operation (SOP) and energy accounting for SpikingSSM.

Spec: docs/specs/2026-10-07-stage22-sop-energy-design.md, amended by Revision 2 in
docs/plans/2026-10-07-stage22-sop-energy-plan.md. Only the temporal readout spikes (choice C); its output feeds five
1x1 convolutions at the entry of the YOLO-PAFPN neck and nothing else (traced in tests/test_bench_sop.py). This
module counts the MACs those convolutions spend on the spiking channels, the dense MACs of the whole network, and
prices both with the 45 nm energy-per-operation constants of the SNN detection literature. Every result is an
OPERATION-COUNT estimate: memory traffic, which dominates energy on real hardware, is not modelled, and pricing
sparse MACs below dense ones assumes hardware that skips zero inputs.
"""
E_MAC_J = 4.6e-12       # 32-bit float multiply (3.7 pJ) + add (0.9 pJ), 45 nm
E_AC_J = 0.9e-12        # 32-bit float add, 45 nm
ENERGY_SOURCE = ("Horowitz, ISSCC 2014, 45 nm, 32-bit float: MAC 4.6 pJ (3.7 mult + 0.9 add), AC 0.9 pJ; "
                 "operation-count estimate, memory traffic not modelled")
# torch.profiler gives these ops FLOPs = 2 x MACs (tests/test_bench_sop.py). Its other FLOP-carrying ops
# (out-of-place add/mul, 1 FLOP per element) are elementwise and are left out of the MAC count.
MAC_OPS = ("aten::conv2d", "aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm")
RATE_CLIPS, RATE_WARMUP, RATE_FRAMES = 16, 32, 64     # Revision 2 §4


def _stage_geometry(detector, in_hw):
    fpn, bb = detector.fpn, detector.backbone
    in_stages = tuple(fpn.in_features)
    dims, strides = bb.get_stage_dims(in_stages), bb.get_strides(in_stages)
    h, w = in_hw
    return in_stages, dims, [(h // st) * (w // st) for st in strides]


def spike_fed_macs(detector, stages, in_hw=(256, 320)) -> dict:
    """{stage: MACs per frame that the neck spends on that stage's output channels}. The YOLO-PAFPN wiring
    (RVT models/detection/yolox_extension/models/yolo_pafpn.py, forward):
        in_stages[2] -> lateral_conv0                 input = x0
        in_stages[1] -> C3_p4.conv1, C3_p4.conv2      input = cat[up(lateral_conv0(x0)), x1]
        in_stages[0] -> C3_p3.conv1, C3_p3.conv2      input = cat[up(reduce_conv1(.)), x2]
    Channel counts, kernel, stride and groups are checked against the modules, so a changed neck fails closed."""
    fpn = detector.fpn
    in_stages, dims, hws = _stage_geometry(detector, in_hw)
    consumers = {2: ((fpn.lateral_conv0.conv,), 0),
                 1: ((fpn.C3_p4.conv1.conv, fpn.C3_p4.conv2.conv), fpn.lateral_conv0.conv.out_channels),
                 0: ((fpn.C3_p3.conv1.conv, fpn.C3_p3.conv2.conv), fpn.reduce_conv1.conv.out_channels)}
    out = {}
    for s in stages:
        if s not in in_stages:
            raise ValueError(f"stage {s} does not feed the neck (in_stages {in_stages})")
        i = in_stages.index(s)
        convs, offset = consumers[i]
        macs = 0
        for conv in convs:
            if (conv.in_channels != offset + dims[i] or conv.kernel_size != (1, 1) or conv.stride != (1, 1)
                    or conv.groups != 1):
                raise ValueError(f"neck wiring changed: stage {s} consumer {conv} (expected 1x1/stride-1/ungrouped "
                                 f"with {offset} + {dims[i]} input channels)")
            macs += hws[i] * dims[i] * conv.out_channels
        out[s] = macs
    return out


def lif_neurons(detector, stages, in_hw=(256, 320)) -> int:
    """LIF neurons per frame: one per channel and position of each spiking stage's output."""
    in_stages, dims, hws = _stage_geometry(detector, in_hw)
    return sum(dims[in_stages.index(s)] * hws[in_stages.index(s)] for s in stages)


def _kernel_macs_per_token(d_ssm, d_state, d_conv, conv_dim) -> int:
    """One scan direction, the part that runs in custom kernels: depthwise causal conv over the xBC channels,
    and the selective scan (state decay, input injection, readout: 3 MACs per state element)."""
    return conv_dim * d_conv + 3 * d_ssm * d_state


def ssm_kernel_macs(detector, in_hw=(256, 320)) -> dict:
    """MACs per frame of the custom SSM kernels, which torch.profiler counts as zero: temporal Mamba-2 layers
    (one direction) and spatial BiMamba scans (two directions). The in/out projections are nn.Linear layers that the
    profiler already counts, so they are NOT added here (Revision 2 §1)."""
    bb = detector.backbone
    h, w = in_hw
    temporal = 0
    for stage_str, block in bb.temporal.items():
        stride = bb.get_strides((int(stage_str),))[0]
        tokens = (h // stride) * (w // stride)
        for layer in getattr(block, "ssm", block).layers:
            conv_dim = layer.d_ssm + 2 * layer.ngroups * layer.d_state
            temporal += tokens * _kernel_macs_per_token(layer.d_ssm, layer.d_state, layer.d_conv, conv_dim)
    spatial = 0
    from event_ssm.models.puressm._scan2d import BiMamba1DScan
    sp = bb.spatial
    for i, stage in enumerate(getattr(sp, "stages", ())):
        tokens = (h // sp.strides[i]) * (w // sp.strides[i])
        for m in stage.modules():
            if isinstance(m, BiMamba1DScan):
                spatial += tokens * 2 * _kernel_macs_per_token(m.d_inner, m.d_state, m.d_conv, m.conv_dim)
    return {"temporal": temporal, "spatial": spatial}


def dense_macs(by_op_flops: dict, kernel_macs: dict) -> dict:
    """Dense MACs per frame: profiler-counted MAC ops (FLOPs / 2) plus the unprofiled SSM kernels."""
    profiled = sum(int(by_op_flops.get(op, 0)) for op in MAC_OPS) // 2
    return {"profiled_macs": profiled, "kernel_macs": dict(kernel_macs),
            "total_macs": profiled + sum(kernel_macs.values()),
            "excluded_flops_by_op": {op: f for op, f in by_op_flops.items() if op not in MAC_OPS}}


def op_class(arm: dict) -> str:
    """'AC' when the neck reads pure binary spikes (spike readout, no residual): a spike only adds weights.
    Otherwise the neck reads real values (graded spikes, the analog membrane, or spikes plus the residual input)
    and every nonzero input costs a multiply-accumulate: 'MAC'."""
    return "AC" if arm["output_mode"] == "spike" and not arm["residual"] else "MAC"


def op_energy(total_macs, fed_macs: dict, rates: dict, op: str, neurons: int) -> dict:
    """Energy per frame of the arm vs the same network priced dense (the PureSSM-equivalent arithmetic). The
    readout's consumers are charged per nonzero input: rate x MACs as accumulates (op 'AC') or as MACs ('MAC');
    each LIF neuron costs one MAC per frame (beta * mem + x). ceiling = the saving at zero activity."""
    if op not in ("AC", "MAC"):
        raise ValueError(f"op must be 'AC' or 'MAC', got {op!r}")
    if set(rates) != set(fed_macs):
        raise ValueError(f"rate stages {sorted(rates)} != spiking stages {sorted(fed_macs)}")
    for s, r in rates.items():
        if not 0.0 <= r <= 1.0:
            raise ValueError(f"rate of stage {s} outside [0, 1]: {r}")
    fed = sum(fed_macs.values())
    if not 0 < fed < total_macs:
        raise ValueError(f"spike-fed MACs {fed} must be positive and below the dense total {total_macs}")
    driven = sum(rates[s] * fed_macs[s] for s in fed_macs)
    macs = total_macs - fed + neurons + (driven if op == "MAC" else 0)
    acs = driven if op == "AC" else 0
    e, e_dense = E_MAC_J * macs + E_AC_J * acs, E_MAC_J * total_macs
    return {"op_class": op, "spike_fed_total": fed, "driven_ops": driven, "neurons": neurons,
            "macs": macs, "acs": acs, "energy_j": e, "energy_dense_j": e_dense,
            "saving": 1.0 - e / e_dense, "ceiling": (fed - neurons) / total_macs,
            "spike_fed_share": fed / total_macs}
