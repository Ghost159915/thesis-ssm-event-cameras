"""Shared Gen1 dataset leaf paths — single source of truth for the RVT-preprocessed representation
and label files inside each recording directory. Imported by make_smoke_dataset.py and
make_train_subset.py so a change to the representation config (dt/nbins) is edited in ONE place."""

LEAF_REPR = "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations.h5"
LEAF_LABELS = "labels_v2/labels.npz"
