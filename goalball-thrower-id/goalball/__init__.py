"""
Goalball thrower identification.

Given a match video and the throw timestamps produced by the upstream throw
detector, name the player who threw each ball and score how confident that
call is.

The public entry points live in ../scripts; this package holds the pieces:

    logs      console output (every user-facing line goes through it)
    config    run settings: config file + CLI overrides, roster
    video     video handle, frame sampling, downscaling
    events    the throws CSV -> Throw objects (timestamps only)
    court     the court polygon, court-plane mapping, end mirroring
    halves    which half is being analysed, and what that does to positions
    gallery   the reference gallery: tag it, save it, load it
    detect    people + pose + ball detection
    pose      "is this a throwing pose" score
    teams     jersey-colour grouping and posture-based attacking team
    reid      appearance embeddings (OSNet / DINOv2)
    predict   the per-throw predictor
    labels    ground-truth label file
    evaluate  accuracy against the labels
"""

__version__ = "1.0.0"
