"""Vision inspection station for the conveyor line.

Two cameras photograph each bottle as it passes x = INSPECTION_X. A small CNN, trained
only on simulator renders, decides whether the bottle is defective, and the robot
rejects the bottles it flags.

    station.py           capture + classify bottles as they pass (or use ground truth with --oracle)
    model.py             the classifier (ResNet-18 shared over both views)
    generate_dataset.py  render a labelled dataset from the simulator
    train.py             train the classifier
    evaluate.py          test-set metrics for the classifier
    evaluate_line.py     run the whole line over many seeds: catches, misses, false rejects
    overlay.py           video overlay (labels, HUD, inspection inset)
"""
