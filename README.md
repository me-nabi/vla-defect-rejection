# VLA Defect Rejection

A simulated quality-control station on a bottling line. Juice bottles ride along a conveyor belt and pass an **inspection station**, where two cameras photograph each bottle. A **CNN trained only on simulated images** decides whether the bottle is defective, and a Franka Panda arm removes the flagged bottles into a cardboard reject box marked REJECT. Good bottles continue round a bend and down an exit belt into a cardboard packing carton. The line stands in a simulated warehouse with pallet racking, pallets of cartons, a fenced robot cell and signage.

All bottles look the same. Which ones are defective, and how, is randomized on every run, so the robot has nothing to go on except what the camera sees. The arm's pick-and-place is scripted. A fine-tuned **[SmolVLA](https://huggingface.co/papers/2506.01844)** vision-language-action policy steers the first approach; it is still a work in progress (see [status](#current-status-and-limitations)).

Built on **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)** (robosuite + MuJoCo) and **[LeRobot](https://github.com/huggingface/lerobot)**.

<p align="center">
  <img src="docs/media/demo.gif" width="768" alt="Inspection line: the classifier flags a chipped and a cracked bottle, the robot rejects them, and the good bottles go round the bend into the crate (3x speed)"><br>
  <em>Left: side view; while the robot rejects a bottle, the panel shows why (here "Liquid below fill line", then "Cap missing"), with the model's confidence and a heatmap of where it saw the defect. Right: the whole line; the three good bottles go round the bend into the packing crate (3× speed).</em>
</p>

**The six defect types** (each with randomized size, position and shape, from obvious to barely visible):

<img src="docs/media/defect_types.png" alt="A good bottle and the six defect types: crack, chip, missing cap, tilted cap, low fill, particle">

## Results

All numbers come from episodes and images the classifier never saw during training.

| Evaluation | Result |
|---|---|
| **Whole line**, classifier decisions + scripted robot, 20 episodes (`inspection.evaluate_line`) | **20 / 20** episodes correct: 28 / 28 defective bottles rejected, 0 / 72 good bottles rejected, 72 / 72 good bottles delivered to the packing crate, 0 bottles on the floor |
| **Classifier**, test set of 2,500 image pairs (`inspection.evaluate`) | 99.6 % accuracy, 99.4 % of defects caught, 0.08 % of good bottles flagged |
| **Classifier**, stress test of 1,500 pairs with harsher lighting, camera jitter and 2× sensor noise than in training | 96.1 % accuracy, 98.3 % of defects caught, 6.0 % of good bottles flagged |
| Robot only, ground-truth decisions, 10 episodes | 10 / 10 (35 / 35 good bottles delivered, 0 on the floor) |

Per defect type on the test set, the share of defective bottles caught: crack 100 %, chip 98.3 %, missing cap 100 %, tilted cap 100 %, low fill 100 %, particle 97.6 %. The misses are tiny chips and particles only about a pixel wide ([examples](docs/media/inspection_test_errors.png)). The stress test shows the main weakness: in conditions outside the training range the model becomes less confident, and good bottles start to get flagged.

**Why was it rejected?** For every bottle the robot removes, the video shows its two inspection photos, the reason in plain words and the model's confidence. A Grad-CAM heatmap shows where in the photos the model found the evidence. In the examples below, the heat sits on the empty upper body for low fill, on the neck for a missing cap, on the flake near the base for a chip (only in the view that sees it), and on the clear glass for a crack:

<img src="docs/media/reject_reasons.png" alt="Reject panels for low fill, missing cap, chip and crack, each with Grad-CAM heatmaps">

---

## How the inspection works

1. **Bottles.** [`tools/make_inspection_bottle.py`](tools/make_inspection_bottle.py) generates the bottle: a lathed glass body, a paper label, juice and a cap. It also contains invisible defect parts. At every reset, [`inspection_defects.py`](libero/libero/envs/inspection_defects.py) picks 0–2 defective bottles and a defect for each. Every bottle, good or bad, also gets harmless variation that a detector must not confuse with defects: air bubbles (which can look like particles), condensation droplets (which can look like chips), a shifted label, and fill level, juice colour and glass tint within spec.
2. **Station.** Two cameras face each other across the belt, so both sides of a bottle are seen. When a bottle reaches the station (like a light-barrier trigger), both cameras take a photo through a simple camera model, with sensor noise and JPEG compression ([`inspection/station.py`](inspection/station.py)). After that the bottle is tracked by identity, the way real lines track items with belt-encoder counts.
3. **Data.** [`inspection/generate_dataset.py`](inspection/generate_dataset.py) renders labelled pairs with the same cameras and camera model: half good bottles, half spread over the defect types. It randomizes the defects, the harmless features, the bottle pose, neighbouring bottles, the lighting and a few millimetres of camera pose.
4. **Classifier.** A ResNet-18 (ImageNet weights) runs on both views; the two feature vectors are combined and classified into good or one of the 6 defect types ([`inspection/model.py`](inspection/model.py)). P(defective) = 1 − P(good). The decision threshold is the centre of the range of thresholds that score best on the validation set, which leaves the most margin when conditions drift.
5. **Explanation.** The predicted defect type becomes the rejection reason (e.g. "Cap missing", "Liquid below fill line"), and Grad-CAM on the last ResNet block gives a heatmap per photo, normalized across both views so the view without the defect stays dark ([`inspection/model.py`](inspection/model.py), [`inspection/overlay.py`](inspection/overlay.py)).
6. **Robot.** Flagged bottles are picked when they reach the robot. The main belt pauses, the arm grasps the bottle, carries it high, lowers it into a free spot in the bin and releases it upright ([`scripted_pick.py`](scripted_pick.py)).
7. **Conveyor physics.** MuJoCo has no moving surfaces, so the belts are static, near-frictionless geoms. Each bottle touching a belt is pulled towards the local belt velocity (straight, round the bend, down the exit) by a force at its centre of mass, capped at what friction could transmit. A blocked bottle therefore slips instead of being shoved over, a knocked-over bottle keeps riding the belt like an upright one, and a stopped belt holds bottles in place. The bend and exit belt keep running while the main belt pauses for a pick.

---

## What's in this repo

### Simulation (added to LIBERO)

| File | What it does |
|---|---|
| [`libero/libero/envs/problems/conveyor_belt_manipulation.py`](libero/libero/envs/problems/conveyor_belt_manipulation.py) | The `conveyor_belt_manipulation` environment: conveyor physics (friction-limited belt drive along the main belt, the bend and the exit belt), random defect assignment, the inspection and overview cameras, and the success check (every defective bottle in the bin, no good one). |
| [`libero/libero/envs/inspection_defects.py`](libero/libero/envs/inspection_defects.py) | Per-bottle appearance: defects and harmless variation. |
| [`libero/libero/envs/arenas/conveyor_belt_arena.py`](libero/libero/envs/arenas/conveyor_belt_arena.py) | The belt arena: the main belt (`belt_extension` lengthens it upstream), and the 90° bend with guide rails, exit belt and cardboard packing carton after the robot (`BeltExit`). |
| [`libero/libero/envs/arenas/warehouse_props.py`](libero/libero/envs/arenas/warehouse_props.py) | Visual-only warehouse dressing (racking, pallets, fence, signs, cabinet), kept out of the inspection cameras' view. Textures come from [`tools/make_warehouse_textures.py`](tools/make_warehouse_textures.py). |
| [`libero/libero/envs/objects/custom_objects.py`](libero/libero/envs/objects/custom_objects.py) | `InspectionBottle`, `RejectBin`, and the older `GlassBottle` / `GoodBottle` used by the single-bottle task. |
| [`tools/make_inspection_bottle.py`](tools/make_inspection_bottle.py) | Generates the inspection bottle's meshes, label texture and XML. |
| [`libero/libero/bddl_files/defect_rejection/`](libero/libero/bddl_files/defect_rejection/) | Tasks: `multi_bottle_inspection` (the five-bottle inspection line) and `pick_defective_bottle_from_belt` (the older single-bottle task the current SmolVLA checkpoint was trained on). |

### Inspection station ([`inspection/`](inspection/))

| Script | What it does |
|---|---|
| `generate_dataset.py` | Render a labelled dataset (`--stress` for the harsher robustness set) |
| `train.py` | Train the classifier; `--recalibrate` recomputes the threshold on the validation set |
| `evaluate.py` | Test-set metrics, confusion matrix, example and error galleries |
| `evaluate_line.py` | Run the whole line over many seeded episodes and compare with ground truth |
| `station.py`, `model.py`, `overlay.py` | The station (trigger, camera model, decisions), the network, and the video overlay |

### Line and policy scripts

| Script | What it does |
|---|---|
| [`pipeline_demo.py`](pipeline_demo.py) | The inspection line with the scripted robot, recorded to a video. `--oracle` uses ground-truth decisions. |
| [`inference_smolvla.py`](inference_smolvla.py) | Same line, with SmolVLA steering the first approach |
| [`scripted_pick.py`](scripted_pick.py) | The shared scripted pick-and-place |
| [`collect_demos.py`](collect_demos.py), [`convert_demos_to_lerobot.py`](convert_demos_to_lerobot.py), [`train_smolvla.yaml`](train_smolvla.yaml) | SmolVLA data collection, conversion and fine-tuning (single-bottle task) |
| `rollout_smolvla.py`, `debug_*.py`, `test_*.py` | Older single-bottle demo (scripted, no policy despite the name) and debugging scripts |

---

## Setup

Tested with Python 3.12, robosuite 1.4.0, lerobot 0.6.1, PyTorch 2.11 and MuJoCo 3.8 on Linux with an NVIDIA GPU (RTX 3050, 6 GB).

```bash
git clone https://github.com/me-nabi/vla-defect-rejection.git
cd vla-defect-rejection

python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .                       # this repo's libero package
pip install robosuite==1.4.0 "lerobot[smolvla]" h5py imageio imageio-ffmpeg
```

> [!NOTE]
> Make sure `python -c "import libero; print(libero.__path__)"` points at this repo. Installing `lerobot[libero]` pulls in a separate copy of LIBERO (`hf_libero`) that will shadow it.

For headless machines, set `MUJOCO_GL=egl` before running any script.

## Usage

**Inspection station.** Rendering takes a few minutes, training about 15 minutes on an RTX 3050:

```bash
python -m inspection.generate_dataset --out inspection_data/train --n 10000 --seed 0
python -m inspection.generate_dataset --out inspection_data/val   --n 1500  --seed 1
python -m inspection.generate_dataset --out inspection_data/test  --n 2500  --seed 2
python -m inspection.train --epochs 15                 # -> inspection_models/inspection_net.pt
python -m inspection.evaluate                          # test-set metrics
python -m inspection.evaluate_line --episodes 20       # the whole line, many seeds
```

**Run the line and record a video:**

```bash
python pipeline_demo.py --seed 1008          # classifier decisions -> pipeline_demo.mp4
python pipeline_demo.py --seed 1008 --show   # same, plus a live window with the annotated views
python pipeline_demo.py --oracle             # ground-truth decisions (no model needed)
python inference_smolvla.py                  # with SmolVLA for the first approach, live window
python inference_smolvla.py --headless       # same, video only
```

The live window (1600×900) shows a large robot front view with the verdict over each bottle ("REJECT: Cap missing" on the bottle being picked). A side column holds an inspection panel and smaller side and overview cameras. The panel shows the inspection photos, the verdict and confidence, the reason and, during a rejection, the heatmap. Nothing is drawn over the camera images except the bottle tags.

**SmolVLA (single-bottle task):**

```bash
python collect_demos.py                      # scripted demos -> demos/conveyor_demos.hdf5
python convert_demos_to_lerobot.py           # -> lerobot_dataset/
lerobot-train --config_path=train_smolvla.yaml
```

Policy inputs: two cameras (`agentview` and the wrist camera, 256×256) and 8 numbers of robot state (7 joint positions and the gripper). Outputs: a 7-number end-effector action (position change, rotation change, gripper), sent through an operational-space controller. Language instruction: *"pick the defective bottle from the conveyor belt and place it in the reject bin"*.

Datasets and checkpoints are not stored in git, because they are too large; the metric reports in `inspection_models/*.json` are.

---

## Current status and limitations

- **Simulation only.** The classifier was trained and tested on renders from the same simulator and has never seen a real bottle. The stress test shows it is sensitive to conditions outside its training range. Real images, or much broader randomization, would be needed before it means anything on a real line.
- **The robot knows where bottles are.** Defect decisions come only from the inspection photos, but the grasp uses the simulator's bottle position, the job a pick camera or encoder tracking would do on a real line.
- **The learned policy does very little so far.** SmolVLA controls only about the first 5 steps of the first approach before the script takes over. Run on its own, the current checkpoint picks up the bottle in 0 of 8 single-bottle trials. It was trained for about 1.6 epochs on the older single-bottle task (red-marked bottle, belt stopped, a different scene and controller).
- **The single-bottle task and `collect_demos.py` still use the old bottles** and their own older drop routine.

### Roadmap
- [x] Reliable scripted pick-and-place
- [x] Realistic, randomized defects, and a vision inspection station trained on simulated data
- [x] Evaluation over many seeds: line success, missed defects, false rejects
- [ ] Demo collection on the inspection line (moving belt, shared pick-and-place, one controller config), then SmolVLA retraining
- [ ] End-to-end policy control, with the script only as a fallback
- [ ] Broader randomization (textures, backgrounds, camera models) and a real-image test set
- [ ] Datasets and checkpoints published on the Hugging Face Hub

---

## Credits

This project builds directly on the work of others:

- **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)** by Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu and Peter Stone. The `libero/` package, its environments, assets and tooling come from LIBERO. The git history of the original project is available in [the upstream repository](https://github.com/Lifelong-Robot-Learning/LIBERO), and its original README, with installation, datasets and benchmark details, is [there](https://github.com/Lifelong-Robot-Learning/LIBERO#readme).
- **[SmolVLA](https://huggingface.co/papers/2506.01844)** and **[LeRobot](https://github.com/huggingface/lerobot)** by Hugging Face: the policy architecture and the training and dataset framework.
- **[ResNet](https://arxiv.org/abs/1512.03385)** (He et al.), via [torchvision](https://github.com/pytorch/vision) with ImageNet weights: the inspection classifier's backbone.
- **[robosuite](https://github.com/ARISE-Initiative/robosuite)**: the robot simulation framework underneath LIBERO.
- **[MuJoCo](https://mujoco.org/)**: the physics engine.

## Citations

If you use this work, please cite the projects it builds on:

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}

@article{shukor2025smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  author={Shukor, Mustafa and others},
  journal={arXiv preprint arXiv:2506.01844},
  year={2025}
}

@misc{cadene2024lerobot,
  author={Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
  title={LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
  howpublished={\url{https://github.com/huggingface/lerobot}},
  year={2024}
}

@article{zhu2020robosuite,
  title={robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author={Zhu, Yuke and Wong, Josiah and Mandlekar, Ajay and Mart{\'i}n-Mart{\'i}n, Roberto and others},
  journal={arXiv preprint arXiv:2009.12293},
  year={2020}
}

@inproceedings{he2016resnet,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={770--778},
  year={2016}
}

@inproceedings{todorov2012mujoco,
  title={MuJoCo: A physics engine for model-based control},
  author={Todorov, Emanuel and Erez, Tom and Tassa, Yuval},
  booktitle={2012 IEEE/RSJ International Conference on Intelligent Robots and Systems},
  pages={5026--5033},
  year={2012}
}
```

## License

MIT. See [LICENSE](LICENSE). The original LIBERO code is © its authors under the MIT license, and that notice is kept in `LICENSE`.
