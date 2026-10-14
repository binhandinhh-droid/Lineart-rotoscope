# Colored Line-Art Rotoscope

Turn a photo or a video of a person into a **colored line-art animation**: clean ink outlines drawn over flat fills, with one color per body part and clothing item sampled from the source footage, all on a paper-like background.

The pipeline uses a SegFormer human-parsing model to split the person into labeled parts (hair, face, upper clothes, pants, shoes, ...), then draws outlines along the label boundaries, adds optional inner detail strokes (clothing folds), and fills each part with its median color from the original frame.

## Features

- Works on both **images** and **videos** (frame by frame).
- Outlines from semantic label boundaries, plus optional Canny-based detail strokes.
- Face region is kept clean (no detail strokes on the face).
- One flat fill color per body part, with **temporal color smoothing** to reduce flicker in videos.
- Adjustable saturation and brightness of the fill colors.
- Subtle paper texture background.
- Runs on CUDA (with optional FP16), Apple MPS, or CPU.
- Safe with Unicode file paths on Windows.

## How it works

```
frame -> human parsing (SegFormer) -> low-res label map
      -> ink map  (label boundaries + detail strokes)
      -> part color table (median color per label, smoothed over time)
      -> composite: paper -> flat fills -> ink on top
```

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Entry point: file picker, device setup, image/video dispatch |
| `config.py` | Paths, file extensions and the `CFG` settings dictionary |
| `segmenter.py` | `Parser` class wrapping the SegFormer human-parsing model |
| `preprocess.py` | Low-resolution label map and RGB working copies |
| `ink.py` | Outline and detail-stroke generation |
| `colors.py` | Per-part representative colors with temporal smoothing |
| `compose.py` | Paper background and final compositing |
| `pipeline.py` | Single-frame pipeline that ties the steps together |
| `media_io.py` | Image/video reading and writing, file dialog, output paths |
| `utils.py` | Small shared helpers |

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

For GPU acceleration, install a CUDA build of PyTorch first by following the instructions at <https://pytorch.org/get-started/locally/>, then install the remaining requirements.

The model weights (`mattmdjaga/segformer_b2_clothes`) are downloaded automatically from the Hugging Face Hub on the first run.

## Usage

```bash
python main.py
```

1. A file dialog opens. Select an image or a video.
2. The result is saved to `OUTPUT_DIR` as `<original_name>_lineart_color.png` (images) or `.mp4` (videos). Existing files are never overwritten; a numeric suffix is added instead.

Supported inputs:

- Images: `.png .jpg .jpeg .bmp .webp .tif .tiff`
- Videos: `.mp4 .mov .avi .mkv .webm .m4v .wmv .flv`

> The file dialog uses `tkinter`, which is bundled with the standard Python installers on Windows and macOS. On some Linux distributions you may need to install it separately (for example `sudo apt install python3-tk`).

## Configuration

All settings live in [`config.py`](config.py).

Paths (edit these for your machine):

| Setting | Description |
|---|---|
| `OUTPUT_DIR` | Folder where results are written |
| `DIALOG_START_DIR` | Folder the file dialog opens in (ignored if it does not exist) |

Common options in the `CFG` dictionary:

| Key | Description |
|---|---|
| `USE_FP16` | Use half precision (CUDA only) |
| `MAX_FRAMES` | Limit the number of processed video frames (`None` = all) |
| `LINE_MAX_SIDE` | Max side (px) of the low-resolution working image |
| `LINE_THICKNESS_FRAC` / `DETAIL_THICKNESS_FRAC` | Outline and detail stroke thickness, relative to image size |
| `LINE_COLOR` | Ink color (RGB) |
| `DETAIL_LINES` | Enable or disable inner detail strokes |
| `CANNY_LOW` / `CANNY_HIGH` | Edge detector thresholds for detail strokes |
| `MIN_STROKE_LEN` | Remove detail strokes shorter than this many pixels |
| `FACE_LABELS` | Label ids treated as face (no detail strokes inside) |
| `PAPER_COLOR` / `PAPER_NOISE` | Background color and texture strength |
| `COLOR_TEMPORAL_ALPHA` | Color smoothing over time (`1.0` disables smoothing) |
| `FILL_SATURATION` / `FILL_BRIGHTNESS` | Adjust the fill colors |

## Notes and limitations

- The result depends on the quality of the human-parsing model; unusual poses, occlusions or heavy motion blur can produce noisy label boundaries.
- Processing is done frame by frame without optical flow, so some line flicker is possible in videos.
- Only the person is filled and outlined; the background is plain paper.
- Output videos are written with the `mp4v` codec and without audio.

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

- [SegFormer](https://arxiv.org/abs/2105.15203) and the [`mattmdjaga/segformer_b2_clothes`](https://huggingface.co/mattmdjaga/segformer_b2_clothes) model on Hugging Face.
- [PyTorch](https://pytorch.org/), [Transformers](https://github.com/huggingface/transformers) and [OpenCV](https://opencv.org/).
