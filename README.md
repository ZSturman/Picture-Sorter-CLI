# Picture Sorter CLI

Python image/video sorting with object detection, OCR, local inference and regression tests for safe file operations.

**Status:** Classification quality is provisional. Audio sorting, undo/journaling and a distributable app are not implemented. Optional model downloads can be large; test first on copies of your files.

[Project portfolio](https://zacharysturman.com/projects/picture-sorter-cli)

## Quick start

```sh
brew install ffmpeg tesseract
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Usage

```sh
.venv/bin/python pipeline.py --src /path/to/source --dest /path/to/sorted --dry-run
```

## How it works

A CLI routes images through OCR/object classification and samples video frames for captioning. File operations handle collisions and malformed media; dry-run previews planned moves.

## Testing

```sh
.venv/bin/python -m unittest discover -s tests -v
```

See [validation and next steps](docs/NEXT_STEPS.md) for the checks performed on this snapshot. Automated fixture tests do not establish physical-device compatibility.

## Configuration and privacy

This is a curated source snapshot, published November 12, 2026. Configure your own endpoints, credentials and signing identities. Private development history, local environment files, personal recordings, runtime databases, signing materials and deployment automation are excluded. Deployment and device installation are explicit operator actions.

## License and attribution

Existing licenses and third-party notices apply to the files they cover. Publication of this snapshot does not grant a new license to original material; no additional license is granted where none is supplied.
