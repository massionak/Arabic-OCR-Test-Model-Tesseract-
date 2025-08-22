# Arabic OCR TestMOdel

## How to run

1. Put your test images inside `data/samples/`.

2. Run OCR on an image:

```bash
python src/baseline_test.py data/samples/your_image.png
```

The text result will be saved in `runs/txt/` and the TSV (with boxes/confidence) in `runs/tsv/`.

3. (Optional) To visualize the boxes on the image:

```bash
python src/visualize_tsv.py data/samples/your_image.png
```

The visualization will be saved in `runs/vis/`.
