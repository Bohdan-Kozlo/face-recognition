# Face Recognition Authentication

An prototype for one-to-one face verification. Users will enroll with three
face images and later authenticate with an email address and a new face image.

## Fine-tuning ResNet18

The project uses ImageNet-pretrained ResNet18 with a 512-dimensional ArcFace embedding.
Choose the training scope explicitly:

```powershell
uv run --group training python src/training.py --fine-tuning last-layer --deterministic
```

Use `--fine-tuning all` to update the full backbone. To train without ImageNet
weights, use `--initialization scratch --fine-tuning all`; it defaults to a
learning rate of `0.001`. Training uses the standard `Adam` optimizer for both
ImageNet fine-tuning and training from scratch. Checkpoints include both
choices in their names, for example `checkpoints/resnet18-scratch-all.pt`.
