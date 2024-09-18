# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BaseModel, Input, Path, Secret
import os
import yaml
import subprocess
from zipfile import ZipFile
from huggingface_hub import HfApi

class TrainingOutput(BaseModel):
    weights: Path

# Run in lucataco/sandbox2
def train(
    steps: int = Input(description="Number of training steps. Recommended range 500-4000", ge=10, le=4000, default=1000),
    lora_rank: int = Input(description="LoRA linear value", default=16),
    optimizer: str = Input(description="optimizer", default="adamw8bit"),
    batch_size: int = Input(description="Batch size", default=1),
    sample_steps: int = Input(description="Sample steps", default=20),
    resolution: str = Input(description="Image resolutions for training", default="512,768,1024"),
    input_images: Path = Input(description="A zip/tar file containing the images that will be used for training. File names must be their captions: a_photo_of_TOK.png, etc. Min 12 images required."),
    model_name: str = Input(description="Model name", default="black-forest-labs/FLUX.1-dev"),
    learning_rate: float = Input(description="Learning rate", default=4e-4),

) -> TrainingOutput:
    """Run a single prediction on the model"""
    print("Starting Training")
    # Cleanup previous runs
    os.system("rm -rf trained_model")
    # Cleanup training images (from canceled training runs)
    input_dir = "input_images"
    os.system(f"rm -rf {input_dir}")

    # Update the config file using YAML
    config_path = "config/replicate.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the configuration
    config['config']['process'][0]['model']['name_or_path'] = model_name
    config['config']['process'][0]['save']['save_every'] = steps + 1
    config['config']['process'][0]['train']['steps'] = steps
    config['config']['process'][0]['train']['optimizer'] = optimizer
    config['config']['process'][0]['train']['lr'] = learning_rate
    config['config']['process'][0]['train']['batch_size'] = batch_size
    config['config']['process'][0]['datasets'][0]['resolution'] = [int(res) for res in resolution.split(',')]
    config['config']['process'][0]['network']['linear'] = lora_rank
    config['config']['process'][0]['network']['linear_alpha'] = lora_rank
    config['config']['process'][0]['sample']['sample_steps'] = sample_steps

    # Save config changes
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    
    # Unzip images from input images file to the input_images folder
    input_images = str(input_images)
    print(f"input_images file = {input_images}")
    if input_images.endswith(".zip"):
        print("Detected zip file")
        os.makedirs(input_dir, exist_ok=True)
        with ZipFile(input_images, "r") as zip_ref:
            zip_ref.extractall(input_dir+"/")
        print("Extracted zip file")
    elif input_images.endswith(".tar"):
        print("Detected tar file")
        os.makedirs(input_dir, exist_ok=True)
        os.system(f"tar -xvf {input_images} -C {input_dir}")
        print("Extracted tar file")
    else:
        print("By default - ASSUMING zip file")
        os.makedirs(input_dir, exist_ok=True)
        with ZipFile(input_images, "r") as zip_ref:
            zip_ref.extractall(input_dir+"/")
        print("Extracted ASSUMED zip file")

    # Run - bash train.sh
    subprocess.check_call(["python", "run.py", "config/replicate.yml"], close_fds=False)

    # Tar up the output folder
    output_lora = os.path.join(config['config']['process'][0]['training_folder'], config['config']['name']) #"output/lora"
    os.rename(os.path.join(output_lora, "flux_train_replicate.safetensors"), os.path.join(output_lora, "lora.safetensors"))

    # copy license file to output folder
    if os.path.isfile("lora-license.md"):
        os.system(f"cp lora-license.md {output_lora}/README.md")
    output_tar_path = "/tmp/trained_model.tar"
    os.system(f"tar -czvf {output_tar_path} {output_lora}")
    print(f"Taring {output_tar_path} into to {output_tar_path}")

    # cleanup input_images folder
    os.system(f"rm -rf {input_dir}")

    return TrainingOutput(weights=Path(output_tar_path))
