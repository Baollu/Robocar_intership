from parameters import shrink_factor, image_width, image_height
from model import SegNet, load_model_json
from torchvision.io import read_image
import torch.nn.functional as F
import torchvision
import torch
import os



def _find_latest_checkpoint():
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    checkpoints = sorted([f for f in os.listdir(weights_dir) if f.endswith(".pth.tar")])
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {weights_dir}. Run train.py first.")
    return os.path.join(weights_dir, checkpoints[-1])

model_json = load_model_json()
model = SegNet(in_chn=model_json['in_chn'], out_chn=model_json['out_chn'], BN_momentum=model_json['bn_momentum'])
checkpoint_path = _find_latest_checkpoint()
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, weights_only=True)
model.load_state_dict(checkpoint["state_dict"])
model.eval()



trans = torchvision.transforms.ToPILImage()
def show_image(img):
    out = trans(img.cpu())
    out.show()


def to_even(val):
    return int(val + 1) if int(val) % 2 == 1  else int(val)


def downscale(images, factor=1):
    if len(images.shape) == 3:
        images = images.reshape((1,) + images.shape)
    return F.interpolate(images, (to_even(image_height/factor), to_even(image_width/factor)))


def transform_image(image_name, debug = False):
    images = torch.zeros((1, 3, to_even(image_height/shrink_factor), to_even(image_width/shrink_factor)))
    images[0] = downscale(read_image(image_name).float() / 255.0, factor=shrink_factor)[:, :3, :, :]
    output = model(images)

    res = torch.argmax(output, dim=1).type(torch.long)

    if debug:
        print(f"Output min/max: {output.min():.4f} / {output.max():.4f}")
        print(f"Class 0 (non-road): {(res == 0).sum().item()} pixels")
        print(f"Class 1 (road):     {(res == 1).sum().item()} pixels")
        show_image(res[0].to(torch.float32) * 255)

    return res[0]



if __name__ =="__main__":
    transform_image("DatasetSimuator/ColoredCamera/carcolor0_frame0000.png", debug=True)
