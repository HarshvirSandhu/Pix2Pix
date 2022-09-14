import torch
import torch.nn as nn
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import optim
from torchvision.utils import save_image, make_grid
from PIL import Image
from torchvision.models.segmentation import fcn_resnet101


# making masks
"""img_path = 'C:/Users/harsh/Downloads/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img'
target_img_path='C:/Users/harsh/Downloads/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
mask_path = 'C:/Users/harsh/Downloads/CelebAMask-HQ/CelebAMask-HQ/masks'
counter = 0
total = 0
for i in range(15):

    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    for j in range(i * 2000, (i + 1) * 2000):

        mask = np.zeros((512, 512))

        for l, att in enumerate(atts, 1):
            total += 1
            file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            path = os.path.join(target_img_path, str(i), file_name)

            if os.path.exists(path):
                counter += 1
                sep_mask = np.array(Image.open(path).convert('P'))
                # print(np.unique(sep_mask))

                mask[sep_mask == 225] = l

        plt.imsave('{}/{}.png'.format(mask_path, j), mask)
        print(j)

print(counter, total)"""


class CelebAMasked(Dataset):
    def __init__(self, img_path, target_img_path, transform):
        self.img_path = img_path
        self.target_img_path = target_img_path
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_path))

    def __getitem__(self, item):
        img = Image.open(f'{self.img_path}/{os.listdir(self.img_path)[item]}').convert('RGB')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        trg = Image.open(f'{self.target_img_path}/{os.listdir(self.target_img_path)[item]}').convert('RGB')
        # trg = cv2.cvtColor(trg, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        trg = self.transform(trg)
        return trg, img


image_size = 64
batch_size = 64
num_epochs = 5
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_data = CelebAMasked(img_path='C:/Users/harsh/Downloads/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img',
                        target_img_path='C:/Users/harsh/Downloads/CelebAMask-HQ/CelebAMask-HQ/masks', transform=transform)


# print(img_data.__getitem__(0)[0].shape, '------------------')

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, embed_size, img_size):
        super(Generator, self).__init__()
        self.embed_size = embed_size
        self.img_size = img_size
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=embed_size, out_channels=z_dim * 8, kernel_size=(4, 4), stride=(1, 1),
                               padding=(0, 0), bias=False),
            nn.BatchNorm2d(z_dim * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=z_dim * 8, out_channels=z_dim * 4, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1), bias=False),
            nn.BatchNorm2d(z_dim * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=z_dim * 4, out_channels=z_dim * 2, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1), bias=False),
            nn.BatchNorm2d(z_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=z_dim * 2, out_channels=z_dim, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1), bias=False),
            nn.BatchNorm2d(z_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=z_dim, out_channels=img_channels, kernel_size=(4, 4), stride=(2, 2),
                               padding=(1, 1), bias=False),
            nn.Tanh()
        )

        # self.embed = nn.Embedding(255, embed_size)

    def forward(self, label):
        return self.network(label)


class Critic(nn.Module):
    def __init__(self, img_channels, z_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=img_channels + 3, out_channels=z_dim, kernel_size=(4, 4), stride=(2, 2),
                      padding=(1, 1),
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=z_dim, out_channels=z_dim * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(z_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=z_dim * 2, out_channels=z_dim * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(z_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=z_dim * 4, out_channels=z_dim * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(z_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=z_dim * 8, out_channels=1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0),
                      bias=False),

        )

    def forward(self, x, label):
        x = torch.concat([x, label], dim=1)
        return self.network(x)


# gen = Generator(z_dim=128, img_channels=3, embed_size=512, img_size=64).to(device)
critic = Critic(img_channels=3, z_dim=128)
model = fcn_resnet101(pretrained=False, num_classes=3)


def weight_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if class_name.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)


# gen.apply(weight_init)
critic.apply(weight_init)
model.apply(weight_init)


def gradient_penalty(label, real_img, fake_img, critic):
    b, c, h, w = real_img.shape
    alpha = torch.rand((b, 1, 1, 1)).repeat(1, c, h, w)
    interpolated = alpha * real_img + (1 - alpha) * fake_img
    critic_interpol = critic(interpolated, label)

    gradients = torch.autograd.grad(outputs=critic_interpol, inputs=interpolated,
                                    grad_outputs=torch.ones_like(critic_interpol),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    grad_norm = gradients.norm(2, dim=1)
    return torch.mean((grad_norm - 1) ** 2)


# opt_gen = optim.RMSprop(gen.parameters(), lr=lr)
opt_critic = optim.RMSprop(critic.parameters(), lr=lr)
opt = optim.RMSprop(model.parameters(), lr=lr)

data_load = DataLoader(dataset=img_data, batch_size=batch_size, shuffle=True)
print(len(data_load))
for epoch in range(num_epochs):
    print(f'Epoch: {epoch}')
    for i, (data, label) in enumerate(data_load):
        # data = data.to(device)
        # label = label.to(device)
        print(label.shape)
        # if i >= 300 and i % 300 == 0:
        print(f"{i}/{len(data_load)}")

        for _ in range(5):
            # noise = torch.randn(batch_size, 512, 1, 1).to(device)
            fake = model(label)['out']
            opt_critic.zero_grad()
            critic_fake = critic(fake, label)
            critic_real = critic(data, label)
            critic_real = critic_real.reshape(-1)
            critic_fake = critic_fake.reshape(-1)
            critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake)) + gradient_penalty(label, data, fake,
                                                                                                  critic)
            critic_loss.backward(retain_graph=True)
            opt_critic.step()

        output_gen = critic(fake, label).reshape(-1)
        gen_loss = -torch.mean(output_gen)
        opt.zero_grad()
        gen_loss.backward()
        opt.step()
    # random_img = torch.randn((batch_size, 100, 1, 1)).to(device)
    img = model(label)['out']
    print(img.shape)
    grid = make_grid(img)
    print(grid.shape)
    name = "Conditional gan" + str(epoch // 4) + ".jpg"
    save_image(grid, name)
    torch.save({
        'Epoch': epoch,
        'model_state': model.state_dict()
    }, 'img_trans.pth.tar')
