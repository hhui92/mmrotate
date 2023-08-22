import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


class FreqAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FreqAttention, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.conv_imag = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        # 将输入的图像从空域转换为频域表示, fftn可以处理任意维度的图像
        x_freq = torch.fft.fftn(x, dim=(-2, -1))
        # 改变重新排列张良的维度顺序 对于图像而言输入维度为b c h w 傅里叶变换后维度与输入相同，不需要对维度进行调增
        # x_freq = x_freq.permute(0, 3, 1, 2)

        x_freq_real = x_freq.real
        x_freq_imag = x_freq.imag

        x_freq_real = self.global_avgpool(x_freq_real)
        x_freq_imag = self.global_avgpool(x_freq_imag)

        # 对频域表示的不同频率分量进行加权
        x_freq_real = self.conv_real(x_freq_real)
        x_freq_imag = self.conv_imag(x_freq_imag)

        x_freq_real = self.sigmoid(x_freq_real)
        x_freq_imag = self.sigmoid(x_freq_imag)

        # 将加权后的频域表示应用到原始图像上
        x_atten_real = x * x_freq_real
        x_atten_imag = x * x_freq_imag
        x_atten = x_atten_real + x_atten_imag

        return x_atten


# Load and preprocess the image
image_path = r'/root/autodl-tmp/DIOR/trainval/images/05851.jpg'
image = Image.open(image_path).convert('RGB')  # Convert to rgb
transform = transforms.Compose([
    transforms.ToTensor()            # Convert PIL Image to Torch tensor
])
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform FFT
# image_fft = torch.fft.fft2(image_tensor)
# image_fft_shifted = torch.from_numpy(fftshift(image_fft))  # Shift zero frequency to the center
# img_fft = torch.fft.fftn(image_tensor, dim=(-2, -1))
# img_fft_real = img_fft.real
# img_fft_img = img_fft.imag
# weight1 = torch.tensor(0.3)
# weight2 = torch.tensor(0.7)
# weighted_fft_real = weight1 * img_fft_real + weight2 * img_fft_real
# weighted_fft_img = weight1 * img_fft_img + weight2 * img_fft_img
# weight_fft = torch.complex(weighted_fft_real, weighted_fft_img)

f = FreqAttention(3, 12)
res = f(image_tensor)
res = torch.fft.ifftn(res, dim=(-2, -1))
diff_image = transforms.ToPILImage()(image_tensor.squeeze(0))
diff_image.save("/root/autodl-tmp/DIOR/trainval/ori.png")
out_image = transforms.ToPILImage()(res.squeeze(0))
out_image.save("/root/autodl-tmp/DIOR/trainval/fft.png")


# # Display original, enhanced images, and combined image
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(image_tensor.squeeze().numpy(), cmap='gray')
# plt.title('Original Image')
# plt.subplot(1, 3, 2)
# plt.imshow(combined_image, cmap='gray')
# plt.title('Combined Enhanced Image')
# plt.subplot(1, 3, 3)
# plt.imshow((enhanced_image_1 + enhanced_image_2) / 2, cmap='gray')
# plt.title('Average Enhanced Image')
# plt.show()
