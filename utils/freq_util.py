
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def display_img_tensor(img_tensor):
    img = img_tensor.squeeze(0).detach().cpu()  # shape: (3, H, W)

    # Convert to NumPy and permute channels
    img_np = img.permute(1, 2, 0).numpy()  # shape: (H, W, 3)

    # Plot
    # set image size to 5x5 inches
    plt.figure(figsize=(2, 2))
    plt.imshow(img_np)
    plt.axis('off')
    plt.show()

def image_fft(img):
    """
    Apply 2D FFT to an image tensor in PyTorch.
    
    Args:
        img: Tensor of shape (B, C, H, W), values in float32, typically in [0, 1]

    Returns:
        fft: complex tensor of shape (B, C, H, W) — complex-valued frequency domain
        magnitude: real tensor of shape (B, C, H, W) — magnitude spectrum
    """
    # Compute 2D FFT
    fft = torch.fft.fft2(img)

    # Shift the zero-frequency component to the center of the spectrum
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

    # Compute magnitude (optionally you can do log-magnitude)
    magnitude = torch.abs(fft_shifted)

    return magnitude
    
def low_pass_filter(img_tensor, radius_ratio=0.25, only_freq=False):
    """
    Apply a differentiable low-pass filter to an image tensor using FFT.

    Args:
        img_tensor: (B, C, H, W) tensor, float32 in [0, 1]
        radius_ratio: Fraction of low frequencies to keep (e.g., 0.1 keeps central 10%)

    Returns:
        filtered_img: Tensor (B, C, H, W), low-passed image
    """
    try:
        _, _, _, H, W = img_tensor.shape
    except:
        _, _, H, W = img_tensor.shape

    fft = torch.fft.fft2(img_tensor)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

    # Create circular low-pass mask
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    center_y, center_x = H // 2, W // 2
    radius = radius_ratio * min(H, W)

    dist = torch.sqrt((yy - center_y)**2 + (xx - center_x)**2)
    mask = (dist <= radius).float().to(img_tensor.device)  # shape: (H, W)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Apply mask
    fft_filtered = fft_shifted * mask
    if only_freq:
        return torch.abs(fft_filtered)

    # Inverse FFT
    fft_unshifted = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
    img_filtered = torch.fft.ifft2(fft_unshifted)

    # Return real part (imaginary part should be close to 0)
    return img_filtered.real


def low_pass_filter_channel(img_tensor, radius_ratio=0.1, channel=0, only_freq=False):
    """
    Apply a differentiable low-pass filter to an image tensor using FFT.

    Args:
        img_tensor: (B, C, H, W) tensor, float32 in [0, 1]
        radius_ratio: Fraction of low frequencies to keep (e.g., 0.1 keeps central 10%)

    Returns:
        filtered_img: Tensor (B, C, H, W), low-passed image
    """
    try:
        _, _, _, H, W = img_tensor.shape
    except:
        _, _, H, W = img_tensor.shape

    fft = torch.fft.fft2(img_tensor)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

    # Create circular low-pass mask
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    center_y, center_x = H // 2, W // 2
    radius = radius_ratio * min(H, W)

    dist = torch.sqrt((yy - center_y)**2 + (xx - center_x)**2)
    mask = (dist <= radius).float().to(img_tensor.device)  # shape: (H, W)
    #mask = mask.unsqueeze(0)#.unsqueeze(0)  # (1, 1, H, W)
    
    # Apply mask
    fft_shifted[:,channel] = fft_shifted[:,channel] * mask
    fft_filtered = fft_shifted
    if only_freq:
        return torch.abs(fft_filtered)

    # Inverse FFT
    fft_unshifted = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
    img_filtered = torch.fft.ifft2(fft_unshifted)

    # Return real part (imaginary part should be close to 0)
    return img_filtered.real



def apply_3_stripe_mask(img_tensor, height=512, width=100):
    try:
        _, _, _, H, W = img_tensor.shape
    except:
        _, _, H, W = img_tensor.shape

    fft = torch.fft.fft2(img_tensor)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))


    mask = torch.zeros_like(fft_shifted)
    middle = int(1024/2)
    mask[:, :, middle-height:middle+height,middle-width:middle+width] = 1

    middle_2= 256
    mask[:, :, middle-height:middle+height,middle_2-width:middle_2+width] = 1
    
    middle_3 = 1024-256
    mask[:, :, middle-height:middle+height,middle_3-width:middle_3+width] = 1
    return torch.abs(fft_shifted * mask)




def frequency_square_injection(img_tensor, square_size_ratio=0.25, offset_ratio=0.4, only_freq=False):
    """
    Add 4 symmetric square patches in frequency domain.

    Args:
        img_tensor: (B, C, H, W) tensor
        square_size_ratio: Size of each square as fraction of image size
        offset_ratio: How far from center to place squares (as fraction of H/W)
        only_freq: If True, return the modified frequency magnitude

    Returns:
        Tensor of same shape (B, C, H, W)
    """
    try:
        _, _, _, H, W = img_tensor.shape
    except:
        _, _, H, W = img_tensor.shape

    device = img_tensor.device
    B, C = img_tensor.shape[:2]

    fft = torch.fft.fft2(img_tensor)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

    # Create frequency mask
    mask = torch.zeros((H, W), device=device)

    square_size = int(square_size_ratio * min(H, W))
    offset_y = int(offset_ratio * H)
    offset_x = int(offset_ratio * W)

    center_y, center_x = H // 2, W // 2

    positions = [
        (center_y - offset_y, center_x - offset_x),
        (center_y - offset_y, center_x + offset_x - square_size),
        (center_y + offset_y - square_size, center_x - offset_x),
        (center_y + offset_y - square_size, center_x + offset_x - square_size),
    ]

    for y, x in positions:
        mask[y:y+square_size, x:x+square_size] = 1.0

    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    fft_modified = fft_shifted * (1 + mask)  # Amplify those regions

    if only_freq:
        return torch.abs(fft_modified)

    # Inverse FFT
    fft_unshifted = torch.fft.ifftshift(fft_modified, dim=(-2, -1))
    img_out = torch.fft.ifft2(fft_unshifted)
    return img_out.real


def text_watermark(img_tensor, text="WATERMARK", font_size=50, only_freq=False):
    """
    Apply a differentiable low-pass filter to an image tensor using FFT.

    Args:
        img_tensor: (B, C, H, W) tensor, float32 in [0, 1]
        radius_ratio: Fraction of low frequencies to keep (e.g., 0.1 keeps central 10%)

    Returns:
        filtered_img: Tensor (B, C, H, W), low-passed image
    """

    try:
        _, _, _, H, W = img_tensor.shape
    except:
        _, _, H, W = img_tensor.shape

    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)

    fft = torch.fft.fft2(img_tensor)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    print(fft_shifted.shape)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf ", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate position and draw the text
    # text_width, text_height = draw.textsize(text, font=font)
    x = 50 #(W - text_width) // 2
    y = (H - 50) // 2
    draw.text((x, y), text, fill=int(255 * 1), font=font)

    # Convert to numpy, then to torch tensor
    mask_np = np.array(img) / 255.0
    mask_tensor = torch.tensor(mask_np, dtype=torch.float32)

    # Expand to match input shape
    if len(fft_shifted.shape) == 3:
        mask_tensor = mask_tensor.unsqueeze(0).expand(fft_shifted.shape[0], -1, -1)

    fft_filtered = fft_shifted * (1.0 - mask_tensor.to(fft_shifted.device))
    del mask_tensor

    if only_freq:
        return torch.abs(fft_filtered)

    # Inverse FFT
    fft_unshifted = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
    img_filtered = torch.fft.ifft2(fft_unshifted)

    # Return real part (imaginary part should be close to 0)
    return img_filtered.real


def apply_frequency_watermark(x, watermark_mask, alpha=0.01):
    """
    x: (B, C, H, W) input image
    watermark_mask: (1, 1, H, W) or (B, 1, H, W) float mask [0,1]
    alpha: scaling factor for watermark strength
    """

    # FFT
    x_fft = torch.fft.fft2(x)  # shape (B, C, H, W), complex
    x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))

    # Separate magnitude and phase
    magnitude = torch.abs(x_fft_shifted)
    phase = torch.angle(x_fft_shifted)

    # Superimpose watermark onto magnitude
    magnitude_watermarked = magnitude + alpha * watermark_mask

    # Rebuild complex FFT
    real = magnitude_watermarked * torch.cos(phase)
    imag = magnitude_watermarked * torch.sin(phase)
    x_fft_watermarked = torch.complex(real, imag)

    # Shift back
    x_fft_unshifted = torch.fft.ifftshift(x_fft_watermarked, dim=(-2, -1))

    # iFFT back to spatial domain
    x_watermarked = torch.fft.ifft2(x_fft_unshifted)

    # Take real part (imaginary part should be small)
    return x_watermarked.real




def create_circular_mask(H, W, radius, center=None):
    if center is None:
        center = (H//2, W//2)
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    dist = torch.sqrt((X - center[1])**2 + (Y - center[0])**2)
    mask = torch.exp(-((dist - radius) ** 2) / (2*(radius*0.2)**2))  # Gaussian ring
    mask = mask / mask.max()
    return mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)



def visualize_fft(magnitude):
    """
    Visualizes the log-magnitude spectrum of the image tensor's FFT.

    Args:
        img_tensor: Tensor of shape (B, C, H, W), values in [0, 1]
    """
    B,C,H,W = magnitude.shape
    #log_magnitude = torch.log1p(magnitude)
    log_magnitude = torch.log(magnitude + 1e-10)  # Avoid log(0)
    # Plot per channel
    for c in range(C):
        plt.figure(figsize=(3, 3))
        img_np = log_magnitude[0, c].detach().cpu().numpy()
        plt.imshow(img_np, cmap='gray')
        plt.title(f'Log-Magnitude Spectrum , Channel {c}')
        plt.axis('off')
        plt.show()

def display_gray_img_tensor(img_tensor):
    img = img_tensor.squeeze(0).detach().cpu()  # shape: (1, H, W)

    # Convert to NumPy and permute channels
    img_np = img.permute(1, 2, 0).numpy()  # shape: (H, W)

    # Plot
     # set image size to 5x5 inches
    plt.figure(figsize=(3, 3))
    plt.imshow(img_np, cmap='gray')
    plt.axis('off')
    plt.show()
def rgb_to_grayscale(img):
    # img shape: (B, 3, H, W)
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
    
def create_png_from_fft_magnitude(img_tensor):
    """
    Save the FFT magnitude of a [C, H, W] image tensor as a PNG.
    """
    # Ensure it's batched
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)  # (1, C, H, W)
    
    # Compute FFT magnitude
    magnitude = image_fft(img_tensor)[0]  # Remove batch dim after FFT

    # Collapse channels for saving — take mean or use one channel
    if magnitude.shape[0] == 3:
        mag_vis = magnitude.mean(0)  # (H, W)
    else:
        mag_vis = magnitude[0]  # (H, W)

    # Log scale for visibility
    mag_vis = torch.log1p(mag_vis)

    # Normalize to [0, 255]
    mag_vis -= mag_vis.min()
    mag_vis /= mag_vis.max()
    mag_vis = (mag_vis * 255).clamp(0, 255).byte()

    # Convert to PIL and save
    img = Image.fromarray(mag_vis.cpu().numpy(), mode='L')
    return img

def create_png_from_fft_magnitude_2(magnitude):
    # Collapse channels for saving — take mean or use one channel
    if magnitude.shape[0] == 3:
        mag_vis = magnitude.mean(0)  # (H, W)
    else:
        mag_vis = magnitude[0]  # (H, W)

    # Log scale for visibility
    mag_vis = torch.log(mag_vis + 1e-10)

    # Normalize to [0, 255]
    mag_vis -= mag_vis.min()
    mag_vis /= mag_vis.max()
    mag_vis = (mag_vis * 255).clamp(0, 255).byte()

    # Convert to PIL and save
    img = Image.fromarray(mag_vis.cpu().numpy(), mode='L')
    return img
