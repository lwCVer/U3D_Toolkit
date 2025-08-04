import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from factories.loss_factory import register_loss
from .base_loss import BaseLoss

class L_color(nn.Module):
    """
    Implements the Color Constancy Loss.

    This loss penalizes images with a strong color cast by encouraging the average
    R, G, and B channel values across the image to be similar. It is based on the
    "Gray-World" assumption, which states that the average color of a scene is achromatic (gray).
    """
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the color constancy loss.

        Args:
            x (torch.Tensor): The input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: The color loss value for each image in the batch.
        """
        b, c, h, w = x.shape

        # Calculate the mean of R, G, B channels across spatial dimensions
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)

        # Compute pairwise squared differences between mean channel values
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        
        # Add a small epsilon for numerical stability
        epsilon = 1e-8
        
        # Calculate the root of the sum of squares of the differences
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2) + epsilon, 0.5)

        return k


class L_spa(nn.Module):
    """
    Implements the Spatial Coherence Loss.

    This loss ensures that the enhanced image preserves the spatial structure of the
    original low-light image by maintaining gradient consistency. It penalizes
    differences in gradients between the original and enhanced images.
    """
    def __init__(self):
        super(L_spa, self).__init__()
        # Define non-trainable kernels for calculating gradients in 4 directions
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        
        # Use average pooling to compute loss on downsampled images for efficiency
        self.pool = nn.AvgPool2d(4)

    def forward(self, org: torch.Tensor, enhance: torch.Tensor) -> torch.Tensor:
        """
        Calculates the spatial coherence loss.

        Args:
            org (torch.Tensor): The original low-light image tensor.
            enhance (torch.Tensor): The enhanced image tensor.

        Returns:
            torch.Tensor: The spatial loss value.
        """
        # Move kernels to the correct device
        self.weight_left = self.weight_left.to(org.device)
        self.weight_right = self.weight_right.to(org.device)
        self.weight_up = self.weight_up.to(org.device)
        self.weight_down = self.weight_down.to(org.device)
        
        # Convert images to grayscale by taking the mean across channels
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        # Downsample for efficiency
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        # Calculate gradients in 4 directions for the original image
        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        # Calculate gradients in 4 directions for the enhanced image
        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        # Compute the squared difference of gradients
        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        
        # Sum the differences to get the final loss
        E = D_left + D_right + D_up + D_down
        return E


class L_TV(nn.Module):
    """
    Implements the Total Variation (TV) Loss.

    This loss encourages spatial smoothness in the input tensor `x` by penalizing
    large gradients between adjacent pixels. It is often applied to the
    illumination map in enhancement tasks.
    """
    def __init__(self):
        super(L_TV, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the total variation loss.

        Args:
            x (torch.Tensor): The input tensor, e.g., an illumination map.

        Returns:
            torch.Tensor: The total variation loss value.
        """
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        
        # Calculate the number of pixel differences
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        
        # Calculate horizontal and vertical TV
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        
        # Normalize and sum
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Sa_Loss(nn.Module):
    """
    Implements a Color Richness Loss (named Sa_Loss in original code).

    This loss encourages color vibrancy by maximizing the average L2 distance of
    pixel colors from the image's mean color. A higher value indicates more
    variation from the mean color, preventing the image from becoming gray or washed out.
    """
    def __init__(self):
        super(Sa_Loss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the color richness/saturation loss.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The color richness loss value.
        """
        b, c, h, w = x.shape
        r, g, b = torch.split(x, 1, dim=1)
        
        # Calculate mean color value for the image
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        
        # Calculate pixel-wise deviation from the mean color
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        
        # Calculate the L2 norm (Euclidean distance) from the mean color
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        
        # Average the distances over all pixels
        k = torch.mean(k)
        return k


class L_INT(nn.Module):
    """
    Implements a Luminance Interval Loss.

    This loss guides the enhanced image's brightness to a desirable range, both
    locally and globally. It penalizes patches that are too dark or too bright,
    and also constrains the overall image brightness to a target value.
    """
    def __init__(self, patch_size: int, dark_target: float = 0.3,
                 bright_target: float = 0.7, global_target: float = 0.6,
                 weight_global: float = 0.3):
        """
        Initializes the Luminance Interval Loss.

        Args:
            patch_size (int): The size of the patches for local average calculation.
            dark_target (float): The lower bound for local patch brightness.
            bright_target (float): The upper bound for local patch brightness.
            global_target (float): The target for the overall image brightness.
            weight_global (float): The weight for the global brightness constraint.
        """
        super(L_INT, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.dark_target = dark_target
        self.bright_target = bright_target
        self.global_target = global_target
        self.weight_global = weight_global

    def forward(self, enhanced_img: torch.Tensor) -> torch.Tensor:
        """
        Calculates the luminance interval loss.

        Args:
            enhanced_img (torch.Tensor): The enhanced image tensor.

        Returns:
            torch.Tensor: The luminance interval loss value.
        """
        # 1. Calculate local mean luminance
        local_mean_enhanced = self.pool(torch.mean(enhanced_img, 1, keepdim=True))

        # 2. Penalty for local dark regions
        loss_dark = torch.mean(torch.pow(torch.relu(self.dark_target - local_mean_enhanced), 2))

        # 3. Penalty for local bright regions
        loss_bright = torch.mean(torch.pow(torch.relu(local_mean_enhanced - self.bright_target), 2))

        # 4. Global mean brightness constraint
        global_mean_enhanced = torch.mean(enhanced_img)
        loss_global = torch.pow(global_mean_enhanced - self.global_target, 2)

        # 5. Total loss is the sum of local and weighted global losses
        local_loss = loss_dark + loss_bright
        total_loss = local_loss + self.weight_global * loss_global

        return total_loss


@register_loss('u3lie_loss')
class U3LIE_Loss(BaseLoss):
    """
    Implements the U3LIE (Unified Unsupervised Underwater Image Enhancement) loss function.

    This class combines multiple unsupervised loss components:
    - Total Variation Loss (L_TV): Encourages smoothness in the illumination map.
    - Spatial Coherence Loss (L_spa): Preserves spatial structure from the original image.
    - Color Constancy Loss (L_col): Reduces color cast.
    - Luminance Interval Loss (L_int): Guides brightness to a desirable range.

    The final loss is a weighted sum of these components.
    """
    def __init__(self, weight_tv: float = 100.0, weight_spa: float = 4.0,
                 weight_col: float = 20.0, exp_patch_size: int = 16,
                 weight_int: float = 200.0):
        """
        Initializes the U3LIE loss function.

        Args:
            weight_tv (float): Weight for the Total Variation Loss.
            weight_spa (float): Weight for the Spatial Coherence Loss.
            weight_col (float): Weight for the Color Constancy Loss.
            exp_patch_size (int): Patch size for the Luminance Interval Loss.
            weight_int (float): Weight for the Luminance Interval Loss.
        """
        super(U3LIE_Loss, self).__init__()
        self.weight_tv = weight_tv
        self.weight_spa = weight_spa
        self.weight_col = weight_col
        self.weight_int = weight_int

        self.l_tv = L_TV()
        self.l_spa = L_spa()
        self.l_col = L_color()
        self.l_int = L_INT(exp_patch_size, dark_target=0.5, bright_target=0.6,
                             global_target=0.6, weight_global=0.4)

    def forward(self, model_outputs: tuple, img_lowlight: torch.Tensor):
        """
        Calculates the total U3LIE loss.

        Args:
            model_outputs (tuple): A tuple containing the model's outputs.
                                   Expected to be (enhanced_image, illumination_map_A).
            img_lowlight (torch.Tensor): The original low-light input image.

        Returns:
            tuple: A tuple containing:
                - total_loss (torch.Tensor): The final weighted loss for backpropagation.
                - loss_dict (dict): A dictionary of individual loss components for logging.
        """
        enhanced_image, A = model_outputs
        
        # Calculate individual loss components
        loss_tv = self.weight_tv * self.l_tv(A)
        loss_spa = self.weight_spa * torch.mean(self.l_spa(enhanced_image, img_lowlight))
        loss_col = self.weight_col * torch.mean(self.l_col(enhanced_image))
        loss_int = self.weight_int * self.l_int(enhanced_image)

        # Sum the weighted losses
        total_loss = loss_tv + loss_spa + loss_col + loss_int
        
        # Create a dictionary for logging purposes
        loss_dict = {
            'Loss_Total': total_loss.item(),
            'Loss_TV': loss_tv.item(),
            'Loss_Spa': loss_spa.item(),
            'Loss_Col': loss_col.item(),
            'Loss_Int': loss_int.item(),
        }
        
        return total_loss, loss_dict