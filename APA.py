import cv2
import numpy as np
from PIL import Image


class APA:
    """
    An advanced, adaptive preprocessor based on Gamma+CLAHE, modified for
    optimizing training data.

    Processing Pipeline:
    1.  Denoising (Bilateral Filter).
    2.  Conversion to YCrCb color space.
    3.  Calculation of an adaptive gamma value based on the Y channel's mean.
    4.  Application of adaptive gamma to the Y channel (initial dark area lifting).
    5.  Application of CLAHE to enhance local details.
    6.  Merging channels and converting back to BGR.
    7.  Color Adjustment: Conversion to L*a*b* space and adjustment of the a*
        channel to enhance reds/suppress greens.
    8.  Final Touches:
        a. Conversion to HSV space.
        b. Boosting color saturation to maintain vibrancy.
        c. Reducing the V channel's brightness to suppress highlights and
           overall luminance.
    9.  Conversion back to BGR, then to the final RGB PIL image.
    """
    def __init__(self, denoise_d: int = 5, denoise_sigma_color: int = 75,
                 denoise_sigma_space: int = 75, base_gamma: float = 2.2,
                 clahe_clip_limit: float = 2.5,
                 clahe_tile_grid_size: tuple = (8, 8),
                 saturation_boost: float = 1.2,
                 highlight_suppression: float = 0.6,
                 red_boost_factor: float = 1.2):
        """
        Initializes the advanced preprocessor.

        Args:
            denoise_d (int): Diameter of each pixel neighborhood for the bilateral filter.
            denoise_sigma_color (int): Sigma value in the color space for the bilateral filter.
            denoise_sigma_space (int): Sigma value in the coordinate space for the bilateral filter.
            base_gamma (float): Target gamma value for moderately dark images.
            clahe_clip_limit (float): Contrast limit for CLAHE.
            clahe_tile_grid_size (tuple): Grid size for CLAHE.
            saturation_boost (float): Saturation boost factor. 1.0 means no change.
            highlight_suppression (float): Highlight suppression factor. Values < 1.0
                                           reduce the final brightness. E.g., 0.9 reduces
                                           brightness by 10%.
            red_boost_factor (float): Red enhancement/green suppression factor.
                                      Values > 1.0 will shift the image's hue towards red.
                                      1.0 means no change.
        """
        self.denoise_params = {
            'd': denoise_d,
            'sigmaColor': denoise_sigma_color,
            'sigmaSpace': denoise_sigma_space
        }
        self.base_gamma = base_gamma
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        self.saturation_boost = saturation_boost
        self.highlight_suppression = highlight_suppression
        self.red_boost_factor = red_boost_factor

    def _get_adaptive_gamma(self, y_channel: np.ndarray) -> float:
        """Calculates an adaptive gamma value based on the mean brightness of the Y channel."""
        mean_brightness = np.mean(y_channel) / 255.0
        # Adjust gamma based on the log of the mean brightness
        gamma_adjustment = -0.5 * np.log(mean_brightness + 1e-6)
        adaptive_gamma = self.base_gamma + gamma_adjustment
        return np.clip(adaptive_gamma, 1.5, 3.5)

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        """
        Applies the complete enhancement pipeline to a PIL image.

        Args:
            pil_img (Image.Image): The input PIL image in RGB format.

        Returns:
            Image.Image: The processed PIL image.
        """
        img_cv2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 1. Edge-preserving Denoising
        denoised_img = cv2.bilateralFilter(img_cv2, **self.denoise_params)

        # 2-6. Base Brightness and Contrast Enhancement
        img_ycrcb = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(img_ycrcb)

        adaptive_gamma = self._get_adaptive_gamma(y_channel)
        y_normalized = y_channel / 255.0
        y_gamma_corrected = np.power(y_normalized, 1.0 / adaptive_gamma)
        y_enhanced = np.uint8(y_gamma_corrected * 255)

        clahe_y = self.clahe.apply(y_enhanced)

        merged_ycrcb = cv2.merge([clahe_y, cr_channel, cb_channel])
        enhanced_bgr = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)

        # --- Step 7: Color Adjustment (Enhance Reds / Suppress Greens) ---
        # Convert to L*a*b* space, where the a* channel represents the green-red axis.
        img_lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2Lab)
        l_channel, a_channel, b_channel = cv2.split(img_lab)

        # Convert the a* channel to float for precise calculations.
        # The neutral point of the a* channel is 128. We shift it to be centered at 0,
        # scale it, and then shift it back. This enhances reds (values > 128) and
        # suppresses greens (values < 128).
        a_channel_float = a_channel.astype(np.float32)
        a_channel_adjusted = (a_channel_float - 128.0) * self.red_boost_factor + 128.0

        # Clip the adjusted values to the [0, 255] range and convert back to uint8.
        a_channel_final = np.clip(a_channel_adjusted, 0, 255).astype(np.uint8)

        # Merge channels and convert back to BGR.
        merged_lab = cv2.merge([l_channel, a_channel_final, b_channel])
        tinted_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_Lab2BGR)

        # --- Step 8: Final Touches (Saturation & Brightness Control) ---
        # Convert to HSV space to independently adjust Saturation (S) and Value (V).
        img_hsv = cv2.cvtColor(tinted_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)

        # a. Boost color saturation.
        s_boosted = np.clip(s * self.saturation_boost, 0, 255).astype(np.uint8)

        # b. Suppress overall brightness and highlights.
        v_suppressed = np.clip(v * self.highlight_suppression, 0, 255).astype(np.uint8)

        # Merge the adjusted HSV channels.
        final_hsv = cv2.merge([h, s_boosted, v_suppressed])
        final_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        # 9. Convert to the final PIL image.
        final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(final_rgb)