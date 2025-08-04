import torch
import torchvision
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
from typing import Dict, Tuple, Union, List, Optional
from copy import deepcopy

# ======================================================================
# [NEW] Generic Model Output Extractor
# ======================================================================
def _extract_enhanced_tensor(
    model_output: Union[torch.Tensor, Tuple, List],
    reference_input: torch.Tensor
) -> torch.Tensor:
    """
    Intelligently extracts the main image tensor from a model's output.

    This function enables the rest of the code to handle models that return
    a single tensor or a tuple/list, thereby enhancing the generality of
    the eei_calculator.

    Args:
        model_output: The raw output from the model.
        reference_input: The input tensor provided to the model, used for shape comparison.

    Returns:
        torch.Tensor: The extracted tensor, assumed to be the enhanced image.

    Raises:
        TypeError: If the model output is not a tensor, tuple, or list.
        ValueError: If a suitable image tensor cannot be found in the tuple/list.
    """
    # Case 1: The model output is a single tensor (most common case)
    if isinstance(model_output, torch.Tensor):
        return model_output

    # Case 2: The model output is a tuple or list
    if isinstance(model_output, (tuple, list)):
        candidate_tensors = []
        for item in model_output:
            if (isinstance(item, torch.Tensor) and item.ndim == 4 and
                    item.shape[0] == reference_input.shape[0] and
                    item.shape[1] == reference_input.shape[1]):
                candidate_tensors.append(item)

        if not candidate_tensors:
            raise ValueError(
                "No suitable 4D image tensor found in the model's output tuple/list. "
                f"Output types: {[type(i) for i in model_output]}"
            )
        # Prioritize the candidate that is closest in H, W dimensions to the input tensor.
        if len(candidate_tensors) > 1:
            ref_h, ref_w = reference_input.shape[2:]
            diffs = [abs(t.shape[2] - ref_h) + abs(t.shape[3] - ref_w) for t in candidate_tensors]
            return candidate_tensors[np.argmin(diffs)]
        return candidate_tensors[0]

    # If the output is an unsupported type
    raise TypeError(
        f"Unsupported model output type: {type(model_output)}. "
        "Expected torch.Tensor, tuple, or list."
    )



def process_image_robustly(
    model: nn.Module,
    input_tensor: torch.Tensor,
    patch_size: Optional[int] = None,
    overlap: int = 64,
    pad_factor: int = 8
) -> Tuple[torch.Tensor, float, float]:
    """
    Robustly processes an image, with an option for either direct or patch-based processing.

    If `patch_size` is `None`, it performs inference on the entire image directly.
    If `patch_size` is an integer, it processes large images using an overlapping
    patch strategy with Hanning window fusion.

    Args:
        model (nn.Module): The model to perform inference with.
        input_tensor (torch.Tensor): The input image tensor.
        patch_size (Optional[int], optional): The size of the patches. If None,
                                              full-size inference is performed. Defaults to None.
        overlap (int, optional): The number of overlapping pixels between patches. Defaults to 64.
        pad_factor (int, optional): The padding factor to meet model downsampling requirements. Defaults to 8.

    Returns:
        Tuple[torch.Tensor, float, float]: The processed image, elapsed time (seconds), and peak memory (GB).
    """
    model.eval()
    device = input_tensor.device

    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    if device.type == 'cuda': torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        # Determine processing method based on patch_size 
        if patch_size is None:
            # Direct full-size inference
            model_output = model(input_tensor)
            output = _extract_enhanced_tensor(model_output, input_tensor)
        else:
            # Original patch-based processing logic
            b, c, h, w = input_tensor.shape
            stride = patch_size - overlap

            pad_h = (stride - (h - patch_size) % stride) % stride if h > patch_size else 0
            pad_w = (stride - (w - patch_size) % stride) % stride if w > patch_size else 0
            input_padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), 'reflect')

            _, _, H, W = input_padded.shape
            output_padded = torch.zeros_like(input_padded)
            count_map = torch.zeros_like(input_padded)
            window = torch.hann_window(patch_size, periodic=False, device=device)
            window2d = window[:, None] * window[None, :]

            for i in range(0, H - patch_size + 1, stride):
                for j in range(0, W - patch_size + 1, stride):
                    patch = input_padded[:, :, i:i+patch_size, j:j+patch_size]

                    ph, pw = patch.shape[2], patch.shape[3]
                    PH, PW = ((ph + pad_factor - 1) // pad_factor) * pad_factor, ((pw + pad_factor - 1) // pad_factor) * pad_factor
                    patch_padded_inf = F.pad(patch, (0, PW - pw, 0, PH - ph), 'reflect')

                    model_output = model(patch_padded_inf)
                    enhanced_patch = _extract_enhanced_tensor(model_output, patch_padded_inf)

                    enhanced_patch_unpadded = enhanced_patch[:, :, :ph, :pw]

                    output_padded[:, :, i:i+patch_size, j:j+patch_size] += enhanced_patch_unpadded * window2d
                    count_map[:, :, i:i+patch_size, j:j+patch_size] += window2d

            output_padded /= (count_map + 1e-8)
            output = output_padded[:, :, :h, :w]

    if device.type == 'cuda': torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    mem_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3) if device.type == 'cuda' else 0

    return torch.clamp(output, 0, 1), elapsed_time, mem_gb


# --- get_hardware_and_model_baseline function ---
def get_hardware_and_model_baseline(
    device: torch.device,
    warmup: int = 20,
    runs: int = 100
) -> Dict[str, float]:
    """
    Calculates and returns hardware performance baseline (ref_time) and
    dynamic model baselines (base_flops, base_params, base_mem).
    All baselines are calculated using a standard lightweight model (MobileNetV2)
    at a specified input size.

    Args:
        device (torch.device): The computation device (e.g., torch.device("cuda")).
        warmup (int): Number of warmup runs.
        runs (int): Number of actual test runs.

    Returns:
        Dict[str, float]: A dictionary containing the baseline metrics.
    """
    baseline_input_size = (1, 3, 3840, 2160)
    if isinstance(device, str):
        device = torch.device(device)
    print(f"\n--- Calculating hardware and model baselines (MobileNetV2 @ {baseline_input_size[2]}x{baseline_input_size[3]}) ---")
    try:
        baseline_model = torchvision.models.mobilenet_v2(weights=None).to(device).eval()
    except (AttributeError, TypeError):
        print("Warning: Using legacy torchvision API to load MobileNetV2.")
        baseline_model = torchvision.models.mobilenet_v2(pretrained=False).to(device).eval()

    dummy_input = torch.randn(*baseline_input_size).to(device)

    base_flops_g, base_params_m = None, None
    base_flops_str, base_params_str = "N/A", "N/A"
    try:
        flops, params = profile(deepcopy(baseline_model), inputs=(dummy_input, ), verbose=False)
        base_flops_g = flops / 1e9
        base_params_m = params / 1e6
        base_flops_str, base_params_str = clever_format([flops, params], "%.3f")
        print(f"Dynamic model baseline set: {base_flops_str}, {base_params_str}")
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"\nWarning: OOM during FLOPs and params evaluation. EEI metrics will switch to FLOPs-less mode.")
        else: raise e

    base_mem_gb = 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = baseline_model(dummy_input)
        base_mem_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        print(f"Baseline memory usage (base_mem): {base_mem_gb:.4f} GB")

    with torch.no_grad():
        for _ in range(warmup):
            _ = baseline_model(dummy_input)
        if device.type == 'cuda': torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(runs):
            _ = baseline_model(dummy_input)
        if device.type == 'cuda': torch.cuda.synchronize()
        total_time = time.time() - start_time

    ref_time = total_time / runs
    print(f"Hardware performance baseline (ref_time): {ref_time * 1000:.3f} ms/frame")
    print("--------------------------------------------------------------------")

    del baseline_model, dummy_input

    return {
        'ref_time': ref_time,
        'base_flops_g': base_flops_g,
        'base_params_m': base_params_m,
        'base_mem_gb': base_mem_gb,
        'base_flops_str': base_flops_str,
        'base_params_str': base_params_str,
    }


def _calculate_model_performance_metrics(
    model: torch.nn.Module,
    device: torch.device,
    input_size: Tuple[int, int, int, int],
    patch_size: Optional[int] = None,
    warmup: int = 10,
    runs: int = 50
) -> Dict[str, float]:
    """
    (Internal function) Calculates performance metrics for a given model in a controlled environment.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device): The computation device.
        input_size (Tuple[int, int, int, int]): The input size for standardized testing.
        patch_size (Optional[int], optional): Fallback patch size for tiled processing
                                              when full-size inference causes an OOM error.
                                              If None, OOM will cause a failure. Defaults to None.
        warmup (int, optional): Number of warmup runs. Defaults to 10.
        runs (int, optional): Number of timing runs. Defaults to 50.

    Returns:
        Dict[str, float]: A dictionary containing the model's performance metrics.
    """
    print(f"\n--- Calculating standardized performance metrics for the current model (input size: {input_size[2]}x{input_size[3]}) ---")

    original_mode = model.training
    model.eval()

    dummy_input = torch.randn(*input_size, device=device)
    # Attempt to calculate FLOPs and params on the full-size input
    try:
        flops, params = profile(deepcopy(model), inputs=(dummy_input, ), verbose=False)
        flops_g = flops / 1e9
        params_m = params / 1e6
        flops_print, params_print = clever_format([flops, params], "%.3f")
        print(f"Static Metrics: FLOPs: {flops_print}, Params: {params_print}")
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"\nWarning: OOM occurred, switching to a mode without param and FLOPs calculation.")
            flops_g, params_m, flops_print, params_print = None, None, "OOM", "OOM"
        else: raise e
    except Exception:
        print(f"\nWarning: thop.profile failed, switching to a mode without param and FLOPs calculation.")
        flops_g, params_m, flops_print, params_print = None, None, "Profile-Error", "Profile-Error"

    avg_time, mem_gb = 0, 0
    with torch.no_grad():
        # Warmup loop
        for _ in range(warmup):
            try:
                # Attempt direct inference
                _ = model(dummy_input)
            except RuntimeError as e:
                # If OOM and a patch-based fallback is provided, use it for warmup
                if 'out of memory' in str(e).lower() and patch_size is not None:
                    print(f"OOM during warmup, switching to patch-based processing (patch_size={patch_size}).")
                    torch.cuda.empty_cache()
                    _, _, _ = process_image_robustly(model, dummy_input, patch_size=patch_size)
                else:
                    # If no fallback or not an OOM error, re-raise
                    raise e

        if device.type == 'cuda': torch.cuda.synchronize()

        # Timing loop
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)

        # Check if patch mode should be used from the start (e.g., OOM detected during warmup)
        use_patch_mode = False
        try:
            # Perform a test run to see if it causes OOM
            _ = model(dummy_input)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and patch_size is not None:
                use_patch_mode = True
                print(f"OOM detected during timing, will use patch-based mode (patch_size={patch_size}) for timing.")
            else:
                raise e

        # Choose the timing loop based on whether patch mode is needed
        if use_patch_mode:
            # Timing in patch-based mode
            torch.cuda.empty_cache() # Clear OOM residuals
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            start_time = time.time()
            for _ in range(runs):
                _, _, _ = process_image_robustly(model, dummy_input, patch_size=patch_size)
        else:
            # Timing in full-size mode
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            start_time = time.time()
            for _ in range(runs):
                _ = model(dummy_input)


        if device.type == 'cuda': torch.cuda.synchronize()
        total_time = time.time() - start_time

        avg_time = total_time / runs
        if device.type == 'cuda': mem_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    print(f"Standardized Dynamic Metrics: Inference Time: {avg_time * 1000:.3f} ms/frame, Peak Memory: {mem_gb:.3f} GB")
    print("-" * 68)

    model.train(original_mode)
    del dummy_input
    return {
        'model_flops_g': flops_g,
        'model_params_m': params_m,
        'model_time': avg_time,
        'model_mem_gb': mem_gb,
        'model_flops_str': flops_print,
        'model_params_str': params_print,
    }


def get_all_perf_metrics(
    model: torch.nn.Module,
    device: torch.device,
    test_img_size: Tuple[int, int, int, int],
    patch_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    One-stop function to calculate all hardware baselines and model performance metrics.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device): The computation device.
        test_img_size (Tuple[int, int, int, int]): The image size for testing the model.
        patch_size (Optional[int], optional): Fallback patch size for tiled processing
                                              when full-size inference causes an OOM error.
                                              If None, OOM will cause a failure. Defaults to None.

    Returns:
        Dict[str, float]: A unified dictionary containing all performance metrics.
    """
    # 1. Get hardware and baseline model (MobileNetV2) metrics.
    baseline_metrics = get_hardware_and_model_baseline(device)

    # 2. Get standardized performance metrics for the current model.
    model_metrics = _calculate_model_performance_metrics(model, device, test_img_size, patch_size=patch_size)

    # 3. Merge all metrics and return.
    all_metrics = {**baseline_metrics, **model_metrics}
    return all_metrics

# --- calculate_EEI function (Unchanged) ---
def calculate_EEI(
    quality_metrics: Dict[str, float],
    perf_metrics: Dict[str, float],
    baseline_size=(1, 3, 3840, 2160),
    test_img_size=(1, 3, 3840, 2160),
    target_fps: float = 30.0
) -> float:
    """
    Calculates the Edge Efficiency Index (EEI), where a lower score is better.
    (New version, accepts dictionaries).

    Args:
        quality_metrics (Dict): A dictionary containing the 'pi' score.
        perf_metrics (Dict): A dictionary with all performance metrics (from get_all_perf_metrics).
        baseline_size: The input size used for the baseline model.
        test_img_size: The input size used for the current model.
        target_fps (float): The target frames per second.

    Returns:
        float: The calculated EEI score.
    """
    pi_score = quality_metrics['pi']

    inference_time = perf_metrics['model_time']
    ref_time = perf_metrics['ref_time']
    flops = perf_metrics['model_flops_g']
    params = perf_metrics['model_params_m']
    mem_usage = perf_metrics['model_mem_gb']

    base_flops = perf_metrics['base_flops_g']
    base_params = perf_metrics['base_params_m']
    base_mem = perf_metrics['base_mem_gb']

    img_res = np.prod(test_img_size)
    base_res_scalar = np.prod(baseline_size)

    target_time = 1.0 / target_fps
    ref_factor = ref_time / target_time
    time_factor = (inference_time * target_fps / ref_factor) * (base_res_scalar / img_res)

    complexity_factor = None
    if flops is not None and params is not None and base_flops is not None and base_params is not None and base_flops > 0 and base_params > 0:
        complexity_factor = (flops / base_flops) * 0.5 + (params / base_params) * 0.5

    resource_factor = mem_usage / base_mem if base_mem > 0 else 0

    if complexity_factor is not None:
        efficiency_factor = (0.8 * time_factor) + (0.1 * complexity_factor) + (0.1 * resource_factor)
    else:
        # If complexity cannot be calculated (e.g., due to OOM), re-weight the other factors.
        efficiency_factor = (0.9 * time_factor) + (0.1 * resource_factor)

    eei = pi_score * efficiency_factor
    return eei