import os
import cv2
import numpy as np
import subprocess
import json
from pathlib import Path

###############################################################################
#                     PER-FRAME PYTHON CORRUPTIONS                            #
###############################################################################

def shot_noise(frame, severity=1):
    """
    Implements the same shot noise as in create_mini_ssv2_c.py
    c = [60, 25, 12, 5, 3][severity - 1]
    """
    c_vals = [60, 25, 12, 5, 3]  # severity 1..5
    c = c_vals[severity - 1]
    
    x = np.array(frame, dtype=np.float32) / 255.0
    # Poisson noise
    corrupted = np.random.poisson(x * c) / float(c)
    corrupted = np.clip(corrupted, 0, 1) * 255
    return corrupted.astype(np.uint8)


def generate_random_lines(imshape, slant, drop_length, rain_type):
    """Helper used by the 'rain' function."""
    drops = []
    area = imshape[0] * imshape[1]
    # base number of drops
    no_of_drops = area // 600  

    if rain_type == 1:
        no_of_drops = area // 770
        drop_length = 10
    elif rain_type == 2:
        no_of_drops = area // 770
        drop_length = 30
    elif rain_type == 3:
        no_of_drops = area // 770
        drop_length = 60
    elif rain_type == 4:
        no_of_drops = area // 500
        drop_length = 60
    elif rain_type == 5:
        no_of_drops = area // 400
        drop_length = 80

    for _ in range(no_of_drops):
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x, y))
    return drops, drop_length

def rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops, darken):
    """
    Helper used by the 'rain' function to overlay the rainy lines.
    """
    imshape = image.shape
    rain_mask = np.zeros((imshape[0], imshape[1]), dtype=np.uint8)

    for rd in rain_drops:
        cv2.line(rain_mask, (rd[0], rd[1]),
                 (rd[0] + slant, rd[1] + drop_length),
                 drop_color, drop_width)

    # Expand mask to 3 channels
    rain_mask_3c = np.stack((rain_mask, rain_mask, rain_mask), axis=2)
    # Combine with original
    image_rain = image + (rain_mask_3c * (1 - image / 255.0) * (1 - np.mean(image) / 255.0)).astype(np.uint8)
    # Blur
    blur_rain = cv2.blur(image_rain, (3, 3))

    combined = (blur_rain * (rain_mask_3c / 255.0)) + (image * (1 - (rain_mask_3c / 255.0)))
    combined = combined / 255.0

    means = np.mean(combined, axis=(0, 1), keepdims=True)
    darkened = np.clip((combined - means) * darken + means, 0, 1) * 255
    return darkened.astype(np.uint8)

def rain(frame, severity=2):
    """
    Matches the logic in create_mini_ssv2_c.py
    Severity in [1..5].
    """
    image = np.asarray(frame, dtype=np.uint8)
    slant = -1
    drop_length = 20
    drop_width = 1
    drop_color = (220, 220, 220)
    darken_coefficient = [0.8, 0.8, 0.7, 0.6, 0.5]  # severity mapping

    if slant == -1:
        slant = np.random.randint(-10, 10)

    rain_drops, drop_length = generate_random_lines(image.shape, slant, drop_length, severity)
    output = rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops,
                          darken_coefficient[severity - 1])
    return output


def contrast(frame, severity=1):
    """
    c = [0.5, 0.4, 0.3, 0.2, 0.1][severity - 1]
    """
    c_vals = [0.5, 0.4, 0.3, 0.2, 0.1]
    c = c_vals[severity - 1]

    x = np.array(frame, dtype=np.float32) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    x = (x - means) * c + means
    x = np.clip(x, 0, 1) * 255
    return x.astype(np.uint8)


def brightness(frame, severity=1):
    """
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    """
    c_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    c = c_vals[severity - 1]

    x = np.array(frame, dtype=np.float32) / 255.0
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + c, 0, 1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = np.clip(rgb, 0, 1) * 255
    return rgb.astype(np.uint8)


def saturate(frame, severity=1):
    """
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    """
    c_list = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)]
    sat_mult, sat_add = c_list[severity - 1]

    x = np.array(frame, dtype=np.float32) / 255.0
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult + sat_add, 0, 1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = np.clip(rgb, 0, 1) * 255
    return rgb.astype(np.uint8)


def motion_blur(frames, severity=1):
    """
    motion_overlapping_frames = [1,2,3,4,6] from create_mini_ssv2_c.py
    We'll apply a simple averaging across 2c frames.
    Returns a *shorter* clip because we skip the first/last c frames.
    """
    motion_vals = [1, 2, 3, 4, 6]
    c = motion_vals[severity - 1]

    blurred = []
    for i in range(c, len(frames) - c):
        accum = np.zeros_like(frames[i], dtype=np.float32)
        for j in range(i - c, i + c):
            accum += frames[j].astype(np.float32)
        accum /= (2.0 * c)

        blurred.append(accum.astype(np.uint8))
    return blurred


def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    As in create_mini_ssv2_c.py. Generates a 2D array in [0..1].
    """
    assert (mapsize & (mapsize - 1) == 0), "mapsize must be a power of two"
    maparray = np.empty((mapsize, mapsize), dtype=np.float32)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, -1, axis=0)
        squareaccum += np.roll(squareaccum, -1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)

        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    maparray /= maparray.max()
    return maparray

def fog(frames, severity=1):
    """
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    Applies 'fog' across the entire frame by blending a fractal map.
    """
    c_list = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)]
    factor, wdecay = c_list[severity - 1]

    # Generate fractal map
    fog_map = factor * plasma_fractal(mapsize=256, wibbledecay=wdecay)

    fogged = []
    for frame in frames:
        f = frame.astype(np.float32) / 255.0
        h, w, _ = f.shape

        # Resize the fog_map to match the entire frame
        resized_fog = cv2.resize(fog_map, (w, h), interpolation=cv2.INTER_LINEAR)
        resized_fog_3c = np.dstack([resized_fog]*3)

        max_val = f.max()
        # Add fog
        f += resized_fog_3c
        # scale => f * (max_val / (max_val + factor))
        f = np.clip(f * (max_val / (max_val + factor)), 0, 1)

        f = np.clip(f, 0, 1) * 255
        fogged.append(f.astype(np.uint8))

    return fogged

###############################################################################
#                           FFmpeg-BASED CORRUPTIONS                          #
###############################################################################

def ffmpeg_bit_error(src, dst, severity):
    """
    bit_error: uses the -bsf noise param
    c=[100000, 50000, 30000, 20000, 10000][severity-1]
    """
    c_vals = [100000, 50000, 30000, 20000, 10000]
    noise_val = c_vals[severity - 1]

    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c", "copy", 
        "-bsf", f"noise={noise_val}",
        str(dst)
    ]
    return subprocess.call(cmd)


def ffmpeg_h264_crf(src, dst, severity):
    """
    h264_crf c=[23,30,37,44,51][severity-1]
    """
    c_vals = [23, 30, 37, 44, 51]
    crf = c_vals[severity - 1]
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'",
        "-vcodec", "libx264",
        "-crf", str(crf),
        str(dst)
    ]
    return subprocess.call(cmd)


def ffmpeg_h264_abr(src, dst, severity):
    """
    h264_abr c=[2,4,8,16,32][severity-1]
    We'll read original video bit_rate from ffprobe and divide by c.
    """
    c_vals = [2,4,8,16,32]
    div_factor = c_vals[severity - 1]

    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(src)
    ]
    result = subprocess.Popen(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    data = json.load(result.stdout)

    original_br = float(data['format'].get('bit_rate', 1000000))
    new_br = int(original_br / div_factor)
    new_br_str = str(new_br)

    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'",
        "-vcodec", "libx264",
        "-b:v", new_br_str,
        "-maxrate", new_br_str,
        "-bufsize", new_br_str,
        str(dst)
    ]
    return subprocess.call(cmd)


def ffmpeg_h265_crf(src, dst, severity):
    """
    h265_crf c = [27, 33, 39, 45, 51][severity - 1]
    """
    c_vals = [27, 33, 39, 45, 51]
    crf = c_vals[severity - 1]
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'",
        "-vcodec", "libx265",
        "-crf", str(crf),
        str(dst)
    ]
    return subprocess.call(cmd)


def ffmpeg_h265_abr(src, dst, severity):
    """
    h265_abr c=[2,4,8,16,32][severity-1]
    """
    c_vals = [2,4,8,16,32]
    div_factor = c_vals[severity - 1]

    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(src)
    ]
    result = subprocess.Popen(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    data = json.load(result.stdout)

    original_br = float(data['format'].get('bit_rate', 1000000))
    new_br = int(original_br / div_factor)
    new_br_str = str(new_br)

    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'",
        "-vcodec", "libx265",
        "-b:v", new_br_str,
        "-maxrate", new_br_str,
        "-bufsize", new_br_str,
        str(dst)
    ]
    return subprocess.call(cmd)


def ffmpeg_frame_rate(src, dst, severity):
    """
    frame_rate c=[10,8,6,4,2][severity-1]
    """
    c_vals = [10, 8, 6, 4, 2]
    fps = c_vals[severity - 1]
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'",
        "-vcodec", "libx265",
        "-fps", str(fps),
        str(dst)
    ]
    return subprocess.call(cmd)

###############################################################################
#                           MAIN LOGIC (corruptAll)                           #
###############################################################################

PER_FRAME_FUNCS = {
    "shot_noise":      shot_noise,
    "rain":            rain,
    "contrast":        contrast,
    "brightness":      brightness,
    "saturate":        saturate
}

CLIP_FUNCS = {
    "motion_blur":     motion_blur,
    "fog":             fog
}

FFMPEG_FUNCS = {
    "bit_error":       ffmpeg_bit_error,
    "h264_crf":        ffmpeg_h264_crf,
    "h264_abr":        ffmpeg_h264_abr,
    "h265_crf":        ffmpeg_h265_crf,
    "h265_abr":        ffmpeg_h265_abr,
    "frame_rate":      ffmpeg_frame_rate
}

def apply_per_frame_corruption(frames, func, severity):
    """Applies a single-image function like shot_noise or rain to each frame."""
    corrupted = []
    for f in frames:
        out = func(f, severity=severity)
        corrupted.append(out)
    return corrupted

def apply_clip_corruption(frames, func, severity):
    """For clip-level transformations (motion_blur, fog)."""
    return func(frames, severity=severity)

def apply_ffmpeg_corruption(input_path, output_path, func, severity):
    """Calls the FFmpeg-based corruption."""
    return_code = func(input_path, output_path, severity)
    return return_code

def process_videos(
    input_folder, output_folder,
    corruptions = [
        "shot_noise","rain","contrast","brightness","saturate",
        "motion_blur","fog",
        "bit_error","h264_crf","h264_abr","h265_crf","h265_abr","frame_rate"
    ],
    severity=3,
    start_index=0
):
    """
    Iterate over .mp4/.avi in 'input_folder', apply each corruption
    at the given 'severity' level, and save to 'output_folder'.

    - severity=3 by default
    - We do not resize or crop frames for the per-frame corruptions,
      so we preserve original resolution in Python-based transformations.
    """
    os.makedirs(output_folder, exist_ok=True)

    video_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    video_files.sort()  # for consistent order

    for idx, video_file in enumerate(video_files):
        if idx < start_index:
            continue

        input_path = os.path.join(input_folder, video_file)
        print(f"[{idx}] Processing {video_file} with severity={severity}...")

        # Load frames with OpenCV if we need to apply per-frame or clip-level corruption
        cap = cv2.VideoCapture(input_path)
        orig_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            orig_frames.append(frame)
        cap.release()

        for corruption in corruptions:
            corr_output_folder = os.path.join(output_folder, corruption)
            os.makedirs(corr_output_folder, exist_ok=True)
            output_path = os.path.join(corr_output_folder, video_file)

            if corruption in PER_FRAME_FUNCS:
                func = PER_FRAME_FUNCS[corruption]
                corrupted_frames = apply_per_frame_corruption(orig_frames, func, severity)

                if corrupted_frames:
                    height, width, _ = corrupted_frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # You can detect original FPS if you want:
                    #   original_fps = cap.get(cv2.CAP_PROP_FPS)
                    # but let's default to 24 or the original_fps if it was read before.
                    out_vid = cv2.VideoWriter(output_path, fourcc, 24, (width, height))
                    for f in corrupted_frames:
                        out_vid.write(f)
                    out_vid.release()

                print(f"  -> Saved {corruption} to {output_path}")

            elif corruption in CLIP_FUNCS:
                func = CLIP_FUNCS[corruption]
                corrupted_frames = apply_clip_corruption(orig_frames, func, severity)

                if corrupted_frames:
                    height, width, _ = corrupted_frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_vid = cv2.VideoWriter(output_path, fourcc, 24, (width, height))
                    for f in corrupted_frames:
                        out_vid.write(f)
                    out_vid.release()

                print(f"  -> Saved {corruption} to {output_path}")

            elif corruption in FFMPEG_FUNCS:
                # For FFmpeg corruptions, we don't do Python-based manipulation.
                # We simply call ffmpeg to produce the output file.
                return_code = apply_ffmpeg_corruption(
                    input_path, output_path, FFMPEG_FUNCS[corruption], severity
                )
                if return_code != 0:
                    print(f"  !! FFmpeg error creating {output_path}")
                else:
                    print(f"  -> Saved {corruption} to {output_path}")

            else:
                print(f"Corruption '{corruption}' not recognized. Skipping.")

        print(f"Finished processing {video_file}.\n")


if __name__ == "__main__":
    # Example usage
    input_folder = "../XD-Violence"
    output_folder = "../XD-Violence-Corrupted-3"

    all_corruptions = ["shot_noise","rain","contrast","brightness","saturate",
        "motion_blur","fog",
        "bit_error","frame_rate"
    ]#"h264_crf","h264_abr","h265_crf","h265_abr"

    process_videos(
        input_folder, 
        output_folder,
        corruptions=all_corruptions,
        severity=3, 
        start_index=0
    )
