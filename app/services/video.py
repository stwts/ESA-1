import glob
import itertools
import os
import random
import gc
import shutil
from typing import List
from loguru import logger
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
    concatenate_videoclips,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import ImageFont

from app.models import const
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services.utils import video_effects
from app.utils import utils

class SubClippedVideoClip:
    def __init__(self, file_path, start_time=None, end_time=None, width=None, height=None, duration=None):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
video_codec = "libx264"
fps = 30
# Additional duration (seconds) to capture for each segment, used to compensate for the shortening caused by encoding/concatenation.
TEMP_CLIP_EXTRA_SEC = 1.0 / 5.0

def close_clip(clip):
    if clip is None:
        return
        
    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)
            
        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]
        
    for file in files:
        try:
            os.remove(file)
        except:
            pass

def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file and os.path.exists(bgm_file):
        return bgm_file

    if bgm_type == "random":
        suffix = "*.mp3"
        song_dir = utils.song_dir()
        files = glob.glob(os.path.join(song_dir, suffix))
        return random.choice(files)

    return ""


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    subtitle_path: str | None = None,
) -> str:
    audio_clip = AudioFileClip(audio_file)
    audio_duration = audio_clip.duration
    logger.info(f"audio duration: {audio_duration} seconds")
    # Target duration baseline per clip
    # Prefer evenly splitting the audio across selected videos but cap by max_clip_duration
    req_dur = min(audio_duration / max(1, len(video_paths)), max_clip_duration)
    logger.info(f"target per-clip duration (capped): {req_dur} seconds")
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()

    # Calculate the total duration of the original input material (without trimming)
    original_total_duration = 0.0
    try:
        for _vp in video_paths:
            try:
                _c = VideoFileClip(_vp)
                original_total_duration += float(_c.duration)
            except Exception:
                pass
            finally:
                try:
                    close_clip(_c)
                except Exception:
                    pass
    except Exception:
        pass

    processed_clips = []
    subclipped_items = []
    video_duration = 0

    # Try to align with each subtitle segment duration in sequential mode.
    use_segment_alignment = (
        subtitle_path is not None
        and os.path.exists(subtitle_path)
        and video_concat_mode.value == VideoConcatMode.sequential.value
    )

    def _parse_srt_durations(srt_path: str) -> List[float]:
        durations: List[float] = []
        try:
            with open(srt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return durations

        import re

        time_pattern = re.compile(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})"
        )
        for line in lines:
            m = time_pattern.search(line)
            if not m:
                continue
            h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
            t1 = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
            t2 = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
            if t2 > t1:
                durations.append(t2 - t1)
        return durations

    if use_segment_alignment:
        target_durations = _parse_srt_durations(subtitle_path)
        if not target_durations:
            logger.warning("Subtitle parsing failed or is empty, falling back to the original trimming logic.")
            use_segment_alignment = False

    if use_segment_alignment:
        logger.info("Align segment by segment according to subtitle duration (shorten the target duration of the segment when the video is insufficient).")
        n = min(len(video_paths), len(target_durations))
        for i in range(n):
            video_path = video_paths[i]
            
            clip = VideoFileClip(video_path)
            clip_duration = clip.duration
            clip_w, clip_h = clip.size
            close_clip(clip)

            desired = float(target_durations[i])
            
            effective = min(clip_duration, desired + TEMP_CLIP_EXTRA_SEC)
            if clip_duration < desired:
                logger.warning(
                    f"Paragraph {i+1}: Original video duration {clip_duration:.2f}s < Duration of dubbing {desired:.2f}s, cut off the audio segment as {effective:.2f}s（By shortening the target segment）。"
                )

            start_time = 0.0
            end_time = effective
            subclipped_items.append(
                SubClippedVideoClip(
                    file_path=video_path,
                    start_time=start_time,
                    end_time=end_time,
                    width=clip_w,
                    height=clip_h,
                )
            )

        
        audio_duration = sum([item.end_time - item.start_time for item in subclipped_items])
        logger.info(
            f"After segment alignment: Total video duration = {audio_duration:.2f}s (used for dubbing alignment)."
        )

    else:
       
        remaining_audio_for_planning = audio_duration
        for i, video_path in enumerate(video_paths):
            clip = VideoFileClip(video_path)
            clip_duration = clip.duration
            clip_w, clip_h = clip.size
            close_clip(clip)

            logger.info(f"Processing video {i+1}: {video_path}, duration : {clip_duration:.2f}s")

            if video_concat_mode.value == VideoConcatMode.sequential.value:
               
                segments_left = len(video_paths) - i
                ideal = remaining_audio_for_planning / max(1, segments_left)
                cap = min(max_clip_duration, clip_duration)
                seg_duration = max(0.0, min(cap, ideal))

                start_time = 0.0
                
                end_time = min(clip_duration, seg_duration + TEMP_CLIP_EXTRA_SEC)
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=clip_w,
                        height=clip_h,
                    )
                )
                logger.info(
                    f"  sequential mode，target fragment: {start_time:.2f}s - {end_time:.2f}s (ideal: {ideal:.2f}s, superior limit: {cap:.2f}s)"
                )
                remaining_audio_for_planning = max(0.0, remaining_audio_for_planning - seg_duration)
            else:
                
                start_time = 0.0
                clips_generated = 0
                while start_time < clip_duration:
                    
                    end_time = min(start_time + max_clip_duration + TEMP_CLIP_EXTRA_SEC, clip_duration)
                    if clip_duration - start_time > 0:
                        subclipped_items.append(
                            SubClippedVideoClip(
                                file_path=video_path,
                                start_time=start_time,
                                end_time=end_time,
                                width=clip_w,
                                height=clip_h,
                            )
                        )
                        clips_generated += 1
                        logger.info(
                            f"  Generating clips {clips_generated}: {start_time:.2f}s - {end_time:.2f}s"
                        )
                    start_time = end_time

    # random subclipped_items order（段对齐时不打乱，保持与字幕一致）
    if (not use_segment_alignment) and video_concat_mode.value == VideoConcatMode.random.value:
        random.shuffle(subclipped_items)
        
    logger.debug(f"total subclipped items: {len(subclipped_items)}")
    logger.info(f"Audio duration: {audio_duration}s, Maximum fragment duration: {max_clip_duration}s")
    logger.info(f"Number of input videos: {len(video_paths)}, Number of generated subfragments: {len(subclipped_items)}")
    
    
    for i, subclipped_item in enumerate(subclipped_items):
        remaining = audio_duration - video_duration
        if remaining <= 1e-3:
            break

        logger.debug(
            f"processing clip {i+1}: {subclipped_item.width}x{subclipped_item.height}, current duration: {video_duration:.2f}s, remaining: {remaining:.2f}s"
        )

        try:
            clip = (
                VideoFileClip(subclipped_item.file_path)
                .subclipped(subclipped_item.start_time, subclipped_item.end_time)
            )
            clip_duration = clip.duration
            
            if clip_duration > remaining + 1e-3:
                truncate_to = min(clip_duration, max(0.0, remaining + TEMP_CLIP_EXTRA_SEC))
                clip = clip.subclipped(0, truncate_to)
                clip_duration = clip.duration
            # Not all videos are same size, so we need to resize them
            clip_w, clip_h = clip.size
            if clip_w != video_width or clip_h != video_height:
                clip_ratio = clip.w / clip.h
                video_ratio = video_width / video_height
                logger.debug(f"resizing clip, source: {clip_w}x{clip_h}, ratio: {clip_ratio:.2f}, target: {video_width}x{video_height}, ratio: {video_ratio:.2f}")
                
                if clip_ratio == video_ratio:
                    clip = clip.resized(new_size=(video_width, video_height))
                else:
                    if clip_ratio > video_ratio:
                        scale_factor = video_width / clip_w
                    else:
                        scale_factor = video_height / clip_h

                    new_width = int(clip_w * scale_factor)
                    new_height = int(clip_h * scale_factor)

                    background = ColorClip(size=(video_width, video_height), color=(0, 0, 0)).with_duration(clip_duration)
                    clip_resized = clip.resized(new_size=(new_width, new_height)).with_position("center")
                    clip = CompositeVideoClip([background, clip_resized])
                    
            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            if video_transition_mode.value == VideoTransitionMode.none.value:
                clip = clip
            elif video_transition_mode.value == VideoTransitionMode.fade_in.value:
                clip = video_effects.fadein_transition(clip, 1)
            elif video_transition_mode.value == VideoTransitionMode.fade_out.value:
                clip = video_effects.fadeout_transition(clip, 1)
            elif video_transition_mode.value == VideoTransitionMode.slide_in.value:
                clip = video_effects.slidein_transition(clip, 1, shuffle_side)
            elif video_transition_mode.value == VideoTransitionMode.slide_out.value:
                clip = video_effects.slideout_transition(clip, 1, shuffle_side)
            elif video_transition_mode.value == VideoTransitionMode.shuffle.value:
                transition_funcs = [
                    lambda c: video_effects.fadein_transition(c, 1),
                    lambda c: video_effects.fadeout_transition(c, 1),
                    lambda c: video_effects.slidein_transition(c, 1, shuffle_side),
                    lambda c: video_effects.slideout_transition(c, 1, shuffle_side),
                ]
                shuffle_transition = random.choice(transition_funcs)
                clip = shuffle_transition(clip)

            
            if (not use_segment_alignment) and clip.duration > (max_clip_duration + TEMP_CLIP_EXTRA_SEC + 1e-3):
                clip = clip.subclipped(0, max_clip_duration + TEMP_CLIP_EXTRA_SEC)
                
            # wirte clip to temp file
            clip_file = f"{output_dir}/temp-clip-{i+1}.mp4"
            clip.write_videofile(clip_file, logger=None, fps=fps, codec=video_codec)
            
            close_clip(clip)
        
            processed_clips.append(
                SubClippedVideoClip(
                    file_path=clip_file, duration=clip.duration, width=clip_w, height=clip_h
                )
            )
            video_duration += clip.duration
            logger.info(
                f"Generate temp-clip {i+1}: {clip_file}, duration: {clip.duration:.2f}s, cumulative duration: {video_duration:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"failed to process clip: {str(e)}")
    

    if (not use_segment_alignment) and video_duration < audio_duration:
        logger.warning(
            f"video duration ({video_duration:.2f}s) is shorter than audio duration ({audio_duration:.2f}s). Skipping loop-fill to avoid unintended duplication."
        )
     

    logger.info("starting clip merging process")
    if not processed_clips:
        logger.warning("no clips available for merging")
        logger.info(
            f"Duration Statistics => Original material total duration: {original_total_duration:.2f}s, Total duration after trimming: {video_duration:.2f}s, Audio duration: {audio_duration:.2f}s"
        )
        return combined_video_path
    
    # if there is only one clip, use it directly
    if len(processed_clips) == 1:
        logger.info("using single clip directly")
        shutil.copy(processed_clips[0].file_path, combined_video_path)
        # 清理生成的临时片段文件
        delete_files([processed_clips[0].file_path])
        logger.info(
            f"Duration Statistics => Original material total duration: {original_total_duration:.2f}s, Total duration after trimming: {video_duration:.2f}s, Audio duration: {audio_duration:.2f}s"
        )
        logger.info("video combining completed")
        return combined_video_path
    
    # create initial video file as base
    base_clip_path = processed_clips[0].file_path
    temp_merged_video = f"{output_dir}/temp-merged-video.mp4"
    temp_merged_next = f"{output_dir}/temp-merged-next.mp4"
    
    # copy first clip as initial merged video
    shutil.copy(base_clip_path, temp_merged_video)
    
    # merge remaining video clips one by one
    for i, clip in enumerate(processed_clips[1:], 1):
        logger.info(f"merging clip {i}/{len(processed_clips)-1}, duration: {clip.duration:.2f}s")
        
        try:
            # load current base video and next clip to merge
            base_clip = VideoFileClip(temp_merged_video)
            next_clip = VideoFileClip(clip.file_path)
            
            # merge these two clips
            merged_clip = concatenate_videoclips([base_clip, next_clip])

            # save merged result to temp file
            merged_clip.write_videofile(
                filename=temp_merged_next,
                threads=threads,
                logger=None,
                temp_audiofile_path=output_dir,
                audio_codec=audio_codec,
                fps=fps,
            )
            close_clip(base_clip)
            close_clip(next_clip)
            close_clip(merged_clip)
            
            # replace base file with new merged file
            delete_files(temp_merged_video)
            os.rename(temp_merged_next, temp_merged_video)
            
        except Exception as e:
            logger.error(f"failed to merge clip: {str(e)}")
            continue
    
    # after merging, rename final result to target file name
    os.rename(temp_merged_video, combined_video_path)
    
    # clean temp files
    clip_files = [clip.file_path for clip in processed_clips]
    delete_files(clip_files)
            
    # 输出时长统计
    logger.info(
        f"Duration Statistics => Original material total duration: {original_total_duration:.2f}s, Total duration after trimming: {video_duration:.2f}s, Audio duration: {audio_duration:.2f}s"
    )
 
    logger.info("video combining completed")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # Create ImageFont
    font = ImageFont.truetype(font, fontsize)

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        left, top, right, bottom = font.getbbox(inner_text)
        return right - left, bottom - top

    width, height = get_text_size(text)
    if width <= max_width:
        return text, height

    processed = True

    _wrapped_lines_ = []
    words = text.split(" ")
    _txt_ = ""
    for word in words:
        _before = _txt_
        _txt_ += f"{word} "
        _width, _height = get_text_size(_txt_)
        if _width <= max_width:
            continue
        else:
            if _txt_.strip() == word.strip():
                processed = False
                break
            _wrapped_lines_.append(_before)
            _txt_ = f"{word} "
    _wrapped_lines_.append(_txt_)
    if processed:
        _wrapped_lines_ = [line.strip() for line in _wrapped_lines_]
        result = "\n".join(_wrapped_lines_).strip()
        height = len(_wrapped_lines_) * height
        return result, height

    _wrapped_lines_ = []
    chars = list(text)
    _txt_ = ""
    for word in chars:
        _txt_ += word
        _width, _height = get_text_size(_txt_)
        if _width <= max_width:
            continue
        else:
            _wrapped_lines_.append(_txt_)
            _txt_ = ""
    _wrapped_lines_.append(_txt_)
    result = "\n".join(_wrapped_lines_).strip()
    height = len(_wrapped_lines_) * height
    return result, height


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
):
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # https://github.com/harry0703/MoneyPrinterTurbo/issues/217
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3'
    # write into the same directory as the output file
    output_dir = os.path.dirname(output_file)

    font_path = ""
    if params.subtitle_enabled:
        if not params.font_name:
            params.font_name = "STHeitiMedium.ttc"
        font_path = os.path.join(utils.font_dir(), params.font_name)
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    def create_text_clip(subtitle_item):
        params.font_size = int(params.font_size)
        params.stroke_width = int(params.stroke_width)
        phrase = subtitle_item[1]
        max_width = video_width * 0.9
        wrapped_txt, txt_height = wrap_text(
            phrase, max_width=max_width, font=font_path, fontsize=params.font_size
        )
        interline = int(params.font_size * 0.25)
        size=(int(max_width), int(txt_height + params.font_size * 0.25 + (interline * (wrapped_txt.count("\n") + 1))))

        _clip = TextClip(
            text=wrapped_txt,
            font=font_path,
            font_size=params.font_size,
            color=params.text_fore_color,
            bg_color=params.text_background_color,
            stroke_color=params.stroke_color,
            stroke_width=params.stroke_width,
            # interline=interline,
            # size=size,
        )
        duration = subtitle_item[0][1] - subtitle_item[0][0]
        _clip = _clip.with_start(subtitle_item[0][0])
        _clip = _clip.with_end(subtitle_item[0][1])
        _clip = _clip.with_duration(duration)
        if params.subtitle_position == "bottom":
            _clip = _clip.with_position(("center", video_height * 0.95 - _clip.h))
        elif params.subtitle_position == "top":
            _clip = _clip.with_position(("center", video_height * 0.05))
        elif params.subtitle_position == "custom":
            # Ensure the subtitle is fully within the screen bounds
            margin = 10  # Additional margin, in pixels
            max_y = video_height - _clip.h - margin
            min_y = margin
            custom_y = (video_height - _clip.h) * (params.custom_position / 100)
            custom_y = max(
                min_y, min(custom_y, max_y)
            )  # Constrain the y value within the valid range
            _clip = _clip.with_position(("center", custom_y))
        else:  # center
            _clip = _clip.with_position(("center", "center"))
        return _clip

    video_clip = VideoFileClip(video_path).without_audio()
    audio_clip = AudioFileClip(audio_path).with_effects(
        [afx.MultiplyVolume(params.voice_volume)]
    )

    def make_textclip(text):
        return TextClip(
            text=text,
            font=font_path,
            font_size=params.font_size,
        )

    if subtitle_path and os.path.exists(subtitle_path):
        sub = SubtitlesClip(
            subtitles=subtitle_path, encoding="utf-8", make_textclip=make_textclip
        )
        text_clips = []
        for item in sub.subtitles:
            clip = create_text_clip(subtitle_item=item)
            text_clips.append(clip)
        video_clip = CompositeVideoClip([video_clip, *text_clips])

    bgm_file = get_bgm_file(bgm_type=params.bgm_type, bgm_file=params.bgm_file)
    if bgm_file:
        try:
            bgm_clip = AudioFileClip(bgm_file).with_effects(
                [
                    afx.MultiplyVolume(params.bgm_volume),
                    afx.AudioFadeOut(3),
                    afx.AudioLoop(duration=video_clip.duration),
                ]
            )
            audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
        except Exception as e:
            logger.error(f"failed to add bgm: {str(e)}")

    video_clip = video_clip.with_audio(audio_clip)
    video_clip.write_videofile(
        output_file,
        audio_codec=audio_codec,
        temp_audiofile_path=output_dir,
        threads=params.n_threads or 2,
        logger=None,
        fps=fps,
    )
    video_clip.close()
    del video_clip


def preprocess_video(materials: List[MaterialInfo], clip_duration=4):
    for material in materials:
        if not material.url:
            continue

        ext = utils.parse_extension(material.url)
        try:
            clip = VideoFileClip(material.url)
        except Exception:
            clip = ImageClip(material.url)

        width = clip.size[0]
        height = clip.size[1]
        if width < 480 or height < 480:
            logger.warning(f"low resolution material: {width}x{height}, minimum 480x480 required")
            continue

        if ext in const.FILE_TYPE_IMAGES:
            logger.info(f"processing image: {material.url}")
            # Create an image clip and set its duration to 3 seconds
            clip = (
                ImageClip(material.url)
                .with_duration(clip_duration)
                .with_position("center")
            )
            # Apply a zoom effect using the resize method.
            # A lambda function is used to make the zoom effect dynamic over time.
            # The zoom effect starts from the original size and gradually scales up to 120%.
            # t represents the current time, and clip.duration is the total duration of the clip (3 seconds).
            # Note: 1 represents 100% size, so 1.2 represents 120% size.
            zoom_clip = clip.resized(
                lambda t: 1 + (clip_duration * 0.03) * (t / clip.duration)
            )

            # Optionally, create a composite video clip containing the zoomed clip.
            # This is useful when you want to add other elements to the video.
            final_clip = CompositeVideoClip([zoom_clip])

            # Output the video to a file.
            video_file = f"{material.url}.mp4"
            final_clip.write_videofile(video_file, fps=30, logger=None)
            close_clip(clip)
            material.url = video_file
            logger.success(f"image processed: {video_file}")
    return materials