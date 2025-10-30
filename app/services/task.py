import math
import os.path
import re
from os import path

from loguru import logger
import sys

# Before running the following command, make sure to modify the BASE_PATH parameter according to your actual environment.
BASE_PATH = os.path.join(os.path.sep, "mnt", "ESA")

if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams, MaterialInfo
from app.services import llm, material, subtitle, video, voice
from app.services import state as sm
from app.utils import utils

# Generate video script (subtitles).

# If the user does not provide a script (params.video_script), call the llm.generate_script function.
def generate_script(task_id, params):
    logger.info("\n\n## generating video script")
    video_script = params.video_script.strip()
    if not video_script:
        video_script = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
        )
    else:
        logger.debug(f"video script: \n{video_script}")

    if not video_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video script.")
        return None

    return video_script


def generate_terms(task_id, params, video_script):
    logger.info("\n\n## generating video terms")
    video_terms = params.video_terms
    if not video_terms:
        video_terms = llm.generate_terms(
            video_subject=params.video_subject, video_script=video_script, amount=5
        )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video terms.")
        return None

    return video_terms


def save_script_data(task_id, video_script, video_terms, params):
    script_file = path.join(utils.task_dir(task_id), "script.json")
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }

    with open(script_file, "w", encoding="utf-8") as f:
        f.write(utils.to_json(script_data))


def generate_audio(task_id, params, video_script):
    logger.info("\n\n## generating audio")
    audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
    sub_maker = voice.tts(
        text=video_script,
        voice_name=voice.parse_voice_name(params.voice_name),
        voice_rate=params.voice_rate,
        voice_file=audio_file,
    )
    if sub_maker is None:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error(
            """failed to generate audio:
1. check if the language of the voice matches the language of the video script.
2. check if the network is available. If you are in China, it is recommended to use a VPN and enable the global traffic mode.
        """.strip()
        )
        return None, None, None

    audio_duration = math.ceil(voice.get_audio_duration(sub_maker))
    return audio_file, audio_duration, sub_maker


def generate_subtitle(task_id, params, video_script, sub_maker, audio_file):
    if not params.subtitle_enabled:
        return ""

    subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    subtitle_fallback = False
    if subtitle_provider == "edge":
        voice.create_subtitle(
            text=video_script, sub_maker=sub_maker, subtitle_file=subtitle_path
        )
        if not os.path.exists(subtitle_path):
            subtitle_fallback = True
            logger.warning("subtitle file not found, fallback to whisper")

    if subtitle_provider == "whisper" or subtitle_fallback:
        subtitle.create(audio_file=audio_file, subtitle_file=subtitle_path)
        logger.info("\n\n## correcting subtitle")
        subtitle.correct(subtitle_file=subtitle_path, video_script=video_script)

    subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
    if not subtitle_lines:
        logger.warning(f"subtitle file is invalid: {subtitle_path}")
        return ""

    return subtitle_path


def get_video_materials(task_id, params, video_terms, audio_duration):
    if params.video_source == "local":
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials, clip_duration=params.video_clip_duration
        )
        if not materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "no valid materials found, please check the materials and try again."
            )
            return None
        return [material_info.url for material_info in materials]
    else:
        logger.info(f"\n\n## downloading videos from {params.video_source}")
        downloaded_videos = material.download_videos(
            task_id=task_id,
            search_terms=video_terms,
            source=params.video_source,
            video_aspect=params.video_aspect,
            video_contact_mode=params.video_concat_mode,
            audio_duration=audio_duration * params.video_count,
            max_clip_duration=params.video_clip_duration,
        )
        if not downloaded_videos:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "failed to download videos, maybe the network is not available. if you are in China, please use a VPN."
            )
            return None
        return downloaded_videos


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path
):
    final_video_paths = []
    combined_video_paths = []
    video_concat_mode = (
        params.video_concat_mode if params.video_count == 1 else VideoConcatMode.random
    )
    video_transition_mode = params.video_transition_mode

    _progress = 50
    for i in range(params.video_count):
        index = i + 1
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## combining video: {index} => {combined_video_path}")
        video.combine_videos(
            combined_video_path=combined_video_path,
            video_paths=downloaded_videos,
            audio_file=audio_file,
            video_aspect=params.video_aspect,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_path = path.join(utils.task_dir(task_id), f"final-{index}.mp4")

        logger.info(f"\n\n## generating video: {index} => {final_video_path}")
        video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)

    return final_video_paths, combined_video_paths


def choose_video(params: VideoParams):
    from app.lang.code.langevin_turbo import langevin_
    # Ensure text_list is not None, use an empty list as fallback
    text_list = params.text_list if params.text_list is not None else []
    downloaded_videos = langevin_(num_select=params.num_select, num_candidate_video=params.num_candidate_video, text_list=text_list)
    return downloaded_videos

def start(task_id, params: VideoParams, stop_at: str = "video"):
    logger.info(f"start task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=5)

    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    # 1. Generate script
    video_script = generate_script(task_id, params)
    if not video_script or "Error: " in video_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
        )
        return {"script": video_script}

    # 2. Generate terms
    video_terms = ""
    if params.video_source != "local":
        video_terms = generate_terms(task_id, params, video_script)
        if not video_terms:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            return

    save_script_data(task_id, video_script, video_terms, params)

    if stop_at == "terms":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": video_script, "terms": video_terms}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    # 3. Generate audio
    audio_file, audio_duration, sub_maker = generate_audio(
        task_id, params, video_script
    )
    if not audio_file:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 4. Generate subtitle
    subtitle_path = generate_subtitle(
        task_id, params, video_script, sub_maker, audio_file
    )

    if stop_at == "subtitle":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    # 5. Preprocess video materials: auto-generate local file list if not provided
    if params.video_source == "local" and (params.video_materials is None or len(params.video_materials) == 0):
        base_path = "app/lang/dataset/candidate_video/segment_"
        file_extension = ".mp4"
        params.video_materials = [
            MaterialInfo(
                provider="local",
                url=f"{base_path}{i:03d}{file_extension}",
                duration=0
            ) for i in range(1, params.num_candidate_video + 1)
        ]
        logger.info(f"Auto-generated {len(params.video_materials)} local video materials.")

    # Get video materials (This will be overridden by Langevin optimization)
    original_downloaded_videos = get_video_materials(
        task_id, params, video_terms, audio_duration
    )
    if not original_downloaded_videos:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    # -------------------------- Semantic matching is enabled here --------------------------
    try:
        _semantic_videos = choose_video(params)
        downloaded_videos = _semantic_videos if _semantic_videos else original_downloaded_videos
    except Exception as e:
        logger.warning(f"Semantic matching failed, falling back to original material: {e}")
        downloaded_videos = original_downloaded_videos
    logger.info(f"Original video count: {len(original_downloaded_videos)}")
    logger.info(f"Video count after Langevin optimization: {len(downloaded_videos)}")
    

    logger.info(f"Number of videos returned by Langevin optimization: {len(downloaded_videos)}")
    logger.info(f"Audio duration: {audio_duration}sec")
    logger.info(f"Video segment duration:{params.video_clip_duration}sec")
    logger.info(f"Splicing mode: {params.video_concat_mode}")


    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # 6. Generate final videos
    final_video_paths, combined_video_paths = generate_final_videos(
        task_id, params, downloaded_videos, audio_file, subtitle_path
        #task_id, params, audio_file, subtitle_path
    )

    if not final_video_paths:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": video_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )
    return kwargs


if __name__ == "__main__":
    import json
    
    script_file_path = "run_script.json"
    
    with open(script_file_path, "r", encoding="utf-8") as f:
        script_data = json.load(f)
    
    params_dict = script_data["params"]
    
    params = VideoParams(
        video_subject=params_dict["video_subject"],
        video_script=params_dict["video_script"],
        video_terms=params_dict["video_terms"],
        video_aspect=params_dict["video_aspect"],
        video_concat_mode=params_dict["video_concat_mode"],
        video_transition_mode=params_dict["video_transition_mode"],
        video_clip_duration=params_dict["video_clip_duration"],
        video_count=params_dict["video_count"],
        video_source=params_dict["video_source"],
        video_materials=params_dict.get("video_materials", None),
        video_language=params_dict["video_language"],
        voice_name=params_dict["voice_name"],
        voice_volume=params_dict["voice_volume"],
        voice_rate=params_dict["voice_rate"],
        bgm_type=params_dict["bgm_type"],
        bgm_file=params_dict["bgm_file"],
        bgm_volume=params_dict["bgm_volume"],
        subtitle_enabled=params_dict["subtitle_enabled"],
        subtitle_position=params_dict["subtitle_position"],
        custom_position=params_dict["custom_position"],
        font_name=params_dict["font_name"],
        text_fore_color=params_dict["text_fore_color"],
        text_background_color=params_dict["text_background_color"],
        font_size=params_dict["font_size"],
        stroke_color=params_dict["stroke_color"],
        stroke_width=params_dict["stroke_width"],
        n_threads=params_dict["n_threads"],
        paragraph_number=params_dict["paragraph_number"],
        text_list=params_dict.get("text_list", [])
    )
    
    # Use the task ID from the configuration file or generate a new task ID.
    # Read task_id from run_script.json
    with open("run_script.json", "r", encoding="utf-8") as f:
        script_data = json.load(f)
    task_id = script_data.get("task_id", "default_task_id")  # Use default if not found
    
    print(f"Start running the task: {task_id}")
    print(f"Video theme: {params.video_subject}")
    print(f"Video source: {params.video_source}")
    print(f"Video count: {params.video_count}")
    print(f"Merge mode: {params.video_concat_mode}")
    
    result = start(task_id, params, stop_at="video")
    
    if result:
        print("\nTask completed!")
        print(f"Video generated: {result.get('videos', [])}")
    else:
        print("\nTask failed！")





