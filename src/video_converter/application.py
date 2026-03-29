from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import subprocess
from dataclasses import KW_ONLY, asdict, dataclass, field
from functools import cache, cached_property
from itertools import chain
from logging import Handler
from logging.handlers import RotatingFileHandler
from os import linesep
from pathlib import Path
from shlex import quote
from shutil import which
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, cast

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from _typeshed import DataclassInstance


@dataclass
class LogFileOptions:
    path: Path
    _ = KW_ONLY
    max_kb: int
    backup_count: int
    level: int = logging.DEBUG
    encoding: str = "utf-8"
    append: bool = True

    def create_handler(self) -> Handler:
        handler = RotatingFileHandler(
            self.path,
            mode="a" if self.append else "w",
            encoding=self.encoding,
            maxBytes=self.max_kb * 1024,
            backupCount=self.backup_count,
        )
        handler.setLevel(self.level)
        return handler


def configure_logging(
    console_level: int, log_file_options: LogFileOptions | None = None
) -> None:
    class SuppressFileOnly(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not getattr(record, "file_only", False)

    logging.getLogger().handlers = []
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter(fmt="{levelname:s}: {message:s}", style="{")
    )
    console_handler.addFilter(SuppressFileOnly())
    logging.getLogger().addHandler(console_handler)
    global_level = console_level

    if log_file_options:
        global_level = min(global_level, log_file_options.level)
        file_handler = log_file_options.create_handler()
        file_handler.setFormatter(
            logging.Formatter(
                fmt=(
                    "[{asctime:s}.{msecs:03.0f}]"
                    " [{levelname:s}] {module:s}: {message:s}"
                ),
                datefmt="%Y-%m-%d %H:%M:%S",
                style="{",
            )
        )
        logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(global_level)
    logger.info("logging configured")


@cache
def cli_tool(name: str) -> Path:
    path = which(name)
    if not path:
        msg = f"Required tool not found in path: {name}"
        raise RuntimeError(msg)
    return Path(path)


def cli_string(arguments: Sequence[str | Path]) -> str:
    return " ".join(quote(str(argument)) for argument in arguments)


def log_cli(arguments: Sequence[str | Path]) -> None:
    logger.info(f"Running command: {cli_string(arguments)}")


def dict_to_args(
    data: Mapping[str, str | Path], *, key_suffix: str = ""
) -> list[str | Path]:
    return [
        item
        for key, value in data.items()
        for item in (f"{key}{key_suffix}", value)
    ]


def pretty_dataclass_str(
    obj: DataclassInstance, extra: dict[str, Any] | None = None
) -> str:
    return linesep.join(
        [type(obj).__name__]
        + [
            f"- {key}: {value}"
            for key, value in chain(asdict(obj).items(), (extra or {}).items())
        ]
    )


@dataclass(frozen=True)
class SideData:
    type: str
    max_content: int
    max_average: int


@dataclass(frozen=True)
class TrackMetadata:
    identifier: str
    index: int
    codec_type: str
    codec_name: str
    profile: str
    language: str
    title: str
    channels: int
    color_transfer: str
    color_primaries: str
    width: int
    height: int
    side_data_list: tuple[SideData, ...]

    def is_visual_type(self) -> bool:
        return self.codec_type.casefold() == "video"

    def is_video(self) -> bool:
        return self.is_visual_type() and self.codec_name.casefold() not in [
            "mjpeg",
            "png",
            "gif",
            "jpg",
            "jpeg",
        ]

    def is_audio(self) -> bool:
        return self.codec_type.casefold() == "audio"

    def is_subtitle(self) -> bool:
        return self.codec_type.casefold() == "subtitle"

    def is_pgs_subtitle(self) -> bool:
        return (
            self.is_subtitle()
            and self.codec_name.casefold() == "hdmv_pgs_subtitle"
        )

    def is_hdr(self) -> bool:
        return self.is_video() and (
            self.is_hdr10plus()
            or self.is_dolby_vision()
            or self.color_transfer.casefold() in {"smpte2084", "arib-std-b67"}
            or bool(self.content_light_levels())
            or any(
                side_data.type.casefold() == "mastering display metadata"
                for side_data in self.side_data_list
            )
        )

    def content_light_levels(self) -> str:
        for side_data in self.side_data_list:
            if side_data.type.casefold() == "content light level metadata":
                result = (side_data.max_content, side_data.max_average)
                if any(result):
                    return ",".join(str(value) for value in result)
        return ""

    def is_hdr10(self) -> bool:
        return (
            self.is_video()
            and not self.is_dolby_vision()
            and not self.is_hdr10plus()
            and self.color_transfer.casefold() == "smpte2084"
            and self.color_primaries.casefold() == "bt2020"
        )

    def is_hdr10plus(self) -> bool:
        return self.is_video() and any(
            side_data.type.casefold() == "hdr10+ metadata"
            for side_data in self.side_data_list
        )

    def is_dolby_vision(self) -> bool:
        return self.is_video() and any(
            side_data.type.casefold()
            in {
                "dovi configuration record",
                "dolby vision configuration record",
                "dolby vision rpu metadata",
            }
            for side_data in self.side_data_list
        )

    def is_sdr(self) -> bool:
        return self.is_video() and not self.is_hdr()

    def is_other(self) -> bool:
        return (
            not self.is_video()
            and not self.is_audio()
            and not self.is_subtitle()
        )

    def near_resolution(
        self, target_width: int, target_height: int, *, width_tol: float = 0.1
    ) -> bool:
        return (
            self.height <= target_height
            and abs(self.width - target_width) <= target_width * width_tol
        )

    def resolution_label(self) -> str:
        if self.is_visual_type():
            if any(
                self.near_resolution(width, 2160)
                for width in (3840, 3996, 4096)
            ):
                return "2160p"
            if self.near_resolution(1920, 1080):
                return "1080p"
            if self.near_resolution(1280, 720):
                return "720p"
            return f"{self.width}x{self.height}"
        return ""

    def hdr_label(self, *, simple: bool = False) -> str:
        label = []
        if not simple and self.is_hdr10plus():
            label.append("HDR10+")
        elif not simple and self.is_hdr10():
            label.append("HDR10")
        elif self.is_hdr():
            label.append("HDR")
        elif self.is_video():
            label.append("SDR")

        if not simple and self.is_dolby_vision():
            label.append("DV")

        return " ".join(label)

    def is_atmos(self) -> bool:
        profile = self.profile.casefold()
        return (self.is_audio() and "dolby atmos" in profile) or (
            self.codec_name.casefold() == "eac3" and "joc" in profile
        )

    def audio_codec_score(self) -> int:
        if self.codec_type != "audio":
            return 0

        name = self.codec_name.casefold()
        profile = self.profile.casefold()

        # Lossless tier
        if name.startswith("pcm_"):
            # Companded PCM is lower quality than linear PCM
            score = 60 if name in {"pcm_mulaw", "pcm_alaw"} else 90
        elif name in {
            "truehd",
            "mlp",
            "flac",
            "alac",
            "wavpack",
            "ape",
            "tta",
        }:
            score = 95

        # DTS family (profile distinguishes lossless DTS-HD MA)
        elif name == "dts":
            if "dts-hd ma" in profile:
                score = 94  # lossless
            elif "dts-hd" in profile:  # HRA (lossy but higher quality)
                score = 55
            else:
                score = 45  # DTS core (lossy)

        # High-quality lossy tier
        elif name in {"opus", "eac3", "aac"}:
            score = 50

        # Mid/legacy lossy tier
        elif name in {"ac3", "vorbis", "mp3"}:
            score = 40
        else:
            score = 10

        return score

    def __str__(self) -> str:
        indicator: list[str] = [str(self.index), self.identifier]
        info: list[str] = [
            value
            for value in [
                self.language.upper() if self.language else "",
                " ".join(
                    value
                    for value in [self.codec_name.upper(), self.profile]
                    if value
                ),
                f"{self.channels}ch" if self.channels else "",
                "Atmos" if self.is_atmos() else "",
                self.resolution_label(),
                self.hdr_label(),
            ]
            if value
        ]

        logger.info(
            pretty_dataclass_str(
                self,
                extra={
                    "hdr_label": self.hdr_label(),
                    "resolution_label": self.resolution_label(),
                    "is_atmos": self.is_atmos(),
                },
            )
        )

        return f"#{' '.join(indicator)}: [{', '.join(info)}] {self.title}"


@dataclass
class AudioFilter:
    atmos: bool | None = None
    channel_min: int | None = None
    channel_max: int | None = None
    language: str | None = None


@dataclass
class IdentifierTracker:
    last_index_map: dict[str, int] = field(default_factory=dict)
    PREFIX_MAP: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "video": "v",
            "audio": "a",
            "subtitle": "s",
            "data": "d",
            "attachment": "t",
        }
    )

    def next(self, codec_type: str) -> str:
        prefix = type(self).PREFIX_MAP[codec_type]
        index = self.last_index_map.get(prefix, 0)
        self.last_index_map[prefix] = index + 1
        return f"{prefix}:{index}"


@dataclass(frozen=True)
class MediaInfo:
    tracks: tuple[TrackMetadata, ...]
    track_indexes: Mapping[int, TrackMetadata] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "track_indexes",
            MappingProxyType({track.index: track for track in self.tracks}),
        )

    @classmethod
    def from_path(cls, path: Path) -> MediaInfo:
        arguments: list[str | Path] = [
            cli_tool("ffprobe"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            # "-show_format",
            str(path),
        ]
        log_cli(arguments)
        data = json.loads(subprocess.check_output(arguments))  # noqa: S603

        tracker = IdentifierTracker()

        tracks: list[TrackMetadata] = []

        for track in sorted(
            data["streams"], key=lambda stream: int(stream["index"])
        ):
            if not isinstance(track, dict):
                msg = f"track is not a dict: {track}"
                raise TypeError(msg)
            tags = cast("dict[str, Any]", track.get("tags") or {})
            codec_type = str(track["codec_type"])
            tracks.append(
                TrackMetadata(
                    identifier=tracker.next(codec_type),
                    index=int(track["index"]),
                    codec_type=codec_type,
                    codec_name=str(track.get("codec_name") or ""),
                    profile=str(track.get("profile") or ""),
                    title=str(tags.get("title") or ""),
                    language=(
                        ""
                        if (language := str(tags.get("language", ""))) == "und"
                        else language
                    ),
                    channels=int(track.get("channels") or 0),
                    color_transfer=str(track.get("color_transfer") or ""),
                    color_primaries=str(track.get("color_primaries") or ""),
                    width=int(track.get("width") or 0),
                    height=int(track.get("height") or 0),
                    side_data_list=tuple(
                        [
                            SideData(
                                type=str(entry["side_data_type"]),
                                max_content=int(entry.get("max_content") or 0),
                                max_average=int(entry.get("max_average") or 0),
                            )
                            for entry in (track.get("side_data_list") or [])
                        ]
                    ),
                )
            )
        return MediaInfo(tracks=tuple(tracks))

    def track_from_index(self, index: int) -> TrackMetadata:
        return self.track_indexes[index]

    @cached_property
    def video_tracks(self) -> tuple[TrackMetadata, ...]:
        return tuple([track for track in self.tracks if track.is_video()])

    @cached_property
    def audio_tracks(self) -> tuple[TrackMetadata, ...]:
        return tuple([track for track in self.tracks if track.is_audio()])

    @cached_property
    def atmos_audio_tracks(self) -> tuple[TrackMetadata, ...]:
        return tuple(
            [track for track in self.audio_tracks if track.is_atmos()]
        )

    @cached_property
    def non_atmos_audio_tracks(self) -> tuple[TrackMetadata, ...]:
        return tuple(
            [track for track in self.audio_tracks if not track.is_atmos()]
        )

    @cached_property
    def subtitle_tracks(self) -> tuple[TrackMetadata, ...]:
        return tuple([track for track in self.tracks if track.is_subtitle()])

    @classmethod
    def subtitle_language_sort_key(
        cls, track: TrackMetadata
    ) -> tuple[int, int, str]:
        language = (track.language or "").casefold()
        title = (track.title or "").casefold()
        return (0 if language == "eng" else 1, 0 if title == "" else 1, title)

    @cached_property
    def other_tracks(self) -> tuple[TrackMetadata, ...]:
        return tuple([track for track in self.tracks if track.is_other()])

    def best_video(self) -> TrackMetadata | None:
        # Choosing from HDR only if any are present, gets highest resolution
        # and if multiple of the same exist, the first one is used.
        try:
            return sorted(
                self.video_tracks,
                key=lambda track: (track.is_hdr(), track.width * track.height),
                reverse=True,
            )[
                0
            ]  # will throw if no stream
        except IndexError:
            return None

    def best_audio(
        self, priority_filters: list[AudioFilter]
    ) -> TrackMetadata | None:
        for audio_filter in priority_filters:
            try:
                return sorted(
                    self.audio_tracks_filtered(audio_filter),
                    key=lambda track: (
                        track.channels,
                        track.title,
                        track.audio_codec_score(),
                    ),
                    reverse=True,
                )[0]
            except IndexError:
                continue
        return None

    def audio_tracks_filtered(
        self, audio_filter: AudioFilter
    ) -> list[TrackMetadata]:
        return [
            track
            for track in self.audio_tracks
            if (
                audio_filter.atmos is None
                or audio_filter.atmos == track.is_atmos()
            )
            and (
                audio_filter.channel_min is None
                or track.channels >= audio_filter.channel_min
            )
            and (
                audio_filter.channel_max is None
                or track.channels <= audio_filter.channel_max
            )
            and (
                audio_filter.language is None
                or track.language == audio_filter.language
            )
        ]


@dataclass
class EncoderCliOptions:
    append_label: bool
    extra_av_indexes: list[int]
    crf: int
    preset: str
    copy_video: bool
    no_video: bool
    audio_languages: list[str]
    copy_audio: bool
    no_audio: bool
    no_subtitles: bool
    no_chapters: bool


@dataclass
class EncoderCliBuilder:
    source: Path
    output: Path
    options: EncoderCliOptions
    info: MediaInfo = field(init=False)
    source_file_index: int = field(init=False)
    tracker: IdentifierTracker = field(init=False)
    track_arguments: list[str | Path] = field(init=False)
    title_arguments: dict[str, str] = field(init=False)
    extra_video_tracks: list[TrackMetadata] = field(init=False)
    extra_audio_tracks: list[TrackMetadata] = field(init=False)
    tracks_in_output: list[TrackMetadata] = field(init=False)

    def __post_init__(self) -> None:
        self.info = MediaInfo.from_path(self.source)
        self.source_file_index = 0  # NOTE: assumes only one source file!
        self.tracker = IdentifierTracker()
        self.track_arguments = dict_to_args(
            {
                "-i": str(self.source),
                "-map_metadata:g": "-1",
                "-map_metadata:s": "-1",
                "-map_metadata:p": "-1",
            }
        )
        self.title_arguments = {}
        self.extra_video_tracks = []
        self.extra_audio_tracks = []
        self.tracks_in_output = []

        self.__collect_extra_tracks()

        if not self.options.no_video:
            self.__add_video()
        if not self.options.no_audio:
            self.__add_audio()
        if not self.options.no_subtitles:
            self.__add_subtitles()
        if not self.options.no_chapters:
            self.__add_chapters()
        self.track_arguments.extend(dict_to_args(self.title_arguments))

    def __collect_extra_tracks(self) -> None:
        for index in self.options.extra_av_indexes or []:
            track = self.info.track_from_index(index)
            if track.is_video():
                self.extra_video_tracks.append(track)
            elif track.is_audio():
                self.extra_audio_tracks.append(track)
            else:
                msg = f"Extra track {index} is neither audio or video."
                raise ValueError(msg)

    def __map_track(self, track: TrackMetadata) -> str:
        track_id = self.tracker.next(track.codec_type)
        self.track_arguments.extend(
            ["-map", f"{self.source_file_index}:{track.identifier}"]
        )
        if track.title:
            lower_title = track.title.casefold()
            if any(
                text in lower_title
                for text in ("7.1", "5.1", "stereo", "mono", "atmos", "truehd")
            ):
                logger.warning(
                    f"Stripping title from track #{track_id}: {track.title}"
                )
                title = ""
            else:
                title = track.title
            self.title_arguments[f"-metadata:s:{track_id}"] = f"title={title}"
        if track.language:
            self.track_arguments.extend(
                [f"-metadata:s:{track_id}", f"language={track.language}"]
            )
        return track_id

    def __add_video(self) -> None:
        best_video = self.info.best_video()
        if best_video is None:
            msg = "No video tracks found!"
            raise RuntimeError(msg)
        if self.options.append_label:
            label = " ".join(
                [
                    label
                    for label in [
                        best_video.resolution_label(),
                        best_video.hdr_label(simple=True),
                    ]
                    if label
                ]
            )
            if label:
                label = f" [{label}]"
            self.output = self.output.parent / (
                self.output.stem + label + self.output.suffix
            )
        video_output_tracks: list[TrackMetadata] = [best_video]
        video_output_tracks.extend(
            track
            for track in self.extra_video_tracks
            if track not in video_output_tracks
        )
        first_video_track = True
        for track in video_output_tracks:
            self.tracks_in_output.append(track)
            track_id = self.__map_track(track)

            track_arguments = {
                "-disposition": "default" if first_video_track else "0"
            }
            first_video_track = False
            if self.options.copy_video:
                track_arguments["-codec"] = "copy"
            else:
                track_arguments.update(
                    {
                        "-codec": "libx265",  # HEVC
                        "-crf": str(self.options.crf),
                        "-preset": self.options.preset,
                    }
                )
                if track.is_hdr():
                    if track.is_hdr10plus():
                        logger.warning(
                            "HDR10+ data will be removed from"
                            f" track #{track_id}"
                        )
                    if track.is_dolby_vision():
                        logger.warning(
                            "Dolby Vision data will be removed from"
                            f" track #{track_id}"
                        )
                    x265_params = [
                        "repeat-headers=1",
                        (
                            "master-display="
                            "G(13250,34500)B(7500,3000)R(34000,16000)"
                            "WP(15635,16450)L(10000000,1)"
                        ),
                    ]

                    if cll := track.content_light_levels():
                        x265_params.append(f"max-cll={cll}")

                    track_arguments.update(
                        {
                            "-pix_fmt": "yuv420p10le",
                            "-color_primaries": "bt2020",
                            "-colorspace": "bt2020nc",
                            "-color_trc": "smpte2084",
                            "-x265-params": ":".join(x265_params),
                        }
                    )
                else:
                    track_arguments.update(
                        {
                            "-pix_fmt": "yuv420p",
                            "-color_primaries": "bt709",
                            "-colorspace": "bt709",
                            "-color_trc": "bt709",
                        }
                    )
            self.track_arguments.extend(
                dict_to_args(track_arguments, key_suffix=f":{track_id}")
            )

    def __add_audio(self) -> None:
        has_audio = False
        channels_5_1 = 6
        for language in self.options.audio_languages or []:
            audio_output_tracks = [
                self.info.best_audio(
                    [AudioFilter(channel_min=channels_5_1, language=language)]
                )
            ]
            if audio_output_tracks:
                has_audio = True
            else:
                logger.warning(f"No audio selected for language: {language}")

            audio_output_tracks.extend(
                track
                for track in self.extra_audio_tracks
                if track not in audio_output_tracks
            )
            first_audio_track = True
            for track in audio_output_tracks:
                if track is None:
                    continue
                self.tracks_in_output.append(track)
                track_id = self.__map_track(track)
                track_arguments = {
                    "-disposition": "default" if first_audio_track else "0"
                }
                first_audio_track = False
                if self.options.copy_audio:
                    track_arguments["-codec"] = "copy"
                else:
                    if track.is_atmos():
                        logger.warning(
                            f"Atmos data will be removed from track #{track_id}"
                        )
                    track_arguments.update(
                        {
                            "-codec": "libopus",
                            "-ac": str(track.channels),
                            "-b": f"{track.channels * 80}k",
                        }
                    )
                self.track_arguments.extend(
                    dict_to_args(track_arguments, key_suffix=f":{track_id}")
                )

        if not has_audio:
            msg = "No audio tracks included!"
            raise RuntimeError(msg)

    def __add_subtitles(self) -> None:
        first_english_subtitle_track = True
        for track in sorted(
            self.info.subtitle_tracks, key=MediaInfo.subtitle_language_sort_key
        ):
            self.tracks_in_output.append(track)
            track_id = self.__map_track(track)
            track_arguments = {}
            if track.codec_name.casefold() == "mov_text":
                track_arguments["-codec"] = "srt"  # convert!
            else:
                track_arguments["-codec"] = "copy"
            if (
                first_english_subtitle_track
                and track.language.casefold() == "eng"
            ):
                track_arguments["-disposition"] = "default"
                first_english_subtitle_track = False
            else:
                track_arguments["-disposition"] = "0"
            self.track_arguments.extend(
                dict_to_args(track_arguments, key_suffix=f":{track_id}")
            )

    def __add_chapters(self) -> None:
        self.track_arguments.extend(
            dict_to_args({"-map_chapters": str(self.source_file_index)})
        )

    def arguments(self) -> list[str | Path]:
        return [
            cli_tool("ffmpeg"),
            "-stats",
            "-loglevel",
            "error",
            *self.track_arguments,
            self.output,
        ]


def get_encoder_cli(
    source: Path, output: Path, options: EncoderCliOptions
) -> tuple[list[TrackMetadata], list[str | Path]]:
    cli_builder = EncoderCliBuilder(source, output, options)
    print(f"Output Tracks: {source}")
    for track in cli_builder.tracks_in_output:
        print(f"- {track}")
    print()
    return cli_builder.tracks_in_output, cli_builder.arguments()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=importlib.metadata.metadata(__package__).get("summary")
    )
    log_group = parser.add_argument_group("logging")
    log_group.add_argument(
        "--log-file",
        metavar="FILE",
        help="Path to a file where logs will be written, if specified.",
    )
    log_verbosity_group = log_group.add_mutually_exclusive_group(
        required=False
    )
    log_verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        dest="console_level",
        const=logging.INFO,
        help="Increase console log level to INFO.",
    )
    log_verbosity_group.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        dest="console_level",
        const=logging.ERROR,
        help="Decrease console log level to ERROR.  Overrides -v.",
    )
    log_verbosity_group.add_argument(
        "--debug",
        action="store_const",
        dest="console_level",
        const=logging.DEBUG,
        help="Maximizes console log verbosity to DEBUG.  Overrides -v and -q.",
    )
    parser.add_argument("source", type=Path, help="Source media file.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("probe", help="Probe media.")  # NO VAR

    convert = sub.add_parser("convert", help="Convert media.")

    convert.add_argument(
        "--crf", type=int, default=22, help="The CRF to use with HEVC."
    )
    convert.add_argument(
        "--preset", default="slow", help="The preset to use with HEVC."
    )
    convert.add_argument(
        "--copy-video",
        action="store_true",
        help="Don't re-encode video tracks.",
    )
    convert.add_argument(
        "--no-video", action="store_true", help="Don't include video tracks."
    )
    convert.add_argument(
        "--audio-language",
        action="append",
        default=None,
        help="The desired audio language(s).  Can pass multiple times.",
    )
    convert.add_argument(
        "--copy-audio",
        action="store_true",
        help="Don't re-encode audio tracks.",
    )
    convert.add_argument(
        "--no-audio", action="store_true", help="Don't include audio tracks."
    )
    convert.add_argument(
        "--no-subtitles", action="store_true", help="Don't include subtitles."
    )
    convert.add_argument(
        "--no-chapters", action="store_true", help="Don't include chapters."
    )
    convert.add_argument(
        "-o", "--output", type=Path, help="Output media file.", required=True
    )
    convert.add_argument(
        "-a",
        "--append-label",
        action="store_true",
        help="Append a format related label to the output filename.",
    )
    convert.add_argument(
        "-e",
        "--extra-av",
        type=int,
        action="append",
        help="Extra audio/video track global index.  Can pass multiple times.",
    )
    convert.add_argument(
        "--run",
        action="store_true",
        help="Extra audio/video track global index.  Can pass multiple times.",
    )

    args = parser.parse_args(args=argv)
    configure_logging(
        console_level=args.console_level or logging.WARNING,
        log_file_options=(
            None
            if not args.log_file
            else LogFileOptions(
                path=Path(args.log_file),
                max_kb=512,  # 0 for unbounded size and no rotation
                backup_count=1,  # 0 for no rolling backups
            )
        ),
    )
    exit_code = 0
    if args.command == "probe":
        info = MediaInfo.from_path(args.source)
        print(args.source.name)
        for track in info.tracks:
            print(f"- {track}")
    elif args.command == "convert":
        if args.audio_language is None:
            args.audio_language = ["eng"]

        cli_builder = EncoderCliBuilder(
            args.source,
            args.output,
            EncoderCliOptions(
                append_label=args.append_label,
                extra_av_indexes=args.extra_av,
                crf=args.crf,
                preset=args.preset,
                copy_video=args.copy_video,
                no_video=args.no_video,
                audio_languages=args.audio_language,
                copy_audio=args.copy_audio,
                no_audio=args.no_audio,
                no_subtitles=args.no_subtitles,
                no_chapters=args.no_chapters,
            ),
        )
        print(f"Output Tracks: {args.source}")
        for track in cli_builder.tracks_in_output:
            print(f"- {track}")
        print()
        encoder_cli = cli_builder.arguments()
        print(cli_string(encoder_cli))
        if args.run:
            exit_code = subprocess.run(  # noqa: S603
                encoder_cli, check=False
            ).returncode
    return exit_code
