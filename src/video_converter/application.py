import argparse
import importlib.metadata
import json
import logging
import subprocess
from dataclasses import KW_ONLY, dataclass
from functools import cache, cached_property
from logging import Handler
from logging.handlers import RotatingFileHandler
from pathlib import Path
from shlex import quote
from shutil import which
from typing import Any, Mapping, Sequence, cast

logger = logging.getLogger(__name__)


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
    logging.info("logging configured")


@cache
def cli_tool(name: str) -> Path:
    path = which(name)
    if not path:
        raise RuntimeError(f"Required tool not found in path: {name}")
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


@dataclass(frozen=True)
class SideData:
    type: str
    max_content: int
    max_average: int


@dataclass(frozen=True)
class TrackMetadata:
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

    @classmethod
    def from_json(cls, data: Any) -> TrackMetadata:
        if not isinstance(data, dict):
            raise TypeError(f"data is not a dict: {data}")
        tags = cast(dict[str, Any], data.get("tags") or {})
        return TrackMetadata(
            index=int(data["index"]),
            codec_type=str(data["codec_type"]),
            codec_name=str(data["codec_name"]),
            profile=str(data.get("profile") or ""),
            title=str(tags.get("title") or ""),
            language=(
                ""
                if (language := str(tags.get("language", ""))) == "und"
                else language
            ),
            channels=int(data.get("channels") or 0),
            color_transfer=str(data.get("color_transfer") or ""),
            color_primaries=str(data.get("color_primaries") or ""),
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
            side_data_list=tuple(
                [
                    SideData(
                        type=str(entry["side_data_type"]),
                        max_content=int(entry.get("max_content") or 0),
                        max_average=int(entry.get("max_average") or 0),
                    )
                    for entry in (data.get("side_data_list") or [])
                ]
            ),
        )

    def is_video(self) -> bool:
        return (
            self.codec_type.casefold() == "video"
            and self.codec_name.casefold() != "mjpeg"
        )

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

    def is_2160p(self) -> bool:
        # NOTE: can be UHD 3840x2160 or DCI 4096x2160
        return self.is_video() and self.height == 2160

    def is_atmos(self) -> bool:
        profile = self.profile.casefold()
        return (
            self.is_audio()
            and "dolby atmos" in profile
            or (self.codec_name.casefold() == "eac3" and "joc" in profile)
        )

    def audio_codec_score(self) -> int:
        if self.codec_type != "audio":
            return 0

        name = self.codec_name.casefold()
        profile = self.profile.casefold()

        # Lossless tier
        if name.startswith("pcm_"):
            # Companded PCM is lower quality than linear PCM
            if name in {"pcm_mulaw", "pcm_alaw"}:
                return 60
            return 90
        if name in {"truehd", "mlp", "flac", "alac", "wavpack", "ape", "tta"}:
            return 95

        # DTS family (profile distinguishes lossless DTS-HD MA)
        if name == "dts":
            if "dts-hd ma" in profile:
                return 94  # lossless
            if "dts-hd" in profile:  # HRA (lossy but higher quality)
                return 55
            return 45  # DTS core (lossy)

        # High-quality lossy tier
        if name in {"opus", "eac3", "aac"}:
            return 50

        # Mid/legacy lossy tier
        if name in {"ac3", "vorbis", "mp3"}:
            return 40

        return 10


@dataclass
class AudioFilter:
    atmos: bool | None = None
    channel_min: int | None = None
    channel_max: int | None = None


@dataclass(frozen=True)
class MediaInfo:
    tracks: tuple[TrackMetadata, ...]

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
        data = json.loads(subprocess.check_output(arguments))
        tracks: tuple[TrackMetadata, ...] = tuple(
            TrackMetadata.from_json(track)
            for track in sorted(
                data["streams"], key=lambda stream: int(stream["index"])
            )
        )
        return MediaInfo(tracks=tracks)

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
        except KeyError:
            return None

    def best_audio(
        self, priority_filters: list[AudioFilter]
    ) -> TrackMetadata | None:
        for filter in priority_filters:
            try:
                return sorted(
                    self.audio_tracks_filtered(filter),
                    key=lambda track: (
                        track.channels,
                        track.audio_codec_score(),
                    ),
                    reverse=True,
                )[0]
            except KeyError:
                continue
        return None

    def audio_tracks_filtered(
        self, filter: AudioFilter
    ) -> list[TrackMetadata]:
        return [
            track
            for track in self.audio_tracks
            if (filter.atmos is None or filter.atmos == track.is_atmos())
            and (
                filter.channel_min is None
                or track.channels >= filter.channel_min
            )
            and (
                filter.channel_max is None
                or track.channels <= filter.channel_max
            )
        ]


@dataclass
class EncoderSettings:
    channels: int | None = None
    copy_only: bool = False


def get_encoder_cli(
    input: Path,
    output: Path,
    *,
    crf: int = 22,
    preset: str = "slow",
    no_video: bool = False,
    no_audio: bool = False,
    no_subtitles: bool = False,
    no_chapters: bool = False,
) -> list[str | Path]:

    info = MediaInfo.from_path(input)
    input_file_index = 0  # TODO: assumes only one input file!
    video_index = audio_index = 0  # applies to all inputs, if more added later
    arguments = dict_to_args({"-i": str(input)})
    track_arguments: dict[str, str]
    if not no_video:
        track = info.best_video()
        if track is None:
            raise RuntimeError("No video tracks found!")
        track_id, video_index = f"v:{video_index}", video_index + 1
        arguments.extend(["-map", f"{input_file_index}:{track.index}"])

        track_arguments = {
            "-codec": "libx265",  # HEVC
            "-crf": str(crf),
            "-preset": preset,
            "-disposition": "default",
        }
        if track.is_hdr():
            track_arguments.update(
                {
                    "-pix_fmt": "yuv420p10le",
                    "-color_primaries": "bt2020",
                    "-colorspace": "bt2020nc",
                    "-color_trc": "smpte2084",
                    "-x265-params": (
                        "master-display="
                        + "G(13250,34500)B(7500,3000)R(34000,16000)"
                        + "WP(15635,16450)L(10000000,1)"
                    ),
                }
            )
            if cll := track.content_light_levels():
                track_arguments["-max_cll"] = cll
        else:
            track_arguments.update(
                {
                    "-pix_fmt": "yuv420p",
                    "-color_primaries": "bt709",
                    "-colorspace": "bt709",
                    "-color_trc": "bt709",
                }
            )
        arguments.extend(
            dict_to_args(track_arguments, key_suffix=f":{track_id}")
        )

    if not no_audio:
        audio_best_atmos = info.best_audio([AudioFilter(atmos=True)])
        audio_6ch = info.best_audio(
            [
                AudioFilter(atmos=False, channel_min=6, channel_max=6),
                AudioFilter(atmos=False, channel_min=6),
                AudioFilter(atmos=True, channel_min=6, channel_max=6),
                AudioFilter(atmos=True, channel_min=6),
            ]
        )
        if audio_6ch is None:
            logger.warning("no 5.1+ audio source found; audio won't be 5.1!")
            audio_fallback = (
                info.best_audio([AudioFilter(atmos=False)]) or audio_best_atmos
            )
        else:
            if audio_6ch.is_atmos():
                logger.warning("5.1 audio to be generated from Atmos source.")
            audio_fallback = None  # not needed; we have 5.1

        first_audio_track = True
        for settings, track in [
            (EncoderSettings(channels=6), audio_6ch),
            (EncoderSettings(copy_only=True), audio_best_atmos),
            (EncoderSettings(), audio_fallback),
        ]:
            if track is None:
                continue
            track_id, audio_index = f"a:{audio_index}", audio_index + 1
            arguments.extend(["-map", f"{input_file_index}:{track.index}"])

            track_arguments = {}
            if settings.copy_only:
                track_arguments["-codec"] = "copy"
            else:
                track_arguments["-codec"] = "eac3"
                if (
                    settings.channels is not None
                    and settings.channels != track.channels
                ):
                    channels = settings.channels
                    track_arguments["-ac"] = str(channels)
                else:
                    channels = track.channels
                track_arguments["-b"] = f"{channels * 112}k"
            if first_audio_track:
                first_audio_track = False
                track_arguments["-disposition"] = "default"
            else:
                track_arguments["-disposition"] = "0"
            arguments.extend(
                dict_to_args(track_arguments, key_suffix=f":{track_id}")
            )
    if not no_subtitles:
        arguments.extend(
            dict_to_args({"-codec:s": "copy", "-map": f"{input_file_index}:s"})
        )
    if not no_chapters:
        arguments.extend(
            dict_to_args({"-map_chapters": str(input_file_index)})
        )

    return [
        cli_tool("ffmpeg"),
        "-stats",
        "-loglevel",
        "error",
        *arguments,
        output,
    ]


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
    parser.add_argument("input", type=Path, help="Input media file.")
    parser.add_argument(
        "--crf", type=int, default=22, help="The CRF to use with HEVC."
    )
    parser.add_argument(
        "--preset", default="slow", help="The preset to use with HEVC."
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Don't include video tracks."
    )
    parser.add_argument(
        "--no-audio", action="store_true", help="Don't include audio tracks."
    )
    parser.add_argument(
        "--no-subtitles", action="store_true", help="Don't include subtitles."
    )
    parser.add_argument(
        "--no-chapters", action="store_true", help="Don't include chapters."
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Output media file.", required=True
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
                # append=False
            )
        ),
    )
    print(
        cli_string(
            get_encoder_cli(
                args.input,
                args.output,
                crf=args.crf,
                preset=args.preset,
                no_video=args.no_video,
                no_audio=args.no_audio,
                no_subtitles=args.no_subtitles,
                no_chapters=args.no_chapters,
            )
        )
    )
    return 0
