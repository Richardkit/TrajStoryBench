import os

from .path_utils import get_project_root, assemble_project_path
from .singleton import Singleton
from .utils import (
    _is_package_available,
    encode_file_base64, 
    decode_file_base64,
    make_file_url, 
    parse_json_blob,
    gather_with_concurrency,
)
from .string_utils import extract_boxed_content, dedent, generate_unique_id
from .file_utils import get_file_info, file_lock
from .args_utils import parse_tool_args

_minimal = os.getenv("TRAJSTORY_MINIMAL_IMPORTS") == "1"
if not _minimal:
    from .record_utils import Record, TradingRecords, PortfolioRecords
    from .token_utils import get_token_count
    from .calender_utils import TimeLevel, TimeLevelFormat, get_start_end_timestamp, calculate_time_info, get_standard_timestamp
    from .misc import get_world_size, get_rank
    from .name_utils import get_tag_name, get_newspage_name, get_md5
    from .url_utils import fetch_url
    from .env_utils import get_env
    from .screenshot_utils import ScreenshotService
    from .download_utils import (get_jsonparsed_data,
                                 generate_intervals)
    from .hub_utils import push_to_hub_folder


__all__ = [
    "get_project_root",
    "assemble_project_path",
    "Singleton",
    "_is_package_available",
    "encode_file_base64",
    "decode_file_base64",
    "make_file_url",
    "parse_json_blob",
    "gather_with_concurrency",
    "extract_boxed_content",
    "get_file_info",
    "dedent",
    "file_lock",
    "generate_unique_id",
    "parse_tool_args",
] + ([] if _minimal else [
    "Record", "TradingRecords", "PortfolioRecords", "get_token_count",
    "TimeLevel", "TimeLevelFormat", "get_start_end_timestamp",
    "calculate_time_info", "get_standard_timestamp", "get_world_size",
    "get_rank", "get_tag_name", "get_newspage_name", "get_md5",
    "fetch_url", "get_env", "ScreenshotService", "get_jsonparsed_data",
    "generate_intervals", "push_to_hub_folder",
])
