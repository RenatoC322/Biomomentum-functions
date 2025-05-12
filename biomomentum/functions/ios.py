import pandas as pd
import numpy as np
import os
import concurrent.futures

from tkinter import messagebox
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .utils import _print_progress, inDict, _parse_data_section, _parse_function_meta, _parse_function_name, _parse_info, _find_data_separators, _find_divider_indices

# Constants for parsing
dividers = ["<INFO>", "<END INFO>", "<DATA>", "<END DATA>"]
INDENTATION_DIVIDERS = ["Contacts mapping (x, y, z)", "Surface Normal (x, y, z)"]

def read_mach_1_file(file_path: Union[str, Path], read_data: bool = True, data_headers: Optional[List[str]] = None, show_progress: bool = True) -> Dict[str, Any]:
    """
    Parse a Mach-1 Motion .txt file into a nested dictionary structure.

    Args:
        file_path: Path to the input .txt file.
        read_data: Whether to parse and include <DATA> sections.
        data_headers: Optional list of column names to filter <DATA> columns.
        show_progress: If True, display a progress bar in the terminal.

    Returns:
        A mapping from function instance keys to a dictionary containing:
          - '<INFO>': core info key-values
          - '<FunctionName>': metadata for each function
          - '<DATA>': parsed measurement data (numpy arrays)
    """
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() != ".txt":
        raise FileNotFoundError(f"File not found or invalid extension: {file_path}")

    lines = path.read_text().splitlines()
    divider_idxs = _find_divider_indices(lines, dividers)
    data_flags, data_separators = _find_data_separators(lines, divider_idxs)
    result: Dict[str, Any] = {}
    name_counts: Dict[str, int] = {}
    num_funcs = len(divider_idxs) // 4

    for i in range(num_funcs):
        # Parse core INFO section
        info_section = _parse_info(lines, divider_idxs, i)
        # Identify function name
        func_label = _parse_function_name(lines, divider_idxs[4*i + 1] + 1)
        # Unique key for multiple occurrences
        count = name_counts.get(func_label, 0) + 1
        name_counts[func_label] = count
        key = f"{func_label}-{count}" if read_data else func_label

        # Parse metadata for this function
        meta_full = _parse_function_meta(lines, divider_idxs, i, func_label)
        func_key = f"<{func_label}>"
        meta_content = meta_full.get(func_key, {})

        # Parse data section if requested
        data = {}
        if read_data:
            data = _parse_data_section(
                path, lines, divider_idxs, i, data_flags[i], data_separators, data_headers
            )

        # Build entry: always include INFO and function metadata
        entry: Dict[str, Any] = {
            "<INFO>": info_section,
            func_key: meta_content
        }
        if read_data:
            entry["<DATA>"] = data

        result[key] = entry
        if show_progress:
            _print_progress(i, num_funcs - 1, prefix="Parsing:", suffix="Done", length=40)

    return result

def load_routine(params: Tuple[Path, str, bool, Optional[List[str]], bool]) -> Dict[str, Any]:
    """
    Helper for multiprocessing: unpacks parameters and calls read_mach1_file.

    Args:
        params: Tuple containing:
            - file_path (Path): path to the .txt file
            - key (str): identifier for this file (e.g., filename)
            - read_data (bool): whether to include <DATA>
            - data_headers (Optional[List[str]]): columns to filter
            - show_progress (bool): whether to display progress bar

    Returns:
        Dict mapping key to the parsed file dictionary.
    """
    file_path, key, read_data, data_headers, show_progress = params
    parsed = read_mach_1_file(
        file_path,
        read_data=read_data,
        data_headers=data_headers,
        show_progress=show_progress
    )
    return {key: parsed}
def read_mach_1_files(sources: Union[str, Path, List[Tuple[Path, str]]], read_data: bool = True, data_headers: Optional[List[str]] = None, pattern: str = "*.txt", show_progress: bool = True, use_multiprocessing: bool = False) -> Tuple[Dict[str, Any], Path]:
    """
    Read Mach-1 .txt files into a dictionary using the refactored reader.

    Args:
        sources: One of:
          - Directory path (str or Path) to glob .txt files by pattern.
          - List of (file_path, key) tuples, e.g. output from select_data_file_dir.
        read_data: Whether to include <DATA> sections from each file.
        data_headers: Optional list of column names to filter in <DATA>.
        pattern: Glob pattern to match files (default '*.txt') when a directory is provided.
        show_progress: Whether to display a progress bar during parsing.
        use_multiprocessing: If True, parse files in parallel.

    Returns:
        A tuple:
          - Dict mapping each key to its parsed content.
          - Path of the processed directory (parent of first file).
    """
    results: Dict[str, Any] = {}
    tasks: List[Tuple[Path, str, bool, Optional[List[str]], bool]] = []

    # If sources is a directory, glob files
    if isinstance(sources, (str, Path)):
        dir_path = Path(sources)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {sources}")
        file_list = list(dir_path.glob(pattern))
        if not file_list:
            messagebox.showwarning("Warning", f"No files matching {pattern} in {sources}")
            return {}, dir_path
        for f in file_list:
            tasks.append((f, f.stem, read_data, data_headers, show_progress))
    else:
        # sources should be a list of (Path, key) tuples
        if not isinstance(sources, (list, tuple)) or not sources:
            raise ValueError("sources must be a directory or list of (file_path, key) tuples")
        first = sources[0]
        if not (isinstance(first, (list, tuple)) and len(first) == 2):
            raise ValueError("each element in sources list must be a (file_path, key) tuple")
        for fp, key in sources:
            tasks.append((Path(fp), key, read_data, data_headers, show_progress))
        dir_path = Path(sources[0][0]).parent

    # Execute tasks
    if use_multiprocessing:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for res in executor.map(load_routine, tasks):
                results.update(res)
    else:
        for params in tasks:
            results.update(load_routine(params))

    return results, dir_path

def get_subSurfaces(MAP_file, keyword = "", scanFlag = False):
    """
    Function to separate surfaces from MAP in dictionary

    Args:
        MAP_file:  Name of the MAP to load
        keyword:  Name given to the measurements in the MAP file
        scanFlag: Indicate whether MAP file contains scan data
    
    Returns
        subSurfaces :  Dictionary of all the surfaces identified in the MAP file                         
    """
    subSurfaces = {}
    df_MAP = pd.read_csv(MAP_file , sep="\t",  skiprows = 6, engine="c", na_filter=False, low_memory=False).to_dict()
    df_MAP_info = pd.read_csv(MAP_file , sep="\t",  nrows = 5, usecols = [1], engine="c", na_filter=False, low_memory=False).values
    subSurfaces["MAP-Info"] = {"Software version" : df_MAP_info[0, 0], "Image" : df_MAP_info[3, 0], "Image directory" : os.path.join(os.path.dirname(MAP_file), df_MAP_info[3,0])}
    keyword_flag = inDict(df_MAP, keyword)
    for id in range(len(df_MAP["PointType"])):
        pt_type = df_MAP["PointType"][id]
        if pt_type == 0 or pt_type == 2:
            subSurf_ID = df_MAP["Sub-SurfaceID"][id]
            if pt_type == 2:
                subSurf_ID = "references"
            pt = np.array([df_MAP["PixelX"][id], df_MAP["PixelY"][id]])
            if scanFlag:
                pt_R = np.array([float(df_MAP["ScanX(mm)"][id]), float(df_MAP["ScanY(mm)"][id])])
            pt_ID = df_MAP["PointID"][id]
            flag = inDict(subSurfaces, subSurf_ID)
            if flag:
                subSurfaces[subSurf_ID]["Image Position"].append(pt)
                subSurfaces[subSurf_ID]["Position ID"].append(pt_ID)
                if scanFlag:
                    subSurfaces[subSurf_ID]["Position"].append(pt_R)
                if keyword_flag:
                    subSurfaces[subSurf_ID][keyword].append(float(df_MAP[keyword][id]))
            else:
                if keyword_flag:
                    subSurfaces[subSurf_ID] = {"Image Position": [pt], "Position ID": [pt_ID], keyword: [float(df_MAP[keyword][id])]}
                else:
                    subSurfaces[subSurf_ID] = {"Image Position": [pt], "Position ID": [pt_ID]}         
                if scanFlag:
                    subSurfaces[subSurf_ID].update({"Position" : [pt_R]})
        elif pt_type == 1:
            subSurf_ID = df_MAP["Sub-SurfaceID"][id]
            Pixel = np.array([df_MAP["PixelX"][id], df_MAP["PixelY"][id]])
            flag = inDict(subSurfaces, subSurf_ID)
            if flag:
                bounds_flag = inDict(subSurfaces[subSurf_ID], "Bounds")
                if bounds_flag:
                    subSurfaces[subSurf_ID]["Bounds"].append(Pixel)
                else:
                    subSurfaces[subSurf_ID]["Bounds"] = [Pixel]
            else:
                subSurfaces[subSurf_ID] = {"Bounds": [Pixel]}                      
    return subSurfaces

