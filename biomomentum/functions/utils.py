import re
import numpy as np
import os
import pandas as pd
import tkinter

from pathlib import Path
from tkinter import filedialog, messagebox
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Dict, List, Optional, Tuple

from .stats import rsquared

# Constants for parsing
dividers = ["<INFO>", "<END INFO>", "<DATA>", "<END DATA>"]
INDENTATION_DIVIDERS = ["Contacts mapping (x, y, z)", "Surface Normal (x, y, z)"]

def sorted_alphanumeric(files):
    """
    Sorts alpha numerically files from directory.

    Args:
        files (list): Files from directory to sort.

    Returns:
        files sorted: files sorted.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(files, key=alphanum_key)

def get_super(x):
    """
    Insert super script char on string

    Args:
        x (str): Character to super script.

    Returns:
        Char in super script format.
    """ 
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s)) 
    return x.translate(res)

def inDict(dic, key):
    """
    Checks if key is in dictionary

    Args:
        key (str): Key to check if is in dictionary.

    Returns:
        flag (Bool): True if key is in dictionary.
    """ 
    flag = False
    if key in dic.keys():
        flag = True
    return flag

def select_data_file(file_extension = ".txt"):
    """
    Function that brings a pop up prompt to select the mach-1 file or MAP file

    Args:
        files (list): Files from directory to sort.

    Returns:
        files sorted: files sorted.
    """ 
    pop_window = tkinter.Tk()
    pop_window.withdraw()
    pop_window.attributes('-topmost', True) 
    filename = filedialog.askopenfilename(parent=pop_window, initialdir= "/", title='Please select the Mach-1 .txt file or MAP file')
    if len(filename) == 0 or not filename.endswith(file_extension):
        messagebox.showwarning("Warning", "No mach-1 or MAP file selected!")
        filename = None
    return filename

def select_data_file_dir(keyword = None, read_MAP = False):
    """
    Function that brings a pop up prompt to select a folder contaning multiple mach-1 files or MAP files.

    Args:
        keyword (str): String Name of group of mach-1 files to load in folder.
        read_MAP (Bool): Bool to indicate whether to select mach_1 files or MAP files.

    Returns:
        files (list): files of the mach-1 from the folder or MAP files.
    """
    ext = ".map" if read_MAP else ".txt"
    # Initialize hidden Tkinter dialog
    root = tkinter.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    directory = filedialog.askdirectory(parent=root, initialdir='/', title='Select directory containing Mach-1 files')
    root.destroy()

    if not directory:
        return []

    dir_path = Path(directory)
    files = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.lower().endswith(ext):
            stem = Path(fname).stem
            if keyword is None or keyword in stem:
                files.append((dir_path / fname, stem))
    return files

def _find_divider_indices(lines: List[str], divs: List[str]) -> List[int]:
    """
    Return a list of 1-based line indices where any of the specified dividers appear.

    Args:
        lines: All lines of the file.
        divs: Divider strings to search for.
    """
    return [i for i, line in enumerate(lines, start=1) if any(d in line for d in divs)]

def _find_data_separators(lines: List[str], idxs: List[int]) -> Tuple[List[bool], List[int]]:
    """
    Identify <divider> markers within each function's <DATA> block.

    Args:
        lines: All lines of the file.
        idxs: Indices of all section dividers (<INFO>, <END INFO>, <DATA>, <END DATA>).

    Returns:
        A tuple of:
          - List of booleans indicating if each function has any <divider> tags.
          - List of all line indices where '<divider>' appears.
    """
    sep_idxs = [i for i, line in enumerate(lines, start=1) if '<divider>' in line]
    funcs = len(idxs) // 4
    flags: List[bool] = []
    for i in range(funcs):
        start, end = idxs[4*i + 2], idxs[4*i + 3]
        flags.append(any(start < s < end for s in sep_idxs))
    return flags, sep_idxs

def _parse_info(lines: List[str], idxs: List[int], func_idx: int) -> Dict[str, Any]:
    """
    Parse the <INFO> section lines into a key-value dict, including the 'Date'.

    Args:
        lines: All lines of the file.
        idxs: Indices of section dividers.
        func_idx: Zero-based index of the function block.

    Returns:
        Dict mapping each INFO key to its value as a string.
    """
    info_start = idxs[4*func_idx]      # '<INFO>' line
    info_end = idxs[4*func_idx + 1]    # '<END INFO>' line
    info: Dict[str, Any] = {}
    # lines between INFO and END INFO
    for line in lines[info_start:info_end - 1]:
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[1].strip():
            key, val = parts[0].strip(), parts[1].strip()
            info[key] = val
    return info

def _parse_function_name(lines: List[str], line_no: int) -> str:
    """
    Extract the function name from a line containing '<FunctionName>'.

    Args:
        lines: All lines of the file.
        line_no: 1-based line number where '<FunctionName>' appears.

    Returns:
        The bare function label string.
    """
    return lines[line_no - 1].strip().strip('<>').strip()

def _parse_function_meta(lines: List[str], idxs: List[int], func_idx: int, func_label: str) -> Dict[str, Any]:
    """
    Parse metadata entries for a given function block.

    Handles 'Normal Indentation' specially by parsing coordinate mappings and scalars.

    Args:
        lines: All lines of the file.
        idxs: Indices of all section dividers.
        func_idx: Zero-based index of the function block.
        func_label: The label of the function (e.g., 'Normal Indentation').

    Returns:
        A dict mapping f'<{func_label}>' to another dict of metadata entries.
    """
    func_key = f"<{func_label}>"
    start_meta = idxs[4*func_idx + 1] + 1
    data_start = idxs[4*func_idx + 2]

    # Normal Indentation special handling
    if func_label == "Normal Indentation":
        # find mapping and normal dividers
        mapping_idx = next((i for i, l in enumerate(lines) if INDENTATION_DIVIDERS[0] in l), None)
        normal_idx = next((i for i, l in enumerate(lines) if INDENTATION_DIVIDERS[1] in l), None)
        if mapping_idx is None or normal_idx is None:
            raise ValueError("Indentation delimiters not found")
        meta: Dict[str, Any] = {}
        # general metadata before mapping
        for ln in lines[start_meta:mapping_idx]:
            parts = ln.split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                k, v = parts[0].strip(), parts[1].strip()
                try: v = float(v)
                except: pass
                meta[k] = v
        # mapping points
        mapping_vals = [list(map(float, lines[j].split()))
                        for j in range(mapping_idx + 1, normal_idx)]
        meta[INDENTATION_DIVIDERS[0]] = np.array(mapping_vals)
        # surface normal
        normal_vals = list(map(float, lines[normal_idx + 1].split()))
        meta[INDENTATION_DIVIDERS[1]] = np.array([normal_vals])
        # scalar lines
        for ln in lines[normal_idx + 2: normal_idx + 5]:
            parts = ln.split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                k, v = parts[0].strip(), parts[1].strip()
                try: v = float(v)
                except: pass
                meta[k] = v
    else:
        # general metadata
        meta = {}
        for ln in lines[start_meta:data_start]:
            parts = ln.split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                k, v = parts[0].strip(), parts[1].strip()
                try: v = float(v)
                except: pass
                meta[k] = v
    return {func_key: meta}

def _parse_data_section(path: Path, lines: List[str], idxs: List[int], func_idx: int, has_seps: bool, sep_idxs: List[int], headers: Optional[List[str]]) -> Dict[str, Any]:
    """
    Parse the <DATA> section for a function, returning either a flat dict of numpy arrays
    or a dict of 'Ramp-X' segments when '<divider>' markers are present.

    Args:
        path: Path to the .txt file.
        lines: All lines from the file.
        idxs: List of section divider line indices.
        func_idx: Zero-based index of the function block.
        has_seps: Whether '<divider>' markers appear in this DATA block.
        sep_idxs: All line indices where '<divider>' appears.
        headers: Optional list of column names to include.

    Returns:
        A dict mapping column names to numpy arrays, or 'Ramp-X' to arrays.
    """
    start = idxs[4*func_idx + 2]
    end = idxs[4*func_idx + 3]
    total = max(end - start - 1, 0)
    if total <= 1:
        return {}
    # read headers
    header_df = pd.read_csv(path, sep="\t", skiprows=start, nrows=0, engine="c")
    cols = list(range(len(header_df.columns))) if headers is None else _get_data_columns(lines, start, headers)
    names = list(header_df.columns)
    def clean_arr(d: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
        return {c: np.array([v for v in vals if v != '' and pd.notna(v)])
                for c, vals in d.items()}
    # no separators → flat data
    if not has_seps:
        df = pd.read_csv(path, sep="\t", skiprows=start + 1, nrows=total - 1, usecols=cols, header=None, names=names, engine="c")
        return clean_arr(df.to_dict(orient='list'))

    # with separators → ramps
    segments: List[Tuple[int, int]] = []
    prev = start
    for s in sep_idxs:
        if start < s < end:
            segments.append((prev, s))
            prev = s
    segments.append((prev, end))

    data: Dict[str, Any] = {}
    for idx, (s, e) in enumerate(segments, start=1):
        seg_rows = max(e - s - 1, 0)
        if idx == 1:
            skip = s + 1
            nrows = seg_rows - 1
        else:
            skip = s
            nrows = seg_rows
        if nrows < 1:
            continue
        df = pd.read_csv(path, sep="\t", skiprows=skip, nrows=nrows, usecols=cols, header=None, names=names, engine="c")
        data[f"Ramp-{idx}"] = clean_arr(df.to_dict(orient='list'))
    return data

def _get_data_columns(lines: List[str], header_row: int, selections: Optional[List[str]]) -> Optional[List[int]]:
    """
    Determine column indices for the specified header names.

    Args:
        lines: All file lines.
        header_row: Line index where the header row appears.
        selections: List of column names to include.

    Returns:
        List of integer column indices or None if all columns are used.
    """
    if selections is None:
        return None
    cols = lines[header_row].split("\t")
    cols = [c.split(',')[0] for c in cols]
    return [cols.index(s) for s in selections if s in cols]

def _print_progress(iteration: int, total: int, prefix: str = '', suffix: str = '', decimals: int = 1, length: int = 50, fill: str = '█') -> None:
    """
    Display a progress bar in the terminal.

    Args:
        iteration: Current iteration count.
        total: Total iterations.
        prefix: Text prefix for the bar.
        suffix: Text suffix for the bar.
        decimals: Number of decimal places in the percentage display.
        length: Character length of the bar.
        fill: Character used to fill the bar.
    """
    if total < 1:
        total = 1
    pct = (iteration / total) * 100
    filled = int(length * iteration // total)
    bar = fill * filled + '-' * (length - filled)
    print(f"\r{prefix} |{bar}| {pct:.{decimals}f}% {suffix}", end='')
    if iteration == total:
        print()

def isNegative(posZ):
    """
    Check for positive sign for Z-position

    Args:
        posZ (np.array): Array Z-position (mm)
    
    Returns:
        posZ (np.array): Array Z-position (mm) with positive sign                               
    """
    if abs(posZ[-1]) < abs(posZ[0]):
        posZ = posZ + 2*abs(posZ[0])
    return posZ

def check_data(loadZ, posZ, Rsq_req):
    """
    Checks if data passes statistical tests

    Args:
        loadZ (np.array): Array Z-load (N or gf)
        posZ (np.array): Array Z-position (mm)
        Rsq_req (float): required R**2 value for test to be accepted
    
    Returns:
        req (int): req == 1 it fails the test.                               
    """
    poly_order = 2
    N = len(loadZ)
    X = np.zeros((N, poly_order + 1))
    Y = loadZ
    
    for k in range(N):
        X[k,0] = 1
        for l in range(poly_order):
            X[k, 1 + l] = posZ[k]**(l+1)
    B = np.linalg.solve(np.dot(X.transpose(),X),np.dot(X.transpose(),Y))
    mse = np.sum((Y - np.dot(X,B)**2))/N
    Rsq_adj = rsquared(Y, mse, poly_order)
    if Rsq_adj < Rsq_req:
        req = 0
    else:
        req = 1
    return req

def normalize_signal(signal):
    """
    Function to normalize signals 

    Args:
        Signal : Data to be normalize
    
    Returns:
        Signal normalized 
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(signal.reshape(-1, 1)).flatten()