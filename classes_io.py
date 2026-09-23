

def read_classes(filepath):
    """Read class codes and labels from a TSV file.

    Args:
        filepath (str): Path to the TSV file containing "code<tab>name" rows

    Returns:
        List[dict]: Ordered list of {"code": int, "name": str} dicts

    Raises:
        FileNotFoundError: File doesn't exist at given path
        ValueError: Invalid line format detected (raises on first bad input)
                    expects exactly 2 tab-separated fields per row
"""
import os


def read_classes(filepath):
    b = []
    with open(filepath, "r", encoding="utf-8") as file:
        for lineno,line in enumerate(file,start=1):  
            line=line.rstrip("\n")  
            parts=line.split("\t")
            entry = {"code": int(parts[0]), "name": parts[1]}
            b.append(entry)
    return b


