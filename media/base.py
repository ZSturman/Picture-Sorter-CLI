# media/base.py
import shutil
import os
from pathlib import Path


def validate_directories(src, dest):
    source, target = Path(src).resolve(), Path(dest).resolve()
    if not source.is_dir():
        raise ValueError("Source must be an existing directory")
    if target.exists() and not target.is_dir():
        raise ValueError("Destination must be a directory")
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and destination directories must not overlap")
    return source, target

class BaseHandler:
    def load(self, path: str):
        """Load/validate file; return a media‐specific object."""
        raise NotImplementedError

    def categorize(self, media_obj) -> str:
        """Run analysis/caption/detection and return a category string."""
        raise NotImplementedError

    def save_to(self, dest_root: str, src_path: str, category: str):
        """
        Create dest_root/category and move the file there.
        """
        if not isinstance(category, str) or not category.strip() or category in {'.', '..'} or '/' in category or '\\' in category:
            raise ValueError("Category must be one non-empty folder name")
        source = Path(src_path)
        if source.is_symlink() or not source.is_file():
            raise ValueError("Source must be a regular file, not a symbolic link")
        root = Path(dest_root).resolve()
        dest_dir = root / category
        if dest_dir.is_symlink():
            raise ValueError("Category directory must not be a symbolic link")
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / source.name
        if target.resolve() == source.resolve():
            raise ValueError("Source already occupies the destination")
        # Exclusive creation prevents an existing file from being overwritten.
        with target.open('xb') as output:
            identity = os.fstat(output.fileno())
            try:
                with source.open('rb') as input_file:
                    original = os.fstat(input_file.fileno())
                    shutil.copyfileobj(input_file, output)
                output.flush()
                os.fsync(output.fileno())
                shutil.copystat(source, target)
                current_source = source.stat()
                if (current_source.st_dev, current_source.st_ino, current_source.st_size, current_source.st_mtime_ns) != (original.st_dev, original.st_ino, original.st_size, original.st_mtime_ns):
                    raise RuntimeError('Source changed during copy; original retained')
                source.unlink()
            except BaseException:
                current = target.stat()
                if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                    target.unlink()
                raise
        return str(target)
