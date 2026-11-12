#!/usr/bin/env python3
"""Video-only entry point using the shared caption/safe-move pipeline."""
import sys
from pipeline import main

if __name__ == '__main__':
    sys.exit(main(media_kind='video'))
