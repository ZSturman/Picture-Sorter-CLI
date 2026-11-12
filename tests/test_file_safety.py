import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import subprocess
import sys
from media.base import BaseHandler, validate_directories

class FileSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.dest = self.root / 'dest'
        self.file = self.source / 'same.jpg'
        self.file.write_bytes(b'original media')
    def tearDown(self):
        self.temp.cleanup()
    def test_collision_preserves_both_files(self):
        folder = self.dest / 'category'
        folder.mkdir(parents=True)
        existing = folder / self.file.name
        existing.write_bytes(b'old destination')
        with self.assertRaises(FileExistsError):
            BaseHandler().save_to(self.dest, self.file, 'category')
        self.assertEqual(self.file.read_bytes(), b'original media')
        self.assertEqual(existing.read_bytes(), b'old destination')
    def test_copy_failure_retains_source_and_removes_owned_partial(self):
        def fail(source, target):
            target.write(b'partial')
            raise OSError('simulated disk write failure')
        with patch('media.base.shutil.copyfileobj', side_effect=fail):
            with self.assertRaises(OSError):
                BaseHandler().save_to(self.dest, self.file, 'category')
        self.assertEqual(self.file.read_bytes(), b'original media')
        self.assertFalse((self.dest/'category/same.jpg').exists())
    def test_source_change_is_retained(self):
        def change(source, target):
            target.write(source.read())
            self.file.write_bytes(b'changed by another process')
        with patch('media.base.shutil.copyfileobj', side_effect=change):
            with self.assertRaises(RuntimeError):
                BaseHandler().save_to(self.dest, self.file, 'category')
        self.assertEqual(self.file.read_bytes(), b'changed by another process')
        self.assertFalse((self.dest/'category/same.jpg').exists())
    def test_category_cannot_escape_destination(self):
        for category in ['..', '../outside', 'a/b', 'a\\b', '']:
            with self.assertRaises(ValueError):
                BaseHandler().save_to(self.dest, self.file, category)
        self.assertTrue(self.file.exists())
    def test_symlink_is_not_followed(self):
        self.dest.mkdir()
        (self.dest/'category').symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            BaseHandler().save_to(self.dest,self.file,'category')
        self.assertTrue(self.file.exists())
    def test_cli_rejects_missing_and_overlapping_sources(self):
        for src,dest in [(self.root/'absent',self.dest),(self.source,self.source/'sorted')]:
            result=subprocess.run([sys.executable,'pipeline.py','--src',str(src),'--dest',str(dest)],capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stderr)
        self.assertEqual(self.file.read_bytes(),b'original media')
    def test_short_ocr_uses_source_text_without_generative_claims(self):
        from media.image import summarize_ocr
        text = 'FUNCTIONAL AUDIT SYNTHETIC IMAGE NOT A REAL PHOTO'
        with patch('media.image.summarizer', side_effect=AssertionError('unnecessary model inference')):
            self.assertEqual(summarize_ocr(text), text)
    def test_cli_help_does_not_load_models(self):
        for script in ['pipeline.py','sort_video.py']:
            result=subprocess.run([sys.executable,script,'--help'],capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertNotIn('Downloading',result.stderr)

if __name__ == '__main__': unittest.main()
