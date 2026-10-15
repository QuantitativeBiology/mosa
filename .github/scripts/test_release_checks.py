import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from release_checks import check_distributions, check_source


class ReleaseChecksTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "src/mosa").mkdir(parents=True)
        (self.root / "pyproject.toml").write_text(
            '[project]\nname = "mosalib"\nversion = "1.2.3"\n'
        )
        (self.root / "src/mosa/__init__.py").write_text('__version__ = "1.2.3"\n')
        self.dist = self.root / "dist"
        self.dist.mkdir()

    def make_distributions(
        self, wheel_version="1.2.3", sdist_version="1.2.3", name="mosalib"
    ):
        with zipfile.ZipFile(
            self.dist / "mosalib-1.2.3-py3-none-any.whl", "w"
        ) as archive:
            archive.writestr(
                "mosalib-1.2.3.dist-info/METADATA",
                f"Name: {name}\nVersion: {wheel_version}\n",
            )
        with tarfile.open(self.dist / "mosalib-1.2.3.tar.gz", "w:gz") as archive:
            data = f"Name: {name}\nVersion: {sdist_version}\n".encode()
            member = tarfile.TarInfo("mosalib-1.2.3/PKG-INFO")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))

    def test_matching_release(self):
        self.assertEqual(check_source(self.root, "v1.2.3"), "1.2.3")
        self.make_distributions()
        check_distributions(self.dist, "1.2.3")

    def test_branch_dry_run(self):
        self.assertEqual(check_source(self.root, ""), "1.2.3")

    def test_wrong_or_invalid_tag(self):
        for tag in ("v1.2.4", "version-test", "1.2.3", "v1.2.3rc1", "v01.2.3"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                check_source(self.root, tag)

    def test_source_version_mismatch(self):
        (self.root / "src/mosa/__init__.py").write_text('__version__ = "1.2.2"\n')
        with self.assertRaises(ValueError):
            check_source(self.root, "v1.2.3")

    def test_distribution_version_mismatch(self):
        for wheel, sdist in (("1.2.2", "1.2.3"), ("1.2.3", "1.2.2")):
            with self.subTest(wheel=wheel, sdist=sdist):
                self.make_distributions(wheel, sdist)
                with self.assertRaises(ValueError):
                    check_distributions(self.dist, "1.2.3")

    def test_wrong_distribution_name(self):
        self.make_distributions(name="another-project")
        with self.assertRaises(ValueError):
            check_distributions(self.dist, "1.2.3")

    def test_missing_or_extra_artifacts(self):
        with self.assertRaises(ValueError):
            check_distributions(self.dist, "1.2.3")
        self.make_distributions()
        (self.dist / "unexpected.txt").write_text("unexpected")
        with self.assertRaises(ValueError):
            check_distributions(self.dist, "1.2.3")


if __name__ == "__main__":
    unittest.main()
