"""Tests for serialization features."""

import tempfile
from pathlib import Path

import pytest

from contraqctor.qc._context_extensions import ContextExportableObj
from contraqctor.qc.base import Suite, _TaggedResult
from contraqctor.qc.reporters import ConsoleReporter, HtmlReporter
from contraqctor.qc.serializers import (
    ContextExportableObjSerializer,
    MatplotlibFigureSerializer,
    NumpyArrayImageSerializer,
    PILImageSerializer,
    TypeSerializer,
)


class TestITypeSerializer:
    """Test the ITypeSerializer protocol."""

    def test_protocol_implementation(self):
        class CustomSerializer(TypeSerializer):
            def can_serialize(self, obj):
                return isinstance(obj, dict)

            def serialize_as_bytes(self, obj):
                return {"type": "custom", "data": str(obj)}

            def serialize_as_file(self, obj, output_dir: Path, filename: str):
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / f"{filename}.txt"
                path.write_text(str(obj))
                return {"type": "custom", "path": str(path)}

        serializer = CustomSerializer()
        assert serializer.can_serialize({"key": "value"})
        assert not serializer.can_serialize("string")

        result = serializer.serialize_as_bytes({"key": "value"})
        assert result["type"] == "custom"
        assert "key" in result["data"]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serializer.serialize_as_file({"key": "value"}, Path(tmpdir), "test")
            assert result["type"] == "custom"
            assert Path(result["path"]).exists()


class TestMatplotlibFigureSerializer:
    """Test matplotlib figure serialization."""

    def test_can_serialize_matplotlib_figure(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        serializer = MatplotlibFigureSerializer()
        fig, ax = plt.subplots()
        assert serializer.can_serialize(fig)
        assert not serializer.can_serialize("not a figure")
        plt.close(fig)

    def test_serialize_as_bytes(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        serializer = MatplotlibFigureSerializer()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])

        result = serializer.serialize_as_bytes(fig)
        assert result["type"] == "image"
        assert result["data"].startswith("data:image/png;base64,")
        plt.close(fig)

    def test_serialize_as_file(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        serializer = MatplotlibFigureSerializer()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serializer.serialize_as_file(fig, Path(tmpdir), "test_plot")
            assert result["type"] == "image"
            assert Path(result["path"]).exists()
            assert result["path"].endswith(".png")
        plt.close(fig)


class TestPILImageSerializer:
    """Test PIL image serialization."""

    def test_can_serialize_pil_image(self):
        pytest.importorskip("PIL")
        from PIL import Image

        serializer = PILImageSerializer()
        img = Image.new("RGB", (100, 100), color="red")
        assert serializer.can_serialize(img)
        assert not serializer.can_serialize("not an image")

    def test_serialize_as_bytes(self):
        pytest.importorskip("PIL")
        from PIL import Image

        serializer = PILImageSerializer()
        img = Image.new("RGB", (100, 100), color="red")

        result = serializer.serialize_as_bytes(img)
        assert result["type"] == "image"
        assert result["data"].startswith("data:image/png;base64,")

    def test_serialize_as_file(self):
        pytest.importorskip("PIL")
        from PIL import Image

        serializer = PILImageSerializer()
        img = Image.new("RGB", (100, 100), color="red")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serializer.serialize_as_file(img, Path(tmpdir), "test_image")
            assert result["type"] == "image"
            assert Path(result["path"]).exists()
            assert result["path"].endswith(".png")


class TestNumpyArrayImageSerializer:
    """Test numpy array image serialization."""

    def test_can_serialize_2d_array(self):
        pytest.importorskip("numpy")
        import numpy as np

        serializer = NumpyArrayImageSerializer()
        arr_2d = np.zeros((100, 100), dtype=np.uint8)
        assert serializer.can_serialize(arr_2d)

    def test_can_serialize_3d_rgb_array(self):
        pytest.importorskip("numpy")
        import numpy as np

        serializer = NumpyArrayImageSerializer()
        arr_3d = np.zeros((100, 100, 3), dtype=np.uint8)
        assert serializer.can_serialize(arr_3d)

    def test_cannot_serialize_invalid_arrays(self):
        pytest.importorskip("numpy")
        import numpy as np

        serializer = NumpyArrayImageSerializer()
        assert not serializer.can_serialize(np.zeros((100,)))
        assert not serializer.can_serialize(np.zeros((100, 100, 5)))
        assert not serializer.can_serialize("not an array")

    def test_serialize_as_bytes(self):
        pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        import numpy as np

        serializer = NumpyArrayImageSerializer()
        arr = np.zeros((100, 100, 3), dtype=np.uint8)

        result = serializer.serialize_as_bytes(arr)
        assert result["type"] == "image"
        assert result["data"].startswith("data:image/png;base64,")

    def test_serialize_as_file(self):
        pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        import numpy as np

        serializer = NumpyArrayImageSerializer()
        arr = np.zeros((100, 100, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serializer.serialize_as_file(arr, Path(tmpdir), "test_array")
            assert result["type"] == "image"
            assert Path(result["path"]).exists()
            assert result["path"].endswith(".png")


class TestContextExportableObjSerializer:
    """Test ContextExportableObj serialization."""

    def test_serialize_as_bytes_with_matplotlib(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        serializer = ContextExportableObjSerializer()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])

        context = ContextExportableObj.as_context(fig)
        result = serializer.serialize_as_bytes(context)

        assert "asset" in result
        assert result["asset"]["type"] == "image"
        assert result["asset"]["data"].startswith("data:image/png;base64,")
        plt.close(fig)

    def test_serialize_as_file_with_matplotlib(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        serializer = ContextExportableObjSerializer()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])

        context = ContextExportableObj.as_context(fig)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serializer.serialize_as_file(context, Path(tmpdir), "test")
            assert "asset" in result
            assert result["asset"]["type"] == "image"
            assert Path(result["asset"]["path"]).exists()
        plt.close(fig)

    def test_serialize_multiple_exportable_objects(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        serializer = ContextExportableObjSerializer()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        context = {
            "plot1": ContextExportableObj(fig1),
            "plot2": ContextExportableObj(fig2),
            "text": "some data",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serializer.serialize_as_file(context, Path(tmpdir), "test")
            assert result["plot1"]["type"] == "image"
            assert result["plot2"]["type"] == "image"
            assert result["text"] == "some data"
            assert Path(result["plot1"]["path"]).exists()
            assert Path(result["plot2"]["path"]).exists()
        plt.close(fig1)
        plt.close(fig2)

    def test_add_custom_serializer(self):
        class CustomSerializer(TypeSerializer):
            def can_serialize(self, obj):
                return isinstance(obj, dict) and "custom_type" in obj

            def serialize_as_bytes(self, obj):
                return {"type": "custom", "data": obj["value"]}

            def serialize_as_file(self, obj, output_dir: Path, filename: str):
                return {"type": "custom", "value": obj["value"]}

        serializer = ContextExportableObjSerializer()
        serializer.add_serializer(CustomSerializer())

        obj = {"custom_type": True, "value": "test"}
        context = ContextExportableObj.as_context(obj)

        result = serializer.serialize_as_bytes(context)
        assert result["asset"]["type"] == "custom"
        assert result["asset"]["data"] == "test"


class TestConsoleReporterSerialization:
    """Test ConsoleReporter with serialization."""

    def test_serialize_all_results_not_just_displayed(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        class TestSuite(Suite):
            name = "Test Suite"

            def test_pass_with_image(self):
                fig, ax = plt.subplots()
                context = ContextExportableObj.as_context(fig)
                plt.close(fig)
                return self.pass_test(True, "Passed", context=context)

            def test_fail_with_image(self):
                fig, ax = plt.subplots()
                context = ContextExportableObj.as_context(fig)
                plt.close(fig)
                return self.fail_test(False, "Failed", context=context)

        suite = TestSuite()
        tagged_results = []
        for test_method in suite.get_tests():
            results_iter = suite.run_test(test_method)
            result = next(iter(results_iter))
            tagged_results.append(_TaggedResult(suite=suite, group="Test", result=result, test=test_method))

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ConsoleReporter()
            reporter.report_results(tagged_results, serialize_context_exportable_obj=True, asset_output_dir=tmpdir)

            files = list(Path(tmpdir).glob("*.png"))
            assert len(files) == 2

    def test_default_asset_output_dir(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        class TestSuite(Suite):
            name = "Test Suite"

            def test_fail_with_image(self):
                fig, ax = plt.subplots()
                context = ContextExportableObj.as_context(fig)
                plt.close(fig)
                return self.fail_test(False, "Failed", context=context)

        suite = TestSuite()
        tagged_results = []
        for test_method in suite.get_tests():
            results_iter = suite.run_test(test_method)
            result = next(iter(results_iter))
            tagged_results.append(_TaggedResult(suite=suite, group="Test", result=result, test=test_method))

        default_dir = Path("./report/assets")
        if default_dir.exists():
            import shutil

            shutil.rmtree(default_dir)

        reporter = ConsoleReporter()
        reporter.report_results(tagged_results, serialize_context_exportable_obj=True)

        assert default_dir.exists()
        files = list(default_dir.glob("*.png"))
        assert len(files) >= 1

        import shutil

        shutil.rmtree(default_dir.parent)


class TestHtmlReporterSerialization:
    """Test HtmlReporter with serialization."""

    def test_serialize_context_as_bytes(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        class TestSuite(Suite):
            name = "Test Suite"

            def test_with_image(self):
                fig, ax = plt.subplots()
                context = ContextExportableObj.as_context(fig)
                plt.close(fig)
                return self.pass_test(True, "Passed", context=context)

        suite = TestSuite()
        tagged_results = []
        for test_method in suite.get_tests():
            results_iter = suite.run_test(test_method)
            result = next(iter(results_iter))
            tagged_results.append(_TaggedResult(suite=suite, group="Test", result=result, test=test_method))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.html"
            reporter = HtmlReporter(output_path)
            reporter.report_results(tagged_results, serialize_context_exportable_obj=True)

            assert output_path.exists()
            content = output_path.read_text()
            assert "data:image/png;base64," in content

    def test_no_serialization_shows_raw_context(self):
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        class TestSuite(Suite):
            name = "Test Suite"

            def test_with_image(self):
                fig, ax = plt.subplots()
                context = ContextExportableObj.as_context(fig)
                plt.close(fig)
                return self.pass_test(True, "Passed", context=context)

        suite = TestSuite()
        tagged_results = []
        for test_method in suite.get_tests():
            results_iter = suite.run_test(test_method)
            result = next(iter(results_iter))
            tagged_results.append(_TaggedResult(suite=suite, group="Test", result=result, test=test_method))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.html"
            reporter = HtmlReporter(output_path)
            reporter.report_results(tagged_results, serialize_context_exportable_obj=False)

            assert output_path.exists()
            content = output_path.read_text()
            assert "data:image/png;base64," not in content
            assert "ContextExportableObj" in content
