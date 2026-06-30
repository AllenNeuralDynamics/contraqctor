"""Demo script for testing ContextExportableObj serialization with images."""

import contraqctor.qc as qc


class ImageTestSuite(qc.Suite):
    """Test suite demonstrating image handling in reports."""

    name = "Image Visualization Tests"

    def test_matplotlib_figure(self):
        """Test that creates a matplotlib figure."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Create a simple plot
            fig, ax = plt.subplots(figsize=(8, 6))
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y, label="sin(x)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Sine Wave")
            ax.legend()
            ax.grid(True)

            # Add the figure to context using ContextExportableObj
            context = qc.ContextExportableObj.as_context(fig)
            context["data_points"] = len(x)

            plt.close(fig)

            return self.pass_test(True, "Matplotlib figure created successfully", context=context)
        except ImportError:
            return self.skip_test("Matplotlib not available")

    def test_pil_image(self):
        """Test that creates a PIL image."""
        try:
            from PIL import Image, ImageDraw

            # Create a simple image
            img = Image.new("RGB", (400, 300), color="white")
            draw = ImageDraw.Draw(img)

            # Draw some shapes
            draw.rectangle([50, 50, 350, 250], outline="blue", width=3)
            draw.ellipse([100, 100, 300, 200], fill="lightblue", outline="darkblue", width=2)
            draw.text((150, 130), "Test Image", fill="black")

            # Add the image to context
            context = qc.ContextExportableObj.as_context(img)
            context["image_size"] = img.size

            return self.pass_test(True, "PIL Image created successfully", context=context)
        except ImportError:
            return self.skip_test("PIL not available")

    def test_numpy_array_image(self):
        """Test that creates a numpy array image."""
        try:
            import numpy as np

            # Create a gradient image
            img = np.zeros((200, 300, 3), dtype=np.uint8)
            for i in range(200):
                for j in range(300):
                    img[i, j] = [int(i / 200 * 255), int(j / 300 * 255), 128]

            # Add the image to context
            context = qc.ContextExportableObj.as_context(img)
            context["array_shape"] = str(img.shape)
            context["array_dtype"] = str(img.dtype)

            return self.pass_test(True, "Numpy array image created successfully", context=context)
        except ImportError:
            return self.skip_test("Numpy not available")

    def test_multiple_images(self):
        """Test with multiple images in context."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Create two different plots
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            x = np.linspace(0, 2 * np.pi, 100)
            ax1.plot(x, np.sin(x), "r-", label="sin")
            ax1.set_title("Sine")
            ax1.legend()

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(x, np.cos(x), "b-", label="cos")
            ax2.set_title("Cosine")
            ax2.legend()

            # Create context with multiple images (not using ContextExportableObj.as_context
            # because it only handles one asset)
            context = {
                "sine_plot": qc.ContextExportableObj(fig1),
                "cosine_plot": qc.ContextExportableObj(fig2),
                "note": "Two separate plots",
            }

            plt.close(fig1)
            plt.close(fig2)

            return self.pass_test(True, "Multiple images created successfully", context=context)
        except ImportError:
            return self.skip_test("Matplotlib not available")

    def test_failed_with_image(self):
        """Test that fails but includes an image for debugging."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Create a plot showing the failure
            fig, ax = plt.subplots(figsize=(8, 6))
            x = np.array([1, 2, 3, 4, 5])
            expected = np.array([2, 4, 6, 8, 10])
            actual = np.array([2, 4, 7, 8, 10])  # Deviation at x=3

            ax.plot(x, expected, "g-o", label="Expected")
            ax.plot(x, actual, "r-x", label="Actual")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Data Validation Failure")
            ax.legend()
            ax.grid(True)

            context = qc.ContextExportableObj.as_context(fig)
            context["deviation_index"] = 2
            context["expected_value"] = 6
            context["actual_value"] = 7

            plt.close(fig)

            return self.fail_test(False, "Data validation failed at index 2", context=context)
        except ImportError:
            return self.skip_test("Matplotlib not available")


class DataQualitySuite(qc.Suite):
    """Test suite for data quality checks."""

    name = "Data Quality"

    def test_data_range(self):
        """Test data is within expected range."""
        return self.pass_test(True, "Data within range")

    def test_no_missing_values(self):
        """Test no missing values."""
        return self.pass_test(True, "No missing values found")

    def test_data_type_validation(self):
        """Test data types are correct."""
        return self.warn_test(True, "Some data types could be optimized")


def main():
    """Run the demo and generate both console and HTML reports."""
    print("Running Image Visualization Demo\n")

    # Create runner and add test suites
    runner = qc.Runner()
    runner.add_suite(ImageTestSuite(), "Visualization")
    runner.add_suite(DataQualitySuite(), "Data Quality")

    # Manually run tests and collect results
    from contraqctor.qc.base import ResultsStatistics, _TaggedResult

    tagged_results = []
    for group, suites in runner.suites.items():
        for suite in suites:
            for test_method in suite.get_tests():
                results_iter = suite.run_test(test_method)
                result = next(iter(results_iter))
                tagged_results.append(_TaggedResult(suite=suite, group=group, result=result, test=test_method))

    stats = ResultsStatistics.from_results([tr.result for tr in tagged_results])

    # Display with console reporter (with asset serialization)
    print("=" * 80)
    print("CONSOLE OUTPUT (with asset serialization enabled)")
    print("=" * 80 + "\n")

    console_reporter = qc.ConsoleReporter()
    console_reporter.report_results(
        tagged_results,
        stats,
        serialize_context_exportable_obj=True,
        asset_output_dir="./report/assets",
    )

    # Generate HTML report WITHOUT serialization (baseline)
    print("\n" + "=" * 80)
    print("Generating HTML report WITHOUT serialization...")
    html_reporter_no_serialize = qc.HtmlReporter("test_report_no_serialize.html")
    html_reporter_no_serialize.report_results(
        tagged_results,
        stats,
        serialize_context_exportable_obj=False,
    )
    print("HTML report (without serialization) saved to: test_report_no_serialize.html")

    # Generate HTML report WITH serialization
    print("\n" + "=" * 80)
    print("Generating HTML report WITH serialization...")
    html_reporter_with_serialize = qc.HtmlReporter("test_report_with_serialize.html")
    html_reporter_with_serialize.report_results(
        tagged_results,
        stats,
        serialize_context_exportable_obj=True,
    )
    print("HTML report (with serialization) saved to: test_report_with_serialize.html")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {stats.total}")
    print(f"Passed: {stats.passed}")
    print(f"Failed: {stats.failed}")
    print(f"Errors: {stats.error}")
    print(f"Warnings: {stats.warnings}")
    print(f"Skipped: {stats.skipped}")

    print("\nGenerated files:")
    print("  - ./report/assets/ (serialized image files for console)")
    print("  - test_report_no_serialize.html (shows raw context)")
    print("  - test_report_with_serialize.html (shows embedded images)")
    print("\nOpen the HTML files in your browser to see the difference!")


if __name__ == "__main__":
    main()
