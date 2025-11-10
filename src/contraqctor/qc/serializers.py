import base64
import io
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

from contraqctor.qc._context_extensions import ContextExportableObj


class ITypeSerializer(t.Protocol):
    """Protocol defining the interface for type-specific serializers.

    This protocol ensures that all serializer implementations provide both
    byte-based (for embedding) and file-based (for external storage) serialization.
    """

    def can_serialize(self, obj: t.Any) -> bool:
        """Check if this serializer can handle the given object type.

        Args:
            obj: Object to check.

        Returns:
            bool: True if this serializer can handle the object.
        """
        ...

    def serialize_as_bytes(self, obj: t.Any) -> t.Dict[str, t.Any]:
        """Serialize the object to a dictionary with base64-encoded bytes.

        Used for embedding data directly in reports (e.g., HTML).

        Args:
            obj: Object to serialize.

        Returns:
            Dict containing the serialized representation with base64-encoded data.
        """
        ...

    def serialize_as_file(self, obj: t.Any, output_dir: Path, filename: str) -> t.Dict[str, t.Any]:
        """Serialize the object to a file and return metadata.

        Used for saving data to external files (e.g., for CLI reports).

        Args:
            obj: Object to serialize.
            output_dir: Directory where the file should be saved.
            filename: Name for the output file (without extension).

        Returns:
            Dict containing metadata including 'type', 'path', and other relevant info.
        """
        ...


class TypeSerializer(ABC):
    """Base class for type-specific serializers.

    Implements the ITypeSerializer protocol with abstract methods that must be
    overridden by subclasses.
    """

    @abstractmethod
    def can_serialize(self, obj: t.Any) -> bool:
        """Check if this serializer can handle the given object type.

        Args:
            obj: Object to check.

        Returns:
            bool: True if this serializer can handle the object.
        """
        pass

    @abstractmethod
    def serialize_as_bytes(self, obj: t.Any) -> t.Dict[str, t.Any]:
        """Serialize the object to a dictionary with base64-encoded bytes.

        Used for embedding data directly in reports (e.g., HTML).

        Args:
            obj: Object to serialize.

        Returns:
            Dict containing the serialized representation with base64-encoded data.
        """
        pass

    @abstractmethod
    def serialize_as_file(self, obj: t.Any, output_dir: Path, filename: str) -> t.Dict[str, t.Any]:
        """Serialize the object to a file and return metadata.

        Used for saving data to external files (e.g., for CLI reports).

        Args:
            obj: Object to serialize.
            output_dir: Directory where the file should be saved.
            filename: Name for the output file (without extension).

        Returns:
            Dict containing metadata including 'type', 'path', and other relevant info.
        """
        pass


class MatplotlibFigureSerializer(TypeSerializer):
    """Serializer for matplotlib Figure objects."""

    def can_serialize(self, obj: t.Any) -> bool:
        """Check if object is a matplotlib Figure."""
        try:
            import matplotlib.figure

            return isinstance(obj, matplotlib.figure.Figure)
        except ImportError:
            return False

    def serialize_as_bytes(self, obj: t.Any) -> t.Dict[str, t.Any]:
        """Serialize matplotlib Figure to base64-encoded PNG.

        Args:
            obj: Matplotlib Figure object.

        Returns:
            Dict with 'type' and 'data' keys containing base64-encoded PNG.
        """
        buf = io.BytesIO()
        obj.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        return {"type": "image", "data": f"data:image/png;base64,{img_base64}"}

    def serialize_as_file(self, obj: t.Any, output_dir: Path, filename: str) -> t.Dict[str, t.Any]:
        """Serialize matplotlib Figure to a PNG file.

        Args:
            obj: Matplotlib Figure object.
            output_dir: Directory where the file should be saved.
            filename: Name for the output file (without extension).

        Returns:
            Dict with 'type' and 'path' keys.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filename}.png"
        obj.savefig(output_path, format="png", bbox_inches="tight")

        return {"type": "image", "path": str(output_path)}


class PILImageSerializer(TypeSerializer):
    """Serializer for PIL Image objects."""

    def can_serialize(self, obj: t.Any) -> bool:
        """Check if object is a PIL Image."""
        try:
            from PIL import Image

            return isinstance(obj, Image.Image)
        except ImportError:
            return False

    def serialize_as_bytes(self, obj: t.Any) -> t.Dict[str, t.Any]:
        """Serialize PIL Image to base64-encoded PNG.

        Args:
            obj: PIL Image object.

        Returns:
            Dict with 'type' and 'data' keys containing base64-encoded PNG.
        """
        buf = io.BytesIO()
        obj.save(buf, format="PNG")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        return {"type": "image", "data": f"data:image/png;base64,{img_base64}"}

    def serialize_as_file(self, obj: t.Any, output_dir: Path, filename: str) -> t.Dict[str, t.Any]:
        """Serialize PIL Image to a PNG file.

        Args:
            obj: PIL Image object.
            output_dir: Directory where the file should be saved.
            filename: Name for the output file (without extension).

        Returns:
            Dict with 'type' and 'path' keys.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filename}.png"
        obj.save(output_path, format="PNG")

        return {"type": "image", "path": str(output_path)}


class NumpyArrayImageSerializer(TypeSerializer):
    """Serializer for numpy arrays representing images."""

    def can_serialize(self, obj: t.Any) -> bool:
        """Check if object is a numpy array that could be an image.

        Valid image arrays are 2D (grayscale) or 3D with 3 or 4 channels (RGB/RGBA).
        """
        try:
            import numpy as np

            if not isinstance(obj, np.ndarray):
                return False

            if obj.ndim == 2:
                return True
            elif obj.ndim == 3 and obj.shape[2] in (3, 4):
                return True

            return False
        except ImportError:
            return False

    def _normalize_array(self, obj: t.Any) -> t.Any:
        """Normalize array to uint8 range."""
        import numpy as np

        if obj.dtype != np.uint8:
            if obj.max() <= 1.0:
                return (obj * 255).astype(np.uint8)
            else:
                return obj.astype(np.uint8)
        return obj

    def _array_to_pil(self, obj: t.Any):
        """Convert numpy array to PIL Image."""
        from PIL import Image

        obj = self._normalize_array(obj)

        if obj.ndim == 2:
            return Image.fromarray(obj, mode="L")
        elif obj.shape[2] == 3:
            return Image.fromarray(obj, mode="RGB")
        elif obj.shape[2] == 4:
            return Image.fromarray(obj, mode="RGBA")
        else:
            raise ValueError(f"Unexpected array shape: {obj.shape}")

    def serialize_as_bytes(self, obj: t.Any) -> t.Dict[str, t.Any]:
        """Serialize numpy array to base64-encoded PNG using PIL.

        Args:
            obj: Numpy array representing an image.

        Returns:
            Dict with 'type' and 'data' keys containing base64-encoded PNG.
        """
        img = self._array_to_pil(obj)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        return {"type": "image", "data": f"data:image/png;base64,{img_base64}"}

    def serialize_as_file(self, obj: t.Any, output_dir: Path, filename: str) -> t.Dict[str, t.Any]:
        """Serialize numpy array to a PNG file.

        Args:
            obj: Numpy array representing an image.
            output_dir: Directory where the file should be saved.
            filename: Name for the output file (without extension).

        Returns:
            Dict with 'type' and 'path' keys.
        """
        img = self._array_to_pil(obj)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{filename}.png"
        img.save(output_path, format="PNG")

        return {"type": "image", "path": str(output_path)}


KNOWN_SERIALIZERS: t.List[TypeSerializer] = [
    MatplotlibFigureSerializer(),
    PILImageSerializer(),
    NumpyArrayImageSerializer(),
]


class ContextExportableObjSerializer:
    """Serializer for ContextExportableObj instances.

    This class recursively searches through dictionaries and serializes
    ContextExportableObj instances based on their wrapped object types.

    Attributes:
        serializers: List of ITypeSerializer instances to use for serialization.
    """

    def __init__(self, serializers: t.Optional[t.List[ITypeSerializer]] = None):
        """Initialize the serializer with optional custom type serializers.

        Args:
            serializers: List of ITypeSerializer instances. If None, uses default
                serializers for matplotlib figures and PIL/numpy images.
        """
        if serializers is None:
            self.serializers: t.List[ITypeSerializer] = t.cast(t.List[ITypeSerializer], KNOWN_SERIALIZERS.copy())
        else:
            self.serializers = serializers

    def add_serializer(self, serializer: ITypeSerializer) -> None:
        """Add a custom type serializer.

        Args:
            serializer: ITypeSerializer instance to add.
        """
        self.serializers.append(serializer)

    def serialize_as_bytes(self, context: t.Any) -> t.Any:
        """Recursively serialize ContextExportableObj instances to base64-encoded bytes.

        This method walks through the context structure and serializes any
        ContextExportableObj instances it finds to base64-encoded data.

        Args:
            context: Context data to serialize (can be dict, list, or any value).

        Returns:
            Serialized context with ContextExportableObj instances replaced by
            their base64-encoded representations.
        """
        if isinstance(context, dict):
            result = {}
            for key, value in context.items():
                if isinstance(value, ContextExportableObj):
                    result[key] = self._serialize_exportable_obj_as_bytes(value)
                else:
                    result[key] = self.serialize_as_bytes(value)
            return result
        elif isinstance(context, (list, tuple)):
            return type(context)(self.serialize_as_bytes(item) for item in context)
        elif isinstance(context, ContextExportableObj):
            return self._serialize_exportable_obj_as_bytes(context)
        else:
            return context

    def serialize_as_file(
        self, context: t.Any, output_dir: Path, base_filename: str = "asset", counter: t.Optional[dict] = None
    ) -> t.Any:
        """Recursively serialize ContextExportableObj instances to files.

        This method walks through the context structure and serializes any
        ContextExportableObj instances it finds to external files.

        Args:
            context: Context data to serialize (can be dict, list, or any value).
            output_dir: Directory where files should be saved.
            base_filename: Base name for output files.
            counter: Optional counter dict to track file numbers (for internal use).

        Returns:
            Serialized context with ContextExportableObj instances replaced by
            their file path representations.
        """
        if counter is None:
            counter = {"count": 0}

        if isinstance(context, dict):
            result = {}
            for key, value in context.items():
                if isinstance(value, ContextExportableObj):
                    result[key] = self._serialize_exportable_obj_as_file(value, output_dir, base_filename, counter)
                else:
                    result[key] = self.serialize_as_file(value, output_dir, base_filename, counter)
            return result
        elif isinstance(context, (list, tuple)):
            return type(context)(self.serialize_as_file(item, output_dir, base_filename, counter) for item in context)
        elif isinstance(context, ContextExportableObj):
            return self._serialize_exportable_obj_as_file(context, output_dir, base_filename, counter)
        else:
            return context

    def _serialize_exportable_obj_as_bytes(self, obj: ContextExportableObj) -> t.Any:
        """Serialize a single ContextExportableObj to base64-encoded bytes.

        Args:
            obj: ContextExportableObj to serialize.

        Returns:
            Serialized representation of the wrapped object, or the original
            object if no serializer can handle it.
        """
        wrapped_obj = obj.asset

        for serializer in self.serializers:
            if serializer.can_serialize(wrapped_obj):
                return serializer.serialize_as_bytes(wrapped_obj)

        # If no serializer can handle it, return the object as-is
        return wrapped_obj

    def _serialize_exportable_obj_as_file(
        self, obj: ContextExportableObj, output_dir: Path, base_filename: str, counter: dict
    ) -> t.Any:
        """Serialize a single ContextExportableObj to a file.

        Args:
            obj: ContextExportableObj to serialize.
            output_dir: Directory where the file should be saved.
            base_filename: Base name for the output file.
            counter: Counter dict to track file numbers.

        Returns:
            Serialized representation with file path, or the original
            object if no serializer can handle it.
        """
        wrapped_obj = obj.asset

        for serializer in self.serializers:
            if serializer.can_serialize(wrapped_obj):
                filename = f"{base_filename}_{counter['count']}"
                counter["count"] += 1
                return serializer.serialize_as_file(wrapped_obj, output_dir, filename)

        # If no serializer can handle it, return the object as-is
        return wrapped_obj
