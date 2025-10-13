from .base import DataStream

_ICON_MAP = {
    False: "📄",
    True: "📂",
    None: "❓",
}


def _get_node_icon(node: DataStream, show_missing_indicator: bool) -> str:
    """Determines the icon for a data stream node based on its type and data presence."""
    node_icon = _ICON_MAP[node.is_collection]
    if not node.has_data and show_missing_indicator:
        node_icon += _ICON_MAP[None]
    return node_icon


def _build_line_prefix(parents: list[bool], is_last: bool) -> str:
    """Builds the line prefix for a node based on its position in the tree."""
    line_prefix = ""
    for parent_is_last in parents[:-1]:
        line_prefix += "    " if parent_is_last else "│   "
    if parents:
        branch = "└── " if is_last else "├── "
        line_prefix += branch
    return line_prefix


def _build_node_label(node: DataStream, show_type: bool, show_params: bool) -> str:
    """Builds the label for a data stream node, optionally including type and parameters."""
    node_label = node.name
    if show_type:
        node_label += f" [{node.__class__.__name__}]"
    if show_params and hasattr(node, "reader_params") and node.reader_params:
        node_label += f" ({node.reader_params})"
    return node_label


def print_data_stream_tree(
    node: DataStream,
    prefix: str = "",
    is_last: bool = True,
    parents: list[bool] = [],
    show_params: bool = False,
    show_type: bool = False,
    show_missing_indicator: bool = True,
) -> str:
    """Generates a tree representation of a data stream hierarchy.

    Creates a formatted string displaying the hierarchical structure of a data stream
    and its children as a tree with branch indicators and icons.

    Args:
        node: The data stream node to start printing from.
        prefix: Prefix string to prepend to each line, used for indentation.
        is_last: Whether this node is the last child of its parent.
        parents: List tracking whether each ancestor was a last child, used for drawing branches.
        show_params: Whether to render parameters of the datastream.
        show_type: Whether to render the class name of the datastream.
        show_missing_indicator: Whether to render the missing data indicator.

    Returns:
        str: A formatted string representing the data stream tree.

    Examples:
        ```python
        from contraqctor.contract import Dataset, csv, json
        from contraqctor.contract.utils import print_data_stream_tree

        csv_stream = csv.Csv("data", reader_params=csv.CsvParams(path="data.csv"))
        json_stream = json.Json("config", reader_params=json.JsonParams(path="config.json"))
        dataset = Dataset("experiment", [csv_stream, json_stream], version="1.0.0")

        tree = print_data_stream_tree(dataset)
        print(tree)
        ```
    """
    node_icon = _get_node_icon(node, show_missing_indicator)
    line_prefix = _build_line_prefix(parents, is_last)
    node_label = _build_node_label(node, show_type, show_params)

    tree_representation = f"{line_prefix}{node_icon} {node_label}\n"

    if node.is_collection and node.has_data:
        for i, child in enumerate(node.data):
            child_is_last = i == len(node.data) - 1
            tree_representation += print_data_stream_tree(
                child,
                prefix="",
                is_last=child_is_last,
                parents=parents + [is_last],
                show_params=show_params,
                show_type=show_type,
                show_missing_indicator=show_missing_indicator,
            )

    return tree_representation


def _get_html_header() -> str:
    """Returns the HTML header with CSS styles for the tree visualization."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Stream Tree</title>
    <style>
        body {
            font-family: 'Courier New', monospace;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .tree {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            white-space: pre;
        }
        .tree-node {
            position: relative;
            display: inline;
        }
        .tooltip {
            position: relative;
            cursor: help;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 8px 12px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            white-space: normal;
            width: 300px;
            opacity: 0;
            transition: opacity 0.3s;
            font-family: Arial, sans-serif;
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        .no-description {
            font-style: italic;
            color: #999;
        }
    </style>
</head>
<body>
    <div class="tree">
"""


def _get_tooltip_text(node: DataStream) -> str:
    """Generates the HTML content for a node's tooltip."""
    import html as html_module

    parts = []
    typ = getattr(node, "__class__", None)
    if typ:
        parts.append(f"Type: {html_module.escape(typ.__name__)}")
    else:
        parts.append("Type: Unknown")

    description = getattr(node, "description", None)
    if description:
        escaped_desc = html_module.escape(str(description))
        parts.append(f"Description: {escaped_desc}")
    else:
        parts.append("Description: <em>No description available</em>")

    return "<br>".join(parts)


def print_data_stream_tree_html(
    node: DataStream,
    is_last: bool = True,
    parents: list[bool] | None = None,
    show_params: bool = False,
    show_type: bool = False,
    show_missing_indicator: bool = True,
) -> str:
    """Generates an HTML tree representation of a data stream hierarchy with tooltips.

    Creates a formatted HTML string displaying the hierarchical structure of a data stream
    and its children as a tree with branch indicators, icons, and tooltips showing descriptions.

    Args:
        node: The data stream node to start printing from.
        is_last: Whether this node is the last child of its parent.
        parents: List tracking whether each ancestor was a last child, used for drawing branches.
        show_params: Whether to render parameters of the datastream.
        show_type: Whether to render the class name of the datastream.
        show_missing_indicator: Whether to render the missing data indicator.

    Returns:
        str: A formatted HTML string representing the data stream tree with tooltips.

    Examples:
        ```python
        from contraqctor.contract import Dataset, csv, json
        from contraqctor.contract.utils import print_data_stream_tree_html

        csv_stream = csv.Csv("data", reader_params=csv.CsvParams(path="data.csv"))
        json_stream = json.Json("config", reader_params=json.JsonParams(path="config.json"))
        dataset = Dataset("experiment", [csv_stream, json_stream], version="1.0.0")

        html = print_data_stream_tree_html(dataset)
        with open("tree.html", "w") as f:
            f.write(html)
        ```
    """
    import html as html_module

    if parents is None:
        parents = []

    html_header = _get_html_header() if not parents else ""
    node_icon = _get_node_icon(node, show_missing_indicator)
    line_prefix = _build_line_prefix(parents, is_last)
    node_label = _build_node_label(node, show_type, show_params)
    tooltip_text = _get_tooltip_text(node)

    node_label_escaped = html_module.escape(node_label)
    line_prefix_escaped = html_module.escape(line_prefix)

    tooltip_span = f'<span class="tooltiptext">{tooltip_text}</span>'
    node_content = f"{node_icon} {node_label_escaped}{tooltip_span}"
    tree_representation = f'{html_header}{line_prefix_escaped}<span class="tooltip">{node_content}</span>\n'

    if node.is_collection and node.has_data:
        for i, child in enumerate(node.data):
            child_is_last = i == len(node.data) - 1
            tree_representation += print_data_stream_tree_html(
                child,
                is_last=child_is_last,
                parents=parents + [is_last],
                show_params=show_params,
                show_type=show_type,
                show_missing_indicator=show_missing_indicator,
            )

    if not parents:
        tree_representation += """    </div>
</body>
</html>
"""

    return tree_representation
