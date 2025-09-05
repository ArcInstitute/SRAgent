#!/usr/bin/env python
"""
Utilities for workflow graph visualization and writing
"""

import os
import sys
from typing import Any, Callable


def write_workflow_graph(graph: Any, output_file: str) -> None:
    """
    Write a workflow graph to a file in various formats

    Args:
        graph: The workflow graph object (LangGraph StateGraph)
        output_file: Output file path with extension (.png, .svg, .pdf, .mermaid)

    Raises:
        Exception: If graph writing fails
    """
    file_ext = os.path.splitext(output_file)[1].lower()

    try:
        if file_ext == ".mermaid":
            # Write Mermaid format
            mermaid_graph = graph.get_graph().draw_mermaid()
            with open(output_file, "w") as f:
                f.write(mermaid_graph)
        elif file_ext == ".png":
            # Write PNG format
            graph_image = graph.get_graph().draw_png()
            with open(output_file, "wb") as f:
                f.write(graph_image)
        elif file_ext == ".svg":
            # Write SVG format
            graph_image = graph.get_graph().draw_svg()
            with open(output_file, "wb") as f:
                f.write(graph_image)
        elif file_ext == ".pdf":
            # Write PDF format
            graph_image = graph.get_graph().draw_pdf()
            with open(output_file, "wb") as f:
                f.write(graph_image)
        else:
            # Default to Mermaid format
            mermaid_graph = graph.get_graph().draw_mermaid()
            with open(output_file, "w") as f:
                f.write(mermaid_graph)

        print(f"Graph written to {output_file}")

    except Exception as e:
        print(f"Error writing graph: {e}", file=sys.stderr)
        sys.exit(1)


def handle_write_graph_option(
    graph_creator: Callable, output_file: str, *args, **kwargs
) -> None:
    """
    Handle the --write-graph command line option

    Args:
        graph_creator: Function that creates the graph/agent/workflow
        output_file: Output file path
        *args: Arguments to pass to graph_creator
        **kwargs: Keyword arguments to pass to graph_creator
    """
    try:
        # Create the graph/agent/workflow
        obj = graph_creator(*args, **kwargs)

        # Check if it's a compiled graph with get_graph method
        if hasattr(obj, "get_graph"):
            write_workflow_graph(obj, output_file)
        # Check if it's a LangGraph StateGraph
        elif hasattr(obj, "compile"):
            # Compile and then write
            compiled_graph = obj.compile()
            write_workflow_graph(compiled_graph, output_file)
        # Check if it's a React agent or similar with graph property
        elif hasattr(obj, "graph"):
            write_workflow_graph(obj.graph, output_file)
        # For agents that don't have graph structure, create a simple representation
        else:
            # Create a simple text representation
            agent_type = type(obj).__name__
            simple_graph = f"""graph TD
    A[Start] --> B[{agent_type}]
    B --> C[End]
    """

            file_ext = os.path.splitext(output_file)[1].lower()
            if file_ext != ".mermaid":
                # Convert extension to .mermaid for agents without graph structure
                base_name = os.path.splitext(output_file)[0]
                output_file = base_name + ".mermaid"
                print(
                    f"Agent doesn't have graph structure, writing as Mermaid to {output_file}"
                )

            with open(output_file, "w") as f:
                f.write(simple_graph)
            print(f"Simple agent representation written to {output_file}")

    except Exception as e:
        print(f"Error creating or writing graph: {e}", file=sys.stderr)
        sys.exit(1)
