# functions
def save_graph_image(graph, outfile: str = "graph_image.png") -> None:
    """
    Save the langgraph graph as a PNG image.
    Args:
        graph: langgraph graph object
        outfile: path to save the image
    """
    with open(outfile, "wb") as file:
        png_data = graph.get_graph().draw_mermaid_png()
        file.write(png_data)


# main
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
