import re
from typing import Any, Optional
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown
from SRAgent.agents.utils import set_model


# functions
def create_step_summary_chain(
    model: Optional[str] = None, max_tokens: int = 45
) -> RunnableSequence:
    """
    Create a chain of tools to summarize each step in a workflow.
    Args:
        model: The OpenAI model to use for the language model.
        max_tokens: The maximum number of tokens to use for the summary.
    Returns:
        A chain of tools to summarize each step in a workflow.
    """
    # Create the prompt template
    template = "\n".join(
        [
            "Concisely summarize the provided step in the langgraph workflow.",
            f"The summary must be {max_tokens} tokens or less.",
            'Do not use introductory words such as "The workflow step involves"',
            "Write your output as plain text instead of markdown.",
            "#-- The workflow step --#",
            "{step}",
        ]
    )
    prompt = PromptTemplate(input_variables=["step"], template=template)

    # Initialize the language model
    model = set_model(agent_name="step_summary", max_tokens=max_tokens)

    # Return the LLM chain
    return prompt | model


def format_agent_message(message_content: str, agent_name: str) -> str:
    """
    Format agent message content for better readability with rich markup.
    Args:
        message_content: The raw message content
        agent_name: The name of the agent
    Returns:
        Formatted message content with rich markup
    """
    # Clean up common patterns
    content = message_content.strip()

    # extract content from message_content if complex string
    match = re.search(r"content='(.*?)'", str(message_content), re.DOTALL)
    if match:
        content = match.group(1)

    # Convert literal \n to actual newlines and handle other escape sequences
    content = (
        content.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\'", "'")
        .replace('\\"', '"')
    )

    # Check for error messages
    if content.startswith("Error:") or content.startswith("I am currently unable"):
        return f"[red]{content}[/red]"

    # limit the string to 100 characters
    if len(content) > 100:
        content = content[:100].rstrip() + "..."

    # General formatting for all agents
    lines = content.split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if line:
            # Section headers (lines ending with colon and short enough)
            if line.endswith(":") and len(line) < 60:
                formatted_lines.append(f"[bold yellow]{line}[/bold yellow]")
            # Key-value pairs (contain colon but don't end with colon)
            elif (
                ":" in line
                and not line.endswith(":")
                and len(line.split(":", 1)[0]) < 40
            ):
                parts = line.split(":", 1)
                formatted_lines.append(
                    f"[bold cyan]{parts[0]}:[/bold cyan] {parts[1].strip()}"
                )
            # Bullet points
            elif line.startswith(("•", "-", "*")) or line.startswith(
                ("  •", "  -", "  *")
            ):
                formatted_lines.append(f"[green]{line}[/green]")
            # Lines that look like accession numbers or IDs
            elif re.match(
                r"^[A-Z]{2,3}[0-9]{6,}", line.split()[0] if line.split() else ""
            ):
                formatted_lines.append(f"[cyan]{line}[/cyan]")
            # Regular lines
            else:
                formatted_lines.append(line)
        else:
            # Preserve empty lines for spacing
            formatted_lines.append("")

    return "\n".join(formatted_lines)


def display_step_simple(console: Console, step_cnt: int, step: Any):
    """
    Display a step in simple format for --no-summaries mode.
    """
    if isinstance(step, dict) and "messages" in step:
        if isinstance(step["messages"], list) and step["messages"]:
            msg = step["messages"][-1]
        elif isinstance(step["messages"], str):
            msg = step["messages"]
        if hasattr(msg, "content"):
            agent_name = getattr(msg, "name", "agent")
            if not agent_name:
                agent_name = "No agent"
            msg = format_agent_message(msg.content.strip(), agent_name)
        else:
            msg = "No message content"
        if msg == "":
            msg = "No message content"
        # don't show the first step, since just a repeat of the query
        if step_cnt == 1:
            return
        else:
            step_cnt -= 1
        # print the step
        console.print(f"[bold green]✅ Step {step_cnt}[/bold green]\n[dim]{msg}[/dim]")


async def create_agent_stream(
    input,
    create_agent_func,
    config: dict = {},
    summarize_steps: bool = False,
    no_progress: bool = False,
) -> str:
    """
    Create an Entrez agent and stream the steps.
    Args:
        input: Input message to the agent.
        create_agent_func: Function to create the agent.
        config: Configuration for the agent.
        summarize_steps: Whether to summarize the steps.
        no_progress: Whether to disable progress bar display.
    Returns:
        The final step message.
    """
    # create entrez agent
    agent = create_agent_func(return_tool=False)

    # create step summary chain
    step_summary_chain = create_step_summary_chain() if summarize_steps else None

    # Initialize rich console
    console = Console(stderr=True)

    # Print header
    console.print(
        Panel.fit(
            f"[bold green]🤖 SRAgent Processing Request[/bold green]\n\n"
            f"[yellow]Query:[/yellow] {input['messages'][0].content}",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()

    # invoke agent
    step_cnt = 0
    final_step = ""

    if no_progress:
        # No progress bar mode
        async for step in agent.astream(input, stream_mode="values", config=config):
            step_cnt += 1
            final_step = step

            # summarize step
            if step_summary_chain:
                # Handle different step formats
                step_messages = None
                if isinstance(step, dict) and "messages" in step:
                    step_messages = step.get("messages")
                elif isinstance(step, list):
                    step_messages = step
                else:
                    step_messages = str(step)

                msg = await step_summary_chain.ainvoke({"step": step_messages})
                console.print(
                    f"[bold green]✅ Step {step_cnt}:[/bold green] {msg.content}"
                )
            else:
                # No summaries mode - use simple display
                display_step_simple(console, step_cnt, step)
    else:
        # With progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Processing...", total=None)

            async for step in agent.astream(input, stream_mode="values", config=config):
                step_cnt += 1
                final_step = step

                # Update progress
                progress.update(
                    task, description=f"[cyan]Step {step_cnt}: Processing..."
                )

                # summarize step
                if step_summary_chain:
                    # Handle different step formats
                    step_messages = None
                    if isinstance(step, dict) and "messages" in step:
                        step_messages = step.get("messages")
                    elif isinstance(step, list):
                        step_messages = step
                    else:
                        step_messages = str(step)

                    msg = await step_summary_chain.ainvoke({"step": step_messages})
                    console.print(
                        f"[bold green]✅ Step {step_cnt}:[/bold green] {msg.content}"
                    )
                else:
                    # No summaries mode - use simple display
                    display_step_simple(console, step_cnt, step)

    console.print()  # Add spacing before final results

    # get final step, and handle different types
    try:
        final_step = final_step["agent"]["messages"][-1].content
    except KeyError:
        try:
            final_step = final_step["messages"][-1].content
        except (KeyError, IndexError, AttributeError):
            if isinstance(final_step, str):
                return final_step
            elif isinstance(final_step, list):
                return final_step
            return str(final_step)
    except TypeError:
        # Handle workflows that return lists directly (like tissue ontology)
        if isinstance(final_step, list):
            return final_step
        return str(final_step)
    return final_step


def display_final_results(results: Any, title: str = "✨ Final Results ✨") -> None:
    """
    Display final results with rich formatting in a consistent format.

    Args:
        results: The results to display (can be string, list, or other types)
        title: The title to display as a header
    """
    console = Console()

    if results:
        # Convert results to string if needed
        if isinstance(results, list):
            # Handle list of results (like Uberon IDs)
            if all(isinstance(item, str) for item in results):
                formatted_results = "\n".join(f"- {item}" for item in results)
            else:
                formatted_results = str(results)
        else:
            formatted_results = str(results)

        # Display title as header
        console.print(f"\n[bold green]{title}[/bold green]")

        # Display results without box
        if any(char in formatted_results for char in ["#", "*", "-", "1.", "2."]):
            console.print(Markdown(formatted_results))
        else:
            console.print(formatted_results)
    else:
        console.print(f"\n[bold green]{title}[/bold green]")
        console.print("[red]No results returned[/red]")
