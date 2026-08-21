"""A minimal Flower AgentApp."""

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context

MODEL = "openai/gpt-5.6-sol"

app = AgentApp()


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Send the configured input to the model."""
    prompt = context.run_config.get("agent.input")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("agent.input must be a non-empty string")

    response = agent.responses.create(
        {
            "model": MODEL,
            "input": prompt.strip(),
            "stream": True,
        }
    )
    print(response["output_text"])
