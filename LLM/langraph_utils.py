try:
    from langgraph.graph import StateGraph
except Exception:  # pragma: no cover - optional dependency
    StateGraph = None


def build_generation_graph(model, tokenizer, device):
    """Return a Langraph graph that generates a single action token."""
    if StateGraph is None:
        raise ImportError("langraph is required for LLM agents")

    from typing import Dict

    # Define the transformation function used by the single node
    def generate_fn(data: Dict[str, str]) -> Dict[str, str]:
        prompt = data["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=1)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"action": text}

    # Build a minimal state graph with one node
    graph = StateGraph(input_schema={"prompt": str}, output_schema={"action": str})
    graph.add_node("generate", generate_fn)
    graph.set_entry_point("generate")
    graph.set_finish_point("generate")
    return graph.compile()
