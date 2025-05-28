import asyncio
import os
import subprocess
from typing import Any, AsyncIterator, List, TypedDict, Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.schema import StreamEvent
from langchain_community.tools import DuckDuckGoSearchRun
import pydantic
import yaml

import tempfile

from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from acp_sdk.models import Message, MessagePart, Metadata
from acp_sdk.server import Server

server = Server()

class AgentState(TypedDict):
    idea: str
    brand_names: str
    features: str
    copy_text: str
    competitors: str
    tech_requirements: str
    final_report: str
    software_architecture: str


class LogMessage(TypedDict):
    message: str


with open('./agents.yaml', 'r', encoding='utf-8') as config:
    agents_config = yaml.safe_load(config)


chat_model = init_chat_model(
    os.getenv("LLM_MODEL", "ollama:granite3.3:2b"),
    temperature=0.6,
    base_url=os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
)


def get_role(agent_name: str):
    print("Getting role")
    role = agents_config.get(agent_name).get("role")
    return role


def get_goal(agent_name: str):
    print("Getting goal")
    return agents_config.get(agent_name).get("goal")


async def feature_extractor(state: AgentState):
    print("Trying feature extraction")
    product_manager = create_react_agent(
        chat_model,
        tools=[],
        prompt=f"Your role is: {get_role('feature_focused_product_manager')}"
    )

    result = await product_manager.ainvoke({"messages": {"role": "user", "content": get_goal("feature_focused_product_manager").format(idea=state.get("idea"))}})
    output = result.get("messages")[-1].content

    # output = "test 123"
    output = {"features": output}
    print(output)
    return output


async def brand_namer(state: AgentState):
    print("Trying brand name generation")

    class BrandNameResponse(pydantic.BaseModel):
        brand_name_suggestions: List[str]

    brand_name_specialist = create_react_agent(
        chat_model,
        tools=[DuckDuckGoSearchRun()],
        prompt=f"Your role is: {get_role('brand_name_specialist')}",
    )
    result = await brand_name_specialist.ainvoke({"messages": {"role": "user", "content": get_goal("brand_name_specialist").format(idea=state.get("features"))}})
    output = result.get("messages")[-1].content

    output = {"brand_names": output}
    return output

async def subtitle_generator(state: AgentState):
    print("Trying subtitle generation")
    copy_writer = create_react_agent(
        chat_model,
        tools=[],
        prompt=f"Your role is: {get_role('copy_writer')}",
    )

    result = await copy_writer.ainvoke(
        {
            "messages": 
                {"role": "user", "content": get_goal("copy_writer").format(brand_names=state.get("brand_name") ,idea=state.get("features"))}
        }
    )

    output = result.get("messages")[-1].content
    output = {"copy_text": output}
    return output

async def competitor_analyzer(state: AgentState):
    print("Trying competitor analysis")
    competitor_analyst = create_react_agent(
        chat_model,
        tools=[DuckDuckGoSearchRun()],
        prompt=f"Your role is: {get_role('competitor_analyst')}"
    )

    result = await competitor_analyst.ainvoke({"messages": {"role": "user", "content": get_goal("competitor_analyst").format(idea=state.get("idea"))}})
    output = result.get("messages")[-1].content

    output = {"competitors": output}
    return output

async def software_requirement_generator(state: AgentState):
    print("Trying competitor analysis")
    software_system_architect = create_react_agent(
        chat_model,
        tools=[],
        prompt=f"Your role is: {get_role('software_system_architect')}"
    )

    result = await software_system_architect.ainvoke({"messages": {"role": "user", "content": get_goal("software_system_architect").format(idea=state.get("idea"))}})
    output = result.get("messages")[-1].content

    output = {"tech_requirements": output}
    return output

async def system_architecture_generator(state: AgentState):
    print("Trying architecture generation")
    qwen_chat_model = init_chat_model(
        "ollama:qwen2.5-coder:1.5b",
        temperature=0.6,
        base_url=os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    )


    software_team_lead = create_react_agent(
        qwen_chat_model,
        tools=[],
        prompt=f"Your role is: {get_role('software_team_lead')}"
    )

    result = await software_team_lead.ainvoke({"messages": {"role": "user", "content": get_goal('software_team_lead').format(idea=state.get("features"), tech_requirements=state.get("tech_requirements"))}})
    output = result.get("messages")[-1].content


    # remove parentheses (breaks messages)
    output = output.replace("(", "&#40;")
    output = output.replace(")", "&#41;")

    # remove triple dash
    output = output.replace("---", "--")
    output = output.replace("participates", "participant")

    output = {"software_architecture": output}
    return output

async def report_generator(state: AgentState) -> Dict[str, AsyncIterator[StreamEvent]]:
    print("Trying competitor analysis")
    business_analyst_report_writer = create_react_agent(
        chat_model,
        tools=[],
        prompt=f"Your role is: {get_role('business_analyst_report_writer')}"
    )
    input_message =         {
        "messages": {
            "role": "user",
            "content": get_goal("business_analyst_report_writer")
            .format(
                idea=state.get("idea"),
                features=state.get("features"),
                name_ideas=state.get("brand_names"),
                copy_text=state.get("copy_text"),
                competitors=state.get("competitors"),
                tech_requirements=state.get("tech_requirements"),
                software_architecture=state.get("software_architecture")
            )
        }
    }
    print("Input message:\n" + str(input_message))

    result = business_analyst_report_writer.astream_events(
        {
            "messages": {
                "role": "user",
                "content": get_goal("business_analyst_report_writer")
                    .format(
                        idea=state.get("idea"),
                        features=state.get("features"),
                        name_ideas=state.get("brand_names"),
                        copy_text=state.get("copy_text"),
                        competitors=state.get("competitors"),
                        tech_requirements=state.get("tech_requirements"),
                        software_architecture=state.get("software_architecture")
                    )
            }
        })

    output = {"final_report": result}
    return output


@server.agent(metadata=Metadata(
    ui={"type": "hands-off", "user_greeting": "What project would you like to plan out today?"},
    tags=["custom", "acp"],
    framework="Homemade"
))
async def product_planner(input: List[Message], context: Any) -> AsyncIterator:
    """
        Perform product planning with AI.

        Agents will perform tasks with the aim of collaborating to produce a write-up.

        DuckDuckGo will be used for competitor analysis and to avoid suggesting already-used names.

        This agent uses 2 models - one for writing, and a second one for generating system architecture diagrams.
        The first agent defaults to ollama:granite3.3:2b, but can be overridden by the `LLM_MODEL` environment variable.
        The second one, that generates a diagram must be `qwen2.5-coder:7b`.

        A full PDF will be generated, and placed in a temp directory, assuming this agent was run on bare-metal. The path to this PDF will be appended to the end of the output.
    """

    idea = input[-1].parts[-1].content

    graph = StateGraph(AgentState)

    graph.add_node(feature_extractor, feature_extractor.__name__)
    graph.add_node(brand_namer, brand_namer.__name__)
    graph.add_node(subtitle_generator, subtitle_generator.__name__)
    graph.add_node(competitor_analyzer, competitor_analyzer.__name__)
    graph.add_node(software_requirement_generator, software_requirement_generator.__name__)
    graph.add_node(system_architecture_generator, system_architecture_generator.__name__)
    graph.add_node(report_generator, report_generator.__name__, defer=True)

    graph.add_edge(START, competitor_analyzer.__name__)
    graph.add_edge(START, feature_extractor.__name__)

    graph.add_edge(feature_extractor.__name__, brand_namer.__name__)
    graph.add_edge(feature_extractor.__name__, subtitle_generator.__name__)
    graph.add_edge(feature_extractor.__name__, software_requirement_generator.__name__)

    graph.add_edge(software_requirement_generator.__name__, system_architecture_generator.__name__)

    graph.add_edge(system_architecture_generator.__name__, report_generator.__name__)
    graph.add_edge(competitor_analyzer.__name__, report_generator.__name__)
    graph.add_edge(brand_namer.__name__, report_generator.__name__)
    graph.add_edge(subtitle_generator.__name__, report_generator.__name__)

    graph.add_edge(report_generator.__name__, END)

    graph = graph.compile()

    def add_section(section_name: str, section_content: str) -> str:
        section = f"""
        # {section_name}

        {section_content}
        """

        return "\n".join([line.lstrip() for line in section.split("\n")])

    result: AgentState  # set type
    output_markdown = ""
    outputs = {
        "Feature analysis": "",
        "Competitor analysis": "",
        "Technical requirements": "",
        "System architecture": "",
        "Brand name analysis": "",
        "Copy text samples": "",
        "Final report": ""
    }

    async for event in graph.astream({"idea": idea}):
        for node, result in event.items():
            match node:
                case feature_extractor.__name__:
                    outputs["Feature analysis"] = result["features"]
                    yield LogMessage(message=f"📦 Generated product features :::: {result['features'].replace("#", "")}")


                case brand_namer.__name__:
                    print("\n\n\n\n\n\nResult: " + result["brand_names"])
                    outputs["Brand name analysis"] = result["brand_names"]
                    yield LogMessage(message=f"📛 Generated potential product names :::: {result['brand_names'].replace("#", "")}")

                case subtitle_generator.__name__:
                    outputs["Copy text samples"] = result["copy_text"]
                    yield LogMessage(message=f"✏️ Generated copy text :::: {result['copy_text'].replace("#", "")}")

                case competitor_analyzer.__name__:
                    outputs["Competitor analysis"] = result["competitors"]
                    yield LogMessage(message=f"🥸 Generated competitor analysis :::: {result['competitors'].replace("#", "")}")

                case software_requirement_generator.__name__:
                    outputs["Technical requirements"] = result["tech_requirements"]
                    yield LogMessage(message=f"🤓 Generated technical requirements :::: {result['tech_requirements'].replace("#", "")}")

                case system_architecture_generator.__name__:
                    outputs["System architecture"] = result["software_architecture"]
                    yield LogMessage(message=f"👷 Generated system architecture :::: {result['software_architecture'].replace("#", "")}")

                case report_generator.__name__:
                    yield LogMessage(message="👑 Preparing final report...")
                    await asyncio.sleep(0)

                    final_report = ""

                    async for event in result["final_report"]:
                        if event.get("event", "") == "on_chat_model_stream":
                            if "data" in event and "chunk" in event["data"]:
                                content = event["data"]["chunk"].content
                                yield MessagePart(content=str(content))
                                final_report = final_report + content

                        await asyncio.sleep(0)
                    outputs["Final report"] = final_report

                    # if running in docker container, ignore the next part
                    if not os.getenv("CONTAINER", False):
                        for title, content in outputs.items():
                            output_markdown += add_section(title, content)

                        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_md_file:
                            temp_md_file.write(output_markdown)
                            temp_md_path = temp_md_file.name

                        with tempfile.NamedTemporaryFile(suffix='.pdf', prefix="product_planning_", delete=False) as temp_pdf_file:
                            temp_pdf_path = temp_pdf_file.name
                            print(temp_pdf_path)

                        subprocess.run([
                            "pandoc", "-t", "html", "--metadata", "title=Product plan", temp_md_path, "-o", temp_pdf_path, "--css", "./gfm.css", "-F", "mermaid-filter"
                        ], check=True)

                        yield MessagePart(content=f"\n\n---\nSaved document to `{temp_pdf_path}`.\n\n---\n")
                    else:
                        yield MessagePart(content=f"\n\n---\n\nTo check a mermaid chart, you can view it in [https://mermaid.live](https://mermaid.live)")

def main():
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))



if __name__ == "__main__":
    main()

