import asyncio
from typing import Any, List

from app.agents.base import AgentBase
from app.utils.streaming import CustomAsyncIteratorCallbackHandler
from prisma.models import Workflow
from customgpt_client import CustomGPT
import logging
logging.basicConfig(level=logging.INFO)
class WorkflowBase:
    def __init__(
        self,
        workflow: Workflow,
        callbacks: List[CustomAsyncIteratorCallbackHandler],
        session_id: str,
        enable_streaming: bool = False,
    ):
        self.workflow = workflow
        self.enable_streaming = enable_streaming
        self.session_id = session_id
        self.callbacks = callbacks

    async def arun(self, input: Any):
        self.workflow.steps.sort(key=lambda x: x.order)
        logging.info(self.workflow.steps)
        previous_output = input
        steps_output = {}
        stepIndex = 0

        for step in self.workflow.steps:
            agent = await AgentBase(
                agent_id=step.agentId,
                enable_streaming=True,
                callback=self.callbacks[stepIndex],
                session_id=self.session_id,
            ).get_agent()
            logging.info(agent)
            if agent.llms[0].llm.provider in ["CUSTOMGPT"]:
                CustomGPT.api_key = agent.llms[0].llm.apiKey
                project_id = agent.llms[0].llm.options['project_id']
                session_id = CustomGPT.Conversation.create(name='SuperAgent', project_id=project_id).parsed.data.session_id
                stream = True if stepIndex == len(self.workflow.steps) - 1
                task = asyncio.ensure_future(
                    CustomGPT.Conversation.asend(project_id=project_id, session_id=session_id, prompt=previous_output, stream=stream)
                )

                await task
                agent_response = task.result()

                previous_output = agent_response
                steps_output[step.order] = agent_response


            else:
                task = asyncio.ensure_future(
                    agent.acall(
                        inputs={"input": previous_output},
                    )
                )

                await task
                agent_response = task.result()
                previous_output = agent_response.get("output")
                steps_output[step.order] = agent_response

            logging.info(steps_output)
            stepIndex += 1
        return {"steps": steps_output, "output": previous_output}
