# product_planning_agent 

A LangGraph agent built to create a write-up of some product analysis of a given product idea.

### BeeAI UI

- Go to http://localhost:8333/agents
- Click on "Import agents"
- Use the GitHub option, and paste `https://github.com/gal/product_planning_agent` for the repository URL.


### BeeAI CLI

- Run `beeai add https://github.com/gal/product_planning_agent`

### If any of the above fail

- You can clone this repo - `git clone git@github.com:gal/product_planning_agent`
- Navigate to it - `cd translation_agent`
- Build and publish to BeeAI's local docker image registry. - `beeai build`
- The above command will output an image tag e.g. `beeai.local/product_planning_agent:latest`
    - Run `beeai add <image tag>` to make this agent callable.

