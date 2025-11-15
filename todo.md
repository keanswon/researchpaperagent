1. make it a stream

2. get benchmarks

    cost:
        - token cost
        - latency
        - api costs (usd)

    performance:
        - task success rate
        - accuracy for subtasks

3. optimize 
    - optimize chunking + embedding
    - maybe add another model

4. change metadata
    - right now, it's using the pdf name
    - we want the research paper's name / author / date


issues:
    - model loading takes a long time
    - retrieval about one liners is not great, but it works alright for more conceptual ideas