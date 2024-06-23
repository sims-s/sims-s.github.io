---
title: "Spice Rack Search Assistant"
excerpt: "It helps you find your spices<br/><img src='/images/RackPic.jpg'>"
collection: project
---
## Spice Rack Search Assistant
![](/images/RackPicLit.jpg)
This is a voice searchable spice rack mainly powered by LLMs and a bit of pytorch/CV2. 

[Here's some of its functionality](https://www.youtube.com/watch?v=qXvxqRdytYA&list=PLbq19pSjbXP_jyKa3y9LYxd5qOoS3Omvs&pp=gAQBiAQB):
* [Scan a spice into the rack](https://youtu.be/qXvxqRdytYA)
* [Search for a spice](https://youtu.be/kipiX0R6iUo)
* [More general search works ok too](https://youtu.be/SmP8thLU6Zo)
* [Remove a spice](https://youtu.be/yFJg5PPse2s)
* [Feedback system](https://youtu.be/_LslLYXnXuI)
* [Turn off the lights](https://youtu.be/kdfq1wFl9TQ)
* [It doesn't hallucinate spices that aren't there](https://youtu.be/JpSfvQR0EFs)


### Architecture
![](/images/RackDiagram.png)
#### Tasks/Tools Called
All llm outputs are tool calls, jsons, or booleans.

* Retriever: tool with folowing parameters:
  * `mode`: `search` for selected spices or pick a `random` subset that meets other criteria.
  * `query`: string name of spice being searched
  * `time_filter`: either `recent` or `old`; filters to 20% most/least recently used spices based on last time spice was searched.
  * `top_k`: can be -1 for everything


* Locate: Find the location of the spice(s) in the rack; light up LED for that cubby.
* Check: Check if a spice is in the rack. Verbalize whether or not the spice is in the rack. (i.e. do I need to add the spice to my shopping list?)
* Lights: Adjust the backlights next to the camera. Automatically on from 10AM - 8PM. Can be set on/off for arbitrary amounts of time. (Function call in hours; tested across minutes/seconds/days/etc)
* Feedback: Start microphone listening; record feedback; add to RAG feedback database.
* Remove: This is the complicated one - first, it's nice to have a confirm sequence since who knows what might go wrong. Second, how to handle redundant copies of a spice? User must select, and that gets used as a confirmation. So either way, only one extra step. In a picture:  
![](/images/RemoveDiagram.png)

#### Camera Pipeline
Automatic realtime spice detection pipeline:
![](/images/CameraDiagram.png)
