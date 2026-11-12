"""Independent vocabulary/templates for post-construction tool selection diagnosis."""
import random
from tool_prune import TOOLS
REQUESTS=[
['Will it be sunny in Lagos on Saturday?', 'Do I need a raincoat this evening?', 'How cold will it get overnight?', 'What is the forecast for my trip?'],
['Put lunch with Ada on Tuesday.', 'Move my dentist visit to Friday.', 'What events are on my agenda?', 'Book a slot with the team next week.'],
['Write a note to my boss and send it.', 'Find the receipt in my mailbox.', 'Reply to the last letter from Ada.', 'Check whether the client wrote back.'],
['How do I get to the airport by car?', 'Where is the nearest pharmacy?', 'Navigate from home to the station.', 'Show me the way to the museum.'],
['What is 24 times 17?', 'Calculate 18 percent of 350.', 'Divide 840 by seven.', 'Add 127 and 368.'],
['Find recent articles about solar power.', 'Look online for the history of jazz.', 'Search the internet for new robotics papers.', 'Find current information about volcanoes.'],
['Open the report on my computer.', 'Find the budget spreadsheet in my files.', 'Search the project folder for notes.', 'Read the local contract document.'],
['Wake me up in twenty minutes.', 'Set an alarm for 7 am.', 'Start a five minute countdown.', 'Remind me when ten minutes have elapsed.']]
def holdout():
    rng=random.Random(3333); out=[]
    for target,requests in enumerate(REQUESTS):
        for request in requests:
            for pos in range(4):
                others=rng.sample([i for i in range(len(TOOLS)) if i!=target],3); others.insert(pos,target)
                prompt='Choose the best tool for the request. Answer with one letter.\nRequest: '+request+'\n'+''.join(f'{chr(65+j)}. {TOOLS[i][0]}: {TOOLS[i][1]}\n' for j,i in enumerate(others))+'Answer:'
                out.append((prompt,pos))
    rng.shuffle(out); return out
