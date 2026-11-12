"""Our own synthetic tool-selection generator, with disjoint query wording per split."""
import random

# Each entry: identifier, capability, calibration/dev/holdout request phrases.
TOOLS=[
 ('forecast','Predict future weather for a place', ['forecast the weather for {x}','will it rain in {x} tomorrow','what weather is expected in {x} this weekend']),
 ('weather_history','Retrieve past recorded weather', ['find past weather records for {x}','how hot was {x} last Monday','what was the temperature in {x} yesterday']),
 ('create_event','Create a new calendar event', ['put a meeting in the calendar for {x}','schedule a new appointment for {x}','add a lunch meeting to my agenda for {x}']),
 ('find_event','Search existing calendar events', ['look up my calendar event for {x}','find when the meeting about {x} occurs','what is on my agenda concerning {x}']),
 ('send_email','Compose and send an email', ['send an email about {x}','write and send a note about {x}','email my team regarding {x}']),
 ('search_email','Find an existing email message', ['search my mailbox for {x}','find an old email mentioning {x}','look through received messages about {x}']),
 ('route','Compute directions between places', ['get directions to {x}','plan a driving route to {x}','how do I travel to {x} from here']),
 ('place_search','Find a place or business', ['find a nearby {x}','locate a business called {x}','where is the closest {x}']),
 ('calculator','Perform arithmetic calculations', ['calculate {x}','compute the result of {x}','what is the numerical answer to {x}']),
 ('currency','Convert between currencies using exchange rates', ['convert currency: {x}','exchange these currencies: {x}','how much is {x} in the other currency']),
 ('web_search','Search public websites for information', ['search the web for {x}','find online information about {x}','look on the internet for {x}']),
 ('local_search','Search locally stored files and documents', ['search local documents for {x}','find a file on my computer about {x}','locate the saved report mentioning {x}']),
 ('set_alarm','Set an alarm or countdown timer', ['set an alarm for {x}','start a timer for {x}','wake me or alert me at {x}']),
 ('translate','Translate text between human languages', ['translate this phrase: {x}','render this sentence in French: {x}','give me the Spanish translation of {x}']),
 ('summarize','Summarize provided text', ['summarize this text: {x}','give a concise summary of {x}','briefly outline the following passage: {x}']),
 ('image_search','Search for images matching a description', ['find pictures of {x}','search images showing {x}','show me photographs of {x}']),
]
VALUES=[['Lagos','Nairobi','Accra','London','Abuja','Paris'],['our project','the client','the review','the report','my team','next week'],['25 times 12','18 plus 41','300 divided by 5','12 percent of 90'],['50 dollars to euros','100 euros to pounds','200 yen to dollars'],['coffee shop','airport','pharmacy','library'],['ten minutes','seven o clock','half an hour'],['hello my friend','the train is late','a short note']]

def make(split,n,seed):
    ix={'calib':0,'dev':1,'heldout':2}[split]; rng=random.Random(seed); rows=[]
    for i in range(n):
        target=i%len(TOOLS); tool,desc,phrases=TOOLS[target]
        values=VALUES[2] if tool=='calculator' else VALUES[3] if tool=='currency' else VALUES[4] if tool=='place_search' else VALUES[5] if tool=='set_alarm' else VALUES[6] if tool in ('translate','summarize') else VALUES[0] if tool in ('forecast','weather_history','route') else VALUES[1]
        text=phrases[ix].format(x=rng.choice(values)); absent=(i%5==0)
        distractors=[j for j in range(len(TOOLS)) if j!=target]; rng.shuffle(distractors)
        # Prefer semantically neighboring tools so task requires discrimination.
        neighbor=target^1
        picks=[]
        if neighbor<len(TOOLS) and neighbor!=target: picks.append(neighbor)
        picks += [j for j in distractors if j not in picks]
        chosen=picks[:4 if absent else 3]
        if not absent: chosen.append(target)
        rng.shuffle(chosen); answer=4 if absent else chosen.index(target)
        rows.append({'id':f'{split}-{i}','inst':'Choose the best tool for the request, or choose none if no tool fits.', 'text':text,'names':[TOOLS[j][0]+': '+TOOLS[j][1] for j in chosen]+['None of these tools can do this'], 'answer':answer})
    rng.shuffle(rows); return rows
