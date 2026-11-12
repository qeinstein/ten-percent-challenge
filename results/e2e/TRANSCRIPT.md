# Shipped commands, end to end, on a clean copy of the artifact (`ART`)

These requests are hand-written and appear in no dataset.

## Voice task (default `task_default.json`), 25 annotations from the MASSIVE-train pool (`voice_ann.jsonl`)

| request | before | after feedback | after empty feedback |
|---|---|---|---|
| how long should i boil an egg for | Question | **Cooking** | Question |
| what can i make with chicken and rice | Lists | **Cooking** | Lists |
| order me a large pepperoni pizza | SmartHome | Food (miss) | SmartHome |
| can you get me some thai food delivered | SmartHome | **Takeaway** | SmartHome |
| wake me up at six thirty | Alarm | Alarm | Alarm |
| turn off the kitchen lights | SmartHome | SmartHome | SmartHome |
| play some jazz | Music | Music | Music |
| add eggs to my shopping list | Lists | Lists | Lists |

- `feedback.py` printed `learned labels ['Cooking', 'Takeaway'] -> ART/feedback_state.pt (39849 bytes); weights unchanged`.
- `shasum` of `bits.bin`, `scales.bin` and `small.pt` was identical before and after.
- After the empty feedback, `diff -r` against the release showed no differences.

## A different task on the same weights (`--task bank_task.json`): a bank support router the model was never set up for

Six hand-written annotations taught a new team, `Loans` (`bank_ann.jsonl`).

| message | before | after feedback |
|---|---|---|
| am i eligible for a loan | Cards | **Loans** |
| how much would monthly repayments be on a 10k loan | Transfers | **Loans** |
| my card was stolen | Cards | Cards |
| send 50 pounds to john | Transfers | Transfers |
| show me last month's statement | Account | Account |

## CPU

`answer.py --device cpu` works (about 80 s for two requests, including about 1 min of loading).
