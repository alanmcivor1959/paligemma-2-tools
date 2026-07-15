# PaliGemma-2

## Exploration

### Preliminary Prompts

Various potential prompts were considered before the following were
selected. Note that they capture the fact that bicycle and scooter
require a person to be using them:

| Class | Prompt |
|-------|--------|
| person | person |
| bicycle | person riding bicycle |
| scooter | person on scooter |

### Prompt Ordering

This considers the three prompts "person riding bicycle", "person,
"person on scooter" (indicated by b, p, s respectively) in different
orders. Result is Y for good, YY for very good, and N for bad, based
on a visual comparison.

| Order | Result |
|-------|--------|
| bps   | Y      |
| pbs   | N      |
| psb   | YY     |
| bsp   | Y      |
| sbp   | N      |
| spb   | N |

NB: spb is better for b since gets more.

NB: These results are from just 4 test images, all from one location.

Whether the variation is due to problems in the image encoder, text
encoder, or decoder needs investigation.

### Alternate Prompts

Gemini suggests that there could be bbox duplication since all labels
contain "person". Alternatives would be:

1. Agent + Vehicle combo: pedestrian, cyclist, scooter rider
2. Structured attributes: person, bicycle [ ridden ], scooter [ridden]

Initial testing showed that "pedestrian" is a fail, so is the
structured approach. Also "cyclist" is invalid because it could be a
person not riding a bicycle. It also suggested "scooterist", which is
a fail, presumably because it is a made up word.

Later, Gemini suggested "person riding an e-scooter" was fine because
"PaliGemma2 understands complex natural language relationships"

The core dataset for object detection used in pre-training was
OpenImages. This includes person (fpv) and bicycle (fpv), independent
of whether in use or not. If ridden, bbox only includes bicycle not
rider too. There was no class for scooter. For fine-tuning the mix
variant, COCO-35L and OWL-ViTv2 were used. **Need to determine if
OWL-ViTv2 provided anything related to scooters, or if we could
generate sometime with it.**


## Futher Work

1. Try separate prompts for bicycle and scooter, with downstream
   matching with associated person to eliminate double counts.
2. Other prompts. like "person walking" instead of "person".
3. Try bigger models with quantisation to fit in. 
4. Explore 664 models.
