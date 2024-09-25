# Semantic Splitting

Here's a semanting splitting concept with WordLlama.

We develop an algorithm for semanting splitting using cross-similarity with savitzky-golay smoothing.

## The process we define goes as follows:
- Load the raw text
- Split the text on newlines
- Join small segments and split large ones to create more consistent small chunks
- Embed the chunks
- Compute the cross similarity matrix
- Compute a windowed average along the diagonal
- Find local minima using a savitzky-golay filter
- Reconstruct text chunks
- Clean up small text chunks


```python
from wordllama import WordLlama

wl = WordLlama.load()
```

Load the text (Lord of the Rings - The Fellowship of the Rings)


```python
import chardet

filename = "/home/lee/Downloads/lotr_fellowship.txt"
with open(filename, "rb") as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

print(f"Detected encoding: {encoding}")

with open(filename, "r", encoding=encoding) as file:
    txt = file.read()
```

    Detected encoding: Windows-1252


## Cleaning

Newlines are a good starting place for semantic splitting. Ideally, we want to preserve this natural split, while also trying to work the text into a more consistent chunk size.

Sometimes, we have multiple newlines in a row, or short sections between newlines (eg. title pages, etc). We want to gradually batch these up to balanced larger segments. After that we want to consider chunks that are longer than our target, and reduce those into sentences before batching them back together.

Our goal is to maintain natural semantic information in the initial splitting, while also breaking down text into more consistent sizes to help improve consistency in window functions we will use to produce final chunk sizes.


```python
import matplotlib.pyplot as plt
from itertools import chain
from wordllama.algorithms import split_sentences, constrained_coalesce

def flatten(nested_list):
    return chain.from_iterable(nested_list)

TARGET_SIZE = 64 # small strings of 
FINAL_TARGET_SIZE = 4096

def constrained_split(x, target_size=FINAL_TARGET_SIZE):
    """
    Split sentence level, then coalesce
    """
    x = split_sentences(x)
    batches = constrained_coalesce(x, target_size, separator=" ")
    return list(map("".join, batches))

# first split on newlines, recombine small segments
lotr_lines = txt.splitlines()

# coalesce splits to target size
# this increases fragmented strings to granular semantically meaningful sizes
lotr_lines = constrained_coalesce(lotr_lines, TARGET_SIZE, separator="\n")

# break down large chunks above final target size
# retain paragraphs, except if too big, then break on sentences
lotr_lines = [constrained_split(x) if len(x) > FINAL_TARGET_SIZE else [x] for x in lotr_lines]

# flatten nested lists
lotr_lines = list(flatten(lotr_lines))

# remove empties and pure whitespace
lotr_lines = list(filter(lambda x: len(x.strip()) > 0, lotr_lines))

# plot the distribution of splits
lotr_chars = list(map(len, lotr_lines))
bins = plt.hist(lotr_chars, bins=150)
```


    
![png](output_5_0.png)
    


## Embed the paragraphs

Use the wordllama embedding method to embed the list of strings. We are using the default 256-dim model.


```python
lotr_vec = wl.embed(lotr_lines)
lotr_vec.shape
```




    (3925, 256)



# Compute the cross similarity

We use the vector_similarity method to compute the vector similarity between each line and every other line in the corpus. This shows banding of similar regions of text that forms the basis of our extraction.


```python
sim = wl.vector_similarity(lotr_vec, lotr_vec)
```


```python
import numpy as np
from wordllama.algorithms.find_local_minima import window_average
import matplotlib.pyplot as plt

# Apply window averaging
window_size = 3
averaged_values = window_average(sim, window_size)
print(np.std(sim), averaged_values.shape)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(sim, cmap='viridis')
ax1.set_title('Original Matrix')
plt.colorbar(im1, ax=ax1)

ax2.plot(averaged_values, 'r-', label=f'Window Size: {window_size}')
ax2.set_title('Averaged Values')
ax2.legend()
ax2.set_xlabel('Matrix Index')
ax2.set_ylabel('Averaged Value')

plt.tight_layout()
plt.show()

```

    0.12451021 (3925,)



    
![png](output_10_1.png)
    


## Signal processing

This part is a bit of art and practicality. Clearly the plot shows a lot of reasonable candidates for splitting. We need to interpret it in such a way that balances text length with the semantic pattern from the text. Our goal is not to only split based on the embedding signals, but also find a reasonable balance in the text length as well. Most of our text chunks should be of a consistent size, utilizing the minima to provide some of the information for segmentation.

Savitzky-Golay is a good choice for processing the signal, because it has zero phase shift and polynomial smoothing. Additionally, it can produce derivatives so that we can find the local minima which we will use for locating splitting indexes.


```python
from wordllama.algorithms.find_local_minima import find_local_minima
```

## Reconstruction

Now we have a window size that we can use to balance the tension between chunk size consistency and semantic segmentation. Larger window sizes perform more smoothing and filter out smaller signals, so more importance is given toward consistent sizes. Smaller window sizes capture finer fluctuations and give more importance to splitting on semantic signals.

We add indexes at 0 and the end and zip the shifted lists of roots. Now we have pairs if indexes where we can split the text.


```python
window_size = 3

sim_avg = window_average(sim, window_size)
x = np.arange(len(sim_avg))
roots, y = find_local_minima(x, sim_avg, poly_order=2, window_size=7, dec=1)
x_idx, = np.where(y < np.quantile(sim_avg, 0.4))

sem_split = np.round(roots).astype(int).tolist()
sem_split = [x for i,x in enumerate(sem_split) if i in x_idx]
slices = list(zip([0] + sem_split, sem_split + [len(lotr_lines)]))

text_span = []
for s in slices:
    a, b = s
    text = constrained_coalesce(lotr_lines[a:b], FINAL_TARGET_SIZE)
    text_span.extend(text)

# clean up small sections
text_span = constrained_coalesce(text_span, FINAL_TARGET_SIZE)
print(f"Number of semantic splits: {len(slices)}")
print(f"Total number of text spans: {len(text_span)}")
print(f"Original number of strings: {len(lotr_lines)}")
print(f"Semantic split rate: {100 * len(slices)/len(text_span):.1f}%")
text_chars = list(map(len, text_span))

h = plt.hist(text_chars, bins=100)
plt.xlabel("Num chars in text section")
t = plt.title("Histogram of num chars in each text section")
```

    Number of semantic splits: 468
    Total number of text spans: 351
    Original number of strings: 3925
    Semantic split rate: 133.3%



    
![png](output_14_1.png)
    


## View splits


```python
import html
import re

def escape_markdown(text):
    text = html.unescape(text)
    text = re.sub(r'([-*_])\1{2,}', lambda m: '\\' + m.group(0), text)
    return text

def format_text_for_markdown(text):
    """Format text, preserving newlines and spaces."""
    text = html.escape(text)
    # Then escape markdown
    text = escape_markdown(text)
    # Preserve newlines and spaces
    lines = text.split('\n')
    formatted_lines = ['    ' + line if line.strip() else '' for line in lines]
    return '\n'.join(formatted_lines)

def display_strings_markdown_preserved(strings):
    """
    Display a list of strings in a GitHub-friendly markdown format, preserving formatting.
    
    Args:
    strings (list): A list of strings to display.
    """
    markdown_content = "# Text Spans\n\n"
    for i, string in enumerate(strings, 1):
        formatted_string = format_text_for_markdown(string)
        markdown_content += f"-----------\n\n{formatted_string}\n\n"
    
    from IPython.display import Markdown, display
    display(Markdown(markdown_content))

# format output
display_strings_markdown_preserved(text_span[100:120])
```


# Text Spans

-----------

         Frodo and Sam stood as if enchanted. The wind puffed out. The leaves hung silently again on stiff branches. There was another burst of song, and then suddenly, hopping and dancing along the path, there appeared above the reeds an old battered hat with a tall crown and a long blue feather stuck in the band. With another hop and a bound there came into view a man, or so it seemed. At any rate he was too large and heavy for a hobbit, if not quite tall enough for one of the Big People, though he made noise enough for one, slumping along with great yellow boots on his thick legs, and charging through grass and rushes like a cow going down to drink. He had a blue coat and a long brown beard; his eyes were blue and bright, and his face was red as a ripe apple, but creased into a hundred wrinkles of laughter. In his hands he carried on a large leaf as on a tray a small pile of white water-lilies.
         'Help!' cried Frodo and Sam running towards him with their hands stretched out.
         'Whoa! Whoa! steady there!' cried the old man, holding up one hand, and they stopped short, as if they had been struck stiff. 'Now, my little fellows, where be you a-going to, puffing like a bellows? What's the matter here then? Do you know who I am? I'm Tom Bombadil. Tell me what's your trouble! Tom's in a hurry now. Don't you crush my lilies!'
         'My friends are caught in the willow-tree,' cried Frodo breathlessly.
         'Master Merry's being squeezed in a crack!' cried Sam.
         'What?' shouted Tom Bombadil, leaping up in the air. 'Old Man Willow? Naught worse than that, eh? That can soon be mended. I know the tune for him. Old grey Willow-man! I'll freeze his marrow cold, if he don't behave himself. I'll sing his roots off. I'll sing a wind up and blow leaf and branch away. Old Man Willow!' Setting down his lilies carefully on the grass, he ran to the tree. There he saw Merry's feet still sticking out – the rest had already been drawn further inside. Tom put his mouth to the crack and began singing into it in a low voice. They could not catch the words, but evidently Merry was aroused. His legs began to kick. Tom sprang away, and breaking off a hanging branch smote the side of the willow with it. 'You let them out again, Old Man Willow!' he said. 'What be you a-thinking of? You should not be waking. Eat earth! Dig deep! Drink water! Go to sleep! Bombadil is talking!' He then seized Merry's feet and drew him out of the suddenly widening crack.
         There was a tearing creak and the other crack split open, and out of it Pippin sprang, as if he had been kicked. Then with a loud snap both cracks closed fast again. A shudder ran through the tree from root to tip, and complete silence fell.
         'Thank you!' said the hobbits, one after the other.

-----------

         Tom Bombadil burst out laughing. 'Well, my little fellows!' said he, stooping so that he peered into their faces. 'You shall come home with me! The table is all laden with yellow cream, honeycomb, and white bread and butter. Goldberry is waiting. Time enough for questions around the supper table. You follow after me as quick as you are able!' With that he picked up his lilies, and then with a beckoning wave of his hand went hopping and dancing along the path eastward, still singing loudly and nonsensically.
         Too surprised and too relieved to talk, the hobbits followed after him as fast as they could. But that was not fast enough. Tom soon disappeared in front of them, and the noise of his singing got fainter and further away. Suddenly his voice came floating back to them in a loud halloo!

              Hop along, my little friends, up the Withywindle!
               Tom's going on ahead candles for to kindle.
               Down west sinks the Sun: soon you will be groping.
               When the night-shadows fall, then the door will open,
               Out of the window-panes light will twinkle yellow.
               Fear no alder black! Heed no hoary willow!
               Fear neither root nor bough! Tom goes on before you.
               Hey now! merry dot! We'll be waiting for you!


         After that the hobbits heard no more. Almost at once the sun seemed to sink into the trees behind them. They thought of the slanting light of evening glittering on the Brandywine River, and the windows of Bucklebury beginning to gleam with hundreds of lights. Great shadows fell across them; trunks and branches of trees hung dark and threatening over the path. White mists began to rise and curl on the surface of the river and stray about the roots of the trees upon its borders. Out of the very ground at their feet a shadowy steam arose and mingled with the swiftly falling dusk.
         It became difficult to follow the path, and they were very tired. Their legs seemed leaden. Strange furtive noises ran among the bushes and reeds on either side of them; and if they looked up to the pale sky, they caught sight of queer gnarled and knobbly faces that gloomed dark against the twilight, and leered down at them from the high bank and the edges of the wood. They began to feel that all this country was unreal, and that they were stumbling through an ominous dream that led to no awakening.
         Just as they felt their feet slowing down to a standstill, they noticed that the ground was gently rising. The water began to murmur. In the darkness they caught the white glimmer of foam, where the river flowed over a short fall. Then suddenly the trees came to an end and the mists were left behind. They stepped out from the Forest, and found a wide sweep of grass welling up before them. The river, now small and swift, was leaping merrily down to meet them, glinting here and there in the light of the stars, which were already shining in the sky.
         The grass under their feet was smooth and short, as if it had been mown or shaven. The eaves of the Forest behind were clipped, and trim as a hedge. The path was now plain before them, well-tended and bordered with stone. It wound up on to the top of a grassy knoll, now grey under the pale starry night; and there, still high above them on a further slope, they saw the twinkling lights of a house. Down again the path went, and then up again, up a long smooth hillside of turf, towards the light. Suddenly a wide yellow beam flowed out brightly from a door that was opened. There was Tom Bombadil's house before them, up, down, under hill. Behind it a steep shoulder of the land lay grey and bare, and beyond that the dark shapes of the Barrow-downs stalked away into the eastern night.
         They all hurried forward, hobbits and ponies. Already half their weariness and all their fears had fallen from them. _Hey! Come merry dol!_ rolled out the song to greet them.

              Hey! Come derry dol! Hop along, my hearties!

-----------

               Hobbits! Ponies all! We are fond of parties.
               Now let the fun begin! Let us sing together!

         Then another clear voice, as young and as ancient as Spring, like the song of a glad water flowing down into the night from a bright morning in the hills, came falling like silver to meet them:

              Now let the song begin! Let us sing together
               Of sun, stars, moon and mist, rain and cloudy weather,
               Light on the budding leaf, dew on the feather,
               Wind on the open hill, bells on the heather,
               Reeds by the shady pool, lilies on the water:
               Old Tom Bombadil and the River-daughter!


         And with that song the hobbits stood upon the threshold, and a golden light was all about them.


                               _Chapter 7_

-----------

                In the House of Tom Bombadil

         The four hobbits stepped over the wide stone threshold, and stood still, blinking. They were in a long low room, filled with the light of lamps swinging from the beams of the roof; and on the table of dark polished wood stood many candles, tall and yellow, burning brightly.
         In a chair, at the far side of the room facing the outer door, sat a woman. Her long yellow hair rippled down her shoulders; her gown was green, green as young reeds, shot with silver like beads of dew; and her belt was of gold, shaped like a chain of flag-lilies set with the pale-blue eyes of forget-me-nots. About her feel in wide vessels of green and brown earthenware, white water-lilies were floating, so that she seemed to be enthroned in the midst of a pool.
         'Enter, good guests!' she said, and as she spoke they knew that it was her clear voice they had heard singing. They came a few timid steps further into the room, and began to bow low, feeling strangely surprised and awkward, like folk that, knocking at a cottage door to beg for a drink of water, have been answered by a fair young elf-queen clad in living flowers. But before they could say anything, she sprang lightly up and over the lily-bowls, and ran laughing towards them; and as she ran her gown rustled softly like the wind in the flowering borders of a river.
         'Come dear folk!' she said, taking Frodo by the hand. 'Laugh and be merry! I am Goldberry, daughter of the River.' Then lightly she passed them and closing the door she turned her back to it, with her white arms spread out across it. 'Let us shut out the night!' she said. 'For you are still afraid, perhaps, of mist and tree-shadows and deep water, and untame things. Fear nothing! For tonight you are under the roof of Tom Bombadil.'
         The hobbits looked at her in wonder; and she looked at each of them and smiled. 'Fair lady Goldberry!' said Frodo at last, feeling his heart moved with a joy that he did not understand. He stood as he had at times stood enchanted by fair elven-voices; but the spell that was now laid upon him was different: less keen and lofty was the delight, but deeper and nearer to mortal heart; marvellous and yet not strange. 'Fair lady Goldberry!' he said again. 'Now the joy that was hidden in the songs we heard is made plain to me.
              O slender as a willow-wand! O clearer than clear water!
               O reed by the living pool! Fair River-daughter!
               O spring-time and summer-time, and spring again after!
               O wind on the waterfall, and the leaves' laughter!'

         Suddenly he stopped and stammered, overcome with surprise to hear himself saying such things. But Goldberry laughed.
         'Welcome!' she said. 'I had not heard that folk of the Shire were so sweet-tongued. But I see you are an elf-friend; the light in your eyes and the ring in your voice tells it. This is a merry meeting! Sit now, and wait for the Master of the house! He will not be long. He is tending your tired beasts.'
         The hobbits sat down gladly in low rush-seated chairs, while Goldberry busied herself about the table; and their eyes followed her, for the slender grace of her movement filled them with quiet delight. From somewhere behind the house came the sound of singing. Every now and again they caught, among many a _derry dol_ and a _merry dol_ and a _ring a ding dillo_ the repeated words:

              Old Tom Bombadil is a merry fellow;
               Bright blue his jacket is, and his boots are yellow.


-----------

         'Fair lady!' said Frodo again after a while. 'Tell me, if my asking does not seem foolish, who is Tom Bombadil?'
         'He is,' said Goldberry, staying her swift movements and smiling.
         Frodo looked at her questioningly. 'He is, as you have seen him,' she said in answer to his look. 'He is the Master of wood, water, and hill.'
         'Then all this strange land belongs to him?'
         'No indeed!' she answered, and her smile faded. 'That would indeed be a burden,' she added in a low voice, as if to herself. 'The trees and the grasses and all things growing or living in the land belong each to themselves. Tom Bombadil is the Master. No one has ever caught old Tom walking in the forest, wading in the water, leaping on the hill-tops under light and shadow. He has no fear. Tom Bombadil is master.'
         A door opened and in came Tom Bombadil. He had now no hat and his thick brown hair was crowned with autumn leaves. He laughed, and going to Goldberry, took her hand.
         'Here's my pretty lady!' he said, bowing to the hobbits. 'Here's my Goldberry clothed all in silver-green with flowers in her girdle! Is the table laden? I see yellow cream and honeycomb, and white bread, and butter; milk, cheese, and green herbs and ripe berries gathered. Is that enough for us? Is the supper ready?'
         'It is,' said Goldberry; 'but the guests perhaps are not?'
         Tom clapped his hands and cried: 'Tom, Tom! your guests are tired, and you had near forgotten! Come now, my merry friends, and Tom will refresh you! You shall clean grimy hands, and wash your weary faces; cast off your muddy cloaks and comb out your tangles!'

-----------

         He opened the door, and they followed him down a short passage and round a sharp turn. They came to a low room with a sloping roof (a penthouse, it seemed, built on to the north end of the house). Its walls were of clean stone, but they were mostly covered with green hanging mats and yellow curtains. The floor was flagged, and strewn with fresh green rushes. There were four deep mattresses, each piled with white blankets, laid on the floor along one side. Against the opposite wall was a long bench laden with wide earthenware basins, and beside it stood brown ewers filled with water, some cold, some steaming hot. There were soft green slippers set ready beside each bed.
         Before long, washed and refreshed, the hobbits were seated at the table, two on each side, while at either end sat Goldberry and the Master. It was a long and merry meal. Though the hobbits ate, as only famished hobbits can eat, there was no lack. The drink in their drinking-bowls seemed to be clear cold water, yet it went to their hearts like wine and set free their voices. The guests became suddenly aware that they were singing merrily, as if it was easier and more natural than talking.
         At last Tom and Goldberry rose and cleared the table swiftly. The guests were commanded to sit quiet, and were set in chairs, each with a footstool to his tired feet. There was a fire in the wide hearth before them, and it was burning with a sweet smell, as if it were built of apple-wood. When everything was set in order, all the lights in the room were put out, except one lamp and a pair of candles at each end of the chimney-shelf. Then Goldberry came and stood before them, holding a candle; and she wished them each a good night and deep sleep.
         'Have peace now,' she said, 'until the morning! Heed no nightly noises! For nothing passes door and window here save moonlight and starlight and the wind off the hill-top. Good night!' She passed out of the room with a glimmer and a rustle. The sound of her footsteps was like a stream falling gently away downhill over cool stones in the quiet of night.
         Tom sat on a while beside them in silence, while each of them tried to muster the courage to ask one of the many questions he had meant to ask at supper. Sleep gathered on their eyelids. At last Frodo spoke:
         'Did you hear me calling, Master, or was it just chance that brought you at that moment?'
         Tom stirred like a man shaken out of a pleasant dream. 'Eh, what?' said he. 'Did I hear you calling? Nay, I did not hear: I was busy singing. Just chance brought me then, if chance you call it. It was no plan of mine, though I was waiting for you. We heard news of you, and learned that you were wandering. We guessed you'd come ere long down to the water: all paths lead that way, down to Withywindle. Old grey Willow-man, he's a mighty singer; and it's hard for little folk to escape his cunning mazes. But Tom had an errand there, that he dared not hinder.' Tom nodded as if sleep was taking him again; but he went on in a soft singing voice:

              I had an errand there: gathering water-lilies,
               green leaves and lilies white to please my pretty lady,
               the last ere the year's end to keep them from the winter,
               to flower by her pretty feet tilt the snows are melted.
               Each year at summer's end I go to find them for her,
               in a wide pool, deep and clear, far down Withywindle;
               there they open first in spring and there they linger latest.
               By that pool long ago I found the River-daughter,
               fair young Goldberry sitting in the rushes.
               Sweet was her singing then, and her heart was beating!

-----------

         He opened his eyes and looked at them with a sudden glint of blue:
              And that proved well for you – for now I shall no longer
               go down deep again along the forest-water,
               not while the year is old. Nor shall I be passing
               Old Man Willow's house this side of spring-time,
               not till the merry spring, when the River-daughter
               dances down the withy-path to bathe in the water.


-----------

         He fell silent again; but Frodo could not help asking one more question: the one he most desired to have answered. 'Tell us, Master,' he said, 'about the Willow-man. What is he? I have never heard of him before.'
         'No, don't!' said Merry and Pippin together, sitting suddenly upright. 'Not now! Not until the morning!'
         'That is right!' said the old man. 'Now is the time for resting. Some things are ill to hear when the world's in shadow. Sleep till the morning-light, rest on the pillow! Heed no nightly noise! Fear no grey willow!' And with that he took down the lamp and blew it out, and grasping a candle in either hand he led them out of the room.
         Their mattresses and pillows were soft as down, and the blankets were of white wool. They had hardly laid themselves on the deep beds and drawn the light covers over them before they were asleep.
         In the dead night, Frodo lay in a dream without light. Then he saw the young moon rising; under its thin light there loomed before him a black wall of rock, pierced by a dark arch like a great gate. It seemed to Frodo that he was lifted up, and passing over he saw that the rock-wall was a circle of hills, and that within it was a plain, and in the midst of the plain stood a pinnacle of stone, like a vast tower but not made by hands. On its top stood the figure of a man. The moon as it rose seemed to hang for a moment above his head and glistened in his white hair as the wind stirred it. Up from the dark plain below came the crying of fell voices, and the howling of many wolves. Suddenly a shadow, like the shape of great wings, passed across the moon. The figure lifted his arms and a light flashed from the staff that he wielded. A mighty eagle swept down and bore him away. The voices wailed and the wolves yammered. There was a noise like a strong wind blowing, and on it was borne the sound of hoofs, galloping, galloping, galloping from the East. 'Black Riders!' thought Frodo as he wakened, with the sound of the hoofs still echoing in his mind. He wondered if he would ever again have the courage to leave the safety of these stone walls. He lay motionless, still listening; but all was now silent, and at last he turned and fell asleep again or wandered into some other unremembered dream.
         At his side Pippin lay dreaming pleasantly; but a change came over his dreams and he turned and groaned. Suddenly he woke, or thought he had waked, and yet still heard in the darkness the sound that had disturbed his dream: _tip-tap, squeak_: the noise was like branches fretting in the wind, twig-fingers scraping wall and window: _creak, creak, creak._ He wondered if there were willow-trees close to the house; and then suddenly he had a dreadful feeling that he was not in an ordinary house at all, but inside the willow and listening to that horrible dry creaking voice laughing at him again. He sat up, and felt the soft pillows yield to his hands, and he lay down again relieved. He seemed to hear the echo of words in his ears: 'Fear nothing! Have peace until the morning! Heed no nightly noises!' Then he went to sleep again.
         It was the sound of water that Merry heard falling into his quiet sleep: water streaming down gently, and then spreading, spreading irresistibly all round the house into a dark shoreless pool. It gurgled under the walls, and was rising slowly but surely. 'I shall be drowned!' he thought. It will find its way in, and then I shall drown.' He felt that he was lying in a soft slimy bog, and springing up he set his fool on the corner of a cold hard flagstone. Then he remembered where he was and lay down again. He seemed to hear or remember hearing: 'Nothing passes doors or windows save moonlight and starlight and the wind off the hill-top.' A little breath of sweet air moved the curtain. He breathed deep and fell asleep again.
         As far as he could remember, Sam slept through the night in deep content, if logs are contented.

-----------

         They woke up, all four at once, in the morning light. Tom was moving about the room whistling like a starling. When he heard them stir he clapped his hands, and cried: 'Hey! Come merry dol! derry dol! My hearties!' He drew back the yellow curtains, and the hobbits saw that these had covered the windows, at either end of the room, one looking east and the other looking west.
         They leapt up refreshed. Frodo ran to the eastern window, and found himself looking into a kitchen-garden grey with dew. He had half expected to see turf right up to the walls, turf all pocked with hoof-prints. Actually his view was screened by a tall line of beans on poles; but above and far beyond them the grey top of the hill loomed up against the sunrise. It was a pale morning: in the East, behind long clouds like lines of soiled wool stained red at the edges, lay glimmering deeps of yellow. The sky spoke of rain to come; but the light was broadening quickly, and the red flowers on the beans began to glow against the wet green leaves.
         Pippin looked out of the western window, down into a pool of mist. The Forest was hidden under a fog. It was like looking down on to a sloping cloud-roof from above. There was a fold or channel where the mist was broken into many plumes and billows; the valley of the Withywindle. The stream ran down the hill on the left and vanished into the white shadows. Near at hand was a flower-garden and a clipped hedge silver-netted, and beyond that grey shaven grass pale with dew-drops. There was no willow-tree to be seen.
         'Good morning, merry friends!' cried Tom, opening the eastern window wide. A cool air flowed in; it had a rainy smell. 'Sun won't show her face much today. I'm thinking. I have been walking wide, leaping on the hilltops, since the grey dawn began, nosing wind and weather, wet grass underfoot, wet sky above me. I wakened Goldberry singing under window; but nought wakes hobbit-folk in the early morning. In the night little folk wake up in the darkness, and sleep after light has come! Ring a ding dillo! Wake now, my merry friends! Forget the nightly noises! Ring a ding dillo del! derry del, my hearties! If you come soon you'll find breakfast on the table. If you come late you'll get grass and rain-water!'

-----------

         Needless to say – not that Tom's threat sounded very serious – the hobbits came soon, and left the table late and only when it was beginning lo look rather empty. Neither Tom nor Goldberry were there. Tom could be heard about the house, clattering in the kitchen, and up and down the stairs, and singing here and there outside. The room looked westward over the mist-clouded valley, and the window was open. Water dripped down from the thatched eaves above. Before they had finished breakfast the clouds had joined into an unbroken roof, and a straight grey rain came softly and steadily down. Behind its deep curtain the Forest was completely veiled.
         As they looked out of the window there came falling gently as if it was flowing down the rain out of the sky, the clear voice of Goldberry singing up above them. They could hear few words, but it seemed plain to them that the song was a rain-song, as sweet as showers on dry hills, that told the tale of a river from the spring in the highlands to the Sea far below. The hobbits listened with delight; and Frodo was glad in his heart, and blessed the kindly weather, because it delayed them from departing. The thought of going had been heavy upon him from the moment he awoke; but he guessed now that they would not go further that day.
         The upper wind settled in the West and deeper and wetter clouds rolled up to spill their laden rain on the bare heads of the Downs. Nothing could be seen all round the house but falling water. Frodo stood near the open door and watched the white chalky path turn into a little river of milk and go bubbling away down into the valley. Tom Bombadil came trotting round the corner of the house, waving his arms as if he was warding off the rain – and indeed when he sprang over the threshold he seemed quite dry, except for his boots. These he took off and put in the chimney-corner. Then he sat in the largest chair and called the hobbits to gather round him.

-----------

         'This is Goldberry's washing day,' he said, 'and her autumn-cleaning. Too wet for hobbit-folk – let them rest while they are able! It's a good day for long tales, for questions and for answers, so Tom will start the talking.'
         He then told them many remarkable stories, sometimes half as if speaking to himself, sometimes looking at them suddenly with a bright blue eye under his deep brows. Often his voice would turn to song, and he would get out of his chair and dance about. He told them tales of bees and flowers, the ways of trees, and the strange creatures of the Forest, about the evil things and good things, things friendly and things unfriendly, cruel things and kind things, and secrets hidden under brambles.
         As they listened, they began to understand the lives of the Forest, apart from themselves, indeed to feel themselves as the strangers where all other things were at home. Moving constantly in and out of his talk was Old Man Willow, and Frodo learned now enough to content him, indeed more than enough, for it was not comfortable lore. Tom's words laid bare the hearts of trees and their thoughts, which were often dark and strange, and filled with a hatred of things that go free upon the earth, gnawing, biting, breaking, hacking, burning: destroyers and usurpers. It was not called the Old Forest without reason, for it was indeed ancient, a survivor of vast forgotten woods; and in it there lived yet, ageing no quicker than the hills, the fathers of the fathers of trees, remembering times when they were lords. The countless years had filled them with pride and rooted wisdom, and with malice. But none were more dangerous than the Great Willow: his heart was rotten, but his strength was green; and he was cunning, and a master of winds, and his song and thought ran through the woods on both sides of the river. His grey thirsty spirit drew power out of the earth and spread like fine root-threads in the ground, and invisible twig-fingers in the air, till it had under its dominion nearly all the trees of the Forest from the Hedge to the Downs.
         Suddenly Tom's talk left the woods and went leaping up the young stream, over bubbling waterfalls, over pebbles and worn rocks, and among small flowers in close grass and wet crannies, wandering at last up on to the Downs. They heard of the Great Barrows, and the green mounds, and the stone-rings upon the hills and in the hollows among the hills. Sheep were bleating in flocks. Green walls and white walls rose. There were fortresses on the heights. Kings of little kingdoms fought together, and the young Sun shone like fire on the red metal of their new and greedy swords. There was victory and defeat; and towers fell, fortresses were burned, and flames went up into the sky. Gold was piled on the biers of dead kings and queens; and mounds covered them, and the stone doors were shut; and the grass grew over all. Sheep walked for a while biting the grass, but soon the hills were empty again. A shadow came out of dark places far away, and the bones were stirred in the mounds. Barrow-wights walked in the hollow places with a clink of rings on cold fingers, and gold chains in the wind.' Stone rings grinned out of the ground like broken teeth in the moonlight.

-----------

         The hobbits shuddered. Even in the Shire the rumour of the Barrow-wights of the Barrow-downs beyond the Forest had been heard. But it was not a tale that any hobbit liked to listen to, even by a comfortable fireside far away. These four now suddenly remembered what the joy of this house had driven from their minds: the house of Tom Bombadil nestled under the very shoulder of those dreaded hills. They lost the thread of his tale and shifted uneasily, looking aside at one another.
         When they caught his words again they found that he had now wandered into strange regions beyond their memory and beyond their waking thought, into limes when the world was wider, and the seas flowed straight to the western Shore; and still on and back Tom went singing out into ancient starlight, when only the Elf-sires were awake. Then suddenly he slopped, and they saw that he nodded as if he was falling asleep. The hobbits sat still before him, enchanted; and it seemed as if, under the spell of his words, the wind had gone, and the clouds had dried up, and the day had been withdrawn, and darkness had come from East and West, and all the sky was filled with the light of white stars.
         Whether the morning and evening of one day or of many days had passed Frodo could not tell. He did not feel either hungry or tired, only filled with wonder. The stars shone through the window and the silence of the heavens seemed to be round him. He spoke at last out of his wonder and a sudden fear of that silence:
         'Who are you, Master?' he asked.
         'Eh, what?' said Tom sitting up, and his eyes glinting in the gloom. 'Don't you know my name yet? That's the only answer. Tell me, who are you, alone, yourself and nameless? But you are young and I am old. Eldest, that's what I am. Mark my words, my friends: Tom was here before the river and the trees; Tom remembers the first raindrop and the first acorn. He made paths before the Big People, and saw the little People arriving. He was here before the Kings and the graves and the Barrow-wights. When the Elves passed westward, Tom was here already, before the seas were bent. He knew the dark under the stars when it was fearless – before the Dark Lord came from Outside.'

-----------

         A shadow seemed to pass by the window, and the hobbits glanced hastily through the panes. When they turned again, Goldberry stood in the door behind, framed in light. She held a candle, shielding its flame from the draught with her hand; and the light flowed through it, like sunlight through a white shell.
         'The rain has ended,' she said; 'and new waters are running downhill, under the stars. Let us now laugh and be glad!'
         'And let us have food and drink!' cried Tom. 'Long tales are thirsty. And long listening's hungry work, morning, noon, and evening!' With that he jumped out of his chair, and with a bound took a candle from the chimney-shelf and lit it in the flame that Goldberry held; then he danced about the table. Suddenly he hopped through the door and disappeared.
         Quickly he returned, bearing a large and laden tray. Then Tom and Goldberry set the table; and the hobbits sat half in wonder and half in laughter: so fair was the grace of Goldberry and so merry and odd the caperings of Tom. Yet in some fashion they seemed to weave a single dance, neither hindering the other, in and out of the room, and round about the table; and with great speed food and vessels and lights were set in order. The boards blazed with candles, white and yellow. Tom bowed to his guests. 'Supper is ready,' said Goldberry; and now the hobbits saw that she was clothed all in silver with a white girdle, and her shoes were like fishes' mail. But Tom was all in clean blue, blue as rain-washed forget-me-nots, and he had green stockings.
         It was a supper even better than before. The hobbits under the spell of Tom's words may have missed one meal or many, but when the food was before them it seemed at least a week since they had eaten. They did not sing or even speak much for a while, and paid close attention to business. But after a time their hearts and spirit rose high again, and their voices rang out in mirth and laughter.
         After they had eaten, Goldberry sang many songs for them, songs that began merrily in the hills and fell softly down into silence; and in the silences they saw in their minds pools and waters wider than any they had known, and looking into them they saw the sky below them and the stars like jewels in the depths. Then once more she wished them each good night and left them by the fireside. But Tom now seemed wide awake and plied them with questions.
         He appeared already to know much about them and all their families, and indeed to know much of all the history and doings of the Shire down from days hardly remembered among the hobbits themselves. It no longer surprised them; but he made no secret that he owed his recent knowledge largely to Farmer Maggot, whom he seemed to regard as a person of more importance than they had imagined. 'There's earth under his old feet, and clay on his fingers; wisdom in his bones, and both his eyes are open,' said Tom. It was also clear that Tom had dealings with the Elves, and it seemed that in some fashion, news had reached him from Gildor concerning the flight of Frodo.
         Indeed so much did Tom know, and so cunning was his questioning, that Frodo found himself telling him more about Bilbo and his own hopes and fears than he had told before even to Gandalf. Tom wagged his head up and down, and there was a glint in his eyes when he heard of the Riders.

-----------

         'Show me the precious Ring!' he said suddenly in the midst of the story: and Frodo, to his own astonishment, drew out the chain from his pocket, and unfastening the Ring handed it at once to Tom.
         It seemed to grow larger as it lay for a moment on his big brown-skinned hand. Then suddenly he put it to his eye and laughed. For a second the hobbits had a vision, both comical and alarming, of his bright blue eye gleaming through a circle of gold. Then Tom put the Ring round the end of his little finger and held it up to the candlelight. For a moment the hobbits noticed nothing strange about this. Then they gasped. There was no sign of Tom disappearing!
         Tom laughed again, and then he spun the Ring in the air – and it vanished with a flash. Frodo gave a cry – and Tom leaned forward and handed it back to him with a smile.
         Frodo looked at it closely, and rather suspiciously (like one who has lent a trinket to a juggler). It was the same Ring, or looked the same and weighed the same: for that Ring had always seemed to Frodo to weigh strangely heavy in the hand. But something prompted him to make sure. He was perhaps a trifle annoyed with Tom for seeming to make so light of what even Gandalf thought so perilously important. He waited for an opportunity, when the talk was going again, and Tom was telling an absurd story about badgers and their queer ways – then he slipped the Ring on.
         Merry turned towards him to say something and gave a start, and checked an exclamation. Frodo was delighted (in a way): it was his own ring all right, for Merry was staring blankly at his chair, and obviously could not see him. He got up and crept quietly away from the fireside towards the outer door.
         'Hey there!' cried Tom, glancing towards him with a most seeing look in his shining eyes. 'Hey! Come Frodo, there! Where be you a-going? Old Tom Bombadil's not as blind as that yet. Take off your golden ring! Your hand's more fair without it. Come back! Leave your game and sit down beside me! We must talk a while more, and think about the morning. Tom must teach the right road, and keep your feet from wandering.'
         Frodo laughed (trying to feel pleased), and taking off the Ring he came and sat down again. Tom now told them that he reckoned the Sun would shine tomorrow, and it would be a glad morning, and setting out would be hopeful. But they would do well to start early; for weather in that country was a thing that even Tom could not be sure of for long, and it would change sometimes quicker than he could change his jacket. 'I am no weather-master,' he said; 'nor is aught that goes on two legs.'
         By his advice they decided to make nearly due North from his house, over the western and lower slopes of the Downs: they might hope in that way to strike the East Road in a day's journey, and avoid the Barrows. He told them not to be afraid – but to mind their own business.
         'Keep to the green grass. Don't you go a-meddling with old stone or cold Wights or prying in their houses, unless you be strong folk with hearts that never falter!' He said this more than once; and he advised them to pass barrows by on the west-side, if they chanced to stray near one. Then he taught them a rhyme to sing, if they should by ill-luck fall into any danger or difficulty the next day.

              Ho! Tom Bombadil, Tom Bombadillo!
               By water, wood and hill, by the reed and willow,
               By fire, sun and moon, harken now and hear us!
               Come, Tom Bombadil, for our need is near us!

         When they had sung this altogether after him, he clapped them each on the shoulder with a laugh, and taking candles led them back to their bedroom.

-----------



                               _Chapter 8_
                Fog on the Barrow-Downs

         That night they heard no noises. But either in his dreams or out of them, he could not tell which, Frodo heard a sweet singing running in his mind; a song that seemed to come like a pale light behind a grey rain-curtain, and growing stronger to turn the veil all to glass and silver, until at last it was rolled back, and a far green country opened before him under a swift sunrise.
         The vision melted into waking; and there was Tom whistling like a tree-full of birds; and the sun was already slanting down the hill and through the open window. Outside everything was green and pale gold.
         After breakfast, which they again ate alone, they made ready to say farewell, as nearly heavy of heart as was possible on such a morning: cool, bright, and clean under a washed autumn sky of thin blue. The air came fresh from the North-west. Their quiet ponies were almost frisky, sniffing and moving restlessly. Tom came out of the house and waved his hat and danced upon the doorstep, bidding the hobbits to get up and be off and go with good speed.
         They rode off along a path that wound away from behind the house, and went slanting up towards the north end of the hill-brow under which it sheltered. They had just dismounted to lead their ponies up the last steep slope, when suddenly Frodo stopped.
         'Goldberry!' he cried. 'My fair lady, clad all in silver green! We have never said farewell to her, nor seen her since the evening!' He was so distressed that he turned back; but at that moment a clear call came rippling down. There on the hill-brow she stood beckoning to them: her hair was flying loose, and as it caught the sun it shone and shimmered. A light like the glint of water on dewy grass flashed from under her feet as she danced.
         They hastened up the last slope, and stood breathless beside her. They bowed, but with a wave of her arm she bade them look round; and they looked out from the hill-top over lands under the morning. It was now as clear and far-seen as it had been veiled and misty when they stood upon the knoll in the Forest, which could now be seen rising pale and green out of the dark trees in the West. In that direction the land rose in wooded ridges, green, yellow, russet under the sun, beyond which lay hidden the valley of the Brandywine. To the South, over the line of the Withywindle, there was a distant glint like pale glass where the Brandywine River made a great loop in the lowlands and flowed away out of the knowledge of the hobbits. Northward beyond the dwindling downs the land ran away in flats and swellings of grey and green and pale earth-colours, until it faded into a featureless and shadowy distance. Eastward the Barrow-downs rose, ridge behind ridge into the morning, and vanished out of eyesight into a guess: it was no more than a guess of blue and a remote white glimmer blending with the hem of the sky, but it spoke to them, out of memory and old tales, of the high and distant mountains.

-----------

         They took a deep draught of the air, and felt that a skip and a few stout strides would bear them wherever they wished. It seemed fainthearted to go jogging aside over the crumpled skirts of the downs towards the Road, when they should be leaping, as lusty as Tom, over the stepping stones of the hills straight towards the Mountains.
         Goldberry spoke to them and recalled their eyes and thoughts. 'Speed now, fair guests!' she said. 'And hold to your purpose! North with the wind in the left eye and a blessing on your footsteps! Make haste while the Sun shines!' And to Frodo she said: 'Farewell, Elf-friend, it was a merry meeting!'
         But Frodo found no words to answer. He bowed low, and mounted his pony, and followed by his friends jogged slowly down the gentle slope behind the hill. Tom Bombadil's house and the valley, and the Forest were lost to view. The air grew warmer between the green walls of hillside and hillside, and the scent of turf rose strong and sweet as they breathed. Turning back, when they reached the bottom of the green hollow, they saw Goldberry, now small and slender like a sunlit flower against the sky: she was standing still watching them, and her hands were stretched out towards them. As they looked she gave a clear call, and lifting up her hand she turned and vanished behind the hill.
         Their way wound along the floor of the hollow, and round the green feet of a steep hill into another deeper and broader valley, and then over the shoulder of further hills, and down their long limbs, and up their smooth sides again, up on to new hill-tops and down into new valleys. There was no tree nor any visible water: it was a country of grass and short springy turf, silent except for the whisper of the air over the edges of the land, and high lonely cries of strange birds. As they journeyed the sun mounted, and grew hot. Each time they climbed a ridge the breeze seemed to have grown less. When they caught a glimpse of the country westward the distant Forest seemed to be smoking, as if the fallen rain was steaming up again from leaf and root and mould. A shadow now lay round the edge of sight, a dark haze above which the upper sky was like a blue cap, hot and heavy.

-----------

         About mid-day they came to a hill whose top was wide and flattened, like a shallow saucer with a green mounded rim. Inside there was no air stirring, and the sky seemed near their heads. They rode across and looked northwards. Then their hearts rose, for it seemed plain that they had come further already than they had expected. Certainly the distances had now all become hazy and deceptive, but there could be no doubt that the Downs were coming to an end. A long valley lay below them winding away northwards, until it came to an opening between two steep shoulders. Beyond, there seemed to be no more hills. Due north they faintly glimpsed a long dark line. That is a line of trees,' said Merry, 'and that must mark the Road. All along it for many leagues east of the Bridge there are trees growing. Some say they were planted in the old days.'
         'Splendid!' said Frodo. 'If we make as good going this afternoon as we have done this morning, we shall have left the Downs before the Sun sets and be jogging on in search of a camping place.' But even as he spoke he turned his glance eastwards, and he saw that on that side the hills were higher and looked down upon them; and all those hills were crowned with green mounds, and on some were standing stones, pointing upwards like jagged teeth out of green gums.
         That view was somehow disquieting; so they turned from the sight and went down into the hollow circle. In the midst of it there stood a single stone, standing tall under the sun above, and at this hour casting no shadow. It was shapeless and yet significant: like a landmark, or a guarding finger, or more like a warning. But they were now hungry, and the sun was still at the fearless noon; so they set their backs against the east side of the stone. It was cool, as if the sun had had no power to warm it; but at that time this seemed pleasant. There they took food and drink, and made as good a noon-meal under the open sky as anyone could wish; for the food came from 'down under Hill'. Tom had provided them with plenty for the comfort of the day. Their ponies unburdened strayed upon the grass.
         Riding over the hills, and eating their fill, the warm sun and the scent of turf, lying a little too long, stretching out their legs and looking at the sky above their noses: these things are, perhaps, enough to explain what happened. However, that may be: they woke suddenly and uncomfortably from a sleep they had never meant to take. The standing stone was cold, and it cast a long pale shadow that stretched eastward over them. The sun, a pale and watery yellow, was gleaming through the mist just above the west wall of the hollow in which they lay; north, south, and east, beyond the wall the fog was thick, cold and white. The air was silent, heavy and chill. Their ponies were standing crowded together with their heads down.

-----------

         The hobbits sprang to their feet in alarm, and ran to the western rim. They found that they were upon an island in the fog. Even as they looked out in dismay towards the setting sun, it sank before their eyes into a white sea, and a cold grey shadow sprang up in the East behind. The fog rolled up to the walls and rose above them, and as it mounted it bent over their heads until it became a roof: they were shut in a hall of mist whose central pillar was the standing stone.
         They felt as if a trap was closing about them; but they did not quite lose heart. They still remembered the hopeful view they had had of the line of the Road ahead, and they still knew in which direction it lay. In any case, they now had so great a dislike for that hollow place about the stone that no thought of remaining there was in their minds. They packed up as quickly as their chilled fingers would work.
         Soon they were leading their ponies in single file over the rim and down the long northward slope of the hill, down into a foggy sea. As they went down the mist became colder and damper, and their hair hung lank and dripping on their foreheads. When they reached the bottom it was so cold that they halted and got out cloaks and hoods, which soon became bedewed with grey drops. Then, mounting their ponies, they went slowly on again, feeling their way by the rise and fall of the ground. They were steering, as well as they could guess, for the gate-like opening at the far northward end of the long valley which they had seen in the morning. Once they were through the gap, they had only lo keep on in anything like a straight line and they were bound in the end to strike the Road. Their thoughts did not go beyond that, except for a vague hope that perhaps away beyond the Downs there might be no fog.
         Their going was very slow. To prevent their getting separated and wandering in different directions they went in file, with Frodo leading. Sam was behind him, and after him came Pippin, and then Merry. The valley seemed to stretch on endlessly. Suddenly Frodo saw a hopeful sign. On either side ahead a darkness began to loom through the mist; and he guessed that they were at last approaching the gap in the hills, the north-gate of the Barrow-downs. If they could pass that, they would be free.

-----------

         'Come on! Follow me!' he called back over his shoulder, and he hurried forward. But his hope soon changed to bewilderment and alarm. The dark patches grew darker, but they shrank; and suddenly he saw, towering ominous before him and leaning slightly towards one another like the pillars of a headless door, two huge standing stones. He could not remember having seen any sign of these in the valley, when he looked out from the hill in the morning. He had passed between them almost before he was aware: and even as he did so darkness seemed to fall round him. His pony reared and snorted, and he fell off. When he looked back he found that he was alone: the others had not followed him. 'Sam!' he called. 'Pippin! Merry! Come along! Why don't you keep up?'
         There was no answer. Fear took him, and he ran back past the stones shouting wildly: 'Sam! Sam! Merry! Pippin!' The pony bolted into the mist and vanished. From some way off, or so it seemed, he thought he heard a cry: 'Hoy! Frodo! Hoy!' It was away eastward, on his left as he stood under the great stones, staring and straining into the gloom. He plunged off in the direction of the call, and found himself going steeply uphill.
         As he struggled on he called again, and kept on calling more and more frantically; but he heard no answer for some time, and then it seemed faint and far ahead and high above him. 'Frodo! Hoy!' came the thin voices out of the mist: and then a cry that sounded like _help, help!_ often repeated, ending with a last _help!_ that trailed off into a long wail suddenly cut short. He stumbled forward with all the speed he could towards the cries; but the light was now gone, and clinging night had closed about him, so that it was impossible to be sure of any direction. He seemed all the time to be climbing up and up.
         Only the change in the level of the ground at his feet told him when he at last came to the top of a ridge or hill. He was weary, sweating and yet chilled. It was wholly dark.
         'Where are you?' he cried out miserably.
         There was no reply. He stood listening. He was suddenly aware that it was getting very cold, and that up here a wind was beginning to blow, an icy wind. A change was coming in the weather. The mist was flowing past him now in shreds and tatters. His breath was smoking, and the darkness was less near and thick. He looked up and saw with surprise that faint stars were appearing overhead amid the strands of hurrying cloud and fog. The wind began to hiss over the grass.
         He imagined suddenly that he caught a muffled cry, and he made towards it; and even as he went forward the mist was rolled up and thrust aside, and the starry sky was unveiled. A glance showed him that he was now facing southwards and was on a round hill-top, which he must have climbed from the north. Out of the east the biting wind was blowing. To his right there loomed against the westward stars a dark black shape. A great barrow stood there.
         'Where are you?' he cried again, both angry and afraid.

-----------

         'Here!' said a voice, deep and cold, that seemed to come out of the ground. 'I am waiting for you!'
         'No!' said Frodo; but he did not run away. His knees gave, and he fell on the ground. Nothing happened, and there was no sound. Trembling he looked up, in time to see a tall dark figure like a shadow against the stars. It leaned over him. He thought there were two eyes, very cold though lit with a pale light that seemed to come from some remote distance. Then a grip stronger and colder than iron seized him. The icy touch froze his bones, and he remembered no more.
         When he came to himself again, for a moment he could recall nothing except a sense of dread. Then suddenly he knew that he was imprisoned, caught hopelessly; he was in a barrow. A Barrow-wight had taken him, and he was probably already under the dreadful spells of the Barrow-wights about which whispered tales spoke. He dared not move, but lay as he found himself: flat on his back upon a cold stone with his hands on his breast.
         But though his fear was so great that it seemed to be part of the very darkness that was round him, he found himself as he lay thinking about Bilbo Baggins and his stories, of their jogging along together in the lanes of the Shire and talking about roads and adventures. There is a seed of courage hidden (often deeply, it is true) in the heart of the fattest and most timid hobbit, wailing for some final and desperate danger to make it grow. Frodo was neither very fat nor very timid; indeed, though he did not know it, Bilbo (and Gandalf) had thought him the best hobbit in the Shire. He thought he had come to the end of his adventure, and a terrible end, but the thought hardened him. He found himself stiffening, as if for a final spring; he no longer felt limp like a helpless prey.
         As he lay there, thinking and getting a hold of himself, he noticed all at once that the darkness was slowly giving way: a pale greenish light was growing round him. It did not at first show him what kind of a place he was in, for the light seemed to be coming out of himself, and from the floor beside him, and had not yet reached the roof or wall. He turned, and there in the cold glow he saw lying beside him Sam, Pippin, and Merry. They were on their backs, and their faces looked deathly pale; and they were clad in white. About them lay many treasures, of gold maybe, though in that light they looked cold and unlovely. On their heads were circlets, gold chains were about their waists, and on their fingers were many rings. Swords lay by their sides, and shields were at their feet. But across their three necks lay one long naked sword.
         Suddenly a song began: a cold murmur, rising and falling. The voice seemed far away and immeasurably dreary, sometimes high in the air and thin, sometimes like a low moan from the ground. Out of the formless stream of sad but horrible sounds, strings of words would now and again shape themselves: grim, hard, cold words, heartless and miserable. The night was railing against the morning of which it was bereaved, and the cold was cursing the warmth for which it hungered. Frodo was chilled to the marrow. After a while the song became clearer, and with dread in his heart he perceived that it had changed into an incantation:

              Cold be hand and heart and bone,
               and cold be sleep under stone:




## Using the algorithm


```python
import chardet

filename = "pg_essay.txt"
with open(filename, "rb") as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

print(f"Detected encoding: {encoding}")

with open(filename, "r", encoding=encoding) as file:
    txt = file.read()
```

    Detected encoding: utf-8



```python
%%time

results = wl.split(txt, target_size=4096)
print(f"Number of segments: {len(results)}")
```

    Number of segments: 26
    CPU times: user 107 ms, sys: 5.08 ms, total: 113 ms
    Wall time: 51 ms



```python
display_strings_markdown_preserved(results[0:5])
```


# Text Spans

-----------



    What I Worked On

    February 2021

    Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.
    The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights.
    The language we used was an early version of Fortran. You had to type programs on punch cards, then stack them in the card reader and press a button to load the program into memory and run it. The result would ordinarily be to print something on the spectacularly loud printer.
    I was puzzled by the 1401. I couldn't figure out what to do with it. And in retrospect there's not much I could have done with it. The only form of input to programs was data stored on punched cards, and I didn't have any data stored on punched cards. The only other option was to do things that didn't rely on any input, like calculate approximations of pi, but I didn't know enough math to do anything interesting of that type. So I'm not surprised I can't remember any programs I wrote, because they can't have done much. My clearest memory is of the moment I learned it was possible for programs not to terminate, when one of mine didn't. On a machine without time-sharing, this was a social as well as a technical error, as the data center manager's expression made clear.
    With microcomputers, everything changed. Now you could have a computer sitting right in front of you, on a desk, that could respond to your keystrokes as it was running instead of just churning through a stack of punch cards and then stopping. [1]
    The first of my friends to get a microcomputer built it himself. It was sold as a kit by Heathkit. I remember vividly how impressed and envious I felt watching him sitting in front of it, typing programs right into the computer.
    Computers were expensive in those days and it took me years of nagging before I convinced my father to buy one, a TRS-80, in about 1980. The gold standard then was the Apple II, but a TRS-80 was good enough. This was when I really started programming. I wrote simple games, a program to predict how high my model rockets would fly, and a word processor that my father used to write at least one book. There was only room in memory for about 2 pages of text, so he'd write 2 pages at a time and then print them out, but it was a lot better than a typewriter.
    Though I liked programming, I didn't plan to study it in college. In college I was going to study philosophy, which sounded much more powerful. It seemed, to my naive high school self, to be the study of the ultimate truths, compared to which the things studied in other fields would be mere domain knowledge. What I discovered when I got to college was that the other fields took up so much of the space of ideas that there wasn't much left for these supposed ultimate truths. All that seemed left for philosophy were edge cases that people in other fields felt could safely be ignored.

-----------

    I couldn't have put this into words when I was 18. All I knew at the time was that I kept taking philosophy courses and they kept being boring. So I decided to switch to AI.
    AI was in the air in the mid 1980s, but there were two things especially that made me want to work on it: a novel by Heinlein called The Moon is a Harsh Mistress, which featured an intelligent computer called Mike, and a PBS documentary that showed Terry Winograd using SHRDLU. I haven't tried rereading The Moon is a Harsh Mistress, so I don't know how well it has aged, but when I read it I was drawn entirely into its world. It seemed only a matter of time before we'd have Mike, and when I saw Winograd using SHRDLU, it seemed like that time would be a few years at most. All you had to do was teach SHRDLU more words.
    There weren't any classes in AI at Cornell then, not even graduate classes, so I started trying to teach myself. Which meant learning Lisp, since in those days Lisp was regarded as the language of AI. The commonly used programming languages then were pretty primitive, and programmers' ideas correspondingly so. The default language at Cornell was a Pascal-like language called PL/I, and the situation was similar elsewhere. Learning Lisp expanded my concept of a program so fast that it was years before I started to have a sense of where the new limits were. This was more like it; this was what I had expected college to do. It wasn't happening in a class, like it was supposed to, but that was ok. For the next couple years I was on a roll. I knew what I was going to do.
    For my undergraduate thesis, I reverse-engineered SHRDLU. My God did I love working on that program. It was a pleasing bit of code, but what made it even more exciting was my belief — hard to imagine now, but not unique in 1985 — that it was already climbing the lower slopes of intelligence.
    I had gotten into a program at Cornell that didn't make you choose a major. You could take whatever classes you liked, and choose whatever you liked to put on your degree. I of course chose "Artificial Intelligence." When I got the actual physical diploma, I was dismayed to find that the quotes had been included, which made them read as scare-quotes. At the time this bothered me, but now it seems amusingly accurate, for reasons I was about to discover.
    I applied to 3 grad schools: MIT and Yale, which were renowned for AI at the time, and Harvard, which I'd visited because Rich Draves went there, and was also home to Bill Woods, who'd invented the type of parser I used in my SHRDLU clone. Only Harvard accepted me, so that was where I went.
    I don't remember the moment it happened, or if there even was a specific moment, but during the first year of grad school I realized that AI, as practiced at the time, was a hoax. By which I mean the sort of AI in which a program that's told "the dog is sitting on the chair" translates this into some formal representation and adds it to the list of things it knows.
    What these programs really showed was that there's a subset of natural language that's a formal language. But a very proper subset. It was clear that there was an unbridgeable gap between what they could do and actually understanding natural language. It was not, in fact, simply a matter of teaching SHRDLU more words. That whole way of doing AI, with explicit data structures representing concepts, was not going to work. Its brokenness did, as so often happens, generate a lot of opportunities to write papers about various band-aids that could be applied to it, but it was never going to get us Mike.

-----------

    So I looked around to see what I could salvage from the wreckage of my plans, and there was Lisp. I knew from experience that Lisp was interesting for its own sake and not just for its association with AI, even though that was the main reason people cared about it at the time. So I decided to focus on Lisp. In fact, I decided to write a book about Lisp hacking. It's scary to think how little I knew about Lisp hacking when I started writing that book. But there's nothing like writing a book about something to help you learn it. The book, On Lisp, wasn't published till 1993, but I wrote much of it in grad school.
    Computer Science is an uneasy alliance between two halves, theory and systems. The theory people prove things, and the systems people build things. I wanted to build things. I had plenty of respect for theory — indeed, a sneaking suspicion that it was the more admirable of the two halves — but building things seemed so much more exciting.
    The problem with systems work, though, was that it didn't last. Any program you wrote today, no matter how good, would be obsolete in a couple decades at best. People might mention your software in footnotes, but no one would actually use it. And indeed, it would seem very feeble work. Only people with a sense of the history of the field would even realize that, in its time, it had been good.
    There were some surplus Xerox Dandelions floating around the computer lab at one point. Anyone who wanted one to play around with could have one. I was briefly tempted, but they were so slow by present standards; what was the point? No one else wanted one either, so off they went. That was what happened to systems work.
    I wanted not just to build things, but to build things that would last.
    In this dissatisfied state I went in 1988 to visit Rich Draves at CMU, where he was in grad school. One day I went to visit the Carnegie Institute, where I'd spent a lot of time as a kid. While looking at a painting there I realized something that might seem obvious, but was a big surprise to me. There, right on the wall, was something you could make that would last. Paintings didn't become obsolete. Some of the best ones were hundreds of years old.
    And moreover this was something you could make a living doing. Not as easily as you could by writing software, of course, but I thought if you were really industrious and lived really cheaply, it had to be possible to make enough to survive. And as an artist you could be truly independent. You wouldn't have a boss, or even need to get research funding.
    I had always liked looking at paintings. Could I make them? I had no idea. I'd never imagined it was even possible. I knew intellectually that people made art — that it didn't just appear spontaneously — but it was as if the people who made it were a different species. They either lived long ago or were mysterious geniuses doing strange things in profiles in Life magazine. The idea of actually being able to make art, to put that verb before that noun, seemed almost miraculous.
    That fall I started taking art classes at Harvard. Grad students could take classes in any department, and my advisor, Tom Cheatham, was very easy going. If he even knew about the strange classes I was taking, he never said anything.
    So now I was in a PhD program in computer science, yet planning to be an artist, yet also genuinely in love with Lisp hacking and working away at On Lisp. In other words, like many a grad student, I was working energetically on multiple projects that were not my thesis.
    I didn't see a way out of this situation. I didn't want to drop out of grad school, but how else was I going to get out? I remember when my friend Robert Morris got kicked out of Cornell for writing the internet worm of 1988, I was envious that he'd found such a spectacular way to get out of grad school.

-----------

    Then one day in April 1990 a crack appeared in the wall. I ran into professor Cheatham and he asked if I was far enough along to graduate that June. I didn't have a word of my dissertation written, but in what must have been the quickest bit of thinking in my life, I decided to take a shot at writing one in the 5 weeks or so that remained before the deadline, reusing parts of On Lisp where I could, and I was able to respond, with no perceptible delay "Yes, I think so. I'll give you something to read in a few days."
    I picked applications of continuations as the topic. In retrospect I should have written about macros and embedded languages. There's a whole world there that's barely been explored. But all I wanted was to get out of grad school, and my rapidly written dissertation sufficed, just barely.
    Meanwhile I was applying to art schools. I applied to two: RISD in the US, and the Accademia di Belli Arti in Florence, which, because it was the oldest art school, I imagined would be good. RISD accepted me, and I never heard back from the Accademia, so off to Providence I went.
    I'd applied for the BFA program at RISD, which meant in effect that I had to go to college again. This was not as strange as it sounds, because I was only 25, and art schools are full of people of different ages. RISD counted me as a transfer sophomore and said I had to do the foundation that summer. The foundation means the classes that everyone has to take in fundamental subjects like drawing, color, and design.
    Toward the end of the summer I got a big surprise: a letter from the Accademia, which had been delayed because they'd sent it to Cambridge England instead of Cambridge Massachusetts, inviting me to take the entrance exam in Florence that fall. This was now only weeks away. My nice landlady let me leave my stuff in her attic. I had some money saved from consulting work I'd done in grad school; there was probably enough to last a year if I lived cheaply. Now all I had to do was learn Italian.
    Only stranieri (foreigners) had to take this entrance exam. In retrospect it may well have been a way of excluding them, because there were so many stranieri attracted by the idea of studying art in Florence that the Italian students would otherwise have been outnumbered. I was in decent shape at painting and drawing from the RISD foundation that summer, but I still don't know how I managed to pass the written exam. I remember that I answered the essay question by writing about Cezanne, and that I cranked up the intellectual level as high as I could to make the most of my limited vocabulary. [2]
    I'm only up to age 25 and already there are such conspicuous patterns. Here I was, yet again about to attend some august institution in the hopes of learning about some prestigious subject, and yet again about to be disappointed. The students and faculty in the painting department at the Accademia were the nicest people you could imagine, but they had long since arrived at an arrangement whereby the students wouldn't require the faculty to teach anything, and in return the faculty wouldn't require the students to learn anything. And at the same time all involved would adhere outwardly to the conventions of a 19th century atelier. We actually had one of those little stoves, fed with kindling, that you see in 19th century studio paintings, and a nude model sitting as close to it as possible without getting burned. Except hardly anyone else painted her besides me. The rest of the students spent their time chatting or occasionally trying to imitate things they'd seen in American art magazines.
    Our model turned out to live just down the street from me. She made a living from a combination of modelling and making fakes for a local antique dealer. She'd copy an obscure old painting out of a book, and then he'd take the copy and maltreat it to make it look old. [3]

-----------

    While I was a student at the Accademia I started painting still lives in my bedroom at night. These paintings were tiny, because the room was, and because I painted them on leftover scraps of canvas, which was all I could afford at the time. Painting still lives is different from painting people, because the subject, as its name suggests, can't move. People can't sit for more than about 15 minutes at a time, and when they do they don't sit very still. So the traditional m.o. for painting people is to know how to paint a generic person, which you then modify to match the specific person you're painting. Whereas a still life you can, if you want, copy pixel by pixel from what you're seeing. You don't want to stop there, of course, or you get merely photographic accuracy, and what makes a still life interesting is that it's been through a head. You want to emphasize the visual cues that tell you, for example, that the reason the color changes suddenly at a certain point is that it's the edge of an object. By subtly emphasizing such things you can make paintings that are more realistic than photographs not just in some metaphorical sense, but in the strict information-theoretic sense. [4]
    I liked painting still lives because I was curious about what I was seeing. In everyday life, we aren't consciously aware of much we're seeing. Most visual perception is handled by low-level processes that merely tell your brain "that's a water droplet" without telling you details like where the lightest and darkest points are, or "that's a bush" without telling you the shape and position of every leaf. This is a feature of brains, not a bug. In everyday life it would be distracting to notice every leaf on every bush. But when you have to paint something, you have to look more closely, and when you do there's a lot to see. You can still be noticing new things after days of trying to paint something people usually take for granted, just as you can after days of trying to write an essay about something people usually take for granted.
    This is not the only way to paint. I'm not 100% sure it's even a good way to paint. But it seemed a good enough bet to be worth trying.
    Our teacher, professor Ulivi, was a nice guy. He could see I worked hard, and gave me a good grade, which he wrote down in a sort of passport each student had. But the Accademia wasn't teaching me anything except Italian, and my money was running out, so at the end of the first year I went back to the US.





```python

```
