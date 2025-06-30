from duckduckgo_search import DDGS ## DuckDuckGo has changed the Api so we need to update
from fastcore.all import *
from fastai.vision.all import *
import time, json
import matplotlib
import matplotlib.pyplot as plt


# Setup matplotlib
matplotlib.use("TkAgg")
print(matplotlib.get_backend())

# Search images over internet
def search_images(keywords, max_images=200): 
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')

# Search a test bird image
urls = search_images('bird photos', max_images=1)

# Download the test image from ddg
from fastdownload import download_url
dest = './bird.jpg'
download_url(urls[0], dest, show_progress=True)

# Show the test image
im = Image.open(dest)
thumb = im.copy()
thumb.thumbnail((256, 256))

def imshow(img):
    plt.imshow(img)
    plt.axis('off')  
    plt.title("Миниатюра через Matplotlib")
    plt.show()

imshow(thumb)

# Downloads training sets
searches = 'forest', 'bird'
path = Path('bird_or_not')

def prepare_dataset(): 
    for src in searches:
        dest = (path/src)
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f'{src} photo'))
        time.sleep(5)
        resize_images(path/src, max_size=400, dest=path/src)

if any(path.iterdir()):
    print('Dataset already downloaded')
else:
    prepare_dataset()

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print('Failed: ', len(failed))

# Prepare dataset
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)
plt.show()

# Learn the model
learn =vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

# Check our model
is_bird, _, probs = learn.predict(PILImage.create('bird.jpg'))
print(f'This is a: {is_bird}.')
print(f"Probability it's a bird: {probs[0]:.4f}")

print('Done')
