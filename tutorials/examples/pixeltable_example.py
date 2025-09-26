import pixeltable as pxt
import os
import getpass

from pixeltable.functions import openai    

from dotenv import load_dotenv

load_dotenv()


OPENAI_MODEL = "gpt-4.1-mini"
IMAGE_URL = "https://raw.githubusercontent.com/pixeltable/pixeltable/release/docs/resources/images/000000000025.jpg"

# Creating a directory for our tables
pxt.drop_dir('demo', force=True)
pxt.create_dir('demo')

print("Creating a table...")

# Creating a PixelTable
table = pxt.create_table('demo.first', {'input_image': pxt.Image})



### Adding object detection
from pixeltable.functions import huggingface

print("Adding object detection and vision columns...")
# adding ResNet50 model for object detection
table.add_computed_column(
    detections=huggingface.detr_for_object_detection(
        table.input_image,
        model_id='facebook/detr-resnet-50'
    )
)

table.add_computed_column(detections_text=table.detections.label_text)


if "OPENAI_API_KEY" not in os.environ:
    os.environ['OPENAI_API_KEY'] = getpass.getpass("Enter your OpenAI API key: ")

print("Setting up OpenAI vision column...")
table.add_computed_column(
    vision=openai.vision(
        prompt="Describe what's on the image",
        image=table.input_image,
        model=OPENAI_MODEL
    )
)


if __name__ == "__main__":
    
    # inserting the image
    table.insert(input_image=IMAGE_URL)
    breakpoint()
    print(table.select(table.input_image, table.detections_text, table.vision).collect())
