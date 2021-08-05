import sys
from time import time
sys.path.append('/app')
from celeryConfig.celery_app import cellapp
import tempfile
from PIL import Image
import cloudinary.uploader


@cellapp.task(name='tasks.add')
def add(x, y):
    return x + y
 
@cellapp.task(name='tasks.call_process_image', )
def call_process_image(img_link, img_id: str):
    from GFPGAN.main import process_image
    new_img_link = process_image(img_link, img_id=img_id)
    return new_img_link