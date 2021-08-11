from os import error
from fastapi.testclient import TestClient
from pathlib import Path
import time
from requests.models import HTTPError
from PIL import Image
from main import app
import cv2 as cv
import numpy as np
from main import is_url_image
import numpy as np
import torch
from celery.result import AsyncResult

client = TestClient(app)
url = "http://127.0.0.1:8000"
TEST_URL = "https://res.cloudinary.com/dydx43zon/image/upload/v1628517482/test_image_cloudinary_wmgoa9.jpg"


test_image_path = Path("./inputs/whole_imgs/00.jpg")
def test_create_upload_file():
    print("Staring test_create_upload_file test...")

    response = client.post('/image_upload/', 
                            files = {"file": (test_image_path.name, 
                            open(test_image_path, 'rb'), 
                            "image/jpeg")}
                            )

    assert response.status_code == 200
    print(response.content)

def sleep(timeout, retry=3):
    def the_real_decorator(function):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < retry:
                try:
                    value = function(*args, **kwargs)
                    if value is None:
                        return True
                except:
                    print(f'Sleeping for {timeout} seconds')
                    time.sleep(timeout)
                    retries += 1
            return False
        return wrapper
    return the_real_decorator
    

@sleep(3, retry =5)
def does_image_exists(url):
    try:
        ret_val = is_url_image(url)
        if ret_val is False:
            print('Image not found. Trying again...')
            raise HTTPError
        
    except HTTPError as e:
        print('HTTP error:{e}')
        raise HTTPError
    else:
        print("Image found in website")

@sleep(10, retry =10)
def is_celery_task_finished(task_id):
    try:
        res = AsyncResult(task_id)
        if res.ready() is False:
            print('Task not ready. Trying again')
            raise ValueError
        
    except ValueError as e:
        print('HTTP error:{e}')
        raise ValueError
    else:
        print("Task finished website")
        
    

#this test is meant to fail to False
def test_does_image_exist():
    print("Starting does_image_exists Test")
    non_existent_image_url = "https://res.cloudinary.com/dydx43zon/image/upload/v1627664136/fucenobitest.jpg"
    exists = does_image_exists(non_existent_image_url)
    assert exists == False
    print("Test test_does_image_exist: Done")

def test_restoration_endpoint(image_path = None):
    from inference_gfpgan_full import restoration_endpoint
    from main import gfpgan, face_helper

    if image_path is None:
        image_path = '/app/GFPGAN/inputs/cropped_faces/Adele_crop.png'

    image = Image.open(image_path)
    m = restoration_endpoint(gfpgan, face_helper, image)
    assert np.ndarray == type(m)

def test_celery_sum():
    from tasks import add
    print('starting celery sum task test')
    a = add.delay(3,3).get()
    assert a == 6
    print(f'finshed test with result {a}')

def test_process_image():
    from main import process_image
    img_link = "https://res.cloudinary.com/dydx43zon/image/upload/v1627639400/test_image_cloudinary.jpg"
    new_img_link = process_image(img_link, img_id='test_image_path')
    return new_img_link


def test_celery_process_image(img_link=None):
    if img_link is None:
        img_link = "https://res.cloudinary.com/dydx43zon/image/upload/v1627639400/test_image_cloudinary.jpg"
    from tasks import call_process_image
    print('Initiating request')
    result = call_process_image.delay(img_link, 'test_image_path')
    return result


def test_inference_with_cloudinary(url=None):
    if url is None:
        url = "https://res.cloudinary.com/dydx43zon/image/upload/v1628517482/test_image_cloudinary_wmgoa9.jpg"
    #Part 1 of test
    #Post a request to the endpoint
    print('posting request to API endpoint')
    response = client.post('/inference_from_cloudinary', params={'img_link': url})
    assert response.status_code == 200
    print("subtest #1 starting.")

    #Part 2 of test 
    #Build URL for checking if image exists in URL as expected
    print("subtest #2 Starting.") 
    img_id = response.json().get('img_id')
    assert len(img_id) >= 9

    task_id = response.json().get('task_id')
    ret_val = is_celery_task_finished(task_id)

    assert  ret_val== True

    check_img_url = img_id+'_enhanced.jpg'
    url_check = url[:url.rfind('/')]
    url_check = url_check + '/' + check_img_url

    print(f"subtest #2.1 Starting. Checking url:{url_check}")
    ret_val = does_image_exists(url_check)
    assert  ret_val == True
    print("Test test_inference_with_cloudinary: Done")


#TODO
def test_multiple_inference_call(num=5, url=None):
    # simulating multiple users by simulating multiple calls to the inference endpoint
    if url is None:
        url = TEST_URL

    task_id_list = []
    img_id_list = []
    success_list = [0 for i in range(num)]
    for i in range(num):
        response = client.post('/inference_from_cloudinary', params={'img_link': url})
        task_id = response.json().get('task_id')
        img_id = response.json().get('img_id')

        task_id_list.append(task_id)
        img_id_list.append(img_id)

    while all(success_list) is False:
        for idx, (succ, task) in enumerate(zip(success_list, task_id_list)):
            if succ == 0:
                response = client.get('/inference_from_cloudinary', params={'task_id': task})
                success_flag = response.json().get('success')
                if (success_flag == 'true') or (success_flag == True):
                    success_list[idx] = 1
                else:
                    pass
            else:
                pass 
        #print(f'Success list: {success_list}')
    assert True
    print('Multiple inference call finished and Passed')
            
            
        



if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        raise ValueError('No CUDA FOUND. Application will not work.')

    print("Starting tests...")
    print("Test #1")
    test_does_image_exist()

    print("Test #2")
    test_celery_sum()
    
    print("Test #3")
    test_celery_process_image()

    print("Test #4")
    test_inference_with_cloudinary()

    print("Test #5")
    test_multiple_inference_call()
    print("Tests have  Passed")

