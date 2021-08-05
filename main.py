from fastapi import FastAPI, UploadFile, File
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from requests.models import HTTPError
import torch, uvicorn, cloudinary, skimage, uuid, requests, tempfile
from PIL import Image
from io import BytesIO
from starlette.responses import StreamingResponse
import cloudinary.uploader


#These imports fix the celery tasks circular semantics
import sys
sys.path.append('/app')
from celeryConfig.celery_app import cellapp
sys.path.append('/app/GFPGAN')
from inference_gfpgan_full import restoration_endpoint
from archs.gfpganv1_arch import GFPGANv1
from tasks import call_process_image

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello there Mr. Kenobi"}

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/image_upload/")
async def create_upload_file(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    else:
        image = read_imagefile(await file.read())
        restored_image = restoration_endpoint(gfpgan, face_helper, image, image_name=file.filename) 
        return StreamingResponse(BytesIO(restored_image.tobytes()), media_type="image/png")


@app.post("/inference_from_cloudinary")
async def inference_from_cloudinary(img_link: str):
    # This function is part of the the below pipeline
    # app -> cloudinary upload -> return ref_img_link
    # -> inference_from_cloudinary -> return img_id
    #                              -> do GAN Generation
    #                              -> upload GAN image
    #                              -> Send call back that it has finished 
    
    #Generate 10 digit UUID
    img_id = str(uuid.uuid4())[:10]
    queue_position = 1

    #Queue in task
    result = call_process_image.delay(img_link, img_id)
    return {'img_id': img_id, 'queue_position': queue_position, 'task_id': result.id}

@app.get("/inference_from_cloudinary")
async def inference_from_cloudinary(task_id: str):
    res = cellapp.AsyncResult(task_id)
    if res.ready() and (res.state == 'SUCCESS'):
        img_link = res.get()
        print(f'res.get yields {img_link}')
        response = {'success': True, 'img_link': img_link}
    else:
        #task_info = celery_app.control.inspect().scheduled()
        queue_position = 1 
        response = {'success': False, 'queue_position': queue_position}

    return response

def is_url_image(url):    
    # mimetype,encoding = mimetypes.guess_type(url)
    # return (mimetype and mimetype.startswith('image'))
    image_formats = ("image/png", "image/jpeg", "image/jpg")
    r = requests.head(url)
    if r.headers["content-type"] in image_formats:
        return True
    return False

def process_image(img_link, img_id: str = None):
    #Check if image
    if is_url_image(img_link):
        try:
            image = skimage.io.imread(img_link)
            print('Image Retreived')
        except HTTPError as e:
            response = "Error in retrieving image"
            print(f"{response}: {e}")
            return response
    else:
        response = "Doesn't seem to be an image"
        print(response)
        return response
    print('Sending to restoration endpoint')
    restored_image = restoration_endpoint(gfpgan, face_helper, image)
    print('Restored image received. Starting upload process')
    #Generate 10 digit UUID
    if img_id is None: img_id = str(uuid.uuid4())[:10]

    #Convert to JPEG and 
    tmp = tempfile.NamedTemporaryFile()
    img_file = Image.fromarray(restored_image)
    img_file.save(tmp.name, 'jpeg')
    print('Uploading to cloudinary')
    #Upload to cloudinary
    new_img_link = cloudinary.uploader.upload(tmp.name, public_id=(img_id+'_enhanced'))
    print('Image Finished Processing')
    return new_img_link


#model_path = 'GFPGAN/experiments/pretrained_models/GFPGANv1.pth'   
model_path = '/app/GFPGAN/experiments/pretrained_models/GFPGANv1.pth' 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
upscale_factor = 1

# initialize the GFP-GAN
gfpgan = GFPGANv1(
    out_size=512,
    num_style_feat=512,
    channel_multiplier=1,
    decoder_load_path=None,
    fix_decoder=True,
    # for stylegan decoder
    num_mlp=8,
    input_is_latent=True,
    different_w=True,
    narrow=1,
    sft_half=True)

gfpgan.to(device)
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
gfpgan.load_state_dict(checkpoint['params_ema'])
gfpgan.eval()

# initialize face helper
face_helper = FaceRestoreHelper(
    upscale_factor, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png')

# Initialize cloudinary MediaDB connection
cloudinary.config( 
    cloud_name = "dydx43zon", 
    api_key = "674245955597818", 
    api_secret = "gmQARji10740vpZf_fg60SSMha4" 
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)