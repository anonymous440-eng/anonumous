# -*- coding: utf-8 -*-
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionControlNetPipeline, UniPCMultistepScheduler, ControlNetModel, StableDiffusionXLControlNetPipeline

from ip_adapter import IPAdapterXL
import cv2
from PIL import Image, ImageOps
import numpy as np
import imageio
from transformers import pipeline
from diffusers.utils import load_image
import os

from os.path import join as ospj
import click

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
base_model_path2 = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"
device = "cuda"

# load SDXL pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_path, 
                                             variant="fp16", 
                                             use_safetensors=True, 
                                             torch_dtype=torch.float32).to(device)


pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path2,
    controlnet=controlnet,
    use_safetensors=True,
    torch_dtype=torch.float32,
    add_watermarker=False,
).to(device)

def preprocess(mask):
    height, width = mask.shape
    total_padding = 1024 - width
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding

    padded_mask = cv2.copyMakeBorder(mask, 
                                     top=0, bottom=0, 
                                     left=left_padding, right=right_padding, 
                                     borderType=cv2.BORDER_CONSTANT, 
                                     value=0)
    
    
    cv2.imwrite('padded_mask.png', padded_mask)
    kernel = np.ones((17, 17), np.uint8)

    # 흰색 픽셀을 팽창시키는 작업
    dilated_mask = cv2.dilate(padded_mask, kernel, iterations=1)    
    cv2.imwrite('padded_mask_.png', dilated_mask)
    
    padded_mask = Image.fromarray(dilated_mask)
    
    return padded_mask

def depth(depth_path):
    depth_estimator = pipeline('depth-estimation')
    
    depth = cv2.imread(depth_path, 0)
    depth = cv2.resize(depth, (768, 1024), interpolation=cv2.INTER_AREA)
    
    height, width = depth.shape
    total_padding = 1024 - width
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding

    padded_depth = cv2.copyMakeBorder(depth, 
                                     top=0, bottom=0, 
                                     left=left_padding, right=right_padding, 
                                     borderType=cv2.BORDER_CONSTANT, 
                                     value=0)
    
    
    padded_mask = Image.fromarray(padded_depth)
    
    # Invert the image (black to white, white to black)
    inverted_image = ImageOps.invert(padded_mask)
    
    # Apply depth estimation to the inverted image
    image_depth = depth_estimator(inverted_image)['depth']
    
    # Save the depth image
    image_depth.save("image_depth.png")
    
    return image_depth


def resize_image(img, scale=0.5):
    width, height = img.size
    new_size = (int(width * scale), int(height * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS)

def main(mask_path, sketch_path, depth_path, image_prompt_path, prompt, shape_prompt):
    mask = cv2.imread(mask_path, 0)
#     mask = cv2.bitwise_not(mask)
    cv2.imwrite('mask.png', mask)
    mask = cv2.resize(mask, (768, 1024), interpolation=cv2.INTER_AREA)
    
    sketch = cv2.imread(sketch_path)
    sketch = cv2.resize(sketch, (768, 1024), interpolation=cv2.INTER_AREA)
    cv2.imwrite('sketch.png', sketch)

    target_width = 1024
    padding = (target_width - sketch.shape[1]) // 2  # 좌우에 추가할 픽셀 수

    # 좌우 동일한 비율로 흰색(255) 픽셀 추가
    padded_sketch = cv2.copyMakeBorder(
        sketch,
        top=0,
        bottom=0,
        left=padding,
        right=padding,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]  # 흰색
    )

    padded_image_rgb = cv2.cvtColor(padded_sketch, cv2.COLOR_BGR2RGB)
    sketch = Image.fromarray(padded_image_rgb)
    
    imageio.imsave('./sketch.png', sketch)
    
    mask = preprocess(mask)
    depth_map = depth(depth_path)
    
    ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)


    image = Image.open(image_prompt_path)

    images = ip_model.generate(pil_image=image, image=depth_map, mask_image=mask, num_samples=1, num_inference_steps=50, prompt=prompt, shape_prompt="", scale=0.5, controlnet_conditioning_scale=0.7, guidance_scale=7.5)

    # resized_images = [resize_image(img) for img in images]

    outputs = np.concatenate([np.asarray(img) for img in images], axis=1)
    imageio.imsave(output_path, outputs)

if __name__ == "__main__":

    mask_dir = "./DATA/mask"
    sketch_dir = "../NIPS/DATA_Dresscode/person"
    depth_dir = "../NIPS/DATA_Dresscode/depth"
    image_prompt_dir = "../NIPS/DATA_Dresscode/style_"
    output_dir = "../NIPS/ablation_style_600/w_projection_wo_selective"
    
    prompts = [
             'An upper garment adorned with delicate lace detailing for a touch of elegance',
             'A bold upper garment featuring striking metal clasps as its main decorative element',
             'An upper garment with intricate embroidery that exudes a vintage charm',
             'A sleek upper garment highlighted by subtle crystal embellishments',
             'An upper garment with playful character logos, perfect for a whimsical look',
             'A sophisticated upper garment featuring elegant ribbon accents',
             'An upper garment with a romantic mood, adorned with gentle frills',
             'A simple upper garment with a single, eye-catching button as its focal point',
             'An upper garment with a charming mood, decorated with small, delicate beads',
             'A unique upper garment featuring a combination of pleats and metal clasps',
             'An upper garment with a sleek mood, showcasing minimalistic embroidery',
             'A vintage-inspired upper garment with classic button detailing',
             'An upper garment with a bold mood, featuring prominent crystal embellishments',
             'A romantic upper garment adorned with intricate lace and subtle ribbon accents',
             'An upper garment with a sophisticated mood, highlighted by elegant embroidery',
             'A charming upper garment featuring playful character logos and a single button',
             'An upper garment with a sleek mood, decorated with minimal metal clasps',
             'A unique upper garment with a combination of beads and frills for added texture',
             'An upper garment with a vintage mood, showcasing delicate pleats and lace',
             'A simple upper garment with a single, elegant ribbon as its decorative element',
             'An upper garment with a bold mood, featuring striking embroidery and crystal embellishments',
             'A romantic upper garment adorned with gentle frills and subtle beads',
             'An upper garment with a sophisticated mood, highlighted by intricate lace detailing',
             'A unique upper garment featuring a combination of character logos and metal clasps',
             'An upper garment with a sleek mood, showcasing minimalistic ribbon accents',
             'A vintage-inspired upper garment with classic embroidery and button detailing',
             'An upper garment with a bold mood, featuring prominent beads and frills',
             'A romantic upper garment adorned with delicate lace and subtle crystal embellishments',
             'An upper garment with a sophisticated mood, highlighted by elegant pleats and embroidery',
             'A charming upper garment featuring playful character logos and a single metal clasp',
             'An upper garment with a sleek mood, decorated with minimal beads and ribbon accents',
             'A unique upper garment with a combination of lace and buttons for added texture',
             'An upper garment with a vintage mood, showcasing intricate embroidery and frills',
             'A simple upper garment with a single, eye-catching crystal embellishment as its focal point',
             'An upper garment with a bold mood, featuring striking metal clasps and character logos',
             'A romantic upper garment adorned with gentle pleats and subtle ribbon accents',
             'An upper garment with a sophisticated mood, highlighted by elegant lace and beads',
             'A unique upper garment featuring a combination of embroidery and crystal embellishments',
             'An upper garment with a sleek mood, showcasing minimalistic frills and buttons',
             'A vintage-inspired upper garment with classic character logos and metal clasps',
             'An upper garment with a bold mood, featuring prominent ribbon accents and embroidery',
             'A romantic upper garment adorned with delicate beads and subtle lace detailing',
             'An upper garment with a sophisticated mood, highlighted by intricate pleats and crystal embellishments',
             'A charming upper garment featuring playful character logos and a single ribbon',
             'An upper garment with a sleek mood, decorated with minimal metal clasps and frills',
             'A unique upper garment with a combination of buttons and lace for added texture',
             'An upper garment with a vintage mood, showcasing elegant embroidery and beads',
             'A simple upper garment with a single, eye-catching character logo as its focal point',
             'An upper garment with a bold mood, featuring striking crystal embellishments and pleats',
             'A romantic upper garment adorned with gentle ribbon accents and subtle metal clasps',
             'An upper garment adorned with delicate lace and subtle embroidery for a touch of sophistication',
             'An upper garment features bold metal clasps and a sleek logo design',
             'An upper garment with intricate beadwork that exudes a romantic charm',
             'A simple upper garment highlighted by a single elegant ribbon detail',
             'An upper garment with whimsical character embroidery, perfect for a playful look',
             'An upper garment is embellished with crystal elements, adding a hint of ethereal beauty',
             'An upper garment with vintage-inspired buttons for a classic touch',
             'A charming upper garment featuring frills and a single embroidered logo',
             'An upper garment with a minimalist design, accented by a small metal clasp',
             'An upper garment showcases sophisticated pleats and a subtle lace trim',
             'An upper garment with a bold logo and understated bead accents',
             'A romantic upper garment featuring delicate embroidery and a single ribbon detail',
             'This upper garment is adorned with charming frills and a whimsical character motif',
             'An upper garment with sleek metal clasps and a touch of crystal embellishment',
             'A vintage-inspired upper garment with intricate lace and a single button detail',
             'An upper garment featuring bold embroidery and a subtle logo design',
             'An upper garment is highlighted by elegant pleats and a small metal clasp',
             'An upper garment with a simple design, accented by delicate beadwork',
             'A sophisticated upper garment with crystal embellishments and a single embroidered logo',
             'An upper garment featuring charming frills and a whimsical ribbon detail',
             'An upper garment is adorned with sleek metal clasps and a touch of lace',
             'An upper garment with a romantic mood, highlighted by intricate embroidery',
             'A vintage-inspired upper garment featuring bold buttons and a subtle logo',
             'An upper garment with a minimalist design, accented by a single crystal element',
             'This upper garment showcases elegant pleats and a whimsical character motif',
             'An upper garment with sophisticated beadwork and a small metal clasp',
             'A charming upper garment featuring delicate lace and a single ribbon detail',
             'An upper garment with bold embroidery and understated crystal embellishments',
             'An upper garment is adorned with vintage-inspired buttons and a subtle logo',
             'An upper garment with a sleek design, highlighted by a single metal clasp',
             'A romantic upper garment featuring intricate beadwork and a whimsical character motif',
             'An upper garment with elegant pleats and a touch of lace for a sophisticated look',
             'An upper garment showcases charming frills and a single embroidered logo',
             'An upper garment with bold metal clasps and a subtle ribbon detail',
             'A vintage-inspired upper garment featuring delicate embroidery and a small crystal element',
             'An upper garment with a minimalist design, accented by sleek buttons',
             'An upper garment is adorned with sophisticated beadwork and a whimsical character motif',
             'An upper garment with a romantic mood, highlighted by elegant pleats',
             'A charming upper garment featuring bold embroidery and a single metal clasp',
             'An upper garment with intricate lace and understated crystal embellishments',
             'An upper garment showcases vintage-inspired buttons and a subtle logo',
             'An upper garment with a sleek design, highlighted by a single ribbon detail',
             'A romantic upper garment featuring delicate beadwork and a whimsical character motif',
             'An upper garment with elegant frills and a touch of lace for a sophisticated look',
             'An upper garment is adorned with bold metal clasps and a small embroidered logo',
             'An upper garment with a minimalist design, accented by intricate embroidery',
             'A vintage-inspired upper garment featuring charming pleats and a single crystal element',
             'An upper garment with sophisticated beadwork and a subtle ribbon detail',
             'An upper garment showcases bold buttons and a whimsical character motif',
             'An upper garment with a romantic mood, highlighted by delicate lace and a single metal clasp',
             'An upper garment adorned with intricate lace detailing around the neckline',
             'Upper garment featuring a playful character logo embroidered on the chest',
             'A vintage-inspired upper garment with delicate embroidery along the sleeves',
             'Upper garment with a sleek design, highlighted by subtle crystal embellishments',
             'An upper garment with bold metal clasps that add a touch of sophistication',
             'Upper garment featuring a row of charming buttons down the front',
             'A romantic upper garment with flowy frills cascading from the shoulders',
             'Upper garment showcasing elegant pleats at the cuffs',
             'A unique upper garment adorned with a single, striking ribbon bow',
             'Upper garment featuring ethereal beadwork along the hemline',
             'A sophisticated upper garment with minimalistic logo embroidery on the pocket',
             'Upper garment with a vintage charm, highlighted by intricate lace panels',
             'An upper garment with a whimsical character print on the back',
             'Upper garment featuring delicate crystal embellishments at the collar',
             'A sleek upper garment with a single metal clasp at the neckline',
             'Upper garment adorned with romantic embroidery on the front',
             'An upper garment with bold pleats that add an avant-garde touch',
             'Upper garment featuring charming frills around the waist',
             'A sophisticated upper garment with a small, elegant button closure',
             'Upper garment showcasing ethereal lace detailing on the sleeves',
             'A unique upper garment with a character logo subtly placed on the cuff',
             'Upper garment featuring intricate beadwork along the neckline',
             'An upper garment with vintage-inspired metal clasps on the shoulders',
             'Upper garment adorned with a sleek ribbon tie at the back',
             'A romantic upper garment with frills accentuating the hem',
             'Upper garment featuring bold embroidery across the chest',
             'An upper garment with charming buttons lining the side',
             'Upper garment showcasing elegant crystal embellishments on the cuffs',
             'A sophisticated upper garment with a single, striking lace insert',
             'Upper garment adorned with whimsical character embroidery on the sleeve',
             'An upper garment with ethereal pleats flowing from the neckline',
             'Upper garment featuring delicate ribbon accents on the shoulders',
             'A vintage-inspired upper garment with subtle bead detailing',
             'Upper garment showcasing bold metal clasps for a statement look',
             'An upper garment with sleek embroidery running down the back',
             'Upper garment adorned with charming frills at the collar',
             'A unique upper garment with a small character logo on the hem',
             'Upper garment featuring elegant buttons at the back',
             'An upper garment with romantic crystal embellishments scattered across the front',
             'Upper garment showcasing intricate lace details on the pockets',
             'A sophisticated upper garment with bold pleats at the waist',
             'Upper garment adorned with whimsical ribbon bows on the sleeves',
             'An upper garment with charming bead accents on the neckline',
             'Upper garment featuring sleek metal clasps at the cuffs',
             'A vintage-inspired upper garment with delicate embroidery on the back',
             'Upper garment showcasing ethereal frills cascading down the front',
             'An upper garment with elegant buttons accentuating the shoulders',
             'Upper garment adorned with bold character embroidery on the chest',
             'A romantic upper garment with a single crystal embellishment at the collar',
             'Upper garment featuring intricate pleats that add a touch of sophistication',
             'An upper garment adorned with delicate lace trim and a subtle logo',
             'An upper garment features elegant beadwork alongside charming embroidery',
             'An upper garment with a subtle crystal embellishment',
             'A vintage upper garment showcasing decorative metal clasps and a small emblem',
             'An upper garment comes with minimalistic buttons and an embroidered character',
             'An upper garment with sophisticated ribbon detailing and understated frills',
             'The upper garment is adorned with pleats and a touch of gleaming embroidery',
             'A sleek upper garment with a single decorative button and subtle beading',
             'An upper garment with crystal embellishments and a delicate logo',
             'The upper garment features whimsical embroidery alongside a simple metal clasp',
             'An upper garment, accented with lace and a subtle character',
             'An upper garment, adorned with frilled detailing',
             'This bold upper garment pairs minimal metal clasps with decorative pleats',
             'An upper garment enriched with refined ribbon detail and a small embroidered logo',
             'An elegant embroidery and a touch of beadwork',
             'The upper garment is subtly decorated with crystal embellishments and a gentle lace',
             'This upper garment features a charming character and sophisticated button accents',
             'A romantic upper garment with decorative frills and a single crystal element',
             'The upper garment is designed with whimsical embroidery and sleek metal clasps',
             'An upper garment featuring lace and bead detail',
             'An upper garment with simple buttons and a charming embroidered character',
             'A sleek metal clasps and sophisticated pleats highlight',
             'This upper garment features subtle crystal embellishments and decorative frills',
             'An upper garment with a small logo and refined embroidery adds a sophisticated touch',
             'A bold upper garment with decorative beads and a single whimsical character',
             'An upper garment adorned with minimal lace and a crystal accent',
             'A vintage-inspired upper garment featuring pleats and elegant metal clasps',
             'An upper garment with ribbon accents and subtle embroidery',
             'An upper garment showcases simple button detailing alongside delicate frills',
             'An upper garment with understated lace and a charming embroidered logo',
             'A sleek beadwork and a small character logo accent',
             'An ethereal upper garment with crystal embellishments and minimal pleats',
             'An upper garment featuring a decorative metal clasp',
             'The upper garment is decorated with refined buttons and a touch of embroidery',
             'An upper garment featuring lace trim and a unique logo',
             'A sleek upper garment with sophisticated ribbon detailing and a delicate character',
             'This upper garment is subtly embellished with pleats and a small crystal element',
             'An elegant upper garment pairing frills with minimal metal clasps for a refined look',
             'A romantic upper garment with bead accents and a simple embroidered logo',
             'An upper garment featuring a whimsical character and lace',
             'An upper garment with sophisticated pleating and a decorative button',
             'An upper garment is adorned with crystal embellishments and a charming frill',
             'A sleek upper garment with understated embroidery and a subtle metal clasp',
             'An upper garment featuring a small logo and elegant beadwork for a bold touch',
             "An upper garment's refined ribbon and crystal details",
             'A romantic upper garment enriched with delicate lace and a small embroidered character',
             'This charming upper garment features minimal pleats and a whimsical logo',
             'An ethereal upper garment with bead embellishments and decorative ribbon detailing',
             'A vintage-inspired embroidery and a single crystal accent highlight',
             'A bold sophistication with this upper garment featuring refined buttons and frills',
             'Bottoms adorned with delicate lace trim for a touch of elegance',
             'Bottoms featuring bold embroidery of abstract patterns',
             'Bottoms with a charming ribbon detail at the waist',
             'Bottoms embellished with sparkling crystal accents for a sophisticated look',
             'Bottoms with playful character patches sewn onto the pockets',
             'Bottoms featuring vintage-inspired metal clasps along the sides',
             'Bottoms with a sleek design highlighted by subtle pleats',
             'Bottoms adorned with whimsical beadwork along the hem',
             'Bottoms featuring a romantic frill detail at the cuffs',
             'Bottoms with a bold logo print for a statement look',
             'Bottoms featuring elegant embroidery of floral motifs',
             'Bottoms with a charming button detail down the front',
             'Bottoms adorned with ethereal lace panels for a dreamy effect',
             'Bottoms featuring sophisticated crystal embellishments at the waistband',
             'Bottoms with a sleek metal clasp closure for a modern touch',
             'Bottoms adorned with intricate beadwork in geometric patterns',
             'Bottoms featuring a playful ribbon tie at the ankle',
             'Bottoms with a vintage-inspired frill detail at the hem',
             'Bottoms adorned with bold character prints for a fun twist',
             'Bottoms featuring elegant pleats for a refined look',
             'Bottoms with a charming embroidery of whimsical designs',
             'Bottoms adorned with sophisticated metal clasps for a polished finish',
             'Bottoms featuring romantic lace detailing along the sides',
             'Bottoms with a sleek button detail for a minimalist aesthetic',
             'Bottoms adorned with ethereal crystal accents for a magical touch',
             'Bottoms featuring bold beadwork in contrasting colors',
             'Bottoms with a charming ribbon bow at the waist',
             'Bottoms adorned with vintage-inspired embroidery of intricate patterns',
             'Bottoms featuring playful frill details at the pockets',
             'Bottoms with a sophisticated logo embellishment for a luxe feel',
             'Bottoms adorned with elegant lace appliqués for a delicate finish',
             'Bottoms featuring sleek metal clasps for a contemporary edge',
             'Bottoms with a romantic beadwork design along the hem',
             'Bottoms adorned with bold character embroidery for a unique flair',
             'Bottoms featuring charming pleats for a classic touch',
             'Bottoms with a sophisticated crystal embellishment at the cuffs',
             'Bottoms adorned with ethereal ribbon details for a graceful look',
             'Bottoms featuring vintage-inspired button accents for a nostalgic vibe',
             'Bottoms with a sleek embroidery of minimalist motifs',
             'Bottoms adorned with playful beadwork in whimsical shapes',
             'Bottoms featuring elegant frill details for a feminine touch',
             'Bottoms with a charming metal clasp at the ankle',
             'Bottoms adorned with bold lace inserts for a striking contrast',
             'Bottoms featuring sophisticated embroidery of abstract designs',
             'Bottoms with a romantic ribbon detail at the hem',
             'Bottoms adorned with ethereal crystal embellishments for a luminous effect',
             'Bottoms featuring vintage-inspired beadwork for a timeless appeal',
             'Bottoms with a sleek logo print for a modern statement',
             'Bottoms adorned with charming character patches for a playful look',
             'Bottoms featuring elegant lace detailing for a refined elegance',
             'Bottoms adorned with delicate lace trim for a touch of elegance',
             'Bottoms featuring bold embroidery of abstract patterns',
             'Bottoms with a charming ribbon detail at the waist',
             'Bottoms embellished with subtle crystal accents for a sophisticated look',
             'Bottoms showcasing playful character patches for a whimsical touch',
             'Bottoms with sleek metal clasps adding a modern edge',
             'Bottoms featuring vintage-inspired button details',
             'Bottoms adorned with romantic floral embroidery',
             'Bottoms with ethereal beadwork along the hem',
             'Bottoms featuring a bold logo print for a contemporary vibe',
             'Bottoms with elegant pleats that create a refined silhouette',
             'Bottoms adorned with frills for a playful and charming appearance',
             'Bottoms featuring intricate lace panels for a sophisticated touch',
             'Bottoms with sleek, minimalist metal clasps for a modern look',
             'Bottoms showcasing whimsical character embroidery',
             'Bottoms with a vintage-inspired ribbon detail at the hem',
             'Bottoms adorned with subtle bead embellishments for an elegant finish',
             'Bottoms featuring bold crystal accents for a glamorous touch',
             'Bottoms with charming button details along the sides',
             'Bottoms showcasing romantic lace embroidery',
             'Bottoms with sleek pleats for a sophisticated appearance',
             'Bottoms featuring playful frills at the hem',
             'Bottoms adorned with whimsical character patches for a fun look',
             'Bottoms with elegant metal clasps for a polished finish',
             'Bottoms featuring vintage-inspired embroidery for a timeless appeal',
             'Bottoms with bold ribbon details for a striking effect',
             'Bottoms showcasing charming crystal embellishments',
             'Bottoms with subtle beadwork for a refined touch',
             'Bottoms featuring sleek button accents for a modern edge',
             'Bottoms adorned with romantic pleats for a graceful look',
             'Bottoms with playful character details for a unique twist',
             'Bottoms featuring sophisticated lace trim',
             'Bottoms with bold metal clasps for an edgy appearance',
             'Bottoms showcasing vintage-inspired button embellishments',
             'Bottoms with charming embroidery for a delightful finish',
             'Bottoms featuring elegant ribbon details for a refined look',
             'Bottoms adorned with whimsical beadwork for a playful touch',
             'Bottoms with sleek crystal accents for a chic vibe',
             'Bottoms featuring romantic frills for an enchanting appearance',
             'Bottoms with subtle character embroidery for a unique flair',
             'Bottoms adorned with sophisticated pleats for a polished look',
             'Bottoms featuring playful lace details for a fun twist',
             'Bottoms with bold button accents for a striking effect',
             'Bottoms showcasing charming ribbon embellishments',
             'Bottoms with elegant beadwork for a refined finish',
             'Bottoms featuring whimsical crystal details for a magical touch',
             'Bottoms adorned with vintage-inspired frills for a timeless appeal',
             'Bottoms with sleek character patches for a modern twist',
             'Bottoms featuring sophisticated metal clasps for a polished appearance',
             'Bottoms with romantic embroidery for a graceful finish',
             'Bottoms adorned with delicate lace trim for a touch of elegance',
             'Bottoms featuring bold, crystal embellishments for a dazzling effect',
             'Bottoms with intricate embroidery depicting whimsical characters',
             'Bottoms designed with sleek metal clasps for a modern look',
             'Bottoms subtly detailed with a single row of vintage buttons',
             'Bottoms with charming frills along the hemline',
             'Bottoms showcasing sophisticated pleats and a hint of embroidery',
             'Bottoms featuring romantic ribbon accents along the sides',
             'Bottoms with a playful logo embroidered near the pocket',
             'Bottoms adorned with a single, bold beadwork pattern',
             'Bottoms with ethereal lace details around the ankles',
             'Bottoms featuring a sleek, minimalistic button design',
             'Bottoms with a vintage-inspired ribbon bow at the waist',
             'Bottoms showcasing crystal embellishments for a bold statement',
             'Bottoms with charming pleats and a small embroidered logo',
             'Bottoms adorned with a subtle frill trim for added elegance',
             'Bottoms featuring sophisticated metal clasps and delicate lace',
             'Bottoms with a bold, embroidered character motif',
             'Bottoms designed with simple, yet elegant, bead accents',
             'Bottoms showcasing intricate lace detailing for a romantic touch',
             'Bottoms with a sleek, single metal clasp at the waist',
             'Bottoms adorned with vintage-style buttons for a classic look',
             'Bottoms featuring a whimsical character embroidered near the hem',
             'Bottoms with charming ribbon details enhancing the design',
             'Bottoms with bold crystal embellishments forming a unique pattern',
             'Bottoms showcasing a sophisticated pleat design and a small logo',
             'Bottoms with ethereal embroidery along the side seams',
             'Bottoms featuring delicate lace elements for a vintage appeal',
             'Bottoms adorned with a single, elegant beadwork detail',
             'Bottoms with romantic ribbon accents at the pockets',
             'Bottoms showcasing bold metal clasps as a focal point',
             'Bottoms with charming frills and a subtle embroidered motif',
             'Bottoms featuring sleek, sophisticated button details',
             'Bottoms adorned with intricate lace for a touch of elegance',
             'Bottoms with a whimsical embroidered character on the leg',
             'Bottoms showcasing romantic pleats and a vintage logo',
             'Bottoms featuring crystal embellishments for a dazzling look',
             'Bottoms with ethereal ribbon details along the waistband',
             'Bottoms adorned with a bold beadwork design on one side',
             'Bottoms with charming lace accents at the hemline',
             'Bottoms featuring sophisticated metal clasps and elegant pleats',
             'Bottoms with a vintage-inspired button arrangement',
             'Bottoms showcasing whimsical embroidery patterns',
             'Bottoms adorned with sleek, minimalistic ribbon details',
             'Bottoms featuring bold crystal embellishments near the pockets',
             'Bottoms with charming pleats and a subtle lace trim',
             'Bottoms showcasing an elegant embroidered character motif',
             'Bottoms with sophisticated ribbon details at the cuffs',
             'Bottoms adorned with a single, bold metal clasp for modern appeal',
             'Bottoms featuring delicate beadwork and a vintage button design.',
             'Bottoms adorned with delicate lace trim evoke a romantic charm',
             'Bottoms featuring subtle embroidery designs create an elegant look',
             'Bottoms with vibrant character prints add a playful touch',
             'Bottoms embellished with a single row of crystal studs for a sophisticated flair',
             'Bottoms with metallic buttons offer a vintage inspired aesthetic',
             'Bottoms with an understated beadwork detail exude a sleek sophistication',
             'Bottoms adorned with whimsical embroidery patterns for a fun twist',
             'Bottoms featuring a series of tiny metal clasps convey a bold style',
             'Bottoms with frills along the hem create a charming silhouette',
             'Bottoms decorated with intricate logos add a unique brand statement',
             'Bottoms with sparkling beads strategically placed for an ethereal effect',
             'Bottoms highlighted by ribbon detailing bring an elegant touch',
             'Bottoms featuring subtle pleat accents lend an air of sophistication',
             'Bottoms with oversized buttons create a bold, statement look',
             'Bottoms adorned with delicate crystal embellishments for a dreamy feel',
             'Bottoms featuring playful character badges for a quirky appearance',
             'Bottoms with a row of tasteful metal studs exude an edgy vibe',
             'Bottoms with minimalistic embroidery for an effortlessly elegant touch',
             'Bottoms enhanced with a gentle lace overlay for a vintage essence',
             'Bottoms with whimsical ribbon details create a charming appeal',
             'Bottoms featuring frills accentuating the sides offer a playful edge',
             'Bottoms with metallic logos give a modern, sleek appeal',
             'Bottoms subtly decorated with tiny beads for a sophisticated look',
             'Bottoms with decorative embroidery for an intricate design element',
             'Bottoms adorned with small crystal patterns exude an ethereal aura',
             'Bottoms featuring a playful ribbon bow detail enhance its charm',
             'Bottoms with decorative metal clasps for a bold, edgy accent',
             'Bottoms adorned with delicate frills convey a romantic vibe',
             'Bottoms featuring playful character motifs energize the design',
             'Bottoms with classic embroidery patterns capture a vintage spirit',
             'Bottoms with elegant crystal embellishments enhance their opulence',
             'Bottoms featuring exaggerated buttons create a statement aesthetic',
             'Bottoms with subtle lace insets introducing a charming detail',
             'Bottoms accented with tiny ribbons create a whimsical appearance',
             'Bottoms featuring sleek metal clasps elevate the overall design',
             'Bottoms decorated with gentle frill layers evoke a playful mood',
             'Bottoms with tiny embroidered icons for a personalized touch',
             'Bottoms with crystal buttons exude a sophisticated, modern look',
             'Bottoms accented with a series of beads for an imaginative flair',
             'Bottoms featuring whimsical character embroidery inspire joy',
             'Bottoms with subtle lace enhancements for a graceful element',
             'Bottoms adorned with tasteful ribbon accents for a classic style',
             'Bottoms featuring a single frill detail create a minimalist charm',
             'Bottoms with decorative metal buttons offer a vintage appeal',
             'Bottoms with delicate crystal motifs boast an ethereal beauty',
             'Bottoms decorated with character patches for a playful design',
             'Bottoms with understated pleats provide an elegant fashion statement',
             'Bottoms accented with tiny metal clasps for a chic allure',
             'Bottoms featuring delicate beadwork elements enhance the elegance',
             'Bottoms with whimsical ribbon detailing invite a joyful vibe',
             'A dress adorned with delicate lace trim and subtle crystal embellishments for a touch of elegance',
             'A dress featuring intricate embroidery of floral patterns, exuding a romantic charm',
             'A dress with bold metal clasps and a sleek, modern logo design',
             'A dress decorated with playful frills and whimsical character motifs',
             'A dress with a vintage feel, highlighted by ornate buttons and subtle pleats',
             'A dress showcasing a sophisticated ribbon detail and minimalistic beadwork',
             'A dress with ethereal crystal embellishments and a touch of lace for a dreamy look',
             'A dress featuring charming embroidery and a single, elegant metal clasp',
             'A dress with a bold, contemporary logo and sleek ribbon accents',
             'A dress adorned with delicate beads and a hint of frills for a playful touch',
             'A dress with romantic lace detailing and subtle crystal embellishments',
             'A dress featuring vintage-inspired buttons and intricate embroidery',
             'A dress with a sophisticated metal clasp and minimalistic pleats',
             'A dress showcasing whimsical character motifs and charming ribbon details',
             'A dress with sleek, modern logos and a touch of bold beadwork',
             'A dress adorned with ethereal lace and subtle frills for a dreamy appearance',
             'A dress featuring elegant embroidery and a single, sophisticated button',
             'A dress with playful ribbon accents and whimsical crystal embellishments',
             'A dress showcasing vintage-inspired pleats and charming metal clasps',
             'A dress with bold character motifs and sleek, contemporary beadwork',
             'A dress adorned with delicate lace and a hint of romantic embroidery',
             'A dress featuring sophisticated buttons and subtle crystal embellishments',
             'A dress with whimsical frills and a touch of ethereal ribbon detail',
             'A dress showcasing sleek logos and bold metal clasps for a modern look',
             'A dress with charming beadwork and vintage-inspired embroidery',
             'A dress adorned with elegant lace and a single, sophisticated clasp',
             'A dress featuring playful character motifs and subtle pleats',
             'A dress with sleek ribbon accents and bold crystal embellishments',
             'A dress showcasing ethereal embroidery and charming buttons',
             'A dress with whimsical frills and a touch of sophisticated beadwork',
             'A dress adorned with vintage-inspired lace and sleek metal clasps',
             'A dress featuring romantic ribbon details and subtle character motifs',
             'A dress with bold logos and a hint of elegant embroidery',
             'A dress showcasing charming crystal embellishments and sophisticated pleats',
             'A dress with playful buttons and whimsical lace detailing',
             'A dress adorned with sleek beadwork and a single, bold clasp',
             'A dress featuring ethereal frills and subtle ribbon accents',
             'A dress with vintage-inspired embroidery and charming metal clasps',
             'A dress showcasing elegant character motifs and sophisticated crystal embellishments',
             'A dress with whimsical pleats and a touch of bold lace',
             'A dress adorned with romantic buttons and sleek ribbon details',
             'A dress featuring playful beadwork and subtle embroidery',
             'A dress with sophisticated metal clasps and charming frills',
             'A dress showcasing ethereal lace and a single, elegant logo',
             'A dress with bold crystal embellishments and whimsical character motifs',
             'A dress adorned with sleek pleats and vintage-inspired ribbon accents',
             'A dress featuring charming embroidery and a touch of sophisticated buttons',
             'A dress with playful metal clasps and subtle lace detailing',
             'A dress showcasing elegant beadwork and bold frills',
             'A dress with ethereal character motifs and a single, romantic clasp',
             'A dress adorned with delicate lace trim and subtle crystal embellishments',
             'A dress featuring bold embroidery and a charming ribbon detail',
             'A dress with intricate beadwork and a single metal clasp accent',
             'A dress showcasing elegant frills and a whimsical character motif',
             'A dress with vintage-inspired buttons and a touch of romantic embroidery',
             'A dress highlighted by sleek pleats and a sophisticated logo design',
             'A dress with ethereal lace detailing and a hint of crystal sparkle',
             'A dress featuring a playful ribbon and understated bead accents',
             'A dress with bold metal clasps and a touch of embroidery',
             'A dress adorned with charming frills and a subtle character element',
             'A dress with elegant embroidery and a single button detail',
             'A dress showcasing vintage-inspired lace and a hint of crystal embellishment',
             'A dress with sleek ribbon accents and a sophisticated bead design',
             'A dress featuring romantic pleats and a whimsical logo motif',
             'A dress with ethereal metal clasps and a touch of embroidery',
             'A dress adorned with bold frills and a subtle character accent',
             'A dress with charming buttons and a hint of crystal detailing',
             'A dress featuring elegant lace and a single ribbon element',
             'A dress with vintage-inspired embroidery and a touch of beadwork',
             'A dress showcasing sleek metal clasps and a sophisticated logo design',
             'A dress with romantic frills and a whimsical character motif',
             'A dress adorned with ethereal lace and a hint of crystal embellishment',
             'A dress with bold ribbon accents and a charming bead detail',
             'A dress featuring elegant embroidery and a single metal clasp',
             'A dress with vintage-inspired frills and a touch of character',
             'A dress showcasing sleek buttons and a hint of crystal sparkle',
             'A dress with romantic lace detailing and a sophisticated bead design',
             'A dress adorned with ethereal pleats and a whimsical logo motif',
             'A dress with bold metal clasps and a touch of embroidery',
             'A dress featuring charming frills and a subtle character accent',
             'A dress with elegant buttons and a hint of crystal detailing',
             'A dress showcasing vintage-inspired lace and a single ribbon element',
             'A dress with sleek embroidery and a touch of beadwork',
             'A dress adorned with romantic metal clasps and a sophisticated logo design',
             'A dress featuring ethereal frills and a whimsical character motif',
             'A dress with bold lace detailing and a hint of crystal embellishment',
             'A dress with charming ribbon accents and a single bead detail',
             'A dress showcasing elegant embroidery and a touch of metal clasps',
             'A dress with vintage-inspired frills and a subtle character accent',
             'A dress adorned with sleek buttons and a hint of crystal sparkle',
             'A dress featuring romantic lace and a sophisticated bead design',
             'A dress with ethereal pleats and a whimsical logo motif',
             'A dress showcasing bold metal clasps and a touch of embroidery',
             'A dress with charming frills and a subtle character element',
             'A dress adorned with elegant buttons and a hint of crystal detailing',
             'A dress featuring vintage-inspired lace and a single ribbon accent',
             'A dress with sleek embroidery and a touch of beadwork',
             'A dress showcasing romantic metal clasps and a sophisticated logo design',
             'A dress with ethereal frills and a whimsical character motif',
             'A dress adorned with bold lace detailing and a hint of crystal embellishment',
             'A dress adorned with delicate lace accents creates a romantic allure',
             'The dress features a charming ribbon detail at the waist',
             'A dress is elegantly embellished with subtle embroidery along the neckline',
             'A dress with bold crystal embellishments adds a touch of glamour',
             'The dress includes playful buttons down the front',
             'A dress with intricate beadwork creates a sophisticated look',
             'A dress is adorned with a whimsical character patch on the sleeve',
             'A dress featuring elegant metal clasps at the shoulders',
             'The dress is softly accented with frills at the hemline',
             'A dress with vintage-inspired pleats for a timeless feel',
             'A dress has a sleek ribbon detail that enhances its modern appeal',
             'A dress with delicate embroidery that evokes an ethereal mood',
             'The dress features charming crystal embellishments at the cuffs',
             'A dress with bold logos that makes a statement',
             'A dress includes sophisticated metal clasps along the back',
             'A dress with subtle frills on the sleeves for a playful touch',
             'The dress is adorned with elegant beadwork that catches the eye',
             'A dress with a romantic lace overlay on the bodice',
             'This dress features a unique character motif on the pocket',
             'A dress with sleek buttons that add a contemporary edge',
             'The dress includes vintage-inspired embroidery around the collar',
             'A dress with a whimsical ribbon detail on the shoulder',
             'This dress is accented with bold crystal embellishments for a striking effect',
             'A dress featuring delicate lace along the hemline',
             'The dress includes playful pleats for a charming look',
             'A dress with sophisticated metal clasps at the waist',
             'This dress is adorned with subtle buttons that enhance its elegance',
             'A dress with romantic beadwork that shimmers softly',
             'The dress features a unique character design on the neckline',
             'A dress with sleek ribbon accents that add a modern twist',
             'This dress is embellished with elegant embroidery on the sleeves',
             'A dress with bold crystal embellishments for a glamorous touch',
             'The dress includes charming frills at the neckline',
             'A dress with delicate lace detailing for an ethereal feel',
             'A dress features playful buttons that create a whimsical look',
             'A dress with sophisticated beadwork that adds a touch of luxury',
             'The dress is adorned with a vintage-inspired logo on the chest',
             'A dress with sleek metal clasps that enhance its modern appeal',
             'A dress includes romantic embroidery along the hem',
             'A dress featuring bold ribbon details for a striking contrast',
             'The dress is accented with charming crystal embellishments',
             'A dress with elegant lace inserts that create a sophisticated vibe',
             'This dress is adorned with whimsical character patches on the sleeves',
             'A dress with subtle pleats that evoke a vintage mood',
             'The dress features sleek buttons for a minimalist look',
             'A dress with bold beadwork that adds a dramatic flair',
             'A dress includes delicate embroidery around the neckline for a timeless touch',
             'A dress with charming metal clasps at the front',
             'The dress is adorned with playful frills at the waist',
             'A dress featuring elegant ribbon accents for a romantic finish',
             'A chic dress adorned with minimalist lace detailing for a touch of elegance',
             'Bold dress featuring striking embroidery inspired by art deco patterns',
             'Dress embellished with delicate crystal elements, exuding a romantic charm',
             'A dress with playful buttons resembling dainty flowers for a whimsical look',
             'Vintage-inspired dress with intricate frill details for a nostalgic touch',
             'Elegant dress showcasing subtle ribbon accents along the neckline',
             'Dress highlighted by refined metal clasps, offering a hint of sleek sophistication',
             'Sophisticated dress designed with tasteful pleats and a touch of embroidery',
             'A dreamy dress featuring ethereal beadwork reminiscent of a starlit night',
             'Dress accented with cheerful character motifs for a unique fashion statement',
             'Sleek dress with stylish logos subtly integrated into the design',
             'Charming dress with elegant lace frills that add a playful flair',
             'Dress adorned with understated crystal embellishments for an evening allure',
             'Dress showcasing bold embroidery in a modern geometric pattern',
             'Sophisticated dress with metal clasps that bring a contemporary edge',
             'Romantic dress with graceful ribbon ties at the sleeves',
             'A dress with charming frill details and subtle embroidery, perfect for any occasion',
             'Dress featuring classic button details, evoking a sense of timeless style',
             'Elegant dress with delicate pleats running down the skirt',
             'Dress with imaginative character motifs for an avant-garde appeal',
             'Artistic dress with intricate embroidery inspired by nature’s beauty',
             'Dress adorned with sparkling beads for a touch of festive flair',
             'Sophisticated dress with minimalist lace accents that enhance its elegance',
             'Dress highlighted by unique crystal embellishments, creating an ethereal glow',
             'A dress featuring playful ribbon and button details for a youthful vibe',
             'Elegant dress with understated metal clasps, ideal for a chic look',
             'Romantic dress adorned with exquisite lace detailing throughout',
             'Dress showcasing embroidered logos that add a modern twist',
             'Dress with delicate frill accents that offer a hint of drama',
             'A dress featuring intricate pleats that cascade gracefully',
             'Bold dress with striking character-inspired motifs, perfect for standing out',
             'Dress adorned with elegant beads that shimmer with each movement',
             'Dress featuring subtle ribbon elements for a refined touch',
             'Sophisticated dress enhanced by timeless embroidery',
             'Dress with unique metal clasps that provide a stylish focal point',
             'Artistic dress featuring bold lace overlays for added depth',
             'Dress adorned with delicate crystal details that capture the light beautifully',
             'Romantic dress with playful frill accents that add a whimsical touch',
             'Dress showcasing subtle logo embroidery for a modern yet classic look',
             'Ethereal dress with intricate beadwork that tells a story of elegance',
             'Elegant dress with understated ribbon accents flowing down the sleeves',
             'Dress featuring exquisite lace frills for a vintage allure',
             'Dress accented with tasteful metal clasps, perfect for a sophisticated ensemble',
             'A dress with chic pleats and decorative buttons, offering a dash of charm',
             'Dress with character motifs subtly placed for a creative twist',
             'Sophisticated dress adorned with beads that evoke an air of mystery',
             'Dress with playful embroidery that brings a touch of joy',
             'Bold dress featuring modern crystal embellishments for a dazzling effect',
             'Dress with simple lace detailing for a touch of elegance that suits any event',
             'Dress adorned with intricate frills, creating a romantic silhouette that captivates'
    ]
    
    os.makedirs(output_dir, exist_ok=True)

    # 파일 정렬 및 반복 처리
    mask_files = sorted(os.listdir(mask_dir))
    sketch_files = sorted(os.listdir(sketch_dir))
    depth_files = sorted(os.listdir(depth_dir))
    image_prompt_files = sorted(os.listdir(image_prompt_dir))
    num_files = min(len(mask_files), len(sketch_files), len(depth_files), len(image_prompt_files))
    
    print("prompts: ", len(prompts))

    # main 함수 실행
    for i in range(num_files):
        mask_path = os.path.join(mask_dir, mask_files[i])
        sketch_path = os.path.join(sketch_dir, sketch_files[i])
        depth_path = os.path.join(depth_dir, depth_files[i])
        image_prompt_path = os.path.join(image_prompt_dir, image_prompt_files[i % len(image_prompt_files)])  # 순환

        prompt = prompts[i % len(prompts)]  # 순환
        
        print("============")
        print("num_files: ", num_files)
        print("mask_path: ", mask_path)
        print("sketch_path: ", sketch_path)
        print("depth_path: ", depth_path)
        print("image_prompt_path: ", image_prompt_path)
        print("============")

        output_path = os.path.join(output_dir, f"{i:03d}.png")

        try:
            main(mask_path, sketch_path, depth_path, image_prompt_path, prompt, output_path)
            print(f"Processed: {output_path}")
        except Exception as e:
            print(f"Error processing {i}: {e}")

