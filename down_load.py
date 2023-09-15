from huggingface_hub import hf_hub_download

# print(hf_hub_download(repo_id='lllyasviel/ControlNet' , filename='models/control__sd15_canny.pth'))
print(hf_hub_download(repo_id='lambdalabs/stable-diffusion-image-conditioned' , filename='sd-clip_vit-l14-img-embed_full.ckpt'))