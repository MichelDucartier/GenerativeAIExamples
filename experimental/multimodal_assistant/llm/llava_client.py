from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates

import torch

class LlavaClient:
    def __init__(self, model_path):
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, 
                                                                                                   get_model_name_from_path(model_path))
    
    def multimodal_invoke(self, question: str, image: Image, temperature=0.7, top_p = 0.7, num_beams = 5, max_new_tokens=512, conv_mode="llama3"):
        # Process image
        images = [image]
        image_sizes = [x.size for x in images]
        
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)
        
        # Process prompt
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()        
        
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
            )
            
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            processed_outputs = outputs.replace("<|end|>", "").strip()

        return processed_outputs


        