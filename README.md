# Idefics2-OCR

Fine-tuned the ```HuggingFaceM4/idefics2-8b``` model on the nielsr/docvqa_1200_examples_donut dataset for document VQA pairs. Checkout Idefics2-OCR on [Hugging Face](https://huggingface.co/smishr-18/Idefics2-OCR).

Find the rest of training details [here](https://docs.google.com/document/d/1BFApjOfvAsCac6oaLkPAIV4clhTpjflV1tU17LapRcU/edit?usp=sharing).
<p align="center">
  <img src="https://github.com/user-attachments/assets/0adba18a-afad-42e8-b9c7-50d4fec2a84a" alt="Image Description" width="600"/>
</p>


## Finetune

Set your wanb token in the .env file as ```WANDB_API```. 

* Install the requirements
```Python
pip install -r requirements.txt
```

* Finetune Idefics
```Python
python3 idefics2.py --wandb True
```
## Run the app
You don't have to finetune for running the app the model is loaded from Hugging Face.
```Python
python3 app.py
```


