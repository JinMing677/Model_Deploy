from ocr.PaddleOCR.paddleocr import TextRecognition

model = TextRecognition(model_dir='/home/unitx/CJM/test/output/model')
output = model.predict(input="/home/unitx/CJM/train_ocr_data/test2.bmp")
for res in output:
    res.print()