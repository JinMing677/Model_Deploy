from ocr.PaddleOCR.paddleocr import TextRecognition

model = TextRecognition(model_dir='/home/unitx/CJM/test/output/model')
output = model.predict(input="/home/unitx/CJM/Model_Deploy/ocr/infer/det_vis/roi_2.png")
for res in output:
    res.print()

